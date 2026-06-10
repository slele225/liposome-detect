"""The backbone interface + the head-composing detector model.

`Backbone` is the thin contract a feature extractor must satisfy: 2-channel image
in, ONE high-resolution feature map out, with `out_channels` and `out_stride`
declared. `DetectorModel` composes ANY conforming backbone with the four heads, so
swapping the backbone (HRNet today; another tomorrow) needs no change to heads,
loss, decode, or the training harness. A dummy backbone (`dummy.DummyBackbone`)
conforming to this interface is used in tests to prove that.
"""

import abc

import torch.nn as nn

from src.models.heads import HeatmapHead, IntensityHead, OffsetHead


class Backbone(nn.Module, abc.ABC):
    """2-channel image -> single high-res feature map.

    Concrete backbones MUST set the two attributes and implement ``forward``:
        out_channels : int  — channel count C of the returned feature map.
        out_stride   : int  — spatial stride S of the returned feature map.
        forward(x: [B, in_chans, H, W]) -> [B, C, H // S, W // S]
    """

    #: channels of the returned feature map (set by the subclass)
    out_channels: int = 0
    #: spatial stride of the returned feature map (set by the subclass)
    out_stride: int = 1

    @abc.abstractmethod
    def forward(self, x):  # pragma: no cover - interface
        ...


class DetectorModel(nn.Module):
    """Backbone + the four dense heads.

    forward(x: [B, 2, H, W]) -> dict with keys:
        'heatmap' : [B, 1, H/S, W/S]   spot-presence probability in (0,1)
        'offset'  : [B, 2, H/S, W/S]   raw subpixel (dx, dy)
        'lipid'   : [B, 2, H/S, W/S]   channel 0 = mean flux (>0), 1 = log-var
        'protein' : [B, 2, H/S, W/S]   channel 0 = mean flux (>0), 1 = log-var

    ``out_stride`` is taken from the backbone so the harness/decode know the
    map-to-pixel scale.
    """

    def __init__(self, backbone, head_hidden=64, heatmap_prior=0.01,
                 lipid_init_flux=1000.0, protein_init_flux=1000.0):
        super().__init__()
        self.backbone = backbone
        self.out_stride = int(backbone.out_stride)
        c = int(backbone.out_channels)
        self.heatmap_head = HeatmapHead(c, head_hidden, prior=heatmap_prior)
        self.offset_head = OffsetHead(c, head_hidden)
        self.lipid_head = IntensityHead(c, head_hidden, init_flux=lipid_init_flux)
        self.protein_head = IntensityHead(c, head_hidden, init_flux=protein_init_flux)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            'heatmap': self.heatmap_head(feat),
            'offset': self.offset_head(feat),
            'lipid': self.lipid_head(feat),
            'protein': self.protein_head(feat),
        }
