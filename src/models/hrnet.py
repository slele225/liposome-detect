"""HRNet backbone (timm) behind the `Backbone` interface.

HRNet is chosen for its high-resolution-preserving parallel branches — the small
liposome tail is where the lipid channel is faintest and localization is hardest,
so we keep a stride-4 high-res feature rather than a deep low-res one.

We take a SINGLE timm feature level (default the stride-4 branch, ``out_index=1``)
via ``features_only`` and a 2-channel input stem (``in_chans=2``, no pretrained
weights — the two channels are protein+lipid, not RGB). ``out_channels`` /
``out_stride`` are read from timm's ``feature_info`` so the heads/harness adapt to
any HRNet variant.
"""

import timm

from src.models.interface import Backbone


class HRNetBackbone(Backbone):
    """timm HRNet feature extractor returning one high-res feature map.

    Args:
        variant: timm model name (e.g. 'hrnet_w18_small_v2', 'hrnet_w18',
            'hrnet_w32').
        out_index: which timm feature level to return. With timm HRNet the levels
            are strides [2, 4, 8, 16, 32]; index 1 = stride 4 (CenterNet default).
        in_chans: input channels (2 = protein + lipid).
        pretrained: load ImageNet weights (default False; the stem is 2-channel).
    """

    def __init__(self, variant='hrnet_w18_small_v2', out_index=1, in_chans=2,
                 pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            variant, pretrained=pretrained, features_only=True,
            out_indices=(out_index,), in_chans=in_chans)
        self.out_channels = int(self.model.feature_info.channels()[0])
        self.out_stride = int(self.model.feature_info.reduction()[0])
        self.variant = variant

    def forward(self, x):
        return self.model(x)[0]
