"""Dataloader over a Prompt-1 generator dataset (image .npy + label .json pairs).

Reads ``datasets/<name>/images/img_NNNNNN.npy`` (shape (2,H,W): 0=protein,
1=lipid) and the matching ``labels/img_NNNNNN.json``, normalizes per channel, and
rasterizes GT heatmap/offset/intensity targets on the fly (``targets.build_targets``).
The custom ``collate`` stacks dense maps and concatenates per-center supervision
arrays across the batch with a batch index, so variable spot counts are handled
without padding.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.train.targets import build_targets


class SyntheticSpotDataset(Dataset):
    """Generator dataset -> (normalized image, dense targets, per-center arrays)."""

    def __init__(self, root, out_stride, indices=None, heatmap_sigma=1.0,
                 d_ref=100.0, w_max=5.0, norm_mean=(0.0, 0.0), norm_std=(1.0, 1.0)):
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.lbl_dir = self.root / 'labels'
        stems = sorted(p.stem for p in self.img_dir.glob('img_*.npy'))
        if not stems:
            raise FileNotFoundError(f"no img_*.npy under {self.img_dir}")
        if indices is not None:
            stems = [stems[i] for i in indices]
        self.stems = stems
        self.out_stride = int(out_stride)
        self.heatmap_sigma = float(heatmap_sigma)
        self.d_ref = float(d_ref)
        self.w_max = float(w_max)
        self.mean = np.asarray(norm_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(norm_std, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = np.load(self.img_dir / f'{stem}.npy').astype(np.float32)   # (2,H,W)
        lbl = json.loads((self.lbl_dir / f'{stem}.json').read_text())
        img = (img - self.mean) / self.std
        H, W = img.shape[1], img.shape[2]
        t = build_targets(lbl['spots'], (H, W), self.out_stride,
                          self.heatmap_sigma, self.d_ref, self.w_max)
        return {
            'image': torch.from_numpy(img),
            'heatmap': torch.from_numpy(t['heatmap']),
            'pos_weight_map': torch.from_numpy(t['pos_weight_map']),
            'iy': torch.from_numpy(t['iy']),
            'ix': torch.from_numpy(t['ix']),
            'offset': torch.from_numpy(t['offset']),
            'lipid': torch.from_numpy(t['lipid']),
            'protein': torch.from_numpy(t['protein']),
            'spots': lbl['spots'],
            'stem': stem,
            'image_hw': (H, W),
        }


def collate(batch):
    """Stack dense maps; concat per-center arrays with a batch index.

    Returns ``(images, targets, meta)``:
      images  : (B, 2, H, W)
      targets : 'heatmap','pos_weight_map' (B,1,h,w); 'bidx','iy','ix' (N,);
                'offset' (N,2); 'lipid','protein' (N,)
      meta    : {'spots': list[list[dict]], 'stems': list[str], 'image_hw': list}
    """
    images = torch.stack([b['image'] for b in batch])
    heatmap = torch.stack([b['heatmap'] for b in batch])
    pos_weight = torch.stack([b['pos_weight_map'] for b in batch])

    bidx, iy, ix, off, lip, pro = [], [], [], [], [], []
    for bi, b in enumerate(batch):
        n = int(b['iy'].shape[0])
        if n == 0:
            continue
        bidx.append(torch.full((n,), bi, dtype=torch.long))
        iy.append(b['iy'])
        ix.append(b['ix'])
        off.append(b['offset'])
        lip.append(b['lipid'])
        pro.append(b['protein'])

    def _cat(lst, empty):
        return torch.cat(lst) if lst else empty

    targets = {
        'heatmap': heatmap,
        'pos_weight_map': pos_weight,
        'bidx': _cat(bidx, torch.zeros(0, dtype=torch.long)),
        'iy': _cat(iy, torch.zeros(0, dtype=torch.long)),
        'ix': _cat(ix, torch.zeros(0, dtype=torch.long)),
        'offset': _cat(off, torch.zeros((0, 2), dtype=torch.float32)),
        'lipid': _cat(lip, torch.zeros(0, dtype=torch.float32)),
        'protein': _cat(pro, torch.zeros(0, dtype=torch.float32)),
    }
    meta = {
        'spots': [b['spots'] for b in batch],
        'stems': [b['stem'] for b in batch],
        'image_hw': [b['image_hw'] for b in batch],
    }
    return images, targets, meta


def targets_to_device(targets, device):
    """Move all target tensors to ``device`` (returns a new dict)."""
    return {k: v.to(device) for k, v in targets.items()}
