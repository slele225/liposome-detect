"""A tiny stride-4 backbone conforming to the `Backbone` interface, for tests.

Proves the substitutability claim: any module that sets ``out_channels`` /
``out_stride`` and maps [B, in_chans, H, W] -> [B, C, H/4, W/4] plugs into
`DetectorModel`, the loss, decode, and the harness UNCHANGED. Also makes the
forward/loss/decode tests fast (no HRNet instantiation).
"""

import torch.nn as nn
import torch.nn.functional as F

from src.models.interface import Backbone


class DummyBackbone(Backbone):
    """Two stride-2 convs -> stride-4 feature map with ``out_channels`` channels."""

    def __init__(self, in_chans=2, out_channels=16):
        super().__init__()
        self.out_channels = int(out_channels)
        self.out_stride = 4
        self.conv1 = nn.Conv2d(in_chans, out_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x
