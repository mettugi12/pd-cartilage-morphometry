"""Fusion net for v2 PD dual-plane segmentation (Framing A — slab-deblur SVR).

Re-framed as slice-to-volume reconstruction: the input 8 channels are
slab-blurred views of a single underlying 4-class DESS mask, two orthogonal
planes (SAG slab along R-L, COR slab along A-P). The net's job is to invert
the slab-blur forward operator. Receptive field need only span the slab
thickness (~3 mm = ~4 DESS voxels along the blurry axis) plus a few voxels of
cleanup context, so multi-scale features are unnecessary.

``FusionKNet`` (default, ~150K params) routes the 8 input channels through
two slab-axis-aware branches with anisotropic 3D kernels, then concatenates
and applies isotropic fusion convs.

Input layout (post-resample, DESS grid in PIR axcodes):
    DESS axis 0 = P (anatomical A-P), axis 1 = I (S-I), axis 2 = R (R-L).
    channels 0..3 — SAG slab-blur of (FB, FC, TB, TC). Blurry axis = 2 (R-L).
    channels 4..7 — COR slab-blur of (FB, FC, TB, TC). Blurry axis = 0 (A-P).

Per-branch kernel shapes are chosen so each branch does NOT mix info across
its own blurry axis (kernel = 1 along that axis), and IS allowed to mix
along the two sharp axes (kernel = 3).

References
----------
Kasten et al. *End-to-End CNN for 3D Reconstruction of Knee Bones from
Bi-Planar X-Ray Images.* MICCAI 2020.
Rousseau et al. *A non-local approach for image super-resolution using
intermodality priors.* MedIA 2010 (SVR origin).
Gholipour et al. *Robust super-resolution volume reconstruction from slice
acquisitions: application to fetal brain MRI.* IEEE TMI 2010.
Ebner et al. *An automated framework for localization, segmentation and
super-resolution reconstruction of fetal brain MRI.* NeuroImage 2020 (NiftyMIC).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class AnisoConvBlock(nn.Module):
    """Conv3d + GN + LeakyReLU with anisotropic kernel + residual skip.

    Kernel shape encodes which axis is the "slab" axis (kernel=1) and which
    are the sharp in-plane axes (kernel=3). Padding matches.
    """

    def __init__(self, in_c: int, out_c: int, kernel: tuple[int, int, int]):
        super().__init__()
        pad = tuple(k // 2 for k in kernel)
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False)
        self.norm = _gn(out_c)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        # Residual skip: 1x1x1 conv if channel count changes, else identity.
        self.skip = (nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
                     if in_c != out_c else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)) + self.skip(x))


class IsoConvBlock(nn.Module):
    """Isotropic 3x3x3 conv block with residual skip — used in the fusion head."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm = _gn(out_c)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = (nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
                     if in_c != out_c else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)) + self.skip(x))


class FusionKNet(nn.Module):
    """Anisotropic per-plane branches + isotropic fusion head.

    DESS grid axcodes = PIR: axis 0 = P (A-P), axis 1 = I (S-I), axis 2 = R (R-L).

    SAG branch kernel (3, 3, 1): mixes axes 0 (P) and 1 (I); preserves axis 2 (R-L slab).
    COR branch kernel (1, 3, 3): mixes axes 1 (I) and 2 (R); preserves axis 0 (A-P slab).

    With 3 stacked anisotropic convs per branch the effective receptive field
    is ~5 voxels on each sharp axis, and the slab axis is touched only by the
    isotropic fusion convs (which add ~5 voxels there too). That matches the
    physical scale we care about: ~3 mm cartilage thickness ≈ 4 voxels at
    0.7 mm DESS spacing.
    """

    def __init__(self, in_channels: int = 8, out_classes: int = 5,
                 branch_channels: int = 32, fusion_channels: int = 64,
                 n_branch_blocks: int = 3, n_fusion_blocks: int = 2):
        super().__init__()
        assert in_channels == 8, "FusionKNet expects 8 channels (4 SAG + 4 COR)"

        # SAG branch — kernel preserves R-L slab axis (axis 2)
        sag_blocks: list[nn.Module] = []
        prev = 4
        for _ in range(n_branch_blocks):
            sag_blocks.append(AnisoConvBlock(prev, branch_channels, kernel=(3, 3, 1)))
            prev = branch_channels
        self.sag_branch = nn.Sequential(*sag_blocks)

        # COR branch — kernel preserves A-P slab axis (axis 0)
        cor_blocks: list[nn.Module] = []
        prev = 4
        for _ in range(n_branch_blocks):
            cor_blocks.append(AnisoConvBlock(prev, branch_channels, kernel=(1, 3, 3)))
            prev = branch_channels
        self.cor_branch = nn.Sequential(*cor_blocks)

        # Fusion head — isotropic 3x3x3 convs on concatenated features.
        fusion_blocks: list[nn.Module] = []
        prev = 2 * branch_channels
        for i in range(n_fusion_blocks):
            fusion_blocks.append(IsoConvBlock(prev, fusion_channels))
            prev = fusion_channels
        self.fusion = nn.Sequential(*fusion_blocks)

        # 5-class logits head
        self.head = nn.Conv3d(fusion_channels, out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sag = self.sag_branch(x[:, 0:4])
        cor = self.cor_branch(x[:, 4:8])
        merged = torch.cat([sag, cor], dim=1)
        feat = self.fusion(merged)
        return self.head(feat)


def build_fusion_net(in_channels: int = 8, out_classes: int = 5,
                     branch_channels: int = 32, fusion_channels: int = 64,
                     n_branch_blocks: int = 3, n_fusion_blocks: int = 2) -> FusionKNet:
    """Factory for `FusionKNet`."""
    return FusionKNet(
        in_channels=in_channels, out_classes=out_classes,
        branch_channels=branch_channels, fusion_channels=fusion_channels,
        n_branch_blocks=n_branch_blocks, n_fusion_blocks=n_fusion_blocks,
    )


# Backwards-compatible alias — callers that still say build_fusion_unet get
# the K-Net. The old name lingers in train.py / v2_pipeline.py / etc. until
# their factory swap; remove this alias after.
def build_fusion_unet(*args, **kwargs):
    return build_fusion_net(*args, **kwargs)
