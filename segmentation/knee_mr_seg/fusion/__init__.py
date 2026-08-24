"""knee_mr_seg.fusion — v2 PD dual-plane fusion segmentation.

Slice-to-volume reconstruction (Framing A): two orthogonal thick-slab views
of the DESS_mask (1D rect-kernel slab-blurs along R-L and A-P) are fused
into the original DESS_mask via a small anisotropic-aware ConvNet
(`FusionKNet`). At test time the same net consumes D211 (PD SAG) + D212
(PD COR) softmax predictions resampled to the DESS grid — see
``v2_pipeline.run_dualplane_inference``.

Documented in Obsidian: knee-MR/Applications/pd-dualplane-cartilage.md
"""
from .resample import (
    resample_probs_to_dess_grid,
    save_dess_grid_mask_nifti,
    DESS_FUSION_LABELS,
)
from .net import FusionKNet, build_fusion_net, build_fusion_unet
from .v2_pipeline import v2_process_one_patient, run_dualplane_inference

__all__ = [
    "resample_probs_to_dess_grid",
    "save_dess_grid_mask_nifti",
    "DESS_FUSION_LABELS",
    "FusionKNet",
    "build_fusion_net",
    "build_fusion_unet",  # deprecated alias
    "v2_process_one_patient",
    "run_dualplane_inference",
]
