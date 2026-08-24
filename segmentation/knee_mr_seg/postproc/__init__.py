"""Segmentation postprocessing — atlas-based filtering and registration primitives.

Moved from knee-mr-atlas as part of the 2026-05-28 refactor:
- atlas_core.py: tibia-anchor registration toolkit (was knee_mr_atlas.core)
- atlas_filter.py: atlas-based soft-tissue postproc filter (was knee_mr_atlas.atlas_filter)

The same atlas_core primitives are reused by cartilage-morphometry's atlas builders.
"""

from .atlas_core import (
    TIBIA_BONE_LABEL_DEFAULT,
    TARGET_ML,
    TARGET_AP,
    side_from_path,
    load_seg_oriented,
    build_tibia_mesh,
    compute_aniso_scale,
    apply_aniso_scale_mesh,
    register_translation_only,
    template_grid,
    affine_for_case,
    voxelize_mask_to_grid,
    make_grid_affine,
    per_case_transform,
)

__all__ = [
    "TIBIA_BONE_LABEL_DEFAULT",
    "TARGET_ML",
    "TARGET_AP",
    "side_from_path",
    "load_seg_oriented",
    "build_tibia_mesh",
    "compute_aniso_scale",
    "apply_aniso_scale_mesh",
    "register_translation_only",
    "template_grid",
    "affine_for_case",
    "voxelize_mask_to_grid",
    "make_grid_affine",
    "per_case_transform",
]
