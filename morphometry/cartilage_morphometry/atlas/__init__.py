"""Tibia-anchored atlas builders for knee MR.

Builds population probability maps and templates in the canonical tibia-anchor
space (TARGET_ML=75mm, TARGET_AP=55mm). Output goes to E:/KneeMR/Datasets/Bone-Atlas/.

Modules:
- softtissue: 7-class soft-tissue probability atlas from PD nnU-Net Dataset204 preds
- cartilage_atlas: cartilage thickness atlas from DESS 4-class preds (KLG=0 cohort)
- patella_template: patella + patella-cart subchondral template

All three import the underlying registration toolkit from
`knee_mr_seg.postproc` (atlas_core: build_tibia_mesh, compute_aniso_scale,
register_translation_only, voxelize_mask_to_grid, ...). The same primitives
are reused by knee_mr_seg.postproc.atlas_filter at inference time.

Moved from knee-mr-atlas/scripts/build_*.py in the 2026-05-28 refactor.
"""
