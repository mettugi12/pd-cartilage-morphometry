"""Cleanup strategies for the post-densify (bone_mask, cart_mask) pair.

Each cleanup function takes (bone_mask, cart_mask, spacing, cfg) and returns
(bone_mask_out, cart_mask_out, meta_dict). Functions may modify the cart
mask, the bone mask, or both. `meta` records what was removed so the test
harness can quantify the cleanup's effect.

The bone-side `keep_largest_bone_components` is applied unconditionally on
the seg label array BEFORE densification (in pipeline.get_patient_masks) —
not via this registry — because spurious bone islands are always wrong (PD
seg of the knee has exactly one connected component per bone label).
Configurable strategies here operate on the cart mask, where the right
behavior is ambiguous (tibial cart is anatomically two CCs).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

from . import register_cleanup


@register_cleanup("drop_small_components")
def drop_small_components(bone_mask, cart_mask, spacing, cfg):
    """Drop cart connected components smaller than `min_size_vox` (default 100).

    Cart cannot use "keep largest": tibial cart is anatomically two components
    (medial + lateral plateaus, separated by intercondylar eminence).

    PDvDESS 2026-05-21: on knee 9226021_RIGHT_femur, 28 cart voxels at 79 mm
    from bone inflated AP bbox 38→108 mm and dropped sf_y to 0.535 (1.87× AP
    compression), pushing rigid ICP into a 42.6° local minimum (ASSD 6.67 mm).
    Filtering at 100-vox restored sf_y to 1.058.
    """
    min_size_vox = 100   # tunable via subclass if needed
    if cart_mask.sum() == 0:
        return bone_mask, cart_mask, {"cart_voxels_removed": 0, "cart_components_kept": 0}
    cc, n = cc_label(cart_mask)
    if n <= 1:
        return bone_mask, cart_mask, {"cart_voxels_removed": 0, "cart_components_kept": int(n)}
    sizes = np.bincount(cc.ravel())[1:]
    keep_labels = np.where(sizes >= min_size_vox)[0] + 1
    if len(keep_labels) == 0:
        keep_labels = np.array([int(np.argmax(sizes)) + 1])
    out = np.isin(cc, keep_labels)
    removed = int(cart_mask.sum() - out.sum())
    return bone_mask, out, {
        "cart_voxels_removed": removed,
        "cart_components_kept": int(len(keep_labels)),
        "cart_components_total": int(n),
    }


@register_cleanup("filter_near_bone_per_slice")
def filter_near_bone_per_slice(bone_mask, cart_mask, spacing, cfg):
    """Drop cart voxels whose 2D distance to bone in their own sagittal slice
    (axis=2 = ML for OAI-conv NIfTI shape (H, W, D)) exceeds r_mm (default 6).

    Stacks on top of `drop_small_components` to clean cart stragglers that
    survive 3D CC analysis because they're connected via slice adjacency to
    the main slab but spatially far from bone within their own slice.
    """
    r_mm = 6.0
    slice_axis = 2
    if cart_mask.sum() == 0:
        return bone_mask, cart_mask, {"cart_voxels_removed_near_bone_filter": 0}
    out = np.zeros_like(cart_mask, dtype=bool)
    n_slices = cart_mask.shape[slice_axis]
    in_plane_axes = [a for a in range(3) if a != slice_axis]
    in_plane_spacing = (spacing[in_plane_axes[0]], spacing[in_plane_axes[1]])
    for k in range(n_slices):
        c2d = np.take(cart_mask, k, axis=slice_axis)
        b2d = np.take(bone_mask, k, axis=slice_axis)
        if not c2d.any() or not b2d.any():
            continue
        dt_to_bone = distance_transform_edt(~b2d, sampling=in_plane_spacing)
        keep = c2d & (dt_to_bone <= r_mm)
        idx = [slice(None), slice(None), slice(None)]
        idx[slice_axis] = k
        out[tuple(idx)] = keep
    removed = int(cart_mask.sum() - out.sum())
    return bone_mask, out, {"cart_voxels_removed_near_bone_filter": removed}


@register_cleanup("keep_largest_components")
def keep_largest_components(bone_mask, cart_mask, spacing, cfg):
    """Keep only the largest cart CC. NOT recommended for tibial cart (two
    anatomical CCs) — provided for completeness and femur-only experiments.
    """
    if cart_mask.sum() == 0:
        return bone_mask, cart_mask, {"cart_voxels_removed": 0}
    cc, n = cc_label(cart_mask)
    if n <= 1:
        return bone_mask, cart_mask, {"cart_voxels_removed": 0}
    sizes = np.bincount(cc.ravel())[1:]
    biggest = int(np.argmax(sizes)) + 1
    out = cc == biggest
    removed = int(cart_mask.sum() - out.sum())
    return bone_mask, out, {"cart_voxels_removed": removed, "cart_components_total": int(n)}
