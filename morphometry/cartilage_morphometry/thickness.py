"""PD-vs-DESS canonical per-vertex cartilage thickness.

Per-vertex value = neighborhood max of `distance_transform_edt(~bone) | cart`
over a (2r+1)³ kernel. Vectorised via `maximum_filter`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter


def compute_full_bone_thickness(bone_mesh, bone_mask, cart_mask, spacing,
                                max_thickness_mm: float = 6.0,
                                search_radius_voxels: int = 3) -> np.ndarray:
    """Returns thickness_mm (n_verts,) — 0 where no cart in the neighborhood."""
    n = bone_mesh.n_points
    thickness = np.zeros(n, dtype=np.float32)
    if cart_mask.sum() == 0 or bone_mask.sum() == 0:
        return thickness

    bone_bool = bone_mask.astype(bool)
    cart_bool = cart_mask.astype(bool)
    dt_bone = distance_transform_edt(~bone_bool, sampling=spacing)
    thickness_volume = np.where(cart_bool, dt_bone, 0.0).astype(np.float32)
    sr = int(search_radius_voxels)
    nbhd_max = maximum_filter(thickness_volume, size=2 * sr + 1, mode="constant", cval=0.0)

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    verts_voxel = np.round(bone_mesh.points / spacing_arr).astype(int)
    H, W, D = cart_mask.shape
    np.clip(verts_voxel[:, 0], 0, H - 1, out=verts_voxel[:, 0])
    np.clip(verts_voxel[:, 1], 0, W - 1, out=verts_voxel[:, 1])
    np.clip(verts_voxel[:, 2], 0, D - 1, out=verts_voxel[:, 2])
    sampled = nbhd_max[verts_voxel[:, 0], verts_voxel[:, 1], verts_voxel[:, 2]]
    np.minimum(sampled, max_thickness_mm, out=sampled)
    thickness[:] = sampled.astype(np.float32)
    return thickness
