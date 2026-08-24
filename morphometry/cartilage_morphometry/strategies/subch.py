"""Subchondral classifier strategies.

Each returns (subch_prob, is_subchondral) of shape (n_patient_verts,) where
`subch_prob` is in [0, 1] (NaN-free) and `is_subchondral` is uint8.

PDvDESS 2026-05-21 lock: `tpl_nn` is canonical (preserves denudation
analysis). The others are diagnostic / sensitivity-analysis only.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import register_subch


@register_subch("tpl_nn")
def tpl_nn(aligned_points, template_mesh, cart_mask, spacing,
           bone_mesh, thickness, cfg):
    """KDTree NN to template, sample subch_prob, threshold."""
    template_tree = cKDTree(template_mesh.points)
    _, nn_idx = template_tree.query(aligned_points, k=1)
    subch_prob = np.asarray(template_mesh.point_data["subch_prob"])[nn_idx]
    is_subch = (subch_prob >= cfg.subch_threshold).astype(np.uint8)
    return subch_prob.astype(np.float32), is_subch


@register_subch("cart_hit")
def cart_hit(aligned_points, template_mesh, cart_mask, spacing,
             bone_mesh, thickness, cfg):
    """Subch ≡ vertex has thickness > 0. Inflates Δ magnitudes but loses
    denudation analysis (denuded template positions get IDW-filled from
    nearby cart-bearing verts). Diagnostic only.
    """
    is_subch = (thickness > 0).astype(np.uint8)
    return is_subch.astype(np.float32), is_subch


@register_subch("cart_prox")
def cart_prox(aligned_points, template_mesh, cart_mask, spacing,
              bone_mesh, thickness, cfg):
    """Subch ≡ patient vert within cfg.cart_prox_mm of any cart voxel.
    Computed in PATIENT physical-mm frame (uses bone_mesh.points, not
    aligned_points), independent of registration. Useful sensitivity
    analysis: does the subch zone definition matter independently of the
    template?
    """
    cart_idx = np.argwhere(cart_mask.astype(bool))
    if len(cart_idx) == 0:
        zeros = np.zeros(bone_mesh.n_points, dtype=np.uint8)
        return zeros.astype(np.float32), zeros
    cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
    cart_tree = cKDTree(cart_mm)
    d, _ = cart_tree.query(np.asarray(bone_mesh.points), k=1)
    is_subch = (d <= cfg.cart_prox_mm).astype(np.uint8)
    return is_subch.astype(np.float32), is_subch
