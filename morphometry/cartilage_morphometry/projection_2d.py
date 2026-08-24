"""
2D unwrapping of per-vertex thickness for visualization / population maps.

Tibia: split medial/lateral via the template's `compartment` field; AP
       normalised per compartment.
Femur: angular unwrap around a SINGLE STRAIGHT axis parallel to ML, anchored
       at the best-fit circle center of the subchondral verts in the (AP, SI)
       plane (delegated to `subregions.compute_femur_template_subregions`).
       Anterior shaft gap is detected (largest empty θ-histogram run) and
       used to anchor arc_norm = 0.

       The legacy per-ML-slice (AP, SI) centroid approach (pre-0.2.0) gave
       ragged 2D borders because the centroid wobbled across slices and the
       per-slice θ-extent was normalised to a global θ_range. The new design
       gives smooth borders and stable cross-slice anatomy.

Coord convention (matches `standardize_plateau_anisotropic`):
  vert[0] = AP    vert[1] = SI    vert[2] = ML
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, distance_transform_edt

from .subregions import compute_femur_template_subregions


def template_thickness_2d_tibia(template_mesh: pv.PolyData,
                                thickness_per_vert: np.ndarray,
                                grid_size: int = 40):
    """Project tibia template thickness onto a 2D plateau grid.

    Returns (grid (rows=AP, cols=ML), ml_extent_mm, ap_extent_mm).
    Cols 0 = medial; cols max = lateral; rows 0 = anterior; rows max = posterior.
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; ML = verts[:, 2]
    subch_prob = np.asarray(template_mesh.point_data["subch_prob"])
    valid = (subch_prob >= 0.5) & np.isfinite(thickness_per_vert)
    comp = np.asarray(template_mesh.point_data["compartment"])
    grid_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    grid_cnt = np.zeros((grid_size, grid_size), dtype=np.int64)
    ml_extent_mm, ap_extent_mm = 0.0, 0.0

    for label, lo, hi in ((1, 0.0, 0.5), (2, 0.5, 1.0)):  # 1=medial 2=lateral
        m = valid & (comp == label)
        if not m.any():
            continue
        ml_min, ml_max = ML[m].min(), ML[m].max()
        ap_min, ap_max = AP[m].min(), AP[m].max()
        ml_r = max(ml_max - ml_min, 1e-6); ap_r = max(ap_max - ap_min, 1e-6)
        ml_n = lo + (ML[m] - ml_min) / ml_r * (hi - lo)
        ap_n = (AP[m] - ap_min) / ap_r
        ml_b = np.clip((ml_n * grid_size).astype(int), 0, grid_size - 1)
        ap_b = np.clip((ap_n * grid_size).astype(int), 0, grid_size - 1)
        np.add.at(grid_sum, (ap_b, ml_b), thickness_per_vert[m])
        np.add.at(grid_cnt, (ap_b, ml_b), 1)
        ml_extent_mm += ml_r
        ap_extent_mm = max(ap_extent_mm, ap_r)

    grid = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)
    return grid, float(ml_extent_mm), float(ap_extent_mm)


def template_thickness_2d_femur(template_mesh: pv.PolyData,
                                thickness_per_vert: np.ndarray,
                                grid_size: int = 40,
                                n_ml_slices: int = 60,
                                subregions=None):
    """Project femur template thickness onto a 2D unwrapped grid.

    Uses the single-axis angular unwrap from `subregions.compute_femur_template_subregions`
    (best-fit circle axis parallel to ML, anchored at the (AP, SI) cylinder
    centre of the subchondral verts). Returns (grid (rows=arc, cols=ML), ml_mm,
    arc_mm).

    rows: ap_norm = 1 - arc_norm (anterior end at the BOTTOM of the array; with
          imshow(origin='upper') anterior renders at the bottom of the image,
          posterior at the top — same convention as the legacy projection).
    cols: ml_norm 0 = medial, 1 = lateral.

    `subregions`: optional pre-computed `FemurSubregions` (saves work when the
    same template is projected many times — pass it in across batch loops).
    """
    if subregions is None:
        subregions = compute_femur_template_subregions(
            template_mesh, n_ml_slices=n_ml_slices,
        )

    valid = (subregions.is_subch & np.isfinite(thickness_per_vert)
             & np.isfinite(subregions.ml_norm) & np.isfinite(subregions.arc_norm))
    if not valid.any():
        return np.full((grid_size, grid_size), np.nan), 0.0, 0.0

    ml = subregions.ml_norm[valid]
    arc = subregions.arc_norm[valid]
    ap_norm = 1.0 - arc  # anterior end at large ap_norm → bottom of grid array
    ml_b = np.clip((ml * grid_size).astype(int), 0, grid_size - 1)
    ap_b = np.clip((ap_norm * grid_size).astype(int), 0, grid_size - 1)

    grid_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    grid_cnt = np.zeros((grid_size, grid_size), dtype=np.int64)
    np.add.at(grid_sum, (ap_b, ml_b), thickness_per_vert[valid])
    np.add.at(grid_cnt, (ap_b, ml_b), 1)
    grid = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)

    # Physical extents for downstream callers
    ML = np.asarray(template_mesh.points)[:, 2]
    sub_idx = np.where(subregions.is_subch)[0]
    ml_r = float(ML[sub_idx].max() - ML[sub_idx].min())
    pts = np.asarray(template_mesh.points)[sub_idx]
    r_pts = np.sqrt((pts[:, 1] - subregions.axis_si) ** 2
                    + (pts[:, 0] - subregions.axis_ap) ** 2)
    arc_mm = subregions.theta_range * float(np.mean(r_pts))
    return grid, ml_r, arc_mm


def patient_thickness_2d_femur(bone_mask: np.ndarray, cart_mask: np.ndarray,
                               spacing: tuple, grid_size: int = 40):
    """PD-vs-DESS-canonical patient-frame 2D unwrap (no template).

    Mirror of `01. Repos/knee-mr-pd2dess/comparison/thickness_map_2d.py`.
    Returns (grid (rows=arc, cols=ML), ml_mm, arc_mm).
    """
    bone_b = bone_mask.astype(bool); cart_b = cart_mask.astype(bool)
    if not bone_b.any() or not cart_b.any():
        return np.full((grid_size, grid_size), np.nan), 0.0, 0.0

    cart_surf = cart_b & ~binary_erosion(cart_b)
    dt_bone = distance_transform_edt(~bone_b, sampling=spacing)
    surf_idx = np.argwhere(cart_surf)
    if len(surf_idx) < 50:
        return np.full((grid_size, grid_size), np.nan), 0.0, 0.0
    h_idx, w_idx, d_idx = surf_idx[:, 0], surf_idx[:, 1], surf_idx[:, 2]
    thick = dt_bone[h_idx, w_idx, d_idx]
    keep = (thick > 0.1) & (thick < 10.0)
    h_idx = h_idx[keep]; w_idx = w_idx[keep]; d_idx = d_idx[keep]; thick = thick[keep]

    n_ml = bone_mask.shape[2]
    h_c = np.full(n_ml, np.nan); w_c = np.full(n_ml, np.nan)
    for d in range(n_ml):
        pts = np.argwhere(bone_b[:, :, d])
        if len(pts) >= 5:
            h_c[d] = pts[:, 0].mean(); w_c[d] = pts[:, 1].mean()
    valid = np.isfinite(h_c)
    if valid.sum() <= 1:
        return np.full((grid_size, grid_size), np.nan), 0.0, 0.0
    idxs = np.arange(n_ml)
    h_c = np.interp(idxs, idxs[valid], h_c[valid])
    w_c = np.interp(idxs, idxs[valid], w_c[valid])

    cart_pts = np.argwhere(cart_b)
    d_min, d_max = cart_pts[:, 2].min(), cart_pts[:, 2].max()
    d_r = max(d_max - d_min, 1)
    d_norm = (d_idx - d_min) / d_r
    ml_extent_mm = d_r * spacing[2]

    # NIfTI axis 0 = AP, axis 1 = SI per template convention; match template's atan2(SI, AP)
    theta = np.arctan2(w_idx - w_c[d_idx], h_idx - h_c[d_idx]) % (2 * np.pi)
    n_bins = 360
    hist, edges = np.histogram(theta, bins=n_bins, range=(0.0, 2 * np.pi))
    is_empty = hist == 0
    if is_empty.any():
        ie2 = np.tile(is_empty.astype(np.int8), 2)
        d2 = np.diff(ie2, prepend=0)
        rs = np.where(d2 == 1)[0]; re = np.where(d2 == -1)[0]
        if len(rs) > len(re):
            re = np.append(re, len(ie2))
        best = int(np.argmax(re - rs))
        arc_start_bin = int((rs[best] + (re - rs)[best]) % n_bins)
        theta_start = float(edges[arc_start_bin])
    else:
        theta_start = 0.0
    theta_shift = (theta - theta_start) % (2 * np.pi)
    theta_range = max(float(theta_shift.max()), 0.01)
    arc_norm = theta_shift / theta_range
    w_norm = 1.0 - arc_norm
    r_pts = np.sqrt((h_idx - h_c[d_idx]) ** 2 + (w_idx - w_c[d_idx]) ** 2)
    arc_extent_mm = float(theta_range * np.mean(r_pts) * spacing[1])

    grid_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    grid_cnt = np.zeros((grid_size, grid_size), dtype=np.int64)
    d_bin = np.clip((d_norm * grid_size).astype(int), 0, grid_size - 1)
    w_bin = np.clip((w_norm * grid_size).astype(int), 0, grid_size - 1)
    np.add.at(grid_sum, (w_bin, d_bin), thick)
    np.add.at(grid_cnt, (w_bin, d_bin), 1)
    grid = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)
    return grid, float(ml_extent_mm), float(arc_extent_mm)
