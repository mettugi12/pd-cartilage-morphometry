"""Δ-map post-processing strategies.

Used by callers doing paired/longitudinal differencing — not by
process_one_patient (which only sees one patient).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import register_smoothing


@register_smoothing("border_preserving_gaussian")
def border_preserving_gaussian(template_mesh, values, sigma_mm, subch_mask=None):
    """Gaussian smoothing on template mesh vertices that **does not bleed
    across the subch boundary** — each vertex gets a weighted mean using only
    its k-nearest neighbors that share its subch status.

    Implementation: for each vertex, query k=24 nearest neighbors in the
    template's point cloud, weight by exp(-d²/(2σ²)), zero-weight neighbors
    whose subch status differs.

    Per PDvDESS 2026-05-10 lock: σ=1.5 mm is the headline-metric smoothing.
    """
    pts = np.asarray(template_mesh.points)
    if subch_mask is None:
        subch_mask = np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5
    subch_mask = np.asarray(subch_mask).astype(bool)
    n = len(pts)

    k = 24
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k)
    w = np.exp(-(d ** 2) / (2.0 * sigma_mm * sigma_mm))
    # Mask out neighbors with different subch status
    same_status = subch_mask[idx] == subch_mask[:, None]
    w = w * same_status
    # Mask out neighbors whose value is NaN (e.g. non-subch verts in the input)
    vals = np.where(np.isnan(values), 0.0, values)
    valid = (~np.isnan(values)).astype(np.float32)
    num = (w * vals[idx]).sum(axis=1)
    den = (w * valid[idx]).sum(axis=1)
    out = np.where(den > 0, num / den, np.nan).astype(np.float32)
    # Preserve NaN at non-subch positions (consistent with input convention)
    if not np.isnan(values).any():
        return out
    out = np.where(np.isnan(values), np.nan, out)
    return out
