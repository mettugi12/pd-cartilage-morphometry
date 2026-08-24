"""Per-template-vertex + 2D-projection correlation metrics.

Both inputs are template-vertex arrays masked NaN outside the subch zone
(that's what process_one_patient / our shared_mesh wrappers return).

- `per_vertex_pearson` — vertex-level r over the overlapping subch zone.
- `per_vertex_mae` — mean |Δ| over the same overlap.
- `r_2d_smooth` — v6 manuscript headline. Projects each thickness array onto
  a 40×40 medial/lateral grid, then computes Pearson r after a
  border-preserving Gaussian smooth at σ=1.5 grid cells (NaN cells stay NaN;
  Gaussian weights are renormalised over the valid mask so no value bleed
  crosses the subch border).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyvista as pv
import scipy.ndimage as ndi

from cartilage_morphometry import (
    template_thickness_2d_femur,
    template_thickness_2d_tibia,
)


# ---------------------------------------------------------------------------
# Vertex-level
# ---------------------------------------------------------------------------
def _paired_finite(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], int(m.sum())


def per_vertex_pearson(a, b) -> dict:
    x, y, n = _paired_finite(a, b)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return {"r": float("nan"), "n": n}
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": n}


def per_vertex_mae(a, b) -> dict:
    x, y, n = _paired_finite(a, b)
    if n == 0:
        return {"mae_mm": float("nan"), "n": 0}
    return {"mae_mm": float(np.mean(np.abs(x - y))), "n": n}


# ---------------------------------------------------------------------------
# 2D projection + border-preserving smooth → headline metric
# ---------------------------------------------------------------------------
def _project_2d(template_mesh: pv.PolyData, thickness: np.ndarray, bone_name: str,
                grid_size: int = 40, femur_subregions=None) -> np.ndarray:
    """Library's template→2D projection (grid_size×grid_size medial/lateral grid).
    Returns a (grid_size, grid_size) array, NaN outside the subch zone."""
    if bone_name == "femur":
        grid, _, _ = template_thickness_2d_femur(
            template_mesh, thickness, grid_size=grid_size, subregions=femur_subregions,
        )
        return grid
    if bone_name == "tibia":
        grid, _, _ = template_thickness_2d_tibia(template_mesh, thickness, grid_size=grid_size)
        return grid
    raise ValueError(f"unknown bone_name {bone_name!r}")


def _border_preserving_smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth with NaN-aware renormalisation.

    Convolve (valid * grid_filled) and (valid) separately with the Gaussian,
    then divide. Cells that started NaN are restored to NaN at the end, so
    no value bleed leaks across the subch border.
    """
    arr = np.asarray(grid, dtype=np.float64)
    nan_mask = ~np.isfinite(arr)
    valid = (~nan_mask).astype(np.float64)
    arr_filled = np.where(nan_mask, 0.0, arr)
    num = ndi.gaussian_filter(arr_filled, sigma=sigma, mode="constant", cval=0.0)
    den = ndi.gaussian_filter(valid, sigma=sigma, mode="constant", cval=0.0)
    out = np.full_like(arr, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    out[nan_mask] = np.nan
    return out


def r_2d_smooth(template_mesh: pv.PolyData,
                thickness_a: np.ndarray, thickness_b: np.ndarray,
                bone_name: str, sigma: float = 1.5,
                grid_size: int = 40,
                femur_subregions=None) -> dict:
    """v6 manuscript headline: 2D-projection r after border-preserving Gaussian.

    Returns {r, n_valid, sigma} where n_valid = # of grid cells with finite
    values in both arrays after the smooth. Pass `femur_subregions` (a
    `FemurSubregions`, NOT FemurEcksteinSubregions — the projection uses the
    angular-unwrap subregions) to skip re-fitting on every call.
    """
    grid_a = _project_2d(template_mesh, thickness_a, bone_name,
                         grid_size=grid_size, femur_subregions=femur_subregions)
    grid_b = _project_2d(template_mesh, thickness_b, bone_name,
                         grid_size=grid_size, femur_subregions=femur_subregions)
    sm_a = _border_preserving_smooth(grid_a, sigma)
    sm_b = _border_preserving_smooth(grid_b, sigma)
    x, y, n = _paired_finite(sm_a, sm_b)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return {"r": float("nan"), "n_valid": n, "sigma": sigma}
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n_valid": n, "sigma": sigma}


def icc_2_1(*_args, **_kwargs):
    raise NotImplementedError
