"""Per-patient baseline 2D-grid cartilage projection (the v3.3 PD-DESS method).

OPT-IN alternative to the atlas-template remap. Selected via
`PipelineConfig.region_projection == "baseline_grid"`. The template path is the
default and is untouched — the web-application pipeline is unaffected by this
module.

Method (restored verbatim from the v3.3 PD-vs-DESS study,
`Studies/PD-vs-DESS/v3.3/analysis/rerun_thickness_mesh.py`): instead of mapping
per-vertex thickness onto a fixed cross-patient atlas template, each knee is
parameterised by a 40x40 medial-lateral (D) x anterior-posterior (W) grid built
from the patient's OWN baseline (00m) bone geometry, and both timepoints are
projected onto that same intrinsic grid. Regional means (cMF, MT, MFTC, ...) are
read off the grid with zero-imputation over the baseline footprint.

Why it exists: in the longitudinal PD-vs-DESS validation the baseline-grid
projection recovers substantially better per-knee agreement with manual QCart
than the atlas-template remap (esp. medial tibia: r~0.47 vs ~0.38, MT SRM ~-1.10
vs ~-0.76), because it avoids the cross-patient template-registration variance on
the flat tibial plate. See Studies/PD-vs-DESS v8 / v8b.

All geometry/binning is normalised (d_norm, w_norm in [0,1]) so it is invariant
to absolute voxel size; the mm->voxel conversion in `project_vertices_to_2d`
must use the SAME spacing the masks were defined at.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

GRID = 40
MIN_THICK_MM = 0.1
MAX_THICK_MM = 10.0

# Region slices on the 40x40 grid (D = medial..lateral rows, W = a-p cols).
D_MED = slice(0, GRID // 2)
D_LAT = slice(GRID // 2, GRID)
W_ANT = slice(0, GRID // 3)
W_CEN = slice(GRID // 3, 2 * GRID // 3)
W_POS = slice(2 * GRID // 3, GRID)
W_WB = slice(int(0.2 * GRID), int(0.8 * GRID))   # central 60% weight-bearing

FC_REGIONS = [("cMF", D_MED, W_WB), ("cLF", D_LAT, W_WB)]
TC_REGIONS = [
    ("MT", D_MED, slice(0, GRID)),
    ("LT", D_LAT, slice(0, GRID)),
    ("aMT", D_MED, W_ANT), ("cMT", D_MED, W_CEN), ("pMT", D_MED, W_POS),
    ("aLT", D_LAT, W_ANT), ("cLT", D_LAT, W_CEN), ("pLT", D_LAT, W_POS),
]
# bone_name -> (compartment tag, region list)
BONE_TO_COMP = {"femur": ("FC", FC_REGIONS), "tibia": ("TC", TC_REGIONS)}


def outer_surface_vox(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask)


def get_bone_centroids(bone_3d: np.ndarray):
    """Per-D-slice (H, W) centroid of the bone mask, NaN-interpolated over gaps."""
    D = bone_3d.shape[0]
    hc = np.full(D, np.nan)
    wc = np.full(D, np.nan)
    for d in range(D):
        pts = np.argwhere(bone_3d[d])
        if len(pts) >= 5:
            hc[d] = pts[:, 0].mean()
            wc[d] = pts[:, 1].mean()
    valid = np.isfinite(hc)
    if valid.sum() > 1:
        idxs = np.arange(D)
        hc = np.interp(idxs, idxs[valid], hc[valid])
        wc = np.interp(idxs, idxs[valid], wc[valid])
    return hc, wc


def fit_circle_2d(x: np.ndarray, y: np.ndarray):
    """Algebraic best-fit circle to 2D points (x, y) -> (a, b, r). Same as
    subregions.fit_circle_2d (copied to keep this module dependency-light)."""
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x, dtype=float)])
    rhs = x.astype(float) ** 2 + y.astype(float) ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = sol
    return float(a), float(b), float(np.sqrt(max(c + a ** 2 + b ** 2, 0.0)))


def _theta_gap_start(theta: np.ndarray, n_ab: int = 360) -> float:
    """Anterior-gap anchor: start angle of the largest empty θ-histogram run."""
    hist_t, edges_t = np.histogram(theta, bins=n_ab, range=(0.0, 2 * np.pi))
    is_empty = hist_t == 0
    if not is_empty.any():
        return 0.0
    ie2 = np.tile(is_empty.astype(np.int8), 2)
    d2 = np.diff(ie2, prepend=0)
    rs = np.where(d2 == 1)[0]; re_ = np.where(d2 == -1)[0]
    if len(rs) > len(re_):
        re_ = np.append(re_, len(ie2))
    rl = re_ - rs
    best = int(np.argmax(rl))
    return float(edges_t[int((rs[best] + rl[best]) % n_ab)])


def compute_ref_geometry(ref_cart: np.ndarray, ref_bone: np.ndarray,
                         compartment: str, laterality: str,
                         femur_unwrap: str = "per_slice") -> dict:
    """Build the per-patient grid geometry from the BASELINE (00m) masks.

    `compartment`: "FC" (femur) or "TC" (tibia, medial/lateral split + per-
    compartment W extents). `femur_unwrap` (FC only):
      "per_slice"       v3.3 default — angular arc around the PER-ML-SLICE bone
                        centroid (centroid wobbles slice-to-slice).
      "best_fit_circle" web-app style — single straight ML axis at the best-fit
                        circle centre of the cartilage surface in the (SI, AP)
                        plane; smoother arc, no per-slice wobble.
    `laterality`: "right_oriented" (all knees, post LEFT->R flip) / "left_native".
    """
    cart = ref_cart.astype(bool)
    bone = ref_bone.astype(bool)
    cart_pts = np.argwhere(cart)
    geo = {"compartment": compartment, "laterality": laterality,
           "femur_unwrap": femur_unwrap}
    geo["d_min"] = int(cart_pts[:, 0].min())
    geo["d_max"] = int(cart_pts[:, 0].max())
    geo["d_range"] = max(geo["d_max"] - geo["d_min"], 1)

    if compartment == "FC":
        geo["h_c"], geo["w_c"] = get_bone_centroids(bone)  # per-slice (always)
        surf = outer_surface_vox(cart)
        coords = np.argwhere(surf)
        if len(coords) == 0:
            geo["theta_start"] = 0.0
            geo["theta_range"] = 2 * np.pi
            geo["h_c_fit"] = geo["w_c_fit"] = 0.0
            return geo
        d_idx, h_idx, w_idx = coords[:, 0], coords[:, 1], coords[:, 2]
        if femur_unwrap == "best_fit_circle":
            # fit ONE circle to the cartilage surface in (AP=w, SI=h); single axis
            w_c, h_c, _r = fit_circle_2d(w_idx.astype(float), h_idx.astype(float))
            geo["h_c_fit"], geo["w_c_fit"] = h_c, w_c
            theta = np.arctan2(h_idx - h_c, w_idx - w_c) % (2 * np.pi)
        else:
            geo["h_c_fit"] = geo["w_c_fit"] = 0.0
            theta = np.arctan2(h_idx - geo["h_c"][d_idx], w_idx - geo["w_c"][d_idx]) % (2 * np.pi)
        geo["theta_start"] = _theta_gap_start(theta)
        ts = (theta - geo["theta_start"]) % (2 * np.pi)
        geo["theta_range"] = max(float(ts.max()), 0.01)
    else:
        d_norm_raw = (cart_pts[:, 0] - geo["d_min"]) / geo["d_range"]
        d_norm_full = d_norm_raw if laterality == "right_oriented" else 1.0 - d_norm_raw
        nbins = max(GRID * 2, 40)
        hist, _ = np.histogram(d_norm_full, bins=nbins, range=(0.0, 1.0))
        lo, hi = nbins // 4, 3 * nbins // 4
        sb = lo + int(np.argmin(hist[lo:hi]))
        geo["split_norm"] = (sb + 0.5) / nbins
        med_w = cart_pts[d_norm_full <= geo["split_norm"], 2]
        lat_w = cart_pts[d_norm_full > geo["split_norm"], 2]
        geo["med_w_min"] = int(med_w.min()) if len(med_w) else 0
        geo["med_w_range"] = max(int(med_w.max()) - geo["med_w_min"], 1) if len(med_w) else 1
        geo["lat_w_min"] = int(lat_w.min()) if len(lat_w) else 0
        geo["lat_w_range"] = max(int(lat_w.max()) - geo["lat_w_min"], 1) if len(lat_w) else 1
    return geo


def vertex_norm_coords(vertex_mm: np.ndarray, thickness_mm: np.ndarray,
                       geo: dict, spacing):
    """Per-vertex normalised grid coords (d_norm, w_norm in [0,1]) + thickness, for
    valid (cartilage-bearing) vertices. This is the continuous form underlying
    `project_vertices_to_2d` — use it for HIGH-RESOLUTION / interpolated rendering
    (the 40x40 binning is only for regional means). Returns (d_norm, w_norm, t)."""
    empty = (np.zeros(0), np.zeros(0), np.zeros(0))
    if len(vertex_mm) == 0:
        return empty
    sz, sy, sx = spacing
    v = np.asarray(vertex_mm)
    d_vox = v[:, 0] / sz; h_vox = v[:, 1] / sy; w_vox = v[:, 2] / sx
    thickness_mm = np.asarray(thickness_mm)
    valid = (thickness_mm > MIN_THICK_MM) & (thickness_mm < MAX_THICK_MM)
    d_vox, h_vox, w_vox, t = d_vox[valid], h_vox[valid], w_vox[valid], thickness_mm[valid]
    if len(t) < 10:
        return empty

    d_norm_raw = (d_vox - geo["d_min"]) / geo["d_range"]
    d_norm_full = d_norm_raw if geo["laterality"] == "right_oriented" else 1.0 - d_norm_raw

    if geo["compartment"] == "FC":
        if geo.get("femur_unwrap") == "best_fit_circle":
            theta = np.arctan2(h_vox - geo["h_c_fit"], w_vox - geo["w_c_fit"]) % (2 * np.pi)
        else:
            hc, wc = geo["h_c"], geo["w_c"]
            d_c = np.clip(d_vox.astype(int), 0, len(hc) - 1)
            theta = np.arctan2(h_vox - hc[d_c], w_vox - wc[d_c]) % (2 * np.pi)
        ts = (theta - geo["theta_start"]) % (2 * np.pi)
        arc = ts / geo["theta_range"]
        d_norm = d_norm_full
        w_norm = 1.0 - np.clip(arc, 0, 1)
    else:
        sn = geo["split_norm"]
        med = d_norm_full <= sn
        d_norm = np.empty_like(d_norm_full)
        d_norm[med] = d_norm_full[med] / sn * 0.5
        d_norm[~med] = 0.5 + (d_norm_full[~med] - sn) / max(1.0 - sn, 1e-6) * 0.5
        w_norm = np.empty(len(d_vox), dtype=np.float64)
        w_norm[med] = (w_vox[med] - geo["med_w_min"]) / geo["med_w_range"]
        w_norm[~med] = (w_vox[~med] - geo["lat_w_min"]) / geo["lat_w_range"]
    return d_norm, np.clip(w_norm, 0, 1), t


def project_vertices_to_2d(vertex_mm: np.ndarray, thickness_mm: np.ndarray,
                           geo: dict, spacing, grid: int = GRID) -> np.ndarray:
    """Bin per-vertex thickness onto the grid×grid normalised grid (verbatim v3.3,
    default 40×40 for regional means). `grid` is overridable for viz only."""
    d_norm, w_norm, t = vertex_norm_coords(vertex_mm, thickness_mm, geo, spacing)
    if len(t) == 0:
        return np.full((grid, grid), np.nan)
    d_bin = np.clip((d_norm * grid).astype(int), 0, grid - 1)
    w_bin = np.clip((w_norm * grid).astype(int), 0, grid - 1)
    sum_g = np.zeros((grid, grid), dtype=np.float64)
    count = np.zeros((grid, grid), dtype=int)
    np.add.at(sum_g, (d_bin, w_bin), t)
    np.add.at(count, (d_bin, w_bin), 1)
    out = np.full((grid, grid), np.nan)
    m = count > 0
    out[m] = sum_g[m] / count[m]
    return out


def mean_over(grid: np.ndarray, ref_mask: np.ndarray, d_slc, w_slc) -> float:
    """Zero-imputed regional mean over the baseline footprint (verbatim v3.3)."""
    sub = np.zeros_like(ref_mask, dtype=bool)
    sub[d_slc, w_slc] = True
    m = ref_mask & sub
    n = int(m.sum())
    if n == 0:
        return np.nan
    g_zi = np.where(m & np.isfinite(grid), grid, 0.0)
    return float(g_zi[m].sum() / n)


def regional_deltas(bone_name: str, points_mm: np.ndarray,
                    th00: np.ndarray, th48: np.ndarray,
                    bone_mask: np.ndarray, cart_mask: np.ndarray, spacing,
                    laterality: str = "right_oriented",
                    femur_unwrap: str = "per_slice") -> dict:
    """Baseline-grid regional means + deltas for one knee/bone.

    `points_mm` are the shared (00m) sampling vertices that BOTH `th00` and
    `th48` are defined on (e.g. the 00m bone-mesh points, with 48m thickness
    IDW-sampled onto them). `bone_mask`/`cart_mask` are the 00m masks used to
    build the grid geometry. Returns {region: {"00m","48m","d"}, "_footprint_bins"}.
    """
    comp, regions = BONE_TO_COMP[bone_name]
    geo = compute_ref_geometry(cart_mask, bone_mask, comp, laterality, femur_unwrap=femur_unwrap)
    g00 = project_vertices_to_2d(points_mm, th00, geo, spacing)
    g48 = project_vertices_to_2d(points_mm, th48, geo, spacing)
    ref_mask = np.isfinite(g00)
    # Per-vertex normalised coords on the 00m cartilage footprint (for high-res
    # interpolated visualisation; the 40x40 grids above are only for the means).
    th00a, th48a = np.asarray(th00, float), np.asarray(th48, float)
    fp = (th00a > MIN_THICK_MM) & (th00a < MAX_THICK_MM)
    dn, wn, _ = vertex_norm_coords(np.asarray(points_mm)[fp], th00a[fp], geo, spacing)
    out = {"_footprint_bins": int(ref_mask.sum()), "_grid_00m": g00, "_grid_48m": g48,
           "_verts": (dn, wn, th00a[fp], th48a[fp])}
    for name, dsl, wsl in regions:
        m00 = mean_over(g00, ref_mask, dsl, wsl)
        m48 = mean_over(g48, ref_mask, dsl, wsl)
        out[name] = {"00m": m00, "48m": m48,
                     "d": (m48 - m00) if (np.isfinite(m00) and np.isfinite(m48)) else np.nan}
    return out
