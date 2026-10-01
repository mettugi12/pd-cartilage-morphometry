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


def _count_grid(points_mm: np.ndarray, geo: dict, spacing, grid: int = GRID) -> np.ndarray:
    """Number of vertices falling in each grid cell (uses unit thickness so that
    vertex_norm_coords' validity filter keeps every vertex)."""
    if len(points_mm) == 0:
        return np.zeros((grid, grid), dtype=int)
    d_norm, w_norm, _ = vertex_norm_coords(points_mm, np.ones(len(points_mm)), geo, spacing)
    if len(d_norm) == 0:
        return np.zeros((grid, grid), dtype=int)
    d_bin = np.clip((d_norm * grid).astype(int), 0, grid - 1)
    w_bin = np.clip((w_norm * grid).astype(int), 0, grid - 1)
    cnt = np.zeros((grid, grid), dtype=int)
    np.add.at(cnt, (d_bin, w_bin), 1)
    return cnt


def total_bone_area_grids(points_mm: np.ndarray, th00: np.ndarray, th48: np.ndarray, th48_status: np.ndarray,
                          cart_mask: np.ndarray, geo: dict, spacing, r_den_mm: float = 1.5,
                          fine: int = 120, grid: int = GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cell means over the total subchondral bone area of the baseline plate, denuded bone = 0 mm.

    Chondrometrics reports thickness over the total subchondral bone area (ThCtAB), with denuded bone
    counted as 0 mm (OAI kMRI_QCart_Eckstein_Descrip p11-12). The legacy grid averages only vertices
    that carry cartilage, so denuded bone inside a partly covered cell drops out at either visit.

    Vertex set (identical at both visits):
      zone   = bone-surface vertices inside the outer contour of the baseline cartilage plate: the
               cartilage-bearing vertices are rasterised on a fine (`fine` x `fine`) version of the grid,
               closed by one cell and hole-filled; every bone vertex falling in that area belongs to it
               (bone beyond the plate edge and the intercondylar gap stay out).
      00m    = measured thickness; else 0 if no 00m cartilage voxel within `r_den_mm` (denuded);
               else failed ray-cast.
      48m    = status 1 -> thickness; 0 -> 0 (denuded, symmetric rule); -1 -> failed.
      Vertices failed at either visit are dropped from BOTH visits.
    Returns (g00, g48, ref_mask, info) on the `grid` x `grid` baseline grid.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

    pts = np.asarray(points_mm, float)
    th00 = np.asarray(th00); th48 = np.asarray(th48, float); st = np.asarray(th48_status)
    valid = (th00 > MIN_THICK_MM) & (th00 < MAX_THICK_MM)
    dn, wn, _ = vertex_norm_coords(pts, np.ones(len(pts)), geo, spacing)      # every vertex, same order
    sp = np.asarray(spacing, float)
    if geo["compartment"] == "FC":
        fd = np.clip((dn * fine).astype(int), 0, fine - 1); fw = np.clip((wn * fine).astype(int), 0, fine - 1)
        nb0, nb1 = fine, fine
    else:
        # Tibia: the plateau is near-planar, so the plate outline is drawn in the isotropic top view
        # (ML x AP, 1 mm bins). The normalised grid follows the sagittal slice positions, and closing it
        # bridges the slice stripes onto the sloping rim / cortex below the plateau margin.
        lo_ml, lo_ap = pts[:, 0].min(), pts[:, 2].min()
        fd = ((pts[:, 0] - lo_ml) / 1.0).astype(int); fw = ((pts[:, 2] - lo_ap) / 1.0).astype(int)
        nb0, nb1 = int(fd.max()) + 1, int(fw.max()) + 1
    pres = np.zeros((nb0, nb1), bool); pres[fd[valid], fw[valid]] = True
    area = binary_fill_holes(binary_closing(pres, structure=np.ones((3, 3)), iterations=1)) | pres
    zone = area[fd, fw]

    # Articular surface only: non-articular bone (condyle side walls, intercondylar walls) runs roughly
    # parallel to the slices and maps onto the same (slice, arc) bins at a smaller radius from the slice
    # centroid (femur) / lower on the plateau (tibia). A vertex without cartilage counts only if it lies on
    # the outermost surface of its fine bin (within `surf_tol_mm` of the bin maximum).
    surf_tol_mm = 1.5
    if geo["compartment"] == "FC":
        d_vox = pts[:, 0] / sp[0]; h_vox = pts[:, 1] / sp[1]; w_vox = pts[:, 2] / sp[2]
        dc = np.clip(d_vox.astype(int), 0, len(geo["h_c"]) - 1)
        if geo.get("femur_unwrap") == "best_fit_circle":
            hc, wc = geo["h_c_fit"], geo["w_c_fit"]
        else:
            hc, wc = geo["h_c"][dc], geo["w_c"][dc]
        outward = np.hypot((h_vox - hc) * sp[1], (w_vox - wc) * sp[2])     # radius from the slice centroid
    else:
        # plateau faces proximally; the SI sign of the patient volume is not fixed, so take the articular
        # direction from the data: cartilage-bearing vertices lie on the articular side of the bone
        si = pts[:, 1]
        sgn = -1.0 if si[valid].mean() < si.mean() else 1.0
        outward = sgn * si
    fb = fd * nb1 + fw
    bmax = np.full(nb0 * nb1, -np.inf)
    np.maximum.at(bmax, fb[zone], outward[zone])
    articular = outward >= bmax[fb] - surf_tol_mm

    cand = zone & ~valid & articular                                        # articular zone bone w/o baseline cartilage
    zone = zone & (valid | articular)
    near = np.zeros(len(pts), bool)
    if cand.any():
        vox = np.rint(pts[cand] / sp).astype(int)
        cart = np.asarray(cart_mask, bool)
        lo = np.maximum(vox.min(0) - 10, 0); hi = np.minimum(vox.max(0) + 11, cart.shape)
        sub = cart[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dist = distance_transform_edt(~sub, sampling=sp) if sub.any() else np.full(sub.shape, np.inf)
        v = np.clip(vox - lo, 0, np.array(sub.shape) - 1)
        near[np.where(cand)[0]] = dist[v[:, 0], v[:, 1], v[:, 2]] <= r_den_mm
    den00 = cand & ~near
    fail00 = cand & near
    t00 = np.where(valid, th00.astype(float), 0.0)
    t48 = np.where(st == 1, th48, 0.0)
    keep = zone & ~fail00 & (st != -1)

    gd = np.clip((dn * grid).astype(int), 0, grid - 1); gw = np.clip((wn * grid).astype(int), 0, grid - 1)
    cnt = np.zeros((grid, grid)); s0 = np.zeros((grid, grid)); s4 = np.zeros((grid, grid))
    np.add.at(cnt, (gd[keep], gw[keep]), 1)
    np.add.at(s0, (gd[keep], gw[keep]), t00[keep]); np.add.at(s4, (gd[keep], gw[keep]), t48[keep])
    ref = cnt > 0
    g00 = np.where(ref, s0 / np.maximum(cnt, 1), np.nan); g48 = np.where(ref, s4 / np.maximum(cnt, 1), np.nan)
    info = {"n_zone": int(zone.sum()), "n_kept": int(keep.sum()), "n_den00": int((den00 & keep).sum()),
            "n_den48": int((keep & (st == 0)).sum()), "n_fail00": int(fail00.sum()),
            "n_fail48": int((zone & ~fail00 & (st == -1)).sum()),
            "vertex_den00": den00, "vertex_keep": keep, "vertex_zone": zone, "vertex_fail00": fail00}
    return g00, g48, ref, info


def eckstein_first_crossing_masks(points_mm: np.ndarray, th00: np.ndarray, geo: dict, spacing,
                                  fraction: float = 0.75, grid: int = GRID,
                                  zone: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Chondrometrics-style central femoral regions (cMF / cLF) as masks on the baseline grid.

    Chondrometrics (OAI kMRI_QCart_Eckstein_Descrip, Fig. 2): the condyle is split by a plane parallel to
    the femoral shaft at `fraction` (0.60 or 0.75) of the AP distance between the trochlear notch and the
    posterior end of the condyle; cMF runs from the notch plane to that divider. Landmarks come from
    `subregions.find_femur_eckstein_landmarks` on the 00m cartilage-bearing vertices (notch = max-AP vertex
    within +-2 mm of the ML midline; end = max-AP vertex per condyle).

    The condyle is C-shaped, so the divider cuts it twice (distal surface and the posterior curl-back).
    Only cartilage continuous from the notch up to the FIRST crossing is central: per ML row of the grid,
    walk the arc anterior -> posterior and keep the running maximum of the cell AP; once it passes the
    divider, every further cell is posterior even where AP falls back below it.

    `points_mm` in grid order (ML, SI, AP) mm, same vertices and geometry as `regional_deltas`.
    `zone` (optional): cells to be classified even without measured cartilage (e.g. baseline-denuded
    cells); their AP position is taken from the nearest cartilage-bearing cell.
    Returns (cMF mask, cLF mask, info) with masks shaped (grid, grid) over (D rows, W cols).
    """
    import pyvista as pv
    from . import subregions as _sr

    th00 = np.asarray(th00)                     # keep dtype: same validity filter as vertex_norm_coords
    pts = np.asarray(points_mm, float)
    valid = (th00 > MIN_THICK_MM) & (th00 < MAX_THICK_MM)
    dn, wn, _ = vertex_norm_coords(pts[valid], th00[valid], geo, spacing)
    db = np.clip((dn * grid).astype(int), 0, grid - 1)
    wb = np.clip((wn * grid).astype(int), 0, grid - 1)
    s = np.zeros((grid, grid)); c = np.zeros((grid, grid))
    np.add.at(s, (db, wb), pts[valid][:, 2]); np.add.at(c, (db, wb), 1)
    cell_ap = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    if zone is not None and np.isfinite(cell_ap).any():
        from scipy.ndimage import distance_transform_edt
        _, inds = distance_transform_edt(~np.isfinite(cell_ap), return_indices=True)
        fill = np.asarray(zone, bool) & ~np.isfinite(cell_ap)
        cell_ap = cell_ap.copy(); cell_ap[fill] = cell_ap[tuple(inds[:, fill])]

    col_ap = np.array([np.nanmedian(cell_ap[:, k]) if np.isfinite(cell_ap[:, k]).any() else np.nan for k in range(grid)])
    cols = np.arange(grid) if np.nanmean(col_ap[: grid // 4]) < np.nanmean(col_ap[-grid // 4:]) else np.arange(grid)[::-1]
    cum = np.full((grid, grid), np.nan)
    cum[:, cols] = np.maximum.accumulate(np.where(np.isfinite(cell_ap[:, cols]), cell_ap[:, cols], -np.inf), axis=1)

    m = pv.PolyData(np.ascontiguousarray(pts[:, [2, 1, 0]]))      # (AP, SI, ML) as the landmark finder expects
    m.point_data["subch_prob"] = valid.astype(float)
    L = _sr.find_femur_eckstein_landmarks(m, subch_thresh=0.5, ml_slice_tol_mm=2.0)
    ap0 = L["ap_notch"]
    rows = np.zeros((grid, grid), bool)
    masks = []
    for dsl, ap_end in ((slice(0, grid // 2), L["ap_end_med"]), (slice(grid // 2, grid), L["ap_end_lat"])):
        split = ap0 + fraction * (ap_end - ap0)
        r = rows.copy(); r[dsl, :] = True
        masks.append(r & (cell_ap >= ap0) & (cum <= split))
    info = {"ap_notch": ap0, "ap_end_med": L["ap_end_med"], "ap_end_lat": L["ap_end_lat"], "fraction": fraction}
    return masks[0], masks[1], info


def regional_deltas(bone_name: str, points_mm: np.ndarray,
                    th00: np.ndarray, th48: np.ndarray,
                    bone_mask: np.ndarray, cart_mask: np.ndarray, spacing,
                    laterality: str = "right_oriented",
                    femur_unwrap: str = "per_slice",
                    th48_status: np.ndarray | None = None,
                    femur_region: str = "grid",
                    total_bone_area: bool = False) -> dict:
    """Baseline-grid regional means + deltas for one knee/bone.

    `points_mm` are the shared (00m) sampling vertices that BOTH `th00` and
    `th48` are defined on (e.g. the 00m bone-mesh points, with 48m thickness
    IDW-sampled onto them). `bone_mask`/`cart_mask` are the 00m masks used to
    build the grid geometry. Returns {region: {"00m","48m","d"}, "_footprint_bins"}.

    `th48_status` (v9.2 symmetric handling; see shared_mesh._followup_symmetric):
    per-vertex 1 = measured, 0 = denuded (true zero), -1 = failed measurement.
    Cells of the baseline footprint whose follow-up evidence is only failures are
    EXCLUDED from both visits' means instead of being zero-imputed; cells whose
    evidence is denudation are set to 0 explicitly. Legacy behaviour when None.

    `femur_region` (femur only): "grid" = legacy cMF/cLF (medial/lateral half x central
    20-80 % of the whole cartilage arc, trochlea included); "eckstein75" / "eckstein60" =
    Chondrometrics-style notch-anchored first-crossing regions (`eckstein_first_crossing_masks`).
    Tibial regions are unaffected.

    `total_bone_area` (requires `th48_status`): cell means over the total subchondral bone area of the
    baseline plate with denuded bone = 0 mm at either visit, on one vertex set for both visits
    (Chondrometrics ThCtAB convention; see `total_bone_area_grids`). Default off (legacy: only
    cartilage-bearing vertices, failures handled per cell).
    """
    comp, regions = BONE_TO_COMP[bone_name]
    geo = compute_ref_geometry(cart_mask, bone_mask, comp, laterality, femur_unwrap=femur_unwrap)
    g00 = project_vertices_to_2d(points_mm, th00, geo, spacing)
    ref_mask = np.isfinite(g00)
    tab_info = None
    frac_failed = frac_denuded = np.nan
    if total_bone_area:
        if th48_status is None:
            raise ValueError("total_bone_area requires th48_status (symmetric follow-up handling)")
        g00, g48, ref_mask, tab_info = total_bone_area_grids(points_mm, th00, th48, th48_status,
                                                             cart_mask, geo, spacing)
    elif th48_status is None:
        g48 = project_vertices_to_2d(points_mm, th48, geo, spacing)
    else:
        st = np.asarray(th48_status)
        pts = np.asarray(points_mm)
        th48a_ = np.asarray(th48, float)
        meas = st == 1
        g48 = project_vertices_to_2d(pts[meas], th48a_[meas], geo, spacing) if meas.any() else np.full((GRID, GRID), np.nan)
        den_c = _count_grid(pts[st == 0], geo, spacing)
        fail_c = _count_grid(pts[st == -1], geo, spacing)
        nomeas = ~np.isfinite(g48)
        denuded_cell = nomeas & (den_c > 0) & (den_c >= fail_c)
        failed_cell = nomeas & ~denuded_cell
        g48 = g48.copy()
        g48[denuded_cell] = 0.0
        n_fp = max(int(ref_mask.sum()), 1)
        frac_failed = float((ref_mask & failed_cell).sum() / n_fp)
        frac_denuded = float((ref_mask & denuded_cell).sum() / n_fp)
        ref_mask = ref_mask & ~failed_cell        # symmetric exclusion (both visits)
    # Per-vertex normalised coords on the 00m cartilage footprint (for high-res
    # interpolated visualisation; the 40x40 grids above are only for the means).
    th00a, th48a = np.asarray(th00, float), np.asarray(th48, float)
    fp = (th00a > MIN_THICK_MM) & (th00a < MAX_THICK_MM)
    dn, wn, _ = vertex_norm_coords(np.asarray(points_mm)[fp], th00a[fp], geo, spacing)
    out = {"_footprint_bins": int(ref_mask.sum()), "_grid_00m": g00, "_grid_48m": g48,
           "_verts": (dn, wn, th00a[fp], th48a[fp]),
           "_frac_failed_cells": frac_failed, "_frac_denuded_cells": frac_denuded,
           "_tab_info": tab_info}
    for name, dsl, wsl in regions:
        m00 = mean_over(g00, ref_mask, dsl, wsl)
        m48 = mean_over(g48, ref_mask, dsl, wsl)
        out[name] = {"00m": m00, "48m": m48,
                     "d": (m48 - m00) if (np.isfinite(m00) and np.isfinite(m48)) else np.nan}
    if bone_name == "femur" and femur_region != "grid":
        frac = {"eckstein75": 0.75, "eckstein60": 0.60}[femur_region]
        mk_med, mk_lat, info = eckstein_first_crossing_masks(points_mm, th00, geo, spacing, fraction=frac,
                                                             zone=ref_mask)
        info["cells_cMF"] = int((ref_mask & mk_med).sum()); info["cells_cLF"] = int((ref_mask & mk_lat).sum())
        full = (slice(0, GRID), slice(0, GRID))
        for name, mk in (("cMF", mk_med), ("cLF", mk_lat)):
            m00 = mean_over(g00, ref_mask & mk, *full)
            m48 = mean_over(g48, ref_mask & mk, *full)
            out[name] = {"00m": m00, "48m": m48,
                         "d": (m48 - m00) if (np.isfinite(m00) and np.isfinite(m48)) else np.nan}
        out["_femur_region_info"] = info
    return out
