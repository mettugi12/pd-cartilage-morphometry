"""Web-viewer export helpers for cartilage morphometry.

Plain (no Modal import) CPU utilities that turn pipeline results into the VTP /
PNG artifacts the web `MorphometryViewer` consumes. Shared by:
  - `modal_app/morphometry_predictor.py`  (production, runs on Modal)
  - `scripts/regen_morphometry_local.py`  (local-only dev regeneration)

The subchondral threshold is exposed as a *display* parameter. The 3D template
is exported as ONE full bone mesh per bone carrying `subch_prob` plus a stack of
per-threshold thickness arrays (`thickness_T050 … thickness_T080`), so the
frontend can switch threshold levels client-side with no refetch. Each level is
an independent IDW remap (values recompute with the threshold, not just extent).
"""
from __future__ import annotations

import numpy as np


# Discrete display thresholds. Single source of truth; mirrored in the frontend
# (MorphometryViewer.tsx THRESHOLDS). Default level is 0.65.
THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_THRESHOLD: float = 0.65


def threshold_key(t: float) -> str:
    """Canonical short label for a threshold, e.g. 0.65 -> 'T065'.

    Used both for VTP point-data array names and for the PNG / manifest keys so
    the frontend can map a slider value to the right artifact deterministically.
    """
    return f"T{int(round(t * 100)):03d}"


# ---------------------------------------------------------------------------
# Mesh I/O helpers (moved verbatim from morphometry_predictor.py)
# ---------------------------------------------------------------------------
def mesh_to_vtp_bytes(mesh) -> bytes:
    """Save a PyVista PolyData to VTP bytes (in-memory)."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    try:
        mesh.save(tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def largest_mesh_component(mesh):
    """Keep only the largest connected component of a PyVista PolyData.

    Removes spurious bone islands left after marching-cubes on a densified
    mask. Returns the input unchanged if it already has 0 or 1 CCs.
    """
    try:
        return mesh.connectivity(extraction_mode="largest")
    except TypeError:
        # Older PyVista API: returns multiple, take the biggest by point count
        out = mesh.connectivity()
        ids = out["RegionId"]
        counts = np.bincount(ids.astype(np.int64))
        biggest = int(np.argmax(counts))
        return out.threshold(value=(biggest - 0.5, biggest + 0.5),
                             scalars="RegionId").extract_surface()


def split_bone_by_cart(mesh, thickness, cart_mask=None):
    """Split a full bone mesh into two DISJOINT parts via a cell-level threshold:

      bone_shell : cells where NOT all 3 verts are in the cart region.
                   No scalars (rendered gray).
      cart_part  : cells where ALL 3 verts are in the cart region. Carries
                   `thickness_mm` as a point-data scalar (sentinel 0 for
                   non-cart-bearing subch verts -> frontend renders gray).

    `cart_mask` (optional): explicit per-vertex bool mask for which verts
    belong to the cart region. When given, the threshold uses it directly so
    the cart submesh can include verts with thickness=0 (template case:
    subchondral zone verts where the patient had no cart mapped) instead of
    restricting to `thickness > 0`.

    When `cart_mask` is None, falls back to `np.isfinite(thickness) &
    (thickness > 0)` -- the patient case.

    Disjoint matters: an overlapping shell + cart Z-fight on the cart-bearing
    surface, painting stripes. PyVista's threshold + extract_surface preserves
    point-data arrays automatically (no manual reindexing).
    """
    if mesh is None:
        return None, None
    t = np.asarray(thickness, dtype=np.float32)
    if cart_mask is not None:
        valid = np.asarray(cart_mask, dtype=bool)
    else:
        valid = np.isfinite(t) & (t > 0)
    if not valid.any():
        # all bone, no cart
        shell = mesh.copy()
        for name in list(shell.point_data.keys()):
            shell.point_data.remove(name)
        for name in list(shell.cell_data.keys()):
            shell.cell_data.remove(name)
        return shell, None

    work = mesh.copy()
    work["thickness_mm"] = t   # ensure scalar is on the mesh; carried through
    work["__valid"] = valid.astype(np.uint8)

    cart = work.threshold(
        0.5, scalars="__valid", all_scalars=True,
    ).extract_surface()
    shell = work.threshold(
        0.5, scalars="__valid", all_scalars=True, invert=True,
    ).extract_surface()

    for sub in (cart, shell):
        if sub is None:
            continue
        if "__valid" in sub.point_data:
            sub.point_data.remove("__valid")

    # Shell has no need for thickness -- strip to keep payload minimal.
    if shell is not None and "thickness_mm" in shell.point_data:
        shell.point_data.remove("thickness_mm")

    return (
        shell if shell is not None and shell.n_points else None,
        cart if cart is not None and cart.n_points else None,
    )


# ---------------------------------------------------------------------------
# Patient / template VTP builders
# ---------------------------------------------------------------------------
def build_patient_vtps(bone_mesh):
    """Patient bone mesh -> (shell, cart) disjoint VTPs.

    Applies largest-CC cleanup first so floating marching-cubes islands are
    dropped from BOTH the 3D shell and downstream 2D projection. Cart = verts
    with thickness > 0 (threshold-independent for the patient frame).

    Returns (pmesh_full, pthick, shell, cart) so callers can reuse the cleaned
    mesh + thickness for the 2D projection.
    """
    pmesh_full = largest_mesh_component(bone_mesh)
    pthick = np.asarray(pmesh_full["thickness_mm"], dtype=np.float32)
    shell, cart = split_bone_by_cart(pmesh_full, pthick)
    return pmesh_full, pthick, shell, cart


def build_full_template_vtp(template_mesh, template_thickness_by_t,
                            thresholds=THRESHOLDS, bone=None):
    """Build ONE full template bone mesh carrying every display level.

    Point-data arrays attached:
      - `subch_prob`            : copied from the template (fixed, per vertex).
      - `thickness_T050 … _T080`: per-threshold IDW thickness over ALL template
                                  verts. Sentinel 0 for in-zone-but-unmapped or
                                  out-of-zone verts (the frontend masks by
                                  `subch_prob >= T`, so out-of-zone values are
                                  never shown).
      - `region_T050 … _T080`   : per-threshold subregion id (1..6, 0=non-subch)
                                  for the 정량 분석 3D split view. Only attached
                                  when `bone` ('femur'|'tibia') is given.

    `template_thickness_by_t`: dict {threshold_float: array(n_template,)}. Each
    array may contain NaN (out-of-zone, from pipeline masking) -- those are
    converted to sentinel 0 here so the VTP stays NaN-free for vtk.js.

    Returns a pv.PolyData (the template mesh with the arrays attached).
    """
    out = template_mesh.copy()
    n = out.n_points

    if "subch_prob" not in out.point_data:
        raise ValueError("template_mesh missing 'subch_prob' point-data field")
    out["subch_prob"] = np.asarray(out.point_data["subch_prob"], dtype=np.float32)

    # Drop any pre-existing thickness/region array so we control exactly what ships.
    for name in list(out.point_data.keys()):
        if name.startswith("thickness") or name.startswith("region_T"):
            out.point_data.remove(name)

    label_fn = {"femur": _femur_six_labels, "tibia": _tibia_six_labels}.get(bone)
    for t in thresholds:
        arr = template_thickness_by_t.get(t)
        if arr is None:
            arr = np.zeros(n, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] != n:
            raise ValueError(
                f"thickness array for T={t} has {arr.shape[0]} verts, "
                f"template has {n}")
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        out.point_data[f"thickness_{threshold_key(t)}"] = arr
        if label_fn is not None:
            labels, _ = label_fn(template_mesh, t)
            out.point_data[f"region_{threshold_key(t)}"] = labels.astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# 2D canonical thickness figure (shared with morphometry_predictor)
# ---------------------------------------------------------------------------
def render_panels_png(grids, vmin=0.0, vmax=3.0, subch_threshold=0.5,
                      style="fill", return_panel_bbox=False):
    """Render a 2x2 figure (rows=femur/tibia, cols=patient/template) from
    pre-projected grids. `grids` keys: 'femur_patient','femur_template',
    'tibia_patient','tibia_template' (any may be None). Columns are the shared
    slice-axis ML so a sagittal slice = same x on every panel. Returns PNG bytes
    (+ per-panel data-area bbox dict when return_panel_bbox)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contour_style = (style == "contour")

    def _panel_mean(grid):
        finite = grid[np.isfinite(grid) & (grid > 0)]
        return float(finite.mean()) if finite.size else 0.0

    cmap = plt.get_cmap("jet_r").copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")
    layout = [
        (0, 0, "femur_patient", "Femur - patient"),
        (0, 1, "femur_template", "Femur - template"),
        (1, 0, "tibia_patient", "Tibia - patient"),
        (1, 1, "tibia_template", "Tibia - template"),
    ]
    ims = {}
    for r, c, key, title in layout:
        ax = axes[r, c]
        g = grids.get(key)
        if g is None:
            ax.set_axis_off()
            continue
        zone = np.isfinite(g)
        g_disp = np.where(np.isfinite(g) & (g > 1e-6), g, np.nan) if contour_style else g
        im = ax.imshow(g_disp, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="bilinear", aspect="auto")
        ims[key] = im
        if contour_style and zone.any():
            ax.contour(zone.astype(float), levels=[0.5], colors="black", linewidths=1.3)
        ax.set_title(f"{title}\nmean = {_panel_mean(g):.2f} mm", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.97, "L", transform=ax.transAxes, fontsize=10, color="white",
                va="top", ha="left", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
        ax.text(0.98, 0.97, "M", transform=ax.transAxes, fontsize=10, color="white",
                va="top", ha="right", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("thickness (mm)", fontsize=9)

    fig.suptitle(
        f"Cartilage thickness   |   subch >= {subch_threshold:.2f}   |   scale 0-{vmax:.1f} mm",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    panel_bbox = None
    if return_panel_bbox:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        fw, fh = fig.bbox.width, fig.bbox.height
        panel_bbox = {}
        for key, im in ims.items():
            bb = im.get_window_extent(rend)
            panel_bbox[key] = [float(bb.x0 / fw), float(1.0 - bb.y1 / fh),
                               float(bb.x1 / fw), float(1.0 - bb.y0 / fh)]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    data = buf.read()
    if return_panel_bbox:
        return data, panel_bbox
    return data


# ---------------------------------------------------------------------------
# Crosshair-link correspondence: patient subch surface point (world mm) -> its
# template-2D panel (u, v). One-way: MRI slice crosshair -> template-2D marker.
# ---------------------------------------------------------------------------
# Reference threshold for the correspondence sampling — widest zone (max points).
CORR_REF_THRESHOLD = 0.50


def _patient_points_to_world(points, frame):
    """Map patient bone-mesh vertices (flipped physical mm) -> original NIfTI
    world mm matching the slice viewer. Undo the axis-2 ML flip, then affine."""
    pts = np.asarray(points, dtype=np.float64)
    sp = np.asarray(frame["spacing"], dtype=np.float64)
    aff = np.asarray(frame["affine"], dtype=np.float64)
    vox = pts / sp
    if frame.get("flip2"):
        vox[:, 2] = (frame["n2"] - 1) - vox[:, 2]
    homog = np.column_stack([vox, np.ones(len(vox))])
    return (homog @ aff.T)[:, :3]


def panel_coords(bone, mesh, layout_mask, subch_threshold=CORR_REF_THRESHOLD):
    """Per-vertex DISPLAYED canonical panel coordinates (u, v) in [0,1]
    (lateral-left, top-origin) + a base-valid mask.

    This is the CartiMorph mapping: the column `u` is the mesh's OWN INTRINSIC
    ML coordinate (femur normalized arc + ML from `compute_femur_template_subregions`;
    tibia intrinsic AP/ML), NOT a projection onto the patient slice axis. Binning
    the template by the patient slice axis re-aligned the per-slice raycast_2d
    thickness noise into vertical stripes; the template's intrinsic atlas ML is
    decoupled from the patient slice sampling, so that noise averages out per
    column and the panel reads smooth (matches CartiMorph's 2D template maps).

    `layout_mask` is the zone used to normalize the coordinate ranges. Keep it
    FIXED across display thresholds (use the ref/widest zone) so the panel layout
    — and the crosshair line — don't shift when the user changes the threshold.

    Returns (u, v, base_valid); base_valid = layout_mask & in-zone & finite. The
    per-threshold display mask is applied later in `project_by_proj`.
    """
    pts = np.asarray(mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    layout_mask = np.asarray(layout_mask, dtype=bool)
    if bone == "femur":
        from cartilage_morphometry import compute_femur_template_subregions
        m = mesh
        if "subch_prob" not in m.point_data:
            m = mesh.copy()
            m["subch_prob"] = layout_mask.astype(np.float32)
        sr = compute_femur_template_subregions(m, subch_thresh=subch_threshold)
        ml = np.asarray(sr.ml_norm)   # 0 = medial, 1 = lateral
        arc = np.asarray(sr.arc_norm)
        u = 1.0 - ml   # lateral-left (matches canonical g[:, ::-1])
        v = 1.0 - arc  # anterior at the bottom
        base_valid = (layout_mask & np.asarray(sr.is_subch)
                      & np.isfinite(ml) & np.isfinite(arc))
    else:
        lm = layout_mask
        ml_lo = float(ML[lm].min()) if lm.any() else 0.0
        ml_hi = float(ML[lm].max()) if lm.any() else 1.0
        ap_lo = float(AP[lm].min()) if lm.any() else 0.0
        ap_hi = float(AP[lm].max()) if lm.any() else 1.0
        ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
        u = 1.0 - (ML - ml_lo) / ml_r   # lateral-left
        v = (AP - ap_lo) / ap_r
        base_valid = lm & np.isfinite(u) & np.isfinite(v)
    return u, v, base_valid


def project_by_proj(bone, mesh, thickness, display_mask, layout_mask,
                    grid_size=150, smooth_sigma=1.5):
    """Project a thickness map onto a grid via the canonical intrinsic-coordinate
    mapping (columns = intrinsic ML, rows = femur arc / tibia AP). Works for
    TEMPLATE or PATIENT meshes. `layout_mask` fixes the coordinate normalization
    (ref zone); `display_mask` selects which verts are shown at this threshold."""
    from cartilage_morphometry.validation.viz import _smooth_border_preserving
    thickness = np.asarray(thickness, dtype=np.float64)
    n = grid_size
    u, v, base_valid = panel_coords(bone, mesh, layout_mask)
    valid = (base_valid & np.asarray(display_mask, dtype=bool)
             & np.isfinite(thickness)
             & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
    g = np.full((n, n), np.nan)
    if valid.any():
        col = np.clip((u[valid] * n).astype(int), 0, n - 1)
        row = np.clip((v[valid] * n).astype(int), 0, n - 1)
        s = np.zeros((n, n)); cnt = np.zeros((n, n), dtype=np.int64)
        np.add.at(s, (row, col), thickness[valid])
        np.add.at(cnt, (row, col), 1)
        g = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        # Close small gaps (single-cell holes AND thin edge-open stripes) by
        # filling any empty cell within FILL_CELLS of real data with its nearest
        # value. Bounded radius reliably closes medial-edge holes without
        # inflating the outer boundary much.
        from scipy.ndimage import distance_transform_edt
        FILL_CELLS = 3
        present = np.isfinite(g)
        if present.any():
            dist, inds = distance_transform_edt(
                ~present, return_distances=True, return_indices=True)
            fill = (~present) & (dist <= FILL_CELLS)
            g[fill] = g[tuple(inds[:, fill])]
        if smooth_sigma and smooth_sigma > 0:
            g = _smooth_border_preserving(g, sigma=smooth_sigma)
    return g


def _build_correspondence(bone, res, ref_threshold=CORR_REF_THRESHOLD):
    """Per patient subch point: world mm + canonical panel positions in BOTH
    frames. `up`/`vp` = patient-panel (u, v); `ut`/`vt` = template-panel (u, v)
    of the nearest template vertex. Each frame uses its own intrinsic ML, so the
    crosshair line is fit per-frame on the frontend (patient line ≠ template line
    in general — that offset IS the registration error the comparison reveals)."""
    from scipy.spatial import cKDTree
    bone_mesh = res["bone_mesh"]
    template_mesh = res["template_mesh"]
    aligned = np.asarray(res["aligned_pts"])
    thickness = np.asarray(bone_mesh.point_data["thickness_mm"])
    subch_prob_v = np.asarray(res["subch_prob_v"])
    frame = res.get("frame")
    empty = {"mm": [], "up": [], "ut": [], "vp": [], "vt": []}
    if frame is None:
        return empty

    pat_mask = thickness > 0
    tmpl_mask = np.asarray(template_mesh.point_data["subch_prob"]) >= ref_threshold
    u_p, v_p, _ = panel_coords(bone, bone_mesh, pat_mask)
    u_t, v_t, _ = panel_coords(bone, template_mesh, tmpl_mask)
    tree = cKDTree(np.asarray(template_mesh.points))
    _, nn = tree.query(aligned, k=1)

    sel = (subch_prob_v >= ref_threshold) & np.isfinite(u_p) & np.isfinite(v_p)
    if not sel.any():
        return empty
    world = _patient_points_to_world(np.asarray(bone_mesh.points)[sel], frame)
    return {
        "mm": world.round(3).tolist(),
        "up": np.clip(u_p[sel], 0, 1).round(4).tolist(),
        "vp": np.clip(v_p[sel], 0, 1).round(4).tolist(),
        "ut": np.clip(u_t[nn[sel]], 0, 1).round(4).tolist(),
        "vt": np.clip(v_t[nn[sel]], 0, 1).round(4).tolist(),
    }


# ---------------------------------------------------------------------------
# Full web-viewer artifact bundle (shared by Modal + local regen script)
# ---------------------------------------------------------------------------
def _radius_remap_levels(res, thresholds, radius, eps=1e-6, k_fallback=3):
    """Recompute the template thickness levels with the VIEWER radius-ball remap
    (cross-slice blending → no 3 mm staircase). Uses the cached patient inputs
    (aligned_pts, per-vert thickness, subch prob), so it's GPU-free and the study
    pipeline's k-NN remap is untouched. Returns {t: (n_template,) array, NaN out
    of the t-zone}.

    Vectorized: the target↔source distances within `radius` are computed ONCE via
    `cKDTree.sparse_distance_matrix` against the widest source (subch ≥ min t);
    each threshold is then just a column mask (source verts with subch ≥ t) +
    row-normalized sparse mat-vec. Output is identical to the per-threshold ball
    IDW but ~5-10× faster (no per-vertex Python loop, no 7× tree rebuilds)."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    template_mesh = res["template_mesh"]
    tpts = np.asarray(template_mesh.points, dtype=np.float64)
    aligned = np.asarray(res["aligned_pts"], dtype=np.float64)
    thickness = np.asarray(res["bone_mesh"].point_data["thickness_mm"], dtype=np.float64)
    subch_prob_v = np.asarray(res["subch_prob_v"])
    tmpl_subch = np.asarray(template_mesh.point_data["subch_prob"])
    n_t = len(tpts)

    min_t = float(min(thresholds))
    src_sel = subch_prob_v >= min_t
    if not src_sel.any():
        return {t: np.full(n_t, np.nan, dtype=np.float32) for t in thresholds}
    src_pts = aligned[src_sel]; src_thick = thickness[src_sel]; src_sub = subch_prob_v[src_sel]

    src_tree = cKDTree(src_pts); tgt_tree = cKDTree(tpts)
    D = tgt_tree.sparse_distance_matrix(src_tree, radius, output_type="coo_matrix")
    W = sp.csr_matrix((1.0 / (D.data + eps), (D.row, D.col)), shape=(n_t, len(src_pts)))

    # k-NN fallback (widest source) for in-zone targets whose ball is empty.
    kf = min(k_fallback, len(src_pts))
    dk, ik = src_tree.query(tpts, k=kf)
    if kf == 1:
        dk = dk[:, None]; ik = ik[:, None]
    wk = 1.0 / (dk + eps)

    out = {}
    for t in thresholds:
        tmask = tmpl_subch >= t
        col_ok = (src_sub >= t).astype(np.float64)
        Wt = W.multiply(col_ok.reshape(1, -1)).tocsr()
        num = Wt @ src_thick
        den = np.asarray(Wt.sum(axis=1)).ravel()
        res_t = np.full(n_t, np.nan)
        nz = den > 0
        res_t[nz] = num[nz] / den[nz]
        empty = tmask & ~nz
        if empty.any():
            for i in np.where(empty)[0]:
                cols = ik[i]; valid = src_sub[cols] >= t
                if valid.any():
                    w = wk[i][valid]
                    res_t[i] = float((w * src_thick[cols[valid]]).sum() / w.sum())
                else:
                    res_t[i] = 0.0
        out[t] = np.where(tmask, res_t, np.nan).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Quantitative analysis: per-subregion mean thickness + area + volume.
# 6 subregions per bone (anterior/central/posterior x medial/lateral), measured
# PATIENT-TRUE (real thickness + cartilage surface area), with subregion labels
# transferred from the template via the registration NN. Aggregates to compartments
# (MF/LF, MT/LT) then whole bone (FC/TC).
# ---------------------------------------------------------------------------
FEMUR_SUBREGIONS = ("aMF", "cMF", "pMF", "aLF", "cLF", "pLF")
TIBIA_SUBREGIONS = ("aMT", "cMT", "pMT", "aLT", "cLT", "pLT")
_COMPARTMENT_OF = {
    "aMF": "MF", "cMF": "MF", "pMF": "MF", "aLF": "LF", "cLF": "LF", "pLF": "LF",
    "aMT": "MT", "cMT": "MT", "pMT": "MT", "aLT": "LT", "cLT": "LT", "pLT": "LT",
}
_WHOLE_OF = {"MF": "FC", "LF": "FC", "MT": "TC", "LT": "TC"}


def _femur_six_labels(template_mesh, t):
    """Per-template-vertex femur subregion id (1..6 = aMF,cMF,pMF,aLF,cLF,pLF;
    0 = outside subch). A/C/P from `arc_norm` thirds (anterior trochlea →
    posterior condyle); medial/lateral from `ml_norm` (<=0.5 medial)."""
    from cartilage_morphometry import compute_femur_template_subregions
    sr = compute_femur_template_subregions(template_mesh, subch_thresh=t)
    n = template_mesh.n_points
    lab = np.zeros(n, dtype=np.int32)
    sub = np.asarray(sr.is_subch)
    arc = np.asarray(sr.arc_norm); ml = np.asarray(sr.ml_norm)
    valid = sub & np.isfinite(arc) & np.isfinite(ml)
    if valid.any():
        third = np.where(arc < 1.0 / 3, 0, np.where(arc < 2.0 / 3, 1, 2))
        col = third + np.where(ml <= 0.5, 0, 3)        # 0..5
        lab[valid] = (col[valid] + 1).astype(np.int32)
    return lab, FEMUR_SUBREGIONS


def _tibia_six_labels(template_mesh, t):
    """Per-template-vertex tibia subregion id (1..6 = aMT,cMT,pMT,aLT,cLT,pLT;
    0 = outside subch). Uses the template's compartment + AP-third labels."""
    from cartilage_morphometry import compute_tibia_template_subregions
    from cartilage_morphometry.subregions import (
        LBL_MED, LBL_AP_ANTERIOR, LBL_AP_CENTRAL, LBL_AP_POSTERIOR,
    )
    sr = compute_tibia_template_subregions(template_mesh, subch_thresh=t)
    n = template_mesh.n_points
    lab = np.zeros(n, dtype=np.int32)
    region = np.asarray(sr.region); ap3 = np.asarray(sr.ap_third); sub = np.asarray(sr.is_subch)
    med = region == LBL_MED
    for ti, tlbl in enumerate((LBL_AP_ANTERIOR, LBL_AP_CENTRAL, LBL_AP_POSTERIOR)):
        m = sub & (ap3 == tlbl)
        if not m.any():
            continue
        col = ti + np.where(med, 0, 3)
        lab[m] = (col[m] + 1).astype(np.int32)
    return lab, TIBIA_SUBREGIONS


def _vertex_areas(mesh):
    """Per-vertex share of the (triangulated) bone surface area (mm^2): each
    triangle contributes 1/3 of its area to each of its 3 vertices. Marching-cubes
    meshes are all-triangle; falls back to triangulation if not."""
    m = mesh
    faces = np.asarray(m.faces)
    if faces.size == 0 or faces.size % 4 != 0 or not np.all(faces[::4] == 3):
        m = mesh.triangulate()
        faces = np.asarray(m.faces)
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    ca = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    tri = faces.reshape(-1, 4)[:, 1:4]
    pa = np.zeros(m.n_points, dtype=np.float64)
    np.add.at(pa, tri.ravel(), np.repeat(ca / 3.0, 3))
    return pa


def _agg(stats_by_name, group_of):
    """Aggregate {name: {area_mm2, vol_mm3, n}} into groups (sum area+vol, mean=vol/area)."""
    out = {}
    for name, st in stats_by_name.items():
        g = group_of[name]
        d = out.setdefault(g, {"area_mm2": 0.0, "vol_mm3": 0.0, "n": 0})
        d["area_mm2"] += st["area_mm2"]; d["vol_mm3"] += st["vol_mm3"]; d["n"] += st["n"]
    for d in out.values():
        d["mean_mm"] = (d["vol_mm3"] / d["area_mm2"]) if d["area_mm2"] > 0 else 0.0
    return out


def _round_stats(d):
    return {k: {"mean_mm": round(v["mean_mm"], 3), "area_mm2": round(v["area_mm2"], 1),
                "vol_mm3": round(v["vol_mm3"], 1), "n": int(v["n"])}
            for k, v in d.items()}


def compute_quant_stats(results_by_bone, thresholds=THRESHOLDS):
    """Patient-true per-subregion stats for every threshold.

    Returns {threshold_key: {bone: {"subregions":{...}, "compartments":{...},
    "whole":{...}}}}, each leaf = {mean_mm, area_mm2, vol_mm3, n}. Mean thickness
    is area-weighted over cartilage-covered verts (thickness>0); volume = sum
    thickness*area (= ThC*AC); area = cartilage-covered subchondral-bone area."""
    from scipy.spatial import cKDTree
    label_fns = {"femur": _femur_six_labels, "tibia": _tibia_six_labels}
    quant = {threshold_key(t): {} for t in thresholds}
    for bone, res in results_by_bone.items():
        if bone not in label_fns:
            continue
        bone_mesh = res["bone_mesh"]; template = res["template_mesh"]
        aligned = np.asarray(res["aligned_pts"])
        thickness = np.asarray(bone_mesh.point_data["thickness_mm"], dtype=np.float64)
        areas = _vertex_areas(bone_mesh)
        nn = cKDTree(np.asarray(template.points)).query(aligned, k=1)[1]
        for t in thresholds:
            labels, names = label_fns[bone](template, t)
            plabel = labels[nn]                       # patient-vertex subregion id
            cover = (thickness > 0) & (plabel > 0)
            sub_stats = {}
            for ri, name in enumerate(names, start=1):
                m = cover & (plabel == ri)
                AC = float(areas[m].sum())
                VC = float((thickness[m] * areas[m]).sum())
                sub_stats[name] = {"mean_mm": (VC / AC if AC > 0 else 0.0),
                                   "area_mm2": AC, "vol_mm3": VC, "n": int(m.sum())}
            comp = _agg(sub_stats, _COMPARTMENT_OF)
            whole = _agg(comp, _WHOLE_OF)
            quant[threshold_key(t)][bone] = {
                "subregions": _round_stats(sub_stats),
                "compartments": _round_stats(comp),
                "whole": _round_stats(whole),
            }
    return quant


def _project_labels(bone, mesh, labels, layout_mask, n_labels, grid_size=150):
    """Majority-vote subregion label per grid cell (canonical projection). 0=empty."""
    u, v, base = panel_coords(bone, mesh, layout_mask)
    L = np.asarray(labels)
    valid = base & (L > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    n = grid_size
    g = np.zeros((n, n), dtype=np.int32)
    if valid.any():
        col = np.clip((u[valid] * n).astype(int), 0, n - 1)
        row = np.clip((v[valid] * n).astype(int), 0, n - 1)
        cnt = np.zeros((n, n, n_labels + 1), dtype=np.int32)
        np.add.at(cnt, (row, col, L[valid]), 1)
        g = cnt.argmax(axis=2).astype(np.int32)   # 0 where no votes
    return g


def render_quant_panel_png(panels, subch_threshold, vmin=0.0, vmax=3.0, style="fill"):
    """2D template viz (femur | tibia) coloured by thickness with the 6 subregions
    outlined in black + labelled. `panels[bone] = (thick_grid, label_grid, names)`.

    `style`: 'fill' shows non-mapped (denuded) cells red (jet_r 0); 'contour'
    leaves them empty (bone-grey) so only mapped cartilage is coloured."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy.ndimage import distance_transform_edt, gaussian_filter

    contour_style = (style == "contour")
    cmap = plt.get_cmap("jet_r").copy()
    cmap.set_bad(color=(0.92, 0.92, 0.92))
    order = [("femur", "Femur"), ("tibia", "Tibia")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.patch.set_facecolor("white")
    for ax, (bone, title) in zip(axes, order):
        p = panels.get(bone)
        if not p:
            ax.set_axis_off(); continue
        g, lab, names = p
        g_disp = np.where(np.isfinite(g) & (g > 1e-6), g, np.nan) if contour_style else g
        ax.imshow(g_disp, origin="upper", cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="auto")

        # The raw majority-vote label grid is sparse + holey; contouring it draws
        # a loop around every hole/speck. Fill it solid (nearest label) within the
        # displayed thickness footprint so only clean region borders remain.
        zone = np.isfinite(g)
        present = lab > 0
        if present.any():
            _, inds = distance_transform_edt(~present, return_indices=True)
            lab = np.where(zone, lab[tuple(inds)], 0)

        for ri, name in enumerate(names, start=1):
            mask = lab == ri
            if mask.sum() < 4:
                continue
            # Smooth the binary mask before contouring so borders aren't jagged.
            ax.contour(gaussian_filter(mask.astype(float), sigma=1.0),
                       levels=[0.5], colors="black", linewidths=1.2)
            rr, cc = np.where(mask)
            ax.text(cc.mean(), rr.mean(), name, fontsize=9, fontweight="bold",
                    ha="center", va="center", color="black",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.97, "L", transform=ax.transAxes, fontsize=10, color="white",
                va="top", ha="left", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
        ax.text(0.98, 0.97, "M", transform=ax.transAxes, fontsize=10, color="white",
                va="top", ha="right", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
    fig.suptitle(f"Cartilage subregions   |   subch >= {subch_threshold:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Parallel PNG rendering — matplotlib figures are independent, so render the
# 2-styles × 7-thresholds (× panel/quant) set across CPU cores. Worker fns are
# module-level so ProcessPoolExecutor can pickle them.
# ---------------------------------------------------------------------------
def _panel_png_job(job):
    grids, t, s, want_bbox = job
    out = render_panels_png(grids, subch_threshold=t, style=s, return_panel_bbox=want_bbox)
    png, bbox = out if want_bbox else (out, None)
    return (s, threshold_key(t), png, bbox)


def _quant_png_job(job):
    panels, t, s = job
    return (s, threshold_key(t), render_quant_panel_png(panels, subch_threshold=t, style=s))


def _pool_map(fn, jobs):
    """Map `fn` over `jobs` across processes (fork on Modal/Linux). Falls back to
    serial on a single job or if the pool can't start."""
    if len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import os
    from concurrent.futures import ProcessPoolExecutor
    # Warm matplotlib in the parent so forked workers (Modal/Linux) inherit it
    # instead of each re-importing.
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot  # noqa: F401
    n = min(len(jobs), max(1, (os.cpu_count() or 2)), 8)
    try:
        with ProcessPoolExecutor(max_workers=n) as ex:
            return list(ex.map(fn, jobs))
    except Exception:
        return [fn(j) for j in jobs]


def build_quant_artifacts(results_by_bone, thresholds=THRESHOLDS, remap_radius=3.0,
                          styles=("fill", "contour"), tmpl_grids=None, ref_mask=None):
    """Build the 정량 분석 artifacts: patient-true per-subregion stats + one
    subregion-delineated template-2D PNG per (style, threshold).

    Returns (files, patch): `files` = {png_name: bytes}; `patch` = manifest keys
    {quant, quant_subregions, quant_png_levels} where
    quant_png_levels[style][thresholdKey] -> filename. Shared by
    `build_web_artifacts` (pass the already-computed `tmpl_grids`/`ref_mask` to
    avoid recomputing the remap) and by the `--quant_only` fast path.
    """
    bones2 = [b for b in ("femur", "tibia") if b in results_by_bone]
    label_fns = {"femur": _femur_six_labels, "tibia": _tibia_six_labels}

    if tmpl_grids is None or ref_mask is None:
        tmpl_grids = {}; ref_mask = {}
        for b in bones2:
            tmpl = results_by_bone[b]["template_mesh"]
            sp = np.asarray(tmpl.point_data["subch_prob"])
            ref_mask[b] = sp >= CORR_REF_THRESHOLD
            thick_by_t = (_radius_remap_levels(results_by_bone[b], thresholds, remap_radius)
                          if remap_radius else results_by_bone[b]["template_thickness_by_t"])
            tmpl_grids[b] = {t: project_by_proj(b, tmpl, thick_by_t[t], sp >= t, ref_mask[b])
                             for t in thresholds}

    # Per-threshold subregion label grids (style-independent) — compute once.
    label_grids = {b: {} for b in bones2}
    names_by_bone = {}
    for b in bones2:
        for t in thresholds:
            labels, names = label_fns[b](results_by_bone[b]["template_mesh"], t)
            names_by_bone[b] = list(names)
            label_grids[b][t] = _project_labels(b, results_by_bone[b]["template_mesh"],
                                                 labels, ref_mask[b], n_labels=len(names))

    files: dict[str, bytes] = {}
    patch: dict = {
        "quant": compute_quant_stats(results_by_bone, thresholds),
        "quant_subregions": {"femur": list(FEMUR_SUBREGIONS), "tibia": list(TIBIA_SUBREGIONS)},
        "quant_png_levels": {s: {} for s in styles},
    }
    quant_jobs = []
    for s in styles:
        for t in thresholds:
            panels = {b: (tmpl_grids[b][t], label_grids[b][t], names_by_bone[b])
                      for b in bones2}
            quant_jobs.append((panels, t, s))
    for s, tkey, png in _pool_map(_quant_png_job, quant_jobs):
        fname = f"quant_2d_{s}_{tkey}.png"
        files[fname] = png
        patch["quant_png_levels"][s][tkey] = fname
    return files, patch


def build_web_artifacts(results_by_bone, thresholds=THRESHOLDS,
                        styles=("fill", "contour"), remap_radius=3.0):
    """Turn per-bone `process_one_patient_levels` results into the complete set
    of viewer artifacts + a manifest.

    Args:
        results_by_bone: {bone: {"bone_mesh", "template_mesh",
                                 "template_thickness_by_t", "aligned_pts",
                                 "subch_prob_v"}} for femur/tibia.
        thresholds: discrete display thresholds (default THRESHOLDS).
        styles: 2D render styles ("fill", "contour").
        remap_radius: if set, recompute the template levels with the radius-ball
            remap (mm) so the template doesn't inherit the patient's 3 mm slice
            staircase. None keeps the k-NN levels from the pipeline.

    Returns (files, manifest):
        files    : {filename: bytes}  -- VTPs + PNGs to write/upload.
        manifest : dict matching what the frontend MorphometryUrls expects;
                   png_2d_levels[style][threshold_key] -> filename.
    """
    files: dict[str, bytes] = {}
    manifest: dict = {
        "thresholds": [float(t) for t in thresholds],
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold_keys": {f"{t:.2f}": threshold_key(t) for t in thresholds},
        "png_2d_levels": {s: {} for s in styles},
    }

    patient_for_2d: dict = {}
    templates: dict = {}
    for bone, res in results_by_bone.items():
        bone_mesh = res["bone_mesh"]
        template_mesh = res["template_mesh"]
        if remap_radius:
            thick_by_t = _radius_remap_levels(res, thresholds, remap_radius)
        else:
            thick_by_t = res["template_thickness_by_t"]
        templates[bone] = (template_mesh, thick_by_t)

        pmesh_full, pthick, shell, cart = build_patient_vtps(bone_mesh)
        patient_for_2d[bone] = (pmesh_full, pthick)
        if shell is not None:
            files[f"{bone}_patient.vtp"] = mesh_to_vtp_bytes(shell)
            manifest[f"{bone}_patient"] = f"{bone}_patient.vtp"
        if cart is not None:
            files[f"{bone}_patient_cart.vtp"] = mesh_to_vtp_bytes(cart)
            manifest[f"{bone}_patient_cart"] = f"{bone}_patient_cart.vtp"

        full_tmpl = build_full_template_vtp(template_mesh, thick_by_t, thresholds, bone=bone)
        files[f"{bone}_template_full.vtp"] = mesh_to_vtp_bytes(full_tmpl)
        manifest[f"{bone}_template_full"] = f"{bone}_template_full.vtp"

    # Render BOTH the patient frame and the template frame with the CANONICAL
    # CartiMorph projection (each mesh's own intrinsic ML / arc), so the template
    # reads smooth (no per-slice stripes) and matches CartiMorph's 2D maps.
    # Patient and template are unwrapped independently in their own frames; a
    # registration shift shows up as the crosshair landing on different relative
    # positions across the patient-vs-template panels.
    if "femur" in results_by_bone and "tibia" in results_by_bone:
        bones2 = ("femur", "tibia")
        # Patient 2D uses the FULL bone_mesh (consistent indexing with
        # aligned_pts/subch_prob_v for the correspondence).
        pat_mesh = {b: results_by_bone[b]["bone_mesh"] for b in bones2}
        pat_thick = {b: np.asarray(pat_mesh[b].point_data["thickness_mm"]) for b in bones2}
        pat_mask = {b: pat_thick[b] > 0 for b in bones2}
        tmpl_subch_prob = {b: np.asarray(results_by_bone[b]["template_mesh"].point_data["subch_prob"])
                           for b in bones2}
        # Fixed reference zone for layout normalization (stable across thresholds).
        ref_mask = {b: tmpl_subch_prob[b] >= CORR_REF_THRESHOLD for b in bones2}

        # Patient grids (threshold-independent) computed once.
        pat_grids = {
            b: project_by_proj(b, pat_mesh[b], pat_thick[b], pat_mask[b], pat_mask[b])
            for b in bones2
        }
        # Template grids per threshold (layout fixed at ref_mask; display masked at t).
        tmpl_grids = {
            b: {t: project_by_proj(b, results_by_bone[b]["template_mesh"],
                                   templates[b][1][t], tmpl_subch_prob[b] >= t,
                                   ref_mask[b])
                for t in thresholds}
            for b in bones2
        }

        panel_jobs = []
        for si, s in enumerate(styles):
            for ti, t in enumerate(thresholds):
                grids = {
                    "femur_patient": pat_grids["femur"],
                    "femur_template": tmpl_grids["femur"][t],
                    "tibia_patient": pat_grids["tibia"],
                    "tibia_template": tmpl_grids["tibia"][t],
                }
                panel_jobs.append((grids, t, s, si == 0 and ti == 0))
        for s, tkey, png, bbox in _pool_map(_panel_png_job, panel_jobs):
            if bbox is not None:
                manifest["panel_bbox"] = bbox
            fname = f"png_2d_{s}_{tkey}.png"
            files[fname] = png
            manifest["png_2d_levels"][s][tkey] = fname

        # Crosshair-link correspondence (one-way: MRI slice -> 2D line + dot).
        import json as _json
        correspondence = {
            b: _build_correspondence(b, results_by_bone[b])
            for b in bones2
        }
        # Slice geometry in ORIGINAL NIfTI world mm (matches niivue crosshair).
        # A sagittal slice = constant (mm . axis); the frontend projects the
        # crosshair onto this axis ONLY (so in-plane motion doesn't move the
        # line) and snaps to discrete slices (round((t - origin)/spacing)).
        frame0 = results_by_bone["femur"]["frame"]
        aff0 = np.asarray(frame0["affine"], dtype=np.float64)
        kax = int(np.argmax(np.linalg.norm(aff0[:3, :3], axis=0)))
        acol = aff0[:3, kax]
        spacing_w = float(np.linalg.norm(acol))
        axis_u = (acol / (spacing_w + 1e-12))
        correspondence["slice_geom"] = {
            "axis": axis_u.tolist(),
            "spacing": spacing_w,
            "origin": float(aff0[:3, 3] @ axis_u),
            "n": int(frame0.get("n2", 0)),
        }
        files["correspondence.json"] = _json.dumps(correspondence).encode("utf-8")
        manifest["correspondence"] = "correspondence.json"

        # --- Quantitative analysis (정량 분석) --- reuse the already-computed
        # template grids + ref masks so the remap isn't recomputed.
        q_files, q_patch = build_quant_artifacts(
            results_by_bone, thresholds, remap_radius,
            tmpl_grids=tmpl_grids, ref_mask=ref_mask)
        files.update(q_files)
        manifest.update(q_patch)

    return files, manifest
