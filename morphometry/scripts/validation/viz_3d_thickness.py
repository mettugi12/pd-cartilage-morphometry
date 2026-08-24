"""Compare remap methods in 3D + 2D, no IDW averaging.

For one case + bone + timepoint, computes per-template-vert thickness via
four variants and renders both 3D and the proportional 2D projection
side-by-side:

  [PATIENT]            patient bone mesh in patient frame, raycast thickness
  [TEMPLATE k=3 IDW]   current canonical: each template vert = inverse-dist
                         weighted mean of k=3 nearest patient subch verts
  [TEMPLATE k=1 fwd]   "direct mapping" — each template vert takes the
                         thickness of its SINGLE nearest patient subch vert
                         (no averaging, preserves peaks, Voronoi patches)
  [TEMPLATE k=1 push]  scatter / reverse: each patient subch vert pushes its
                         thickness to its single nearest template subch vert;
                         template verts receiving multiple pushes get the mean

3D output: 2 rows (superior + anterior views) x 4 cols.
2D output: 1 row x 4 cols, proportional projection (shared ML/AP bounds).

Usage:
  python -m scripts.viz_3d_thickness --case 9477205_RIGHT --tp 00m
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from cartilage_morphometry import PipelineConfig, TEMPLATE_PATHS
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import process_long_shared_mesh


# ---------------------------------------------------------------------------
# Remap variants
# ---------------------------------------------------------------------------
def remap_k3_idw(template_mesh, aligned_pts, patient_thickness, subch_prob,
                  subch_threshold: float = 0.5, k: int = 3, eps: float = 1e-6):
    """Forward k=3 IDW (current canonical)."""
    n_t = template_mesh.n_points
    is_sp = subch_prob >= subch_threshold
    if not is_sp.any():
        return np.full(n_t, np.nan, dtype=np.float32)
    src_pts = aligned_pts[is_sp]; src_th = patient_thickness[is_sp]
    tpl_pts = np.asarray(template_mesh.points)
    tree = cKDTree(src_pts)
    d, i = tree.query(tpl_pts, k=min(k, len(src_pts)))
    if d.ndim == 1: d = d[:, None]; i = i[:, None]
    w = 1.0 / (d + eps); w /= w.sum(axis=1, keepdims=True)
    return (w * src_th[i]).sum(axis=1).astype(np.float32)


def remap_k1_forward(template_mesh, aligned_pts, patient_thickness, subch_prob,
                      subch_threshold: float = 0.5):
    """Forward k=1: each template vert takes its single nearest patient
    subch vert's thickness. No averaging. Preserves peaks; produces Voronoi
    patches at the template scale."""
    n_t = template_mesh.n_points
    is_sp = subch_prob >= subch_threshold
    if not is_sp.any():
        return np.full(n_t, np.nan, dtype=np.float32)
    src_pts = aligned_pts[is_sp]; src_th = patient_thickness[is_sp]
    tpl_pts = np.asarray(template_mesh.points)
    tree = cKDTree(src_pts)
    _, i = tree.query(tpl_pts, k=1)
    return src_th[i].astype(np.float32)


def remap_k1_scatter(template_mesh, aligned_pts, patient_thickness, subch_prob,
                      subch_threshold: float = 0.5):
    """Reverse k=1: each patient subch vert pushes its thickness to its
    single nearest template subch vert. Template verts receiving multiple
    pushes get the arithmetic mean of those pushes. Template verts that
    receive nothing → NaN. Closely tied to the patient sampling density."""
    n_t = template_mesh.n_points
    tpl_pts = np.asarray(template_mesh.points)
    tpl_subch_mask = (np.asarray(template_mesh.point_data["subch_prob"])
                       >= subch_threshold)
    tpl_subch_idx = np.where(tpl_subch_mask)[0]
    tpl_subch_pts = tpl_pts[tpl_subch_idx]
    is_sp = subch_prob >= subch_threshold
    if not is_sp.any() or not len(tpl_subch_pts):
        return np.full(n_t, np.nan, dtype=np.float32)
    src_pts = aligned_pts[is_sp]; src_th = patient_thickness[is_sp]
    tree = cKDTree(tpl_subch_pts)
    _, i = tree.query(src_pts, k=1)
    sum_arr = np.zeros(len(tpl_subch_pts), dtype=np.float64)
    cnt_arr = np.zeros(len(tpl_subch_pts), dtype=np.int64)
    np.add.at(sum_arr, i, src_th)
    np.add.at(cnt_arr, i, 1)
    out_sub = np.where(cnt_arr > 0, sum_arr / np.maximum(cnt_arr, 1), np.nan)
    out = np.full(n_t, np.nan, dtype=np.float32)
    out[tpl_subch_idx] = out_sub.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 3D rendering (pyvista off-screen)
# ---------------------------------------------------------------------------
_VIEWS = [
    ("superior",  [(0, 0,  250), (0, 0, 0), (0, 1, 0)]),
    ("anterior",  [(0, -250, 0), (0, 0, 0), (0, 0, 1)]),
]


def render_3d_grid(meshes_with_scalars, titles, out_png: Path,
                    vmax: float = 4.0, mask_nan: bool = True):
    """meshes_with_scalars: list of (mesh, scalar_array) tuples — one per col.
    Renders 2 rows (views) x N cols (variants).
    """
    n_cols = len(meshes_with_scalars)
    n_rows = len(_VIEWS)
    pl = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                    window_size=(600 * n_cols, 600 * n_rows))
    for r, (view_name, cam) in enumerate(_VIEWS):
        for c, (mesh, sc) in enumerate(meshes_with_scalars):
            pl.subplot(r, c)
            m = mesh.copy()
            sc_arr = np.asarray(sc, dtype=np.float32).copy()
            if mask_nan:
                sc_arr = np.where(np.isfinite(sc_arr), sc_arr, np.nan)
            m["thickness_mm"] = sc_arr
            pl.add_mesh(m, scalars="thickness_mm", cmap="jet_r",
                         clim=(0, vmax), nan_color=(0.85, 0.85, 0.85),
                         show_scalar_bar=(c == n_cols - 1),
                         scalar_bar_args={"title": "thickness (mm)",
                                            "fmt": "%.1f"},
                         smooth_shading=True)
            if r == 0:
                pl.add_text(titles[c], position="upper_left",
                             font_size=10, color="black")
            if c == 0:
                pl.add_text(view_name, position="upper_right",
                             font_size=10, color="black")
            pl.camera_position = cam
            pl.reset_camera()
    pl.background_color = "white"
    pl.screenshot(str(out_png))
    pl.close()
    print(f"  [ok] {out_png}")


# ---------------------------------------------------------------------------
# 2D proportional projection (single shared ML/AP, isotropic aspect)
# ---------------------------------------------------------------------------
def _proportional_projection(pts: np.ndarray, scalar: np.ndarray,
                              mask: np.ndarray, grid_size: int = 60,
                              padding_frac: float = 0.02, layout=None,
                              agg: str = "mean"):
    AP = pts[:, 0]; ML = pts[:, 2]
    if layout is None:
        v = mask
        if not v.any():
            return np.full((grid_size, grid_size), np.nan), None
        ml_lo, ml_hi = float(ML[v].min()), float(ML[v].max())
        ap_lo, ap_hi = float(AP[v].min()), float(AP[v].max())
        ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
        ml_lo -= ml_r * padding_frac; ml_hi += ml_r * padding_frac
        ap_lo -= ap_r * padding_frac; ap_hi += ap_r * padding_frac
        ml_r = ml_hi - ml_lo; ap_r = ap_hi - ap_lo
        aspect = ml_r / ap_r
        n_cols = grid_size if aspect >= 1.0 else max(int(round(grid_size * aspect)), 1)
        n_rows = max(int(round(grid_size / aspect)), 1) if aspect >= 1.0 else grid_size
        layout = {"ml_lo": ml_lo, "ml_hi": ml_hi, "ap_lo": ap_lo, "ap_hi": ap_hi,
                  "ml_r": ml_r, "ap_r": ap_r, "n_rows": n_rows, "n_cols": n_cols}
    n_rows, n_cols = layout["n_rows"], layout["n_cols"]
    ml_r, ap_r = layout["ml_r"], layout["ap_r"]
    ml_lo, ap_lo = layout["ml_lo"], layout["ap_lo"]

    v = mask & np.isfinite(scalar) if scalar is not None else mask
    if not v.any():
        return np.full((n_rows, n_cols), np.nan), layout
    ml_n = (ML[v] - ml_lo) / ml_r; ap_n = (AP[v] - ap_lo) / ap_r
    ml_b = np.clip((ml_n * n_cols).astype(int), 0, n_cols - 1)
    ap_b = np.clip((ap_n * n_rows).astype(int), 0, n_rows - 1)
    sum_g = np.zeros((n_rows, n_cols), dtype=np.float64)
    cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(sum_g, (ap_b, ml_b), scalar[v] if scalar is not None else 1.0)
    np.add.at(cnt_g, (ap_b, ml_b), 1)
    return np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan), layout


def render_2d_grid(panels, titles, out_png: Path, vmax: float = 4.0):
    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    if n_cols == 1: axes = [axes]
    for c, (g, info) in enumerate(panels):
        im = axes[c].imshow(g, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
                              interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.02).set_label(
            "thickness (mm)", fontsize=8)
        axes[c].set_title(titles[c], fontsize=10)
        axes[c].set_xticks([]); axes[c].set_yticks([])
        for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                                   (0.98, 0.97, "L", "right"),
                                   (0.5, 0.97, "P", "center"),
                                   (0.5, 0.03, "A", "center")):
            axes[c].text(tx, ty, label, transform=axes[c].transAxes,
                          color="white", fontsize=9,
                          va=("top" if ty > 0.5 else "bottom"), ha=ha,
                          bbox=dict(facecolor="black", alpha=0.5,
                                     edgecolor="none", pad=2))
        with np.errstate(invalid="ignore"):
            axes[c].text(0.5, -0.05, info, transform=axes[c].transAxes,
                          fontsize=8, va="top", ha="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  [ok] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--tp", choices=["00m", "48m"], default="00m")
    p.add_argument("--bone", choices=["femur", "tibia"], default="tibia")
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_3d_thickness"))
    p.add_argument("--vmax", type=float, default=4.0)
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    case = cohort[args.case]
    print(f"=== {args.case} {args.bone} {args.tp} ===")

    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)
    r = process_long_shared_mesh(case.seg_00m_path, case.seg_48m_path,
                                  case.side, args.bone, cfg,
                                  compartment_aware=False)

    bone_mesh = r["bone_mesh_00m"]
    aligned_pts = r["aligned_pts_00"]
    subch_prob_pat = r["subch_prob_00_patient"]
    template = pv.read(str(TEMPLATE_PATHS[args.bone]))

    if args.tp == "00m":
        patient_th = r["th00_patient"]
    else:
        patient_th = r["th48_at_00_patient"]

    # 4 thickness variants (one per column)
    th_idw  = remap_k3_idw     (template, aligned_pts, patient_th, subch_prob_pat)
    th_fwd1 = remap_k1_forward (template, aligned_pts, patient_th, subch_prob_pat)
    th_scat = remap_k1_scatter (template, aligned_pts, patient_th, subch_prob_pat)

    # Mask out non-subch template verts for display (NaN -> gray)
    tpl_subch = (np.asarray(template.point_data["subch_prob"]) >= 0.5)
    th_idw_m  = np.where(tpl_subch, th_idw,  np.nan).astype(np.float32)
    th_fwd1_m = np.where(tpl_subch, th_fwd1, np.nan).astype(np.float32)
    th_scat_m = np.where(tpl_subch, th_scat, np.nan).astype(np.float32)

    # Patient thickness for display: keep all bone verts but show NaN where thickness=0
    pat_disp = np.where(patient_th > 0, patient_th, np.nan).astype(np.float32)

    def _stats(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if not len(a): return "no data"
        return f"mean={a.mean():.2f}  >0:{int((a > 0).sum())}/{len(a)}"

    titles = [
        f"PATIENT (raycast)",
        f"TEMPLATE k=3 IDW",
        f"TEMPLATE k=1 fwd (direct)",
        f"TEMPLATE k=1 scatter",
    ]
    info_pat  = _stats(patient_th[patient_th > 0])
    info_idw  = _stats(th_idw_m)
    info_fwd1 = _stats(th_fwd1_m)
    info_scat = _stats(th_scat_m)
    print(f"  patient: {info_pat}")
    print(f"  k=3 IDW: {info_idw}")
    print(f"  k=1 fwd: {info_fwd1}")
    print(f"  k=1 sct: {info_scat}")

    # ------ 3D figure ------
    render_3d_grid(
        [(bone_mesh, pat_disp),
         (template, th_idw_m),
         (template, th_fwd1_m),
         (template, th_scat_m)],
        titles=titles,
        out_png=args.out_dir / f"viz3d_{args.case}_{args.bone}_{args.tp}.png",
        vmax=args.vmax,
    )

    # ------ 2D figure (proportional projection on each side's own bounds) ------
    pat_pts = np.asarray(bone_mesh.points)
    tpl_pts = np.asarray(template.points)
    _, lay_pat = _proportional_projection(pat_pts, None, patient_th > 0)
    _, lay_tpl = _proportional_projection(tpl_pts, None, tpl_subch)
    g_pat, _  = _proportional_projection(pat_pts, patient_th.astype(np.float64),
                                          patient_th > 0, layout=lay_pat)
    g_idw, _  = _proportional_projection(tpl_pts, th_idw.astype(np.float64),
                                          tpl_subch & np.isfinite(th_idw),
                                          layout=lay_tpl)
    g_fwd1, _ = _proportional_projection(tpl_pts, th_fwd1.astype(np.float64),
                                          tpl_subch & np.isfinite(th_fwd1),
                                          layout=lay_tpl)
    g_scat, _ = _proportional_projection(tpl_pts, th_scat.astype(np.float64),
                                          tpl_subch & np.isfinite(th_scat),
                                          layout=lay_tpl)
    render_2d_grid(
        [(g_pat, info_pat),
         (g_idw, info_idw),
         (g_fwd1, info_fwd1),
         (g_scat, info_scat)],
        titles=titles,
        out_png=args.out_dir / f"viz2d_{args.case}_{args.bone}_{args.tp}.png",
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
