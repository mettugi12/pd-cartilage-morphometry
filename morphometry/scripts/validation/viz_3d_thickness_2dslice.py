"""Per-vertex thickness via 2D-per-slice raycast, rendered in 3D + 2D.

For each axis-2 (sagittal) slice with bone + cart:
  1. Smooth bone + cart masks (Gaussian, removes voxel staircase)
  2. Find bone-surface pixels (1-ring around bone_soft level 0.5)
  3. Compute outward 2D normals from -grad(bone_soft)
  4. Cast 2D ray (step 0.05 mm, max 6 mm) on smoothed cart_soft → entry/exit
  5. Record thickness = exit - entry at each surface pixel

Aggregate to 3D bone mesh vertices: for each vert, take its z slice
(round vert_z / sp_z), KD-tree query the nearest 2D-measured surface pixel
in that slice (within `--max_query_mm`), inherit that thickness.

Renders the SAME 3D + 2D figure layout as `viz_3d_thickness.py`:
  [PATIENT 3D raycast]   [PATIENT 2D-per-slice raycast]   [delta]

Usage:
  python -m scripts.viz_3d_thickness_2dslice --case 9477205_RIGHT --bone tibia --tp 00m
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import (
    binary_erosion, distance_transform_edt, gaussian_filter, map_coordinates,
)
from scipy.spatial import cKDTree

from cartilage_morphometry import PipelineConfig, get_thickness
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import _force_ml_flip, _ml_flip_decision


# ---------------------------------------------------------------------------
# 2D-slice raycast (vectorised over surface pixels of one slice)
# ---------------------------------------------------------------------------
def _smooth_bone_normals_2d(bone2d: np.ndarray, sigma: float):
    soft = gaussian_filter(bone2d.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(soft)
    ny = -gy; nx = -gx
    mag = np.sqrt(ny ** 2 + nx ** 2) + 1e-9
    ny /= mag; nx /= mag
    thr = soft >= 0.5
    boundary = thr & ~binary_erosion(thr, structure=np.ones((3, 3)))
    return soft, boundary, ny, nx


def _raycast_2d_batch(ys: np.ndarray, xs: np.ndarray,
                       ny: np.ndarray, nx: np.ndarray,
                       cart_soft: np.ndarray, spacing_yx: tuple,
                       max_mm: float = 6.0, step_mm: float = 0.05,
                       cart_threshold: float = 0.5):
    """Vectorised 2D raycast for an array of starting pixels.
    Returns thickness (mm) per pixel, 0 where no clean 2-edge crossing."""
    if not len(ys):
        return np.zeros(0, dtype=np.float32)
    n_steps = int(np.ceil(max_mm / step_mm))
    ts = np.arange(1, n_steps + 1) * step_mm                         # (S,)
    dy_per_mm = ny / spacing_yx[0]                                   # (P,)
    dx_per_mm = nx / spacing_yx[1]                                   # (P,)
    # Coords (S, P): step s × pixel p
    ys_grid = ys[None, :] + ts[:, None] * dy_per_mm[None, :]
    xs_grid = xs[None, :] + ts[:, None] * dx_per_mm[None, :]
    v = map_coordinates(cart_soft, np.stack([ys_grid.ravel(), xs_grid.ravel()]),
                         order=1, mode="constant", cval=0.0).reshape(n_steps, -1)
    inside = v > cart_threshold                                      # (S, P)

    has_any = inside.any(axis=0)
    first = np.argmax(inside, axis=0)                                # first True idx (or 0 if no Trues)
    # For pixels with no hit, mark first = -1 to skip
    first_mask = has_any                                             # bool (P,)

    # Find first exit (first False at index >= first + 1)
    # Build "after" pattern: a False at position s in (first..n_steps]
    rng = np.arange(n_steps)[:, None]                                # (S, 1)
    after_first = rng >= (first[None, :] + 1)                        # (S, P): True for s > first
    out_seq = (~inside) & after_first                                # (S, P): first False after first hit
    has_exit = out_seq.any(axis=0)
    exit_idx = np.argmax(out_seq, axis=0)
    # If no exit, take last in-cart index = first + run length
    last_in_idx = np.where(has_exit, exit_idx - 1, n_steps - 1)

    entry_mm = ts[first]
    exit_mm = ts[np.clip(last_in_idx, 0, n_steps - 1)]
    thickness = (exit_mm - entry_mm).astype(np.float32)
    thickness[~first_mask] = 0.0
    thickness[thickness < 0] = 0.0
    thickness[thickness >= max_mm] = 0.0
    return thickness


def compute_2dslice_thickness_per_vert(bone_mesh, bone3d, cart3d, spacing,
                                         bone_sigma: float = 1.8,
                                         cart_sigma: float = 0.8,
                                         near_cart_mm: float = 4.0,
                                         max_mm: float = 6.0,
                                         max_query_mm: float = 1.5):
    """Per-3D-vertex thickness from 2D-per-slice raycast.

    Loops over axis-2 slices, runs the 2D raycast on every boundary pixel
    near cart, then for each bone_mesh vert looks up the nearest measured
    surface pixel ON ITS SLICE and inherits that thickness. Verts whose
    nearest 2D pixel is farther than `max_query_mm` (or whose slice had no
    cart) get thickness 0.
    """
    sp_y, sp_x, sp_z = spacing
    n_slices = bone3d.shape[2]
    per_slice = [None] * n_slices

    t0 = time.time()
    n_total_hits = 0; n_total_pts = 0
    for z in range(n_slices):
        b2d = bone3d[:, :, z].astype(bool)
        c2d = cart3d[:, :, z].astype(bool)
        if not b2d.any() or not c2d.any():
            continue
        soft, boundary, ny, nx = _smooth_bone_normals_2d(b2d, sigma=bone_sigma)
        cart_soft = gaussian_filter(c2d.astype(np.float32), sigma=cart_sigma)
        ys, xs = np.where(boundary)
        if not len(ys):
            continue
        # Near-cart filter
        dt_cart = distance_transform_edt(~c2d, sampling=(sp_y, sp_x))
        keep = dt_cart[ys, xs] <= near_cart_mm
        ys = ys[keep]; xs = xs[keep]
        if not len(ys):
            continue
        ny_pts = ny[ys, xs]; nx_pts = nx[ys, xs]
        thick = _raycast_2d_batch(ys.astype(np.float64), xs.astype(np.float64),
                                    ny_pts, nx_pts, cart_soft, (sp_y, sp_x),
                                    max_mm=max_mm)
        per_slice[z] = (ys, xs, thick)
        n_total_pts += len(ys); n_total_hits += int((thick > 0).sum())
    print(f"  2D-per-slice raycast: {n_total_hits}/{n_total_pts} pixel hits "
          f"({100*n_total_hits/max(n_total_pts,1):.1f}%)  in {time.time()-t0:.1f}s")

    # Aggregate to 3D bone mesh
    pts = np.asarray(bone_mesh.points, dtype=np.float64)
    vert_th = np.zeros(len(pts), dtype=np.float32)
    z_idx = np.clip(np.round(pts[:, 2] / sp_z).astype(int), 0, n_slices - 1)
    # Process per slice batch for cache locality
    n_assigned = 0
    for z in range(n_slices):
        if per_slice[z] is None: continue
        mask = z_idx == z
        if not mask.any(): continue
        ys_s, xs_s, ths_s = per_slice[z]
        # Allow nearest from BOTH neighbouring slices in addition to z if very few hits
        pix_mm = np.stack([ys_s * sp_y, xs_s * sp_x], axis=1)
        tree = cKDTree(pix_mm)
        v_yx = pts[mask][:, :2]
        d, idx = tree.query(v_yx, k=1)
        within = d <= max_query_mm
        if within.any():
            assigned_idx = np.where(mask)[0][within]
            vert_th[assigned_idx] = ths_s[idx[within]]
            n_assigned += int(within.sum())
    print(f"  mesh aggregation: {n_assigned}/{len(pts)} verts assigned "
          f"({100*n_assigned/len(pts):.1f}%)")
    return vert_th


# ---------------------------------------------------------------------------
# 3D + 2D rendering (mirror viz_3d_thickness.py layout)
# ---------------------------------------------------------------------------
_VIEWS = [
    ("superior",  [(0, 0,  250), (0, 0, 0), (0, 1, 0)]),
    ("anterior",  [(0, -250, 0), (0, 0, 0), (0, 0, 1)]),
]


def render_3d_grid(mesh: pv.PolyData, scalar_arrays: list, titles: list,
                    out_png: Path, vmax: float = 4.0):
    n_cols = len(scalar_arrays)
    n_rows = len(_VIEWS)
    pl = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                    window_size=(600 * n_cols, 600 * n_rows))
    for r, (view_name, cam) in enumerate(_VIEWS):
        for c, sc in enumerate(scalar_arrays):
            pl.subplot(r, c)
            m = mesh.copy()
            sc_arr = np.asarray(sc, dtype=np.float32).copy()
            sc_arr = np.where(sc_arr > 0, sc_arr, np.nan)
            m["thickness_mm"] = sc_arr
            pl.add_mesh(m, scalars="thickness_mm", cmap="jet_r",
                         clim=(0, vmax), nan_color=(0.85, 0.85, 0.85),
                         show_scalar_bar=(c == n_cols - 1),
                         scalar_bar_args={"title": "thickness (mm)", "fmt": "%.1f"},
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


def _project_2d(bone_mesh, scalar, mask, grid_size=80, layout=None):
    pts = np.asarray(bone_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    if layout is None:
        v = mask
        if not v.any(): return np.full((grid_size, grid_size), np.nan), None
        ml_lo, ml_hi = float(ML[v].min()), float(ML[v].max())
        ap_lo, ap_hi = float(AP[v].min()), float(AP[v].max())
        ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
        aspect = ml_r / ap_r
        n_cols = grid_size if aspect >= 1.0 else max(int(round(grid_size * aspect)), 1)
        n_rows = max(int(round(grid_size / aspect)), 1) if aspect >= 1.0 else grid_size
        layout = {"ml_lo": ml_lo, "ap_lo": ap_lo, "ml_r": ml_r, "ap_r": ap_r,
                  "n_rows": n_rows, "n_cols": n_cols}
    n_rows, n_cols = layout["n_rows"], layout["n_cols"]
    v = mask & np.isfinite(scalar)
    if not v.any(): return np.full((n_rows, n_cols), np.nan), layout
    ml_b = np.clip(((ML[v] - layout["ml_lo"]) / layout["ml_r"] * n_cols).astype(int), 0, n_cols - 1)
    ap_b = np.clip(((AP[v] - layout["ap_lo"]) / layout["ap_r"] * n_rows).astype(int), 0, n_rows - 1)
    sum_g = np.zeros((n_rows, n_cols), dtype=np.float64)
    cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(sum_g, (ap_b, ml_b), scalar[v])
    np.add.at(cnt_g, (ap_b, ml_b), 1)
    return np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan), layout


def render_2d_grid(bone_mesh, scalar_arrays, titles, out_png: Path, vmax=4.0):
    # Define shared 2D layout from "any vert with positive thickness in EITHER method"
    union_mask = np.zeros(bone_mesh.n_points, dtype=bool)
    for sc in scalar_arrays:
        union_mask |= np.asarray(sc) > 0
    _, layout = _project_2d(bone_mesh, scalar_arrays[0].astype(np.float64), union_mask)
    fig, axes = plt.subplots(1, len(scalar_arrays), figsize=(5 * len(scalar_arrays), 5.5))
    if len(scalar_arrays) == 1: axes = [axes]
    for c, sc in enumerate(scalar_arrays):
        g, _ = _project_2d(bone_mesh, sc.astype(np.float64), sc > 0, layout=layout)
        if "delta" in titles[c].lower():
            im = axes[c].imshow(g, origin="upper", cmap="RdBu",
                                  vmin=-vmax/2, vmax=vmax/2,
                                  interpolation="nearest", aspect="equal")
            plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.02).set_label(
                "Δ thickness (mm)", fontsize=8)
        else:
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
            pos = np.asarray(sc)[np.asarray(sc) > 0]
            info = (f"mean(>0)={pos.mean():.2f}mm  n>0={len(pos)}/{len(sc)} "
                     f"({100*len(pos)/len(sc):.0f}%)") if len(pos) else "no hits"
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
    p.add_argument("--bone", choices=["femur", "tibia"], default="tibia")
    p.add_argument("--tp", choices=["00m", "48m"], default="00m")
    p.add_argument("--bone_sigma", type=float, default=1.8)
    p.add_argument("--cart_sigma", type=float, default=0.8)
    p.add_argument("--near_cart_mm", type=float, default=4.0)
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--max_query_mm", type=float, default=1.5)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_3d_thickness_2dslice"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    case = cohort[args.case]
    seg_path = case.seg_00m_path if args.tp == "00m" else case.seg_48m_path
    flip = _ml_flip_decision(case.seg_00m_path)

    print(f"=== {args.case} {args.bone} {args.tp} ===")
    cfg = PipelineConfig(thickness_method="raycast")
    with _force_ml_flip(flip):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, case.side, "pd", args.bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing,
                                            bone_smooth_iters=5)
    print(f"  bone_mesh: {bone_mesh.n_points} verts  "
          f"spacing={tuple(round(s,3) for s in spacing)}")

    # 1) Library 3D raycast
    th_3d = get_thickness("raycast")(bone_mesh, bone3d, cart3d, spacing, cfg).astype(np.float32)
    n3 = int((th_3d > 0).sum())
    m3 = th_3d[th_3d > 0].mean() if n3 else float("nan")
    print(f"  3D raycast: hits {n3}/{len(th_3d)} ({100*n3/len(th_3d):.1f}%)  mean(>0)={m3:.2f}mm")

    # 2) 2D-per-slice raycast aggregated to verts
    th_2d = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm)
    n2 = int((th_2d > 0).sum())
    m2 = th_2d[th_2d > 0].mean() if n2 else float("nan")
    print(f"  2D-slice  : hits {n2}/{len(th_2d)} ({100*n2/len(th_2d):.1f}%)  mean(>0)={m2:.2f}mm")

    delta = th_2d - th_3d

    # Render 3D
    render_3d_grid(
        bone_mesh,
        [th_3d, th_2d, delta],
        titles=["3D raycast (library)", "2D-per-slice raycast", "Δ (2D − 3D)"],
        out_png=args.out_dir / f"viz3d_{args.case}_{args.bone}_{args.tp}.png",
        vmax=args.vmax,
    )

    # Render 2D
    render_2d_grid(
        bone_mesh,
        [th_3d, th_2d, delta],
        titles=["3D raycast (library)", "2D-per-slice raycast", "Δ (2D − 3D)"],
        out_png=args.out_dir / f"viz2d_{args.case}_{args.bone}_{args.tp}.png",
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
