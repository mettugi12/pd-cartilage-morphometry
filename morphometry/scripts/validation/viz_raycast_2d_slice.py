"""2D in-plane raycast visualisation on a single PD slice.

For PD data, voxel spacing is anisotropic (0.357 x 0.357 in-plane,
0.75 mm slice axis). 3D raycast suffers because rays through the slice
axis hit voxel stair-steps. A 2D in-plane raycast uses ONLY the in-plane
0.357 mm resolution → no staircase, much cleaner cartilage thickness.

For one slice (axis-2 index), this script:
  1. Extracts bone (gray) and cart (yellow) 2D masks for that slice
  2. Finds bone-surface pixels (boundary of bone mask)
  3. Computes outward 2D normals via gradient of EDT(bone)
  4. Casts a 2D ray from each surface pixel along its normal
       step 0.1 mm, max 5 mm
       first crossing of cart_mask → d_entry
       first non-cart after entry → d_exit
       thickness = d_exit - d_entry
  5. Draws arrows on the slice: origin = surface pixel, direction = normal,
       length = thickness, color = jet (0..vmax mm). Optional dim gray
       arrows where surface is near cart but raycast missed.

Usage:
  python -m scripts.viz_raycast_2d_slice --case 9477205_RIGHT --slice 113 \
      --subsample 8 --show_misses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import (
    binary_erosion, distance_transform_edt, map_coordinates,
    gaussian_filter,
)
from skimage import measure

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import _force_ml_flip, _ml_flip_decision


def _smooth_bone_and_normals(bone2d: np.ndarray, sigma: float = 1.8):
    """Gaussian-smooth the bone mask to remove the per-voxel staircase, then
    derive outward normals from the gradient of the smoothed scalar field.

    The smoothed mask is a soft float in [0, 1]; the level set at 0.5 is the
    implicit smooth bone surface. The negative gradient of the smooth mask
    points from inside (high) toward outside (low) → outward normal.

    Returns:
      bone_smooth (float 2D)  — soft mask
      boundary (bool 2D)      — pixels where bone_smooth crosses 0.5 (1-pixel ring)
      ny_u, nx_u (float 2D)   — unit outward normals at every pixel
      contour_pts (Nx2 float) — smooth polyline tracing bone_smooth = 0.5,
                                 in (y_idx, x_idx) coordinates (sub-pixel)
    """
    soft = gaussian_filter(bone2d.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(soft)               # interior > exterior, so grad points inward
    ny = -gy; nx = -gx                       # flip → outward
    mag = np.sqrt(ny ** 2 + nx ** 2) + 1e-9
    ny_u = ny / mag; nx_u = nx / mag

    # 1-pixel ring around the 0.5 iso-line (use thresholded mask)
    thr = soft >= 0.5
    boundary = thr & ~binary_erosion(thr, structure=np.ones((3, 3)))

    # Sub-pixel smooth contour via marching squares
    try:
        contours = measure.find_contours(soft, level=0.5)
    except Exception:
        contours = []
    contour_pts = np.concatenate(contours, axis=0) if contours else np.zeros((0, 2))
    return soft, boundary, ny_u, nx_u, contour_pts


def _raycast_2d(origin_yx: tuple, direction_yx: tuple,
                 cart_soft: np.ndarray, spacing_yx: tuple,
                 max_mm: float = 5.0, step_mm: float = 0.05,
                 cart_threshold: float = 0.5):
    """Walk in 2D from `origin_yx` (voxel idx) along `direction_yx` (unit
    vector in voxel idx). `cart_soft` is the Gaussian-smoothed cart mask
    (float 0..1) — we threshold at 0.5 along the ray to find sub-pixel
    entry / exit, no staircase. Returns (entry_mm, exit_mm) or (nan, nan).
    """
    n_steps = int(np.ceil(max_mm / step_mm))
    ts = np.arange(1, n_steps + 1) * step_mm
    dy_per_mm = direction_yx[0] / spacing_yx[0]
    dx_per_mm = direction_yx[1] / spacing_yx[1]
    ys = origin_yx[0] + ts * dy_per_mm
    xs = origin_yx[1] + ts * dx_per_mm
    # order=1 linear interp on the smoothed cart field — sub-pixel crossings
    v = map_coordinates(cart_soft, np.stack([ys, xs]),
                         order=1, mode="constant", cval=0.0)
    inside = v > cart_threshold
    if not inside.any():
        return float("nan"), float("nan")
    # First entry = first True
    first = int(np.argmax(inside))
    # First exit = first False at or after first entry+1
    after = inside[first:]
    if (~after).any():
        last_in = first + int(np.argmax(~after)) - 1
    else:
        last_in = n_steps - 1   # cart extended to ray's max; record it
    entry_mm = float(ts[first])
    exit_mm = float(ts[last_in])
    return entry_mm, exit_mm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--bone", choices=["femur", "tibia"], default="tibia")
    p.add_argument("--tp", choices=["00m", "48m"], default="00m")
    p.add_argument("--slice", type=int, default=113,
                   help="Axis-2 slice index. PD-SAG: ~113 is mid-medial plateau for 9477205_RIGHT.")
    p.add_argument("--subsample", type=int, default=4,
                   help="Show every Nth boundary pixel near cart.")
    p.add_argument("--show_misses", action="store_true")
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--near_cart_mm", type=float, default=4.0,
                   help="Restrict arrows to bone-surface pixels within this distance of cart.")
    p.add_argument("--bone_sigma", type=float, default=1.8,
                   help="Gaussian sigma (in pixels) for smoothing the bone mask. "
                        "Higher = smoother surface, less staircase, but blurrier boundary.")
    p.add_argument("--cart_sigma", type=float, default=0.8,
                   help="Gaussian sigma (in pixels) for smoothing the cart mask used in raycasting.")
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_raycast_2d_slice"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    case = cohort[args.case]
    seg_path = case.seg_00m_path if args.tp == "00m" else case.seg_48m_path
    flip = _ml_flip_decision(case.seg_00m_path)

    print(f"=== {args.case} {args.bone} {args.tp} slice={args.slice} ===")
    cfg = PipelineConfig(thickness_method="raycast")
    with _force_ml_flip(flip):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, case.side, "pd", args.bone, config=cfg)
    print(f"  shape={bone3d.shape}  spacing={tuple(round(s,3) for s in spacing)}")
    sl = args.slice
    bone2d = bone3d[:, :, sl].astype(bool)
    cart2d = cart3d[:, :, sl].astype(bool)
    if not bone2d.any() or not cart2d.any():
        print(f"  [skip] slice {sl} has no bone or no cart"); return
    sp_y, sp_x = spacing[0], spacing[1]   # in-plane spacing (mm)

    # Smooth bone + cart masks (Gaussian, sigma in pixels). Removes the per-
    # voxel staircase so normals are smooth and rays don't randomly flip
    # horizontal/vertical at corner pixels.
    bone_soft, boundary, ny_u, nx_u, bone_contour_yx = _smooth_bone_and_normals(
        bone2d, sigma=args.bone_sigma)
    cart_soft = gaussian_filter(cart2d.astype(np.float32), sigma=args.cart_sigma)
    n_boundary = int(boundary.sum())
    print(f"  bone smoothing sigma={args.bone_sigma}px ({args.bone_sigma * sp_y:.2f}mm); "
          f"cart sigma={args.cart_sigma}px ({args.cart_sigma * sp_y:.2f}mm)")

    # Restrict to boundary pixels NEAR cart (so we don't draw arrows on the
    # whole bone outline, only the articular side)
    dt_cart = distance_transform_edt(~cart2d, sampling=(sp_y, sp_x))
    near = dt_cart <= args.near_cart_mm
    boundary_near = boundary & near
    n_near = int(boundary_near.sum())

    ys, xs = np.where(boundary_near)
    # Subsample (preserve spatial spread by taking every Nth in flat order)
    idx = np.arange(len(ys))[::args.subsample]
    ys, xs = ys[idx], xs[idx]
    n_pts = len(ys)
    print(f"  boundary pixels total: {n_boundary}, near cart: {n_near}, "
          f"sampled (every {args.subsample}): {n_pts}")

    # Raycast each sampled boundary pixel — using SMOOTH cart_soft (sub-pixel
    # entry/exit) and SMOOTH bone-derived normals (no staircase flips)
    thicks = np.zeros(n_pts, dtype=np.float32)
    n_hits = 0
    for k in range(n_pts):
        yi, xi = ys[k], xs[k]
        ny, nx = float(ny_u[yi, xi]), float(nx_u[yi, xi])
        d_in, d_out = _raycast_2d(
            (yi, xi), (ny, nx), cart_soft, (sp_y, sp_x),
            max_mm=args.max_mm, step_mm=0.05)
        if np.isfinite(d_in) and np.isfinite(d_out):
            t = d_out - d_in
            if 0 < t < args.max_mm:
                thicks[k] = t
                n_hits += 1
    print(f"  hits: {n_hits}/{n_pts}  ({100*n_hits/max(n_pts,1):.1f}%)  "
          f"mean(>0)={thicks[thicks>0].mean() if n_hits else float('nan'):.2f}mm")

    # ---------- Render ----------
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    # imshow expects (rows, cols). bone2d shape (Y, X) = (448, 444).
    # Use extent so axes are in mm.
    extent_mm = (0, bone2d.shape[1] * sp_x, bone2d.shape[0] * sp_y, 0)
    ax.imshow(np.where(bone2d, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1,
              alpha=0.3, extent=extent_mm, interpolation="nearest")
    ax.imshow(np.where(cart2d, 1.0, np.nan), cmap="autumn_r", vmin=0, vmax=1,
              alpha=0.65, extent=extent_mm, interpolation="nearest")
    # Smooth bone contour (marching squares on Gaussian-blurred mask) — draws
    # the actual smooth surface the raycast normals are derived from
    if len(bone_contour_yx):
        ax.plot(bone_contour_yx[:, 1] * sp_x, bone_contour_yx[:, 0] * sp_y,
                "-", color="#222222", lw=1.4, alpha=0.9, label="smooth bone surface")
    # Smooth cart contour for comparison
    try:
        for c_yx in measure.find_contours(cart_soft, level=0.5):
            ax.plot(c_yx[:, 1] * sp_x, c_yx[:, 0] * sp_y,
                    "-", color="#cc6600", lw=1.0, alpha=0.7)
    except Exception:
        pass

    norm = Normalize(vmin=0, vmax=args.vmax)
    sm = ScalarMappable(norm=norm, cmap="jet_r")
    # Arrow origins in mm:
    ox = xs * sp_x; oy = ys * sp_y
    # Direction in mm space: just the normal unit vector (already unit-norm in voxel idx,
    # but for plotting we want mm direction — since in-plane spacing is isotropic
    # 0.357 mm, the unit normal in voxel idx is the same direction in mm).
    nx_arr = nx_u[ys, xs]; ny_arr = ny_u[ys, xs]
    # Hits
    hit = thicks > 0
    for k in np.where(hit)[0]:
        L = float(thicks[k])
        ax.annotate("", xy=(ox[k] + nx_arr[k] * L, oy[k] + ny_arr[k] * L),
                     xytext=(ox[k], oy[k]),
                     arrowprops=dict(arrowstyle="->",
                                       color=sm.to_rgba(L),
                                       lw=1.3, mutation_scale=8))
    # Misses
    if args.show_misses:
        miss = ~hit
        L_miss = 0.5   # fixed short length in mm
        for k in np.where(miss)[0]:
            ax.annotate("", xy=(ox[k] + nx_arr[k] * L_miss,
                                 oy[k] + ny_arr[k] * L_miss),
                         xytext=(ox[k], oy[k]),
                         arrowprops=dict(arrowstyle="->",
                                           color="#555555", lw=0.6,
                                           alpha=0.4, mutation_scale=5))

    # Crop to ROI around bone+cart to avoid blank space
    bm = bone2d | cart2d
    ry, rx = np.where(bm)
    pad_mm = 5.0
    ax.set_xlim(rx.min() * sp_x - pad_mm, rx.max() * sp_x + pad_mm)
    ax.set_ylim(ry.max() * sp_y + pad_mm, ry.min() * sp_y - pad_mm)  # inverted (img coords)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"{args.case}  {args.bone}  {args.tp}  slice axis2={args.slice}\n"
                  f"2D in-plane raycast — bone (gray), cart (yellow), arrows colored by thickness\n"
                  f"hits {n_hits}/{n_pts}  mean(>0)={thicks[thicks>0].mean() if n_hits else float('nan'):.2f}mm"
                  f"  (in-plane spacing={sp_y:.3f} mm)", fontsize=11)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("thickness (mm)", fontsize=10)
    fig.tight_layout()
    suffix = "_misses" if args.show_misses else ""
    out = args.out_dir / f"raycast_2d_{args.case}_{args.bone}_{args.tp}_slice{args.slice}_ss{args.subsample}{suffix}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
