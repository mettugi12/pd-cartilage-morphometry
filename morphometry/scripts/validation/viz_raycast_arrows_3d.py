"""3D bone+cart with raycast vectors visualized as arrows.

For one case + bone + timepoint, renders the patient bone (gray) + cart
(yellow) and overlays raycast result as arrows:

  - origin = bone vertex (subsampled for legibility)
  - direction = bone surface normal (outward)
  - length = measured thickness (raycast hit-to-exit distance)
  - color = thickness magnitude (jet, 0..vmax mm)

Verts where raycast MISSED (thickness = 0) are optionally shown as short
thin dim arrows so you can see which directions are missing cart hits.

Usage:
  python -m scripts.viz_raycast_arrows_3d --case 9477205_RIGHT \
      --bone tibia --tp 00m --subsample 6 --show_misses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage import measure
from scipy.ndimage import binary_closing, gaussian_filter

from cartilage_morphometry import PipelineConfig, get_thickness
from cartilage_morphometry import pipeline as _pipeline
from cartilage_morphometry._helpers import create_pv_mesh

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import _force_ml_flip, _ml_flip_decision


def _build_cart_mesh_raw(cart_mask, spacing):
    if cart_mask.sum() == 0: return None
    cv, cf, _, _ = measure.marching_cubes(
        cart_mask.astype(float), level=0.5, spacing=spacing, step_size=1)
    return create_pv_mesh(cv, cf)


def _build_bone_mesh_raw(bone_mask, spacing):
    bone_closed = binary_closing(bone_mask, structure=np.ones((3, 3, 3)), iterations=2)
    bone_smooth = gaussian_filter(bone_closed.astype(float), sigma=1.0)
    bv, bf, _, _ = measure.marching_cubes(bone_smooth, level=0.5, spacing=spacing, step_size=1)
    return create_pv_mesh(bv, bf)


def _arrow_glyph(origins: np.ndarray, normals: np.ndarray, lengths: np.ndarray,
                  scalar: np.ndarray, factor: float = 1.0):
    """Build a pyvista PolyData of arrow glyphs (one arrow per origin).

    Arrow length = lengths[i] * factor (mm). Direction = normals[i].
    `scalar` is carried on each arrow for colormap.
    """
    pts = pv.PolyData(origins.astype(np.float32))
    n = normals.astype(np.float32) * lengths.reshape(-1, 1).astype(np.float32)
    pts["vec"] = n
    pts["thickness_mm"] = scalar.astype(np.float32)
    arrow = pv.Arrow(tip_length=0.25, tip_radius=0.10, shaft_radius=0.04)
    glyphed = pts.glyph(orient="vec", scale="vec", factor=factor, geom=arrow)
    # Copy the thickness scalar onto the glyph output (pv.glyph propagates if
    # it's a point_data already — but to be safe we re-attach via cell mean).
    return glyphed


def _set_camera(pl, mesh, view: str, zoom: float = 1.0):
    pts = np.asarray(mesh.points)
    center = pts.mean(axis=0)
    bounds = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bounds))
    d = 1.4 * diag
    if view == "top_down":
        cam = (center[0], center[1] - d, center[2])
        view_up = (-1.0, 0.0, 0.0)
    elif view == "antero_lateral":
        cam = (center[0] - 0.6 * d, center[1] - 0.4 * d, center[2] + 0.4 * d)
        view_up = (0.0, -1.0, 0.0)
    elif view == "lateral_slice":
        cam = (center[0], center[1], center[2] + 1.4 * d)
        view_up = (0.0, -1.0, 0.0)
    elif view == "oblique_top":
        # Steeper-than-anterolateral oblique that emphasizes the plateau
        cam = (center[0] - 0.5 * d, center[1] - 0.6 * d, center[2] + 0.8 * d)
        view_up = (0.0, 0.0, 1.0)
    pl.camera_position = [cam, tuple(center), view_up]
    pl.camera.zoom(zoom)


def _shot(ax, bone_mesh, cart_mesh, hits_glyph, misses_glyph,
            view: str, title: str, vmax: float, show_bar: bool = False):
    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.set_background("white")
    pl.add_mesh(bone_mesh, color="#dddddd", opacity=0.30,
                 smooth_shading=True, show_scalar_bar=False)
    if cart_mesh is not None and cart_mesh.n_points:
        pl.add_mesh(cart_mesh, color="#f1c40f", opacity=0.45,
                     smooth_shading=False, show_scalar_bar=False)
    if misses_glyph is not None and misses_glyph.n_points:
        pl.add_mesh(misses_glyph, color="#555555", opacity=0.35,
                     show_scalar_bar=False)
    if hits_glyph is not None and hits_glyph.n_points:
        pl.add_mesh(hits_glyph, scalars="thickness_mm",
                     cmap="jet_r", clim=(0, vmax),
                     show_scalar_bar=show_bar,
                     scalar_bar_args={"title": "thickness (mm)",
                                       "fmt": "%.1f"})
    _set_camera(pl, bone_mesh, view, zoom=1.05)
    img = pl.screenshot(return_img=True, transparent_background=False)
    pl.close()
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--bone", choices=["femur", "tibia"], default="tibia")
    p.add_argument("--tp", choices=["00m", "48m"], default="00m")
    p.add_argument("--subsample", type=int, default=8,
                   help="Show every Nth bone vert (default 8).")
    p.add_argument("--show_misses", action="store_true",
                   help="Also render thin dim arrows where raycast returned 0.")
    p.add_argument("--miss_length_mm", type=float, default=0.8,
                   help="Length of miss arrows (no thickness data, fixed length).")
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--cart_only", action="store_true",
                   help="Show arrows ONLY on bone verts whose tpl-NN is in subch (cart-bearing zone).")
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_raycast_arrows_3d"))
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
        bone_mask, cart_mask, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, case.side, "pd", args.bone, config=cfg)
    print(f"  spacing={tuple(round(s,3) for s in spacing)}  "
          f"bone={int(bone_mask.sum())}  cart={int(cart_mask.sum())}")

    # Library mesh + thickness (bone smoothed exactly as the canonical pipeline)
    bone_mesh_lib, _ = _pipeline.build_meshes(
        bone_mask, cart_mask, spacing, bone_smooth_iters=5)
    thickness = get_thickness("raycast")(
        bone_mesh_lib, bone_mask, cart_mask, spacing, cfg)
    bone_mesh_lib["thickness_mm"] = thickness
    print(f"  bone_mesh={bone_mesh_lib.n_points}v  thick>0={int((thickness > 0).sum())}  "
          f"mean(>0)={thickness[thickness > 0].mean():.2f}mm")

    bone_pts = np.asarray(bone_mesh_lib.points, dtype=np.float64)
    normals = np.asarray(bone_mesh_lib.point_normals, dtype=np.float64)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nrm, 1e-9)

    # Optional restriction to cart-bearing zone (so we don't render shaft arrows)
    if args.cart_only:
        # Mark bone verts within 5 mm of any cart voxel as "near cart"
        from scipy.spatial import cKDTree
        cart_idx = np.argwhere(cart_mask.astype(bool))
        if len(cart_idx):
            cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
            d, _ = cKDTree(cart_mm).query(bone_pts, k=1)
            near_cart = d <= 5.0
        else:
            near_cart = np.ones(len(bone_pts), dtype=bool)
        print(f"  cart-zone restriction: keeping {int(near_cart.sum())}/{len(bone_pts)} bone verts")
    else:
        near_cart = np.ones(len(bone_pts), dtype=bool)

    # Subsample for legibility
    keep = np.zeros(len(bone_pts), dtype=bool)
    keep[::args.subsample] = True
    keep &= near_cart
    n_kept = int(keep.sum())
    print(f"  subsample (every {args.subsample}th, intersected w/ zone): {n_kept} verts")

    # Build separate glyphs for hits vs misses
    hits_mask = keep & (thickness > 0)
    miss_mask = keep & (thickness == 0)
    n_hits = int(hits_mask.sum()); n_misses = int(miss_mask.sum())
    print(f"  hits (will draw thickness arrows): {n_hits}")
    print(f"  misses (will draw fixed-length dim arrows): {n_misses}")

    hits_glyph = None
    if n_hits:
        hits_glyph = _arrow_glyph(
            bone_pts[hits_mask], normals[hits_mask],
            thickness[hits_mask].astype(np.float64),
            thickness[hits_mask].astype(np.float64), factor=1.0)
    misses_glyph = None
    if args.show_misses and n_misses:
        miss_len = np.full(n_misses, args.miss_length_mm, dtype=np.float64)
        misses_glyph = _arrow_glyph(
            bone_pts[miss_mask], normals[miss_mask],
            miss_len, miss_len, factor=1.0)

    bone_mesh_render = _build_bone_mesh_raw(bone_mask, spacing)
    cart_mesh = _build_cart_mesh_raw(cart_mask, spacing)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    title_base = (f"{args.case} {args.bone} {args.tp}  spacing="
                   f"{tuple(round(s,2) for s in spacing)}mm  "
                   f"arrows: hits={n_hits}, misses={n_misses if args.show_misses else 'hidden'}  "
                   f"(every {args.subsample}th vert)")
    _shot(axes[0], bone_mesh_render, cart_mesh, hits_glyph, misses_glyph,
           "top_down",       "TOP-DOWN (superior → inferior)",
           args.vmax, show_bar=False)
    _shot(axes[1], bone_mesh_render, cart_mesh, hits_glyph, misses_glyph,
           "oblique_top",    "OBLIQUE (plateau emphasis)",
           args.vmax, show_bar=True)
    _shot(axes[2], bone_mesh_render, cart_mesh, hits_glyph, misses_glyph,
           "lateral_slice",  "LATERAL (slice-direction edge)",
           args.vmax, show_bar=False)
    fig.suptitle(title_base, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    suffix = "_with_misses" if args.show_misses else ""
    suffix += "_cartonly" if args.cart_only else ""
    out = args.out_dir / f"raycast_arrows_{args.case}_{args.bone}_{args.tp}_ss{args.subsample}{suffix}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
