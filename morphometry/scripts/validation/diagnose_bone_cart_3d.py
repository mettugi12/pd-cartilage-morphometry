"""3D bone+cart viz to assess raycast reliability vs PD/DESS slice resolution.

For each case + bone, produces a 1×3 figure showing the patient bone mesh
(gray, semi-transparent) and cart mesh (yellow, more transparent) from 3
camera angles:

  [1] Top-down (superior → inferior)  — shows the plateau cart distribution
  [2] Antero-lateral oblique          — shows surface detail
  [3] Lateral slice-edge view         — shows DISCRETENESS of cart slices
                                        (the source of raycast unreliability)

Cart mesh is built via marching_cubes on the RECON-densified cart_mask
(PD: 4× upsampled to ~0.75 mm slab; DESS: ~0.7 mm native). Even after
densification, slice-direction voxel boundaries create stair-steps that
raycast can miss.
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

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline
from cartilage_morphometry._helpers import create_pv_mesh

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import _force_ml_flip, _ml_flip_decision


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _build_cart_mesh_raw(cart_mask, spacing, smooth_iters=0):
    """marching_cubes on cart_mask with MINIMAL smoothing — preserves the
    voxel-step staircase so we can see what raycast actually has to deal
    with. Library's default uses light smoothing which hides it."""
    if cart_mask.sum() == 0:
        return None
    cv, cf, _, _ = measure.marching_cubes(
        cart_mask.astype(float), level=0.5, spacing=spacing, step_size=1,
    )
    mesh = create_pv_mesh(cv, cf)
    if smooth_iters > 0:
        mesh = mesh.smooth(n_iter=smooth_iters, relaxation_factor=0.1)
    return mesh


def _build_bone_mesh_raw(bone_mask, spacing):
    bone_closed = binary_closing(bone_mask, structure=np.ones((3, 3, 3)), iterations=2)
    bone_smooth = gaussian_filter(bone_closed.astype(float), sigma=1.0)
    bv, bf, _, _ = measure.marching_cubes(
        bone_smooth, level=0.5, spacing=spacing, step_size=1,
    )
    return create_pv_mesh(bv, bf)


def _set_camera(pl, mesh, view: str):
    pts = np.asarray(mesh.points)
    center = pts.mean(axis=0)
    bounds = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bounds))
    d = 1.4 * diag
    if view == "top_down":
        # Superior → inferior, looking down at plateau
        cam = (center[0], center[1] - d, center[2])
        view_up = (-1.0, 0.0, 0.0)   # anterior up
    elif view == "antero_lateral":
        # Anterior-lateral oblique
        cam = (center[0] - 0.6 * d, center[1] - 0.4 * d, center[2] + 0.4 * d)
        view_up = (0.0, -1.0, 0.0)  # superior up
    elif view == "lateral_slice":
        # Pure lateral view, showing slice-direction staircase
        cam = (center[0], center[1], center[2] + 1.4 * d)  # camera lateral (+ML)
        view_up = (0.0, -1.0, 0.0)
    pl.camera_position = [cam, tuple(center), view_up]
    pl.camera.zoom(1.1)


def _shot(ax, bone_mesh, cart_mesh, view: str, title: str):
    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.set_background("white")
    pl.add_mesh(bone_mesh, color="#dddddd", opacity=0.55,
                smooth_shading=True, show_scalar_bar=False)
    if cart_mesh is not None and cart_mesh.n_points:
        pl.add_mesh(cart_mesh, color="#f1c40f", opacity=0.85,
                    smooth_shading=False,  # off → see voxel staircase
                    show_scalar_bar=False)
    _set_camera(pl, bone_mesh, view)
    img = pl.screenshot(return_img=True, transparent_background=False)
    pl.close()
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA[:2])
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--timepoint", choices=["00m", "48m"], default="00m")
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_bone_cart_3d"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None: continue
        seg_path = case.seg_00m_path if args.timepoint == "00m" else case.seg_48m_path
        flip = _ml_flip_decision(case.seg_00m_path)  # consistent flip from 00m
        for bone in args.bones:
            print(f"\n=== {case_tag} {bone} {args.timepoint} ===")
            _pipeline.clear_recon_cache()
            cfg = PipelineConfig(thickness_method="raycast")
            try:
                with _force_ml_flip(flip):
                    bone_mask, cart_mask, spacing, _, _, _ = _pipeline.get_patient_masks(
                        seg_path, case.side, "pd", bone, config=cfg,
                    )
            except Exception as e:
                print(f"  [ERR get_masks] {e}"); continue
            print(f"  spacing={tuple(round(s,3) for s in spacing)}  "
                  f"bone={int(bone_mask.sum())}  cart={int(cart_mask.sum())}")
            bone_mesh = _build_bone_mesh_raw(bone_mask, spacing)
            # Two cart meshes: raw (no smooth, shows staircase) and smoothed (library default)
            cart_raw = _build_cart_mesh_raw(cart_mask, spacing, smooth_iters=0)
            cart_smooth = _build_cart_mesh_raw(cart_mask, spacing, smooth_iters=15)

            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            for r, (cart, label) in enumerate(((cart_raw, "RAW cart mesh — voxel staircase visible"),
                                                (cart_smooth, "SMOOTHED cart mesh — library default"))):
                _shot(axes[r, 0], bone_mesh, cart,  "top_down",       f"{label}\nTOP-DOWN (superior → inferior)")
                _shot(axes[r, 1], bone_mesh, cart,  "antero_lateral", f"{label}\nANTERO-LATERAL OBLIQUE")
                _shot(axes[r, 2], bone_mesh, cart,  "lateral_slice",  f"{label}\nLATERAL VIEW (slice direction)")
            fig.suptitle(f"{case_tag}  {bone}  {args.timepoint} — patient bone (gray) + cart (yellow)  "
                         f"spacing={tuple(round(s,2) for s in spacing)} mm",
                         fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = args.out_dir / f"bone_cart_3d_{case_tag}_{bone}_{args.timepoint}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
