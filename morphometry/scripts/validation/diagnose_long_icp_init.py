"""Show PRE-ICP overlay of 00m + 48m bones — diagnose whether bad rigid ICP
results come from a bad initialization (huge translation between NIfTI affines)
or from segmentation errors (mismatched bone shapes).

For each requested case + bone, produces a 1×3 figure:
  [pre-ICP raw overlay]  [rigid-aligned overlay]  [bounded_affine-aligned overlay]

00m bone colored blue, 48m bone colored red. Where the two overlap looks
purple. Big gaps = bad alignment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    _articular_icp, _articular_mask, _bone_assd, _build_thickness,
    _force_ml_flip, _ml_flip_decision,
)
from cartilage_morphometry.validation.viz import _set_oblique_top_down


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _overlay_screenshot(ax, mesh_00, mesh_48, bone_name, title, articular_radius_mm=5.0):
    """Render 00m (blue) + 48m (red) on same canvas. Title shows ASSD."""
    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.set_background("white")
    pl.add_mesh(mesh_00, color="#1f77b4", opacity=0.55, smooth_shading=True,
                show_scalar_bar=False)
    pl.add_mesh(mesh_48, color="#d62728", opacity=0.55, smooth_shading=True,
                show_scalar_bar=False)
    _set_oblique_top_down(pl, mesh_00, bone_name)
    img = pl.screenshot(return_img=True, transparent_background=False); pl.close()
    img_mirrored = np.fliplr(img)
    ax.imshow(img_mirrored); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for tx, ty, label in ((-0.04, 0.5, "M"), (1.04, 0.5, "L"),
                          (0.5, 1.04, "A"), (0.5, -0.04, "P")):
        ax.text(tx, ty, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontweight="bold")


def _aligned_mesh(mesh48, A, t):
    out = mesh48.copy()
    pts = np.asarray(out.points)
    out.points = pts @ A.T + t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_long_icp"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    print(f"[recon] disk cache: {DEFAULT_RECON_CACHE_DIR}")

    config = PipelineConfig(thickness_method="raycast")
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None: continue
        # v1.2 fix: consistent ML-flip across 00m and 48m (use 00m's empirical decision)
        flip_override = _ml_flip_decision(case.seg_00m_path)
        print(f"\n=== {case_tag}  flip_override={flip_override} (from 00m) ===")
        for bone in args.bones:
            print(f"-- bone={bone} --")
            _pipeline.clear_recon_cache()
            try:
                with _force_ml_flip(flip_override):
                    m00, b00, c00, sp00, _ = _build_thickness(case.seg_00m_path, case.side, "pd", bone, config)
            except Exception as e:
                print(f"  [err 00m] {e}"); continue
            try:
                with _force_ml_flip(flip_override):
                    m48, b48, c48, sp48, _ = _build_thickness(case.seg_48m_path, case.side, "pd", bone, config)
            except Exception as e:
                print(f"  [err 48m] {e}"); continue

            p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
            assd_pre = _bone_assd(p48, p00)
            cen_off = p48.mean(axis=0) - p00.mean(axis=0)
            print(f"  pre-ICP: ASSD={assd_pre:.2f}mm centroid_offset={np.linalg.norm(cen_off):.2f}mm "
                  f"(P,I,R)={tuple(round(x,1) for x in cen_off)}")
            # Rigid art-ICP
            p48_rig, R_rig, t_rig, used_rig, *_, _ = _articular_icp(
                p48, c48, sp48, p00, c00, sp00, articular_radius_mm=5.0, method="rigid")
            assd_rig = _bone_assd(p48_rig, p00)
            print(f"  rigid:   ASSD={assd_rig:.2f}mm |t|={np.linalg.norm(t_rig):.2f}mm "
                  f"rot={float(np.degrees(np.arccos(np.clip((np.trace(R_rig)-1)/2,-1,1)))):.1f}°")
            # Bounded affine
            p48_baf, A_baf, t_baf, used_baf, *_, _ = _articular_icp(
                p48, c48, sp48, p00, c00, sp00, articular_radius_mm=5.0, method="bounded_affine")
            assd_baf = _bone_assd(p48_baf, p00)
            sv_baf = np.linalg.svd(A_baf, compute_uv=False)
            print(f"  baff:    ASSD={assd_baf:.2f}mm |t|={np.linalg.norm(t_baf):.2f}mm "
                  f"sv=[{sv_baf[0]:.2f},{sv_baf[1]:.2f},{sv_baf[2]:.2f}]")

            m48_rig = _aligned_mesh(m48, R_rig, t_rig)
            m48_baf = _aligned_mesh(m48, A_baf, t_baf)

            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            _overlay_screenshot(axes[0], m00, m48, bone,
                f"PRE-ICP (raw NIfTI mm)\nASSD={assd_pre:.2f}mm  centroid_off={np.linalg.norm(cen_off):.1f}mm")
            _overlay_screenshot(axes[1], m00, m48_rig, bone,
                f"RIGID art-ICP (5mm)\nASSD={assd_rig:.2f}mm  rot={float(np.degrees(np.arccos(np.clip((np.trace(R_rig)-1)/2,-1,1)))):.1f}°  |t|={np.linalg.norm(t_rig):.1f}mm")
            _overlay_screenshot(axes[2], m00, m48_baf, bone,
                f"BOUNDED_AFFINE art-ICP (5mm)\nASSD={assd_baf:.2f}mm  sv=[{sv_baf[0]:.2f},{sv_baf[1]:.2f},{sv_baf[2]:.2f}]  |t|={np.linalg.norm(t_baf):.1f}mm")
            fig.suptitle(f"{case_tag}  {bone} — 00m (blue) vs 48m (red) overlay", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = args.out_dir / f"icp_init_{case_tag}_{bone}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"[ok] {out}")


if __name__ == "__main__":
    main()
