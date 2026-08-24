"""Interactive patient-frame longitudinal check — 48m → 00m, NO template.

Registers the 48m bone DIRECTLY onto the 00m bone (articular rigid ICP in
patient mm), IDW-samples 48m thickness onto the 00m bone verts, and shows
patient-frame 2D maps of 00m / 48m / Δ side by side. This bypasses the
per-timepoint template registration entirely, so the 00m↔48m comparison
carries only thickness change, not template-alignment drift.

Opens an interactive matplotlib window (does NOT save). Close it to exit.

Usage:
  python -m scripts.viz_long_patient_frame_interactive --case 9477205_RIGHT
"""
from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    _force_ml_flip, _ml_flip_decision, _articular_icp, _idw_sample, _bone_assd,
)
from cartilage_morphometry.validation.viz import (
    project_thickness_2d_canonical, proportional_projection_2d,
    _smooth_border_preserving,
)
from scripts.viz_3d_thickness_2dslice import compute_2dslice_thickness_per_vert


def _thickness_2d(seg_path, side, bone, flip, args):
    cfg = PipelineConfig(thickness_method="raycast")
    with _force_ml_flip(flip):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, "pd", bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing,
                                            bone_smooth_iters=5)
    th = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm)
    return bone_mesh, bone3d, cart3d, spacing, th


def _proj_patient(bone, bone_mesh, scalar, grid_size, smooth_sigma, layout=None,
                   agg="mean"):
    """Patient-frame 2D projection (canonical: femur unwrap, tibia proportional)."""
    g, layout = project_thickness_2d_canonical(
        bone, bone_mesh, scalar, grid_size=grid_size, smooth_sigma=smooth_sigma,
        subch_prob=(np.abs(scalar) > 0).astype(np.float32), tibia_layout=layout,
        agg=agg)
    return g, layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--bones", nargs="+", choices=["tibia", "femur"],
                   default=["tibia", "femur"])
    p.add_argument("--articular_radius_mm", type=float, default=5.0)
    p.add_argument("--bone_sigma", type=float, default=1.8)
    p.add_argument("--cart_sigma", type=float, default=0.8)
    p.add_argument("--near_cart_mm", type=float, default=4.0)
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--max_query_mm", type=float, default=1.5)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--vdelta", type=float, default=1.5)
    p.add_argument("--grid_size", type=int, default=120)
    p.add_argument("--smooth_sigma", type=float, default=1.2)
    args = p.parse_args()

    # Switch to an interactive backend before creating any figure
    # (cartilage_validation.viz forced Agg at import → no window otherwise).
    for _bk in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            plt.switch_backend(_bk); break
        except Exception:
            continue
    print(f"matplotlib backend = {matplotlib.get_backend()}")

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    case = cohort[args.case]
    flip = _ml_flip_decision(case.seg_00m_path)

    n_bones = len(args.bones)
    fig, axes = plt.subplots(n_bones, 3, figsize=(18, 6 * n_bones), squeeze=False)

    for r, bone in enumerate(args.bones):
        print(f"\n=== {args.case} {bone} ===")
        m00, b00, c00, sp00, th00 = _thickness_2d(case.seg_00m_path, case.side, bone, flip, args)
        m48, b48, c48, sp48, th48 = _thickness_2d(case.seg_48m_path, case.side, bone, flip, args)

        p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
        assd_before = _bone_assd(p48, p00)
        p48_aligned, A, t, used_art, n48a, n00a, _ = _articular_icp(
            p48, c48, sp48, p00, c00, sp00,
            articular_radius_mm=args.articular_radius_mm)
        assd_after = _bone_assd(p48_aligned, p00)
        rot = float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
        print(f"  48m→00m articular ICP: ASSD {assd_before:.2f}→{assd_after:.2f}mm "
              f"rot={rot:.1f}° |t|={np.linalg.norm(t):.2f}mm  (art={used_art})")

        # 48m thickness IDW-sampled onto 00m bone verts (shared sampling surface)
        th48_at_00 = _idw_sample(p00, p48_aligned, th48, k=3).astype(np.float32)
        delta = th48_at_00 - th00

        # Patient-frame 2D maps on the SAME 00m bone (shared layout)
        g00, layout = _proj_patient(bone, m00, th00, args.grid_size, args.smooth_sigma)
        g48, _ = _proj_patient(bone, m00, th48_at_00, args.grid_size, args.smooth_sigma, layout=layout)
        gd, _ = _proj_patient(bone, m00, delta, args.grid_size, args.smooth_sigma, layout=layout)

        for c, (g, title, cmap, vlo, vhi, cblab) in enumerate([
            (g00, f"{bone} 00m", "jet", 0, args.vmax, "mm"),
            (g48, f"{bone} 48m→00m", "jet", 0, args.vmax, "mm"),
            (gd,  f"{bone} Δ (48m−00m)", "RdBu_r", -args.vdelta, args.vdelta, "Δmm"),
        ]):
            ax = axes[r, c]
            im = ax.imshow(g, origin="upper", cmap=cmap, vmin=vlo, vmax=vhi,
                            interpolation="bilinear", aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(cblab, fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            a_y, p_y = (0.97, 0.03) if bone == "femur" else (0.03, 0.97)
            for tx, ty, lab, ha in ((0.98, 0.97, "M", "right"), (0.02, 0.97, "L", "left"),
                                     (0.5, a_y, "A", "center"), (0.5, p_y, "P", "center")):
                ax.text(tx, ty, lab, transform=ax.transAxes, color="white", fontsize=9,
                        va=("top" if ty > 0.5 else "bottom"), ha=ha,
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))
            with np.errstate(invalid="ignore"):
                ax.text(0.5, -0.05, f"mean={np.nanmean(g):+.3f}mm",
                        transform=ax.transAxes, fontsize=8, va="top", ha="center")

    fig.suptitle(f"{args.case}  — patient-frame 48m→00m (articular ICP, NO template)",
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    print("\nOpening interactive viewer — close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
