"""Longitudinal registration diagnostic — 2D-per-slice raycast, 00m vs 48m.

Like `viz_2draycast_to_template` but shows BOTH timepoints so you can spot
per-timepoint template-registration drift. Each timepoint is registered to
the template INDEPENDENTLY (its own bone → template), so a misalignment at
one timepoint stands out.

Per bone, a 2x4 figure:
    Row 0 (3D):  patient 00m | template 00m | patient 48m | template 48m
    Row 1 (2D):  patient 00m | template 00m | patient 48m | template 48m

3D uses the bone-aware oblique camera; 2D uses the proportional projection
(femur M/L flipped, tibia compartment split), via cartilage_validation.viz.

Usage:
  python -m scripts.viz_long_diagnostic_2draycast --case 9477205_RIGHT
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

from cartilage_morphometry import (
    PipelineConfig, TEMPLATE_PATHS, get_subch,
    patient_to_template_align, remap_thickness_to_template,
)
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    _force_ml_flip, _ml_flip_decision, compartment_aware_remap,
)
from cartilage_morphometry.validation.viz import (
    screenshot_3d_thickness, project_thickness_2d_canonical,
)
from scripts.viz_3d_thickness_2dslice import compute_2dslice_thickness_per_vert


def _inherit_compartment(aligned_pts, template_mesh, subch_threshold=0.5):
    tpl_pts = np.asarray(template_mesh.points)
    tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold
    sp_idx = np.where(tpl_subch)[0]
    tree = cKDTree(tpl_pts[sp_idx])
    d, idx = tree.query(np.asarray(aligned_pts), k=1)
    return tpl_comp[sp_idx][idx].astype(np.int8), d.astype(np.float32)


def _process(seg_path, side, bone, template_mesh, args, flip_override):
    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)
    with _force_ml_flip(flip_override):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, "pd", bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing,
                                            bone_smooth_iters=5)
    th_2d = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm)
    aligned_pts, *_ = patient_to_template_align(
        bone_mesh, cart3d, spacing, template_mesh, bone, "pd")
    subch_fn = get_subch(cfg.subch_method)
    subch_prob, _ = subch_fn(aligned_pts, template_mesh, cart3d, spacing,
                              bone_mesh, th_2d, cfg)
    has_comp = "compartment" in template_mesh.point_data
    if has_comp:
        inh, _ = _inherit_compartment(aligned_pts, template_mesh)
        tpl_th, _ = compartment_aware_remap(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, inh, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
    else:
        inh = np.zeros(len(aligned_pts), dtype=np.int8)
        tpl_th, _ = remap_thickness_to_template(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
        tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= cfg.subch_threshold
        tpl_th = tpl_th.copy(); tpl_th[~tpl_subch] = np.nan
    return {"bone_mesh": bone_mesh, "patient_th": th_2d,
            "template_th": tpl_th, "subch_prob": subch_prob,
            "inh_comp": inh, "aligned_pts": aligned_pts}


def _imshow(ax, g, title, vmax):
    im = ax.imshow(g, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
                    interpolation="bilinear", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mm", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    with np.errstate(invalid="ignore"):
        ax.text(0.5, -0.05, f"mean={np.nanmean(g):.2f}mm",
                transform=ax.transAxes, fontsize=8, va="top", ha="center")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--bones", nargs="+", choices=["tibia", "femur"],
                   default=["tibia", "femur"])
    p.add_argument("--bone_sigma", type=float, default=1.8)
    p.add_argument("--cart_sigma", type=float, default=0.8)
    p.add_argument("--near_cart_mm", type=float, default=4.0)
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--max_query_mm", type=float, default=1.5)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--grid_size", type=int, default=120)
    p.add_argument("--smooth_sigma", type=float, default=1.2)
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_long_diagnostic"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    case = cohort[args.case]
    flip = _ml_flip_decision(case.seg_00m_path)   # consistent flip from 00m

    for bone in args.bones:
        template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
        print(f"\n=== {args.case} {bone} ===")
        res = {}
        for tp, seg in (("00m", case.seg_00m_path), ("48m", case.seg_48m_path)):
            print(f"  [{tp}]")
            res[tp] = _process(seg, case.side, bone, template_mesh, args, flip)

        fig, axes = plt.subplots(2, 4, figsize=(26, 13))
        for col, (tp, who) in enumerate([("00m", "patient"), ("00m", "template"),
                                          ("48m", "patient"), ("48m", "template")]):
            r = res[tp]
            if who == "patient":
                img = screenshot_3d_thickness(r["bone_mesh"], r["patient_th"], bone, vmax=args.vmax)
                g, _ = project_thickness_2d_canonical(
                    bone, r["bone_mesh"], r["patient_th"],
                    grid_size=args.grid_size, smooth_sigma=args.smooth_sigma,
                    subch_prob=r["subch_prob"])
                title3d = f"PATIENT {tp} 3D"; title2d = f"PATIENT {tp} 2D"
            else:
                img = screenshot_3d_thickness(template_mesh, r["template_th"], bone, vmax=args.vmax)
                g, _ = project_thickness_2d_canonical(
                    bone, template_mesh, r["template_th"],
                    grid_size=args.grid_size, smooth_sigma=args.smooth_sigma)
                title3d = f"TEMPLATE {tp} 3D"; title2d = f"TEMPLATE {tp} 2D"
            axes[0, col].imshow(img); axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
            axes[0, col].set_title(title3d, fontsize=11)
            _imshow(axes[1, col], g, title2d, args.vmax)

        fig.suptitle(f"{args.case}  {bone}  — longitudinal registration diagnostic "
                      f"(2D-per-slice raycast, independent per-timepoint registration)",
                      fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = args.out_dir / f"longdiag_{args.case}_{bone}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
