"""Interactive shared-mesh longitudinal check — 48m→00m→template (consistent).

The CONSISTENT recipe (v6 shared-mesh), with 2D-per-slice raycast thickness:
  1. 2D-per-slice raycast thickness on the 00m bone (th00) and 48m bone (th48)
  2. 48m → 00m bone articular rigid ICP (patient-to-patient)
  3. resample th48 onto the 00m bone verts  → th48_at_00
  4. register the 00m bone to the template ONCE  → aligned_00
  5. remap BOTH th00 and th48_at_00 to the template using the SAME aligned_00
     + the SAME subch/compartment  → tpl00, tpl48
  6. Δ = tpl48 − tpl00  (template frame; zero inter-timepoint registration drift)

Shows, per bone, two rows in an interactive window:
  Row A (patient frame): 00m | 48m→00m | Δ   (no template)
  Row B (template frame): tpl00 | tpl48 | Δ   (single shared 00m→template map)

Opens an interactive matplotlib window (does NOT save). Close it to exit.

Usage:
  python -m scripts.viz_long_sharedmesh_interactive --case 9477205_RIGHT
"""
from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    _force_ml_flip, _ml_flip_decision, _articular_icp, _idw_sample, _bone_assd,
    compartment_aware_remap,
)
import pyvista as pv
from cartilage_morphometry.validation.viz import project_thickness_2d_canonical
from scripts.viz_3d_thickness_2dslice import compute_2dslice_thickness_per_vert


def _thickness_2d(seg_path, side, bone, flip, args):
    cfg = PipelineConfig(thickness_method="raycast")
    with _force_ml_flip(flip):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, "pd", bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing, bone_smooth_iters=5)
    th = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm)
    return bone_mesh, bone3d, cart3d, spacing, th


def _inherit_compartment(aligned_pts, template_mesh, thr=0.5):
    tpl_pts = np.asarray(template_mesh.points)
    tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= thr
    sp_idx = np.where(tpl_subch)[0]
    d, idx = cKDTree(tpl_pts[sp_idx]).query(np.asarray(aligned_pts), k=1)
    return tpl_comp[sp_idx][idx].astype(np.int8)


def _draw(ax, g, title, cmap, vlo, vhi, cblab, bone):
    im = ax.imshow(g, origin="upper", cmap=cmap, vmin=vlo, vmax=vhi,
                    interpolation="bilinear", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(cblab, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    a_y, p_y = (0.97, 0.03) if bone == "femur" else (0.03, 0.97)
    for tx, ty, lab, ha in ((0.98, 0.97, "M", "right"), (0.02, 0.97, "L", "left"),
                             (0.5, a_y, "A", "center"), (0.5, p_y, "P", "center")):
        ax.text(tx, ty, lab, transform=ax.transAxes, color="white", fontsize=9,
                va=("top" if ty > 0.5 else "bottom"), ha=ha,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))
    with np.errstate(invalid="ignore"):
        ax.text(0.5, -0.05, f"mean={np.nanmean(g):+.3f}mm", transform=ax.transAxes,
                fontsize=8, va="top", ha="center")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="9477205_RIGHT")
    p.add_argument("--bones", nargs="+", choices=["tibia", "femur"],
                   default=["tibia", "femur"])
    p.add_argument("--articular_radius_mm", type=float, default=5.0)
    p.add_argument("--icp_method", default="trimmed_rigid",
                   choices=["rigid", "trimmed_rigid"],
                   help="48m→00m bone ICP method. trimmed_rigid rejects "
                        "non-overlapping points (partial-overlap robust).")
    p.add_argument("--trim_fraction", type=float, default=0.8)
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
    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)

    n_bones = len(args.bones)
    fig, axes = plt.subplots(2 * n_bones, 3, figsize=(18, 11 * n_bones), squeeze=False)

    for bi, bone in enumerate(args.bones):
        print(f"\n=== {args.case} {bone} ===")
        template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
        m00, b00, c00, sp00, th00 = _thickness_2d(case.seg_00m_path, case.side, bone, flip, args)
        m48, b48, c48, sp48, th48 = _thickness_2d(case.seg_48m_path, case.side, bone, flip, args)

        # 2) 48m → 00m bone ICP ; 3) resample th48 onto 00m verts
        p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
        p48_aligned, A, t, used_art, *_ = _articular_icp(
            p48, c48, sp48, p00, c00, sp00,
            articular_radius_mm=args.articular_radius_mm,
            method=args.icp_method, trim_fraction=args.trim_fraction)
        rot = float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
        print(f"  48→00 ICP rot={rot:.1f}° |t|={np.linalg.norm(t):.1f}mm "
              f"ASSD→{_bone_assd(p48_aligned, p00):.2f}mm")
        th48_at_00 = _idw_sample(p00, p48_aligned, th48, k=3).astype(np.float32)

        # 4) 00m → template ONCE
        aligned_00, *_ = patient_to_template_align(m00, c00, sp00, template_mesh, bone, "pd")
        subch_fn = get_subch(cfg.subch_method)
        subch_prob, _ = subch_fn(aligned_00, template_mesh, c00, sp00, m00, th00, cfg)

        # 5) remap BOTH timepoints with the SAME aligned_00 + subch
        has_comp = "compartment" in template_mesh.point_data
        if has_comp:
            inh = _inherit_compartment(aligned_00, template_mesh)
            tpl00, _ = compartment_aware_remap(template_mesh, aligned_00, th00.astype(np.float32),
                                                subch_prob, inh, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
            tpl48, _ = compartment_aware_remap(template_mesh, aligned_00, th48_at_00.astype(np.float32),
                                                subch_prob, inh, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
        else:
            tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= cfg.subch_threshold
            tpl00, _ = remap_thickness_to_template(template_mesh, aligned_00, th00.astype(np.float32),
                                                    subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
            tpl48, _ = remap_thickness_to_template(template_mesh, aligned_00, th48_at_00.astype(np.float32),
                                                    subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
            tpl00 = tpl00.copy(); tpl00[~tpl_subch] = np.nan
            tpl48 = tpl48.copy(); tpl48[~tpl_subch] = np.nan

        # ---- patient-frame row ----
        g00p, lay = project_thickness_2d_canonical(bone, m00, th00, grid_size=args.grid_size,
                                                     smooth_sigma=args.smooth_sigma,
                                                     subch_prob=(th00 > 0).astype(np.float32))
        g48p, _ = project_thickness_2d_canonical(bone, m00, th48_at_00, grid_size=args.grid_size,
                                                   smooth_sigma=args.smooth_sigma,
                                                   subch_prob=(th00 > 0).astype(np.float32), tibia_layout=lay)
        gdp, _ = project_thickness_2d_canonical(bone, m00, (th48_at_00 - th00), grid_size=args.grid_size,
                                                 smooth_sigma=args.smooth_sigma,
                                                 subch_prob=(th00 > 0).astype(np.float32), tibia_layout=lay)
        rA = 2 * bi
        _draw(axes[rA, 0], g00p, f"{bone} PATIENT 00m", "jet", 0, args.vmax, "mm", bone)
        _draw(axes[rA, 1], g48p, f"{bone} PATIENT 48m→00m", "jet", 0, args.vmax, "mm", bone)
        _draw(axes[rA, 2], gdp, f"{bone} PATIENT Δ", "RdBu_r", -args.vdelta, args.vdelta, "Δmm", bone)

        # ---- template-frame row (shared 00m→template map) ----
        g00t, _ = project_thickness_2d_canonical(bone, template_mesh, tpl00, grid_size=args.grid_size,
                                                   smooth_sigma=args.smooth_sigma)
        g48t, _ = project_thickness_2d_canonical(bone, template_mesh, tpl48, grid_size=args.grid_size,
                                                   smooth_sigma=args.smooth_sigma)
        gdt, _ = project_thickness_2d_canonical(bone, template_mesh, (tpl48 - tpl00), grid_size=args.grid_size,
                                                 smooth_sigma=args.smooth_sigma)
        rB = 2 * bi + 1
        _draw(axes[rB, 0], g00t, f"{bone} TEMPLATE 00m", "jet", 0, args.vmax, "mm", bone)
        _draw(axes[rB, 1], g48t, f"{bone} TEMPLATE 48m (shared map)", "jet", 0, args.vmax, "mm", bone)
        _draw(axes[rB, 2], gdt, f"{bone} TEMPLATE Δ", "RdBu_r", -args.vdelta, args.vdelta, "Δmm", bone)

    fig.suptitle(f"{args.case}  — shared-mesh 48m→00m→template (single 00m→template map, NO per-tp drift)",
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    print("\nOpening interactive viewer — close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
