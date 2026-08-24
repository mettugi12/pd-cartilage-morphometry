"""2D-per-slice raycast end-to-end: patient bone → template.

For each case + bone:
  1. Compute per-vertex thickness via 2D-per-slice raycast (smooth bone/cart
     masks, sub-pixel ray entry/exit)
  2. Register patient bone to template (library `patient_to_template_align`)
  3. Compute subch_prob via tpl_nn
  4. For tibia: inherit compartment from template-NN, run
     `compartment_aware_remap` (v1.2). For femur: plain forward IDW.
  5. Render via `cartilage_validation.viz.make_thickness_panel` — the
     canonical 2x2 thickness panel (3D + 2D, proportional projection).

Usage:
  python -m scripts.viz_2draycast_to_template
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from cartilage_morphometry import (
    PipelineConfig,
    TEMPLATE_PATHS,
    get_subch,
    patient_to_template_align,
    remap_thickness_to_template,
)
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    _force_ml_flip, _ml_flip_decision, compartment_aware_remap,
)
from cartilage_morphometry.validation.viz import make_thickness_panel
from scripts.viz_3d_thickness_2dslice import compute_2dslice_thickness_per_vert


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _inherit_compartment(aligned_pts, template_mesh, subch_threshold=0.5):
    tpl_pts = np.asarray(template_mesh.points)
    tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold
    sp_idx = np.where(tpl_subch)[0]
    tree = cKDTree(tpl_pts[sp_idx])
    d, idx = tree.query(np.asarray(aligned_pts), k=1)
    return tpl_comp[sp_idx][idx].astype(np.int8), d.astype(np.float32)


def process_case(case, bone: str, args, template_mesh):
    seg_path = case.seg_00m_path
    flip = _ml_flip_decision(case.seg_00m_path)
    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)
    with _force_ml_flip(flip):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, case.side, "pd", bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing,
                                            bone_smooth_iters=5)

    # 2D-per-slice raycast on patient bone verts
    th_2d = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm,
    )

    # Register to template
    aligned_pts, *_ = patient_to_template_align(
        bone_mesh, cart3d, spacing, template_mesh, bone, "pd")

    # Subch on aligned bone verts (tpl_nn)
    subch_fn = get_subch(cfg.subch_method)
    subch_prob, _ = subch_fn(
        aligned_pts, template_mesh, cart3d, spacing, bone_mesh, th_2d, cfg)

    # IDW remap to template
    has_compartment = "compartment" in template_mesh.point_data
    if has_compartment:
        inh_comp, _ = _inherit_compartment(aligned_pts, template_mesh)
        template_thickness, _ = compartment_aware_remap(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, inh_comp,
            k=cfg.idw_k, subch_threshold=cfg.subch_threshold,
        )
    else:
        inh_comp = np.zeros(len(aligned_pts), dtype=np.int8)
        template_thickness, _ = remap_thickness_to_template(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold,
        )
        tpl_subch_mask = (np.asarray(template_mesh.point_data["subch_prob"])
                           >= cfg.subch_threshold)
        template_thickness = template_thickness.copy()
        template_thickness[~tpl_subch_mask] = np.nan
    return {
        "bone_mesh": bone_mesh, "aligned_pts": aligned_pts,
        "patient_thickness": th_2d, "template_thickness": template_thickness,
        "subch_prob_patient": subch_prob, "inherited_compartment": inh_comp,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["tibia", "femur"],
                   default=["tibia", "femur"])
    p.add_argument("--bone_sigma", type=float, default=1.8)
    p.add_argument("--cart_sigma", type=float, default=0.8)
    p.add_argument("--near_cart_mm", type=float, default=4.0)
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--max_query_mm", type=float, default=1.5)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--grid_size", type=int, default=120,
                   help="2D projection grid (higher = more detail).")
    p.add_argument("--smooth_sigma", type=float, default=1.2,
                   help="Border-preserving Gaussian sigma in grid cells.")
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/viz_2draycast_to_template"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    for bone in args.bones:
        template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
        has_comp = "compartment" in template_mesh.point_data
        summary = []
        for case_tag in args.cases:
            case = cohort.get(case_tag)
            if case is None:
                print(f"[skip] {case_tag}"); continue
            print(f"\n=== {case_tag} {bone} ===")
            try:
                res = process_case(case, bone, args, template_mesh)
            except Exception as e:
                print(f"  [ERR] {e}"); continue
            pat_th = res["patient_thickness"]; tpl_th = res["template_thickness"]
            n_pat = int((pat_th > 0).sum())
            pat_mean = float(pat_th[pat_th > 0].mean()) if n_pat else float("nan")
            if has_comp:
                tpl_comp = np.asarray(template_mesh.point_data["compartment"])
                med = float(np.nanmean(tpl_th[tpl_comp == 1]))
                lat = float(np.nanmean(tpl_th[tpl_comp == 2]))
            else:
                med = lat = float("nan")
            summary.append((case_tag, pat_mean, n_pat, med, lat,
                            float(np.nanmean(tpl_th))))

            # Canonical 2x2 panel
            patient_subch = res["subch_prob_patient"] >= 0.5
            make_thickness_panel(
                case_id=case_tag, bone=bone,
                patient_mesh=res["bone_mesh"],
                patient_thickness=res["patient_thickness"],
                template_mesh=template_mesh,
                template_thickness=res["template_thickness"],
                out_png=args.out_dir / f"viz_{case_tag}_{bone}_00m.png",
                vmax=args.vmax, grid_size=args.grid_size,
                smooth_sigma=args.smooth_sigma,
                patient_subch_mask=patient_subch,
                patient_compartment=(res["inherited_compartment"]
                                       if has_comp else None),
                subtitle=("2D-per-slice raycast -> template "
                           f"({'compartment-aware' if has_comp else 'plain'} IDW)"),
            )
            print(f"  [ok] viz_{case_tag}_{bone}_00m.png")

        if summary:
            print(f"\n=== SUMMARY ({bone} 00m) ===")
            print(f"{'case':>20s}  {'pat_mean':>9s}  {'pat_n>0':>8s}  "
                  f"{'tpl_med':>8s}  {'tpl_lat':>8s}  {'tpl_tot':>8s}")
            for case_tag, pm, pn, m, l, t in summary:
                print(f"{case_tag:>20s}  {pm:9.2f}  {pn:8d}  {m:8.3f}  {l:8.3f}  {t:8.3f}")


if __name__ == "__main__":
    main()
