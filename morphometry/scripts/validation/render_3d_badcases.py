"""Render 3D viz (patient bone + template thickness, deliverable style) for
a list of cases — typically the worst-tibia outliers from a full run.

Re-runs each case through process_{pair,long}_shared_mesh with VTK saving
on, then makes per-case PNGs via cartilage_validation.viz:
  - <out>/pair_<case_id>_<bone>_3d.png        (PD↔DESS, 2×2)
  - <out>/long_<case_tag>_<bone>_3d.png       (00m↔48m, 2×3 with Δ)

Defaults to the worst-r_2d tibia pairs and largest-pMT-thickening longitudinal
cases from `v1_raycast_shared_mesh_FULL.json`. Pass `--cases` to override.

Usage:
    python -m scripts.render_3d_badcases
    python -m scripts.render_3d_badcases --bones tibia femur
    python -m scripts.render_3d_badcases --cases 9739777_RIGHT 9167541_RIGHT --cohort cross_sectional
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_pair_cases, load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    process_long_shared_mesh, process_pair_shared_mesh,
)
from cartilage_morphometry.validation.viz import make_long_3d_figure, make_pair_3d_figure

DEFAULT_REPORT = Path(
    r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Studies/cartilage-validation-pipeline/v1.0/reports/v1_raycast_shared_mesh_FULL.json"
)
DEFAULT_OUT = Path(
    r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Studies/cartilage-validation-pipeline/v1.0/reports/badcase_3d"
)
DEFAULT_VTK = Path(
    r"E:/KneeMR/Studies/cartilage-validation/badcase_vtk"
)


def _worst_pairs(report: dict, n: int, bone: str = "tibia") -> list[str]:
    rows = sorted(report["cross_sectional"]["per_case"],
                  key=lambda c: c["bones"][bone].get("r_2d_smooth", 1.0))
    return [c["case_id"] for c in rows[:n]]


def _worst_long(report: dict, n: int, bone: str = "tibia",
                region: str = "pMT") -> list[str]:
    rows = []
    for c in report["longitudinal"]["per_case"]:
        v = (c["bones"][bone].get("regions_delta") or {}).get(region)
        if v is not None:
            rows.append((c, v))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return [f"{c['pid']}_{c['side']}" for c, _ in rows[:n]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                   help="JSON report to pick worst cases from.")
    p.add_argument("--cases", nargs="+", default=None,
                   help="Explicit case IDs. If unset, picks worst N tibia.")
    p.add_argument("--cohort", choices=["cross_sectional", "longitudinal", "both"],
                   default="both")
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--n_worst", type=int, default=5,
                   help="If --cases not given: pick top-N worst per cohort.")
    p.add_argument("--vtk_dir", type=Path, default=DEFAULT_VTK)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", default="v1_raycast_shared_mesh",
                   choices=["v1_default", "v1_raycast_shared_mesh", "v1_no_cart_cleanup"])
    p.add_argument("--skip_recompute", action="store_true",
                   help="Don't re-process; just re-render from existing VTKs.")
    args = p.parse_args()

    cfgs = {
        "v1_default": PipelineConfig(),
        "v1_raycast_shared_mesh": PipelineConfig(thickness_method="raycast"),
        "v1_no_cart_cleanup": PipelineConfig(cart_cleanup=()),
    }
    config = cfgs[args.config]

    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    # Resolve which cases to render
    cs_cases = lg_cases = []
    if args.cases:
        if args.cohort in ("cross_sectional", "both"):
            cs_cases = list(args.cases)
        if args.cohort in ("longitudinal", "both"):
            lg_cases = list(args.cases)
    else:
        if args.cohort in ("cross_sectional", "both"):
            cs_cases = _worst_pairs(rep, args.n_worst, bone="tibia")
        if args.cohort in ("longitudinal", "both"):
            lg_cases = _worst_long(rep, args.n_worst, bone="tibia", region="pMT")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.vtk_dir.mkdir(parents=True, exist_ok=True)

    # --- enable RECON disk cache (re-use cached PD densifications) ---
    if not args.skip_recompute:
        DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
        _patch_disk_cache_path_unique()
        print(f"[recon] disk cache: {DEFAULT_RECON_CACHE_DIR}")

    print(f"[mode] config={args.config}  shared_mesh=True  bones={args.bones}")
    print(f"[cs] cases: {cs_cases}")
    print(f"[long] cases: {lg_cases}")

    # ---- cross-sectional ----
    if cs_cases:
        all_pairs = {p.case_id: p for p in load_pair_cases()}
        for case_id in cs_cases:
            case = all_pairs.get(case_id)
            if case is None:
                print(f"  [skip] {case_id} not in cohort")
                continue
            for bone in args.bones:
                print(f"\n=== [cs] {case_id} {bone} ===")
                if not args.skip_recompute:
                    try:
                        _pipeline.clear_recon_cache()
                        process_pair_shared_mesh(
                            case.pd_seg_path, case.dess_seg_path,
                            side=case.side_for_pipeline, bone_name=bone, config=config,
                            dess_is_11class=True,
                            out_vtk_dir=args.vtk_dir, case_id=case_id,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        print(f"  [ERR] {case_id} {bone}: {e}")
                        continue
                try:
                    png = make_pair_3d_figure(
                        case_id, args.vtk_dir, bone,
                        args.out_dir / f"pair_{case_id}_{bone}_3d.png",
                    )
                    print(f"  [ok] {png}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"  [ERR-render] {case_id} {bone}: {e}")

    # ---- longitudinal ----
    if lg_cases:
        all_long = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
        for case_tag in lg_cases:
            case = all_long.get(case_tag)
            if case is None:
                print(f"  [skip] {case_tag} not in longitudinal cohort")
                continue
            for bone in args.bones:
                print(f"\n=== [long] {case_tag} {bone} ===")
                if not args.skip_recompute:
                    try:
                        _pipeline.clear_recon_cache()
                        process_long_shared_mesh(
                            case.seg_00m_path, case.seg_48m_path,
                            side=case.side, bone_name=bone, config=config,
                            out_vtk_dir=args.vtk_dir, case_id=case_tag,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        print(f"  [ERR] {case_tag} {bone}: {e}")
                        continue
                try:
                    png = make_long_3d_figure(
                        case_tag, args.vtk_dir, bone,
                        args.out_dir / f"long_{case_tag}_{bone}_3d.png",
                    )
                    print(f"  [ok] {png}")
                except Exception as e:
                    traceback.print_exc()
                    print(f"  [ERR-render] {case_tag} {bone}: {e}")


if __name__ == "__main__":
    main()
