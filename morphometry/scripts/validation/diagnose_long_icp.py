"""Patient-frame Δ diagnostic — ignore the template entirely.

Question: if we just do 48m→00m inter-timepoint registration and look at
Δ = th48_at_00 − th00 on the 00m bone mesh, do we still see implausible
thickening? If yes → registration or seg is the source. If no → it's the
template anchoring.

For each of the 5 worst-pMT tibia cases (where the v1.1 template-frame
showed the largest thickening artifact), run TWO variants:
  - icp_method="rigid"          (current v1.1 default, 6-DOF)
  - icp_method="bounded_affine" (12-DOF affine with SV-clamp ±30 %)

Produces one 2×2 PNG per case (femur top row, tibia bottom row;
rigid left, bounded_affine right), 3D Δ map on the 00m bone mesh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import process_long_shared_mesh
from cartilage_morphometry.validation.viz import _screenshot_to_ax, DELTA_VABS_3D


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _run_one(case, bone_name: str, config: PipelineConfig, icp_method: str):
    """Returns the result dict (bone_mesh_00m + th00 + th48_at_00 + ...)."""
    _pipeline.clear_recon_cache()
    return process_long_shared_mesh(
        case.seg_00m_path, case.seg_48m_path,
        side=case.side, bone_name=bone_name, config=config,
        icp_method=icp_method,
    )


def _render_case(case_tag: str, results_per_bone_per_method: dict, out_png: Path,
                 vabs: float = DELTA_VABS_3D):
    """2-row × 2-col figure:
        row1: femur (rigid Δ, bounded_affine Δ)
        row2: tibia (rigid Δ, bounded_affine Δ)
    All Δ rendered on the 00m bone mesh (patient frame, NO template)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for r, bone in enumerate(("femur", "tibia")):
        for c, method in enumerate(("rigid", "bounded_affine")):
            ax = axes[r, c]
            res = results_per_bone_per_method.get((bone, method))
            if res is None:
                ax.axis("off")
                ax.set_title(f"{bone} / {method}  (failed)", fontsize=10)
                continue
            mesh = res["bone_mesh_00m"].copy()
            delta = res["th48_at_00_patient"] - res["th00_patient"]
            mesh["delta_mm"] = delta.astype(np.float32)
            mean_d = float(delta.mean())
            pos_frac = float((delta > 0.1).mean()) * 100
            neg_frac = float((delta < -0.1).mean()) * 100
            _screenshot_to_ax(
                ax, mesh, bone, "delta_mm",
                "RdBu_r", -vabs, vabs,
                f"{bone} / {method}   mean_Δ={mean_d:+.3f}mm  "
                f"thickening>0.1mm: {pos_frac:.0f}%  thinning<-0.1mm: {neg_frac:.0f}%",
                "Δ thickness (mm)", mask_nan=False,
            )
    fig.suptitle(f"{case_tag}  — patient-frame Δ (48m − 00m), no template",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"],
                   default=["femur", "tibia"])
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
        if case is None:
            print(f"[skip] {case_tag} not in cohort"); continue
        print(f"\n========== {case_tag} ==========")
        results = {}
        for bone in args.bones:
            for method in ("rigid", "bounded_affine"):
                print(f"\n--- {bone} / {method} ---")
                try:
                    res = _run_one(case, bone, config, method)
                    results[(bone, method)] = res
                except Exception as e:
                    print(f"  [ERR] {e}")
        out = args.out_dir / f"diag_{case_tag}.png"
        _render_case(case_tag, results, out)
        print(f"[ok] {out}")


if __name__ == "__main__":
    main()
