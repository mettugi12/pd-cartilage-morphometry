"""
Batch the cartilage thickness pipeline over N OAI longitudinal seg files and
print per-case + summary stats. No visualization, no file output.
"""
import argparse
import contextlib
import io
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

# Allow running this script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from cartilage_morphometry import process_one_patient


def detect_side(filename: str) -> str:
    s = filename.upper()
    if "_LEFT" in s:
        return "LEFT"
    return "RIGHT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_glob",
                   default=r"E:/KneeMR/Datasets/OAI-Longitudinal-Seg/segmentation/00m/*_seg.nii.gz")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--bones", default="femur,tibia")
    p.add_argument("--modality", default="pd", choices=["pd", "dess"])
    args = p.parse_args()

    files = sorted(glob(args.seg_glob))[: args.n]
    bones = args.bones.split(",")
    print(f"Running pipeline on {len(files)} cases × {len(bones)} bones "
          f"(modality={args.modality})\n")

    all_rows = []  # for summary
    for f in files:
        seg_path = Path(f)
        case_id = seg_path.stem.replace(".nii", "")
        side = detect_side(seg_path.name)
        print(f"==== {case_id} ({side}) ====")
        for bone in bones:
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = process_one_patient(
                        seg_path=seg_path, side=side, bone_name=bone,
                        modality=args.modality, out_dir=None,
                    )
            except Exception as e:
                print(f"  {bone}: FAILED ({e})")
                continue
            dt = time.time() - t0
            q = res["quality"]
            t_mesh = res["bone_mesh"]["thickness_mm"]
            t_temp = res["template_thickness"]
            n_with_t = int((t_mesh > 0).sum())
            patient_mean = float(t_mesh[t_mesh > 0].mean()) if n_with_t else 0.0
            tmp_finite = t_temp[np.isfinite(t_temp)]
            template_mean = float(tmp_finite.mean()) if len(tmp_finite) else 0.0
            template_max = float(tmp_finite.max()) if len(tmp_finite) else 0.0
            scale_ml, scale_ap, _ = res["transforms"]["scale_factors"]
            R = res["transforms"]["R_icp"]
            icp_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

            row = {
                "case": case_id, "side": side, "bone": bone,
                "n_patient_subch": q["n_patient_subch"],
                "n_template_subch": q["n_template_subch"],
                "frac_mutual": q["frac_mutual"],
                "subch_dist_mean_mm": q["mean_mapping_dist_mm_subch_only"],
                "subch_dist_median_mm": q["median_mapping_dist_mm_subch_only"],
                "subch_dist_max_mm": q["max_mapping_dist_mm_subch_only"],
                "assd_subch_mm": q["assd_subch_mm"],
                "scale_ml": scale_ml, "scale_ap": scale_ap,
                "icp_rot_deg": icp_deg,
                "patient_mean_mm": patient_mean,
                "template_mean_mm": template_mean,
                "template_max_mm": template_max,
                "n_with_thickness": n_with_t,
                "secs": dt,
            }
            all_rows.append(row)
            print(
                f"  {bone:6s}: subch_n={q['n_template_subch']:>5}/{q['n_patient_subch']:>5}  "
                f"mutual={q['frac_mutual']*100:5.1f}%  "
                f"dist(subch only) mean={q['mean_mapping_dist_mm_subch_only']:.2f}mm "
                f"med={q['median_mapping_dist_mm_subch_only']:.2f}mm "
                f"max={q['max_mapping_dist_mm_subch_only']:.2f}mm  "
                f"ASSD={q['assd_subch_mm']:.2f}mm  "
                f"scale=({scale_ml:.2f},{scale_ap:.2f})  ICP_rot={icp_deg:.1f}°  "
                f"thick: pat_mean={patient_mean:.2f} tmpl_mean={template_mean:.2f}  "
                f"t={dt:.0f}s"
            )
        print()

    # Summary
    if not all_rows:
        return
    print("=== SUMMARY ===")
    for bone in bones:
        b_rows = [r for r in all_rows if r["bone"] == bone]
        if not b_rows:
            continue
        def stat(field):
            vals = np.array([r[field] for r in b_rows], dtype=float)
            return float(vals.mean()), float(vals.std())
        mu, sd = stat("subch_dist_mean_mm")
        amu, asd = stat("assd_subch_mm")
        fmu, fsd = stat("frac_mutual")
        rmu, rsd = stat("icp_rot_deg")
        pmu, psd = stat("patient_mean_mm")
        tmu, tsd = stat("template_mean_mm")
        print(f"\n[{bone}] n={len(b_rows)}")
        print(f"  subch dist (mean across subch verts): {mu:.2f} ± {sd:.2f} mm")
        print(f"  ASSD subch:                            {amu:.2f} ± {asd:.2f} mm")
        print(f"  mutual NN%:                            {fmu*100:.1f} ± {fsd*100:.1f}%")
        print(f"  ICP rotation:                          {rmu:.1f} ± {rsd:.1f}°")
        print(f"  patient mean thickness:                {pmu:.2f} ± {psd:.2f} mm")
        print(f"  template mean thickness:               {tmu:.2f} ± {tsd:.2f} mm")
        print(f"  pat-tmpl bias:                         {pmu - tmu:+.2f} mm")


if __name__ == "__main__":
    main()
