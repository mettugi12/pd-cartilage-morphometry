"""Reproducible held-out segmentation evaluation for Table S1 (v9.1 / Scientific Reports).

Recomputes Dice, 95th-percentile Hausdorff distance (HD95) and average symmetric
surface distance (ASSD) per case and per class from the frozen Dataset204 held-out
test set (44 scans: 12 OAI + 32 hospital), GT vs raw model prediction, on the
native PD/IW voxel grid with header spacing. Reports pooled and by source (OAI /
hospital), because the morphometry cohorts are entirely OAI.

Inputs : E:/KneeMR/eval/PD_seg_test/{labelsTs,predsTs}
Outputs: ../results/v9_1_seg_test_percase.csv, ../results/v9_1_seg_test_table.md

Usage: python eval_seg_test_v9_1.py [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt

EVAL = Path(r"E:/KneeMR/eval/PD_seg_test")
RES = Path(__file__).resolve().parent.parent / "results"
CLASSES = {1: "Patella (bone)", 2: "Femur (bone)", 3: "Tibia (bone)",
           4: "Patellar cartilage", 5: "Femoral cartilage", 6: "Tibial cartilage",
           7: "Medial meniscus", 8: "Lateral meniscus", 9: "Anterior cruciate ligament",
           10: "Posterior cruciate ligament", 11: "Patellar tendon"}


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, iterations=1)


def surface_distances(gt: np.ndarray, pr: np.ndarray, spacing):
    """Symmetric surface-to-surface distances (mm) between two binary masks."""
    sg, sp = surface(gt), surface(pr)
    if not sg.any() or not sp.any():
        return None
    dt_g = distance_transform_edt(~sg, sampling=spacing)
    dt_p = distance_transform_edt(~sp, sampling=spacing)
    return np.concatenate([dt_g[sp], dt_p[sg]])


def metrics(gt: np.ndarray, pr: np.ndarray, spacing):
    inter = np.logical_and(gt, pr).sum()
    denom = gt.sum() + pr.sum()
    dice = 2 * inter / denom if denom else np.nan
    d = surface_distances(gt, pr, spacing)
    if d is None:
        return dice, np.nan, np.nan
    return dice, float(np.percentile(d, 95)), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    a = ap.parse_args()

    cases = sorted(p.name[:-7] for p in (EVAL / "labelsTs").glob("*.nii.gz"))
    if a.n_run:
        cases = cases[: a.n_run]
    rows = []
    for k, c in enumerate(cases, 1):
        g_img = nib.load(str(EVAL / "labelsTs" / f"{c}.nii.gz"))
        p_img = nib.load(str(EVAL / "predsTs" / f"{c}_seg.nii.gz"))
        g = np.asanyarray(g_img.dataobj).astype(np.uint8)
        p = np.asanyarray(p_img.dataobj).astype(np.uint8)
        assert g.shape == p.shape, c
        sp = tuple(float(z) for z in g_img.header.get_zooms()[:3])
        src = "OAI" if c.startswith("90") else "hospital"
        for lbl, name in CLASSES.items():
            dice, hd95, assd = metrics(g == lbl, p == lbl, sp)
            rows.append(dict(case=c, source=src, label=lbl, structure=name, dice=dice, hd95=hd95, assd=assd))
        print(f"[{k}/{len(cases)}] {c} FC dice {rows[-7]['dice']:.3f}", flush=True)

    df = pd.DataFrame(rows)

    def block(sub, tag):
        g = sub.groupby("label").agg(dice_m=("dice", "mean"), dice_s=("dice", "std"),
                                     hd_m=("hd95", "mean"), hd_s=("hd95", "std"),
                                     as_m=("assd", "mean"), as_s=("assd", "std"))
        return g

    pooled, oai, hosp = (block(df, "all"), block(df[df.source == "OAI"], "OAI"),
                         block(df[df.source == "hospital"], "hospital"))
    n_all, n_oai, n_h = df.case.nunique(), df[df.source == "OAI"].case.nunique(), df[df.source == "hospital"].case.nunique()

    md = [f"# Table S1 (v9.1) — Held-out segmentation performance, GT vs prediction (n={n_all}: OAI {n_oai}, hospital {n_h})",
          "", "Recomputed from E:/KneeMR/eval/PD_seg_test (native grid, header spacing). Values mean (SD).", "",
          "| Structure | Dice, all | Dice, OAI | Dice, hospital | HD95 mm, all | ASSD mm, all |", "|---|--:|--:|--:|--:|--:|"]
    for lbl, name in CLASSES.items():
        P, O, H = pooled.loc[lbl], oai.loc[lbl], hosp.loc[lbl]
        md.append(f"| {name} | {P.dice_m:.3f} ({P.dice_s:.3f}) | {O.dice_m:.3f} ({O.dice_s:.3f}) | {H.dice_m:.3f} ({H.dice_s:.3f}) | "
                  f"{P.hd_m:.2f} ({P.hd_s:.2f}) | {P.as_m:.2f} ({P.as_s:.2f}) |")
    macro = df.groupby("case").dice.mean()
    md += ["", f"Macro-average Dice across 11 structures: {macro.mean():.3f} ± {macro.std():.3f} "
               f"(OAI {macro[[c for c in macro.index if c.startswith('90')]].mean():.3f}; "
               f"hospital {macro[[c for c in macro.index if not c.startswith('90')]].mean():.3f})."]
    text = "\n".join(md) + "\n"
    print(text)
    if a.dry_run:
        print("[dry_run] not saved")
        return
    df.to_csv(RES / "v9_1_seg_test_percase.csv", index=False)
    (RES / "v9_1_seg_test_table.md").write_text(text, encoding="utf-8")
    print("wrote", RES / "v9_1_seg_test_table.md")


if __name__ == "__main__":
    main()
