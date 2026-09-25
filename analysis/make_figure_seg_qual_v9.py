"""Supplementary figure: qualitative segmentation validation (post-JMRI revision).

Reviewer 1 & 2 asked for sagittal slices comparing DESS/PD and automated vs manual
contours, and for visual validation of the held-out Dice values.

Layout (3 rows x 3 cols, sagittal slices through the medial compartment):
  row a  held-out OAI knee 9014883 (in Dataset204 labelsTs AND the 12-case paired
         manual substudy):  PD + manual | PD + automated | DESS + manual (same knee)
  row b  held-out hospital case at MEDIAN femoral-cartilage Dice: PD | +manual | +automated
  row c  held-out hospital case at WORST  femoral-cartilage Dice: PD | +manual | +automated

Cartilage contours only (labels 4 patellar, 5 femoral, 6 tibial). Manual = green,
automated = magenta. The slice is chosen per volume as the sagittal plane through the
centroid of the medial meniscus label (7), so PD and DESS panels show the same
compartment although they are separate acquisitions (not voxel-aligned).

Inputs (E:/ only):
  E:/KneeMR/eval/PD_seg_test/{imagesTs,labelsTs,predsTs,dice_full.csv}
  E:/KneeMR/Datasets/segmentation_canonical/legacy/Triple-GT-OAI/9014883_20090219_Lt/{DESS,DESS_mask}.nii.gz
Output:
  ../manuscript/figures/FigureS4_segmentation_validation.png  (+ .tif at 300 dpi)

Usage: python make_figure_seg_qual_v9.py [--dry_run] [--n_run N]
  --dry_run : pick cases and print them, do not render/save
  --n_run   : only consider the first N rows of dice_full.csv (spot-check)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

EVAL = Path(r"E:/KneeMR/eval/PD_seg_test")
TRIPLE = Path(r"E:/KneeMR/Datasets/segmentation_canonical/legacy/Triple-GT-OAI/9014883_20090219_Lt")
OUT = Path(__file__).resolve().parent.parent / "manuscript" / "figures"
OAI_CASE = "9014883_Left"
CART = (4, 5, 6)
MED_MEN = 7
C_MAN, C_AUTO = "#2ecc40", "#ff33cc"


def load_canon(p: Path):
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def sag_slice_index(lab: np.ndarray) -> int:
    m = lab == MED_MEN
    if m.sum() < 50:            # fall back to femoral cartilage centroid
        m = lab == 5
    xs = np.where(m)[0]
    return int(np.round(np.median(xs)))


def show(ax, img, lab_manual, lab_auto, i, zoom, title, crop_ref=None):
    sl = img[i].astype(float).T          # (z, y) -> rows = SI, cols = AP
    lo, hi = np.percentile(sl[sl > 0], [1, 99.5]) if (sl > 0).any() else (0, 1)
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower",
              aspect=zoom[2] / zoom[1], interpolation="nearest")
    for lab, col, ls in ((lab_manual, C_MAN, "-"), (lab_auto, C_AUTO, "--")):
        if lab is None:
            continue
        m = np.isin(lab[i], CART).T.astype(float)
        if m.any():
            ax.contour(m, levels=[0.5], colors=[col], linewidths=0.9, linestyles=ls)
    # crop to the joint: bbox of cartilage in the manual (or auto) label with margin
    ref = crop_ref if crop_ref is not None else (lab_manual if lab_manual is not None else lab_auto)
    m = np.isin(ref[i], CART).T
    if m.any():
        r, c = np.where(m)
        pad = 25
        ax.set_ylim(max(r.min() - pad, 0), min(r.max() + pad, m.shape[0]))
        ax.set_xlim(max(c.min() - pad, 0), min(c.max() + pad, m.shape[1]))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    a = ap.parse_args()

    d = pd.read_csv(EVAL / "dice_full.csv")
    d = d[d["variant"] == "baseline"].copy()
    if a.n_run:
        d = d.head(a.n_run)
    hosp = d[~d["case"].str.startswith("90")].sort_values("dice_05_femur_cart")
    worst = hosp.iloc[0]
    median = hosp.iloc[len(hosp) // 2]
    oai = d[d["case"] == OAI_CASE].iloc[0]
    print(f"OAI held-out {OAI_CASE}: FC Dice {oai['dice_05_femur_cart']:.3f}, TC {oai['dice_06_tibia_cart']:.3f}")
    print(f"hospital median FC Dice: {median['case']}  FC {median['dice_05_femur_cart']:.3f}")
    print(f"hospital worst  FC Dice: {worst['case']}  FC {worst['dice_05_femur_cart']:.3f}")
    print(f"hospital held-out n={len(hosp)}, FC Dice range {hosp['dice_05_femur_cart'].min():.3f}–{hosp['dice_05_femur_cart'].max():.3f}")
    if a.dry_run:
        print("[dry_run] not rendering")
        return

    fig, axes = plt.subplots(3, 3, figsize=(6.75, 7.2), dpi=300)

    # ---- row a: OAI paired knee
    img, z = load_canon(EVAL / "imagesTs" / f"{OAI_CASE}_0000.nii.gz")
    gt, _ = load_canon(EVAL / "labelsTs" / f"{OAI_CASE}.nii.gz")
    pr, _ = load_canon(EVAL / "predsTs" / f"{OAI_CASE}_seg.nii.gz")
    i = sag_slice_index(gt)
    show(axes[0, 0], img, gt, None, i, z, "PD/IW, manual")
    show(axes[0, 1], img, None, pr, i, z, f"PD/IW, automated (held-out; FC Dice {oai['dice_05_femur_cart']:.2f})")
    dimg, dz = load_canon(TRIPLE / "DESS.nii.gz")
    dlab, _ = load_canon(TRIPLE / "DESS_mask.nii.gz")
    j = sag_slice_index(dlab)
    show(axes[0, 2], dimg, dlab, None, j, dz, "DESS, manual (same knee)")

    # ---- rows b, c: hospital median / worst
    for row, rec, tag in ((1, median, "median"), (2, worst, "worst")):
        c = rec["case"]
        img, z = load_canon(EVAL / "imagesTs" / f"{c}_0000.nii.gz")
        gt, _ = load_canon(EVAL / "labelsTs" / f"{c}.nii.gz")
        pr, _ = load_canon(EVAL / "predsTs" / f"{c}_seg.nii.gz")
        i = sag_slice_index(gt)
        show(axes[row, 0], img, None, None, i, z,
             f"PD/IW, hospital ({tag} FC Dice {rec['dice_05_femur_cart']:.2f})", crop_ref=gt)
        show(axes[row, 1], img, gt, None, i, z, "manual")
        show(axes[row, 2], img, None, pr, i, z, "automated")

    for k, ax in enumerate(axes[:, 0]):
        ax.text(-0.04, 1.02, "abc"[k], transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
    fig.tight_layout(w_pad=0.4, h_pad=0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "FigureS4_segmentation_validation.png"
    fig.savefig(png, dpi=300)
    fig.savefig(png.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
