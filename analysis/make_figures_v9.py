"""v9 JMRI figures — progressor-vs-stable discrimination (the new centerpiece).

Reads ../results/v8b_merged_deltas.csv (progressors) + v8b_nonprog_deltas.csv
(stable arm), dedups one-knee-per-subject (seed 42) to match the discrimination
table exactly, and renders:

  Figure5_discrimination.png   1x3 ROC panels (MFTC / cMF / MT): PD-auto vs
                               manual DESS (QCart) detecting radiographic progressors
                               from 48-month thickness change. AUC in legend.
  FigureS2_distributions.png   2x3 (rows: PD-auto, manual DESS (QCart)): overlapping
                               progressor / stable delta distributions with the
                               SDC95 detection limit marked. Shows the stable-
                               knee drift directly. Proposed as supplementary.

Style matches make_figures_v8b.py (house blue/red, frameless legends, 300 dpi,
no suptitle — captions live in the manuscript). Grayscale-safe: the two methods
differ in linestyle (solid vs dashed) and luminance, not hue alone.

Outputs to ../manuscript/figures/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_v8b import dedup, col
from aggregate_discrimination_v8b import auc_mw

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE.parent / "manuscript" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["MFTC", "cMF", "MT"]
BLUE = "#2c6fbb"   # PD-auto (house data color)
DARK = "#444444"   # manual DESS (QCart) (luminance-separated, grayscale-safe)
RED = "#c0392b"    # progressor fill in distribution panels
GRAY = "#7f8c9b"   # stable fill


def load_arms():
    """Primary figures use the label-free QC-pass deduped cohorts written by
    qc_labelfree_v9.py, so figures match the primary tables exactly. Falls back
    to the full-cohort CSVs if the QC outputs don't exist yet."""
    qp, qn = RESULTS / "v9_prog_qcpass.csv", RESULTS / "v9_nonprog_qcpass.csv"
    if qp.exists() and qn.exists():
        prog = list(csv.DictReader(open(qp, encoding="utf-8")))
        nonp = list(csv.DictReader(open(qn, encoding="utf-8")))
        return prog, nonp
    prog = dedup([r for r in csv.DictReader(open(RESULTS / "v8b_merged_deltas.csv", encoding="utf-8"))
                  if r.get("cohort") == "progressor"])
    nonp = dedup([r for r in csv.DictReader(open(RESULTS / "v8b_nonprog_deltas.csv", encoding="utf-8"))
                  if r.get("cohort") == "nonprogressor"])
    return prog, nonp


def roc_points(pos, neg):
    """ROC from score = -delta (more thinning = more progression-like)."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    scores = np.concatenate([-pos, -neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores)
    labels = labels[order]
    tpr = np.concatenate([[0], np.cumsum(labels) / labels.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - labels) / (1 - labels).sum()])
    return fpr, tpr


def fig_roc(prog, nonp):
    # Final print size (JMRI: max 2-column width 17.15 cm = 6.75 in; 600 dpi color);
    # panels lettered a, b, c upper-left outside the frame.
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6))
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        dp_pd = col(prog, f"pd_{reg}_d"); dn_pd = col(nonp, f"pd_{reg}_d")
        dp_qc = col(prog, f"eck_{reg}_d"); dn_qc = col(nonp, f"eck_{reg}_d")
        for dp, dn, color, ls, name in (
                (dp_pd, dn_pd, BLUE, "-", "PD-auto"),
                (dp_qc, dn_qc, DARK, "--", "Manual DESS (QCart)")):
            fpr, tpr = roc_points(dp, dn)
            a = auc_mw(dp, dn)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.2,
                    label=f"{name} (AUC {a:.2f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.5, alpha=0.5)
        n_p = int(np.isfinite(dp_pd).sum()); n_n = int(np.isfinite(dn_pd).sum())
        ax.set_title(f"{reg}  ({n_p} progressor vs {n_n} stable)", fontsize=7.5)
        ax.set_xlabel("False-positive rate (stable knees)", fontsize=7)
        ax.set_ylabel("True-positive rate (progressors)", fontsize=7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        ax.legend(loc="lower right", fontsize=5.8, frameon=False)
        ax.text(-0.22, 1.07, "abc"[i], transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    out = FIGDIR / "Figure5_discrimination.png"
    fig.savefig(out, dpi=600); plt.close(fig)
    return out


def fig_distributions(prog, nonp):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex="col")
    for row, (tag, label) in enumerate((("pd", "PD-auto"), ("eck", "Manual DESS (QCart)"))):
        for j, reg in enumerate(REGIONS):
            ax = axes[row, j]
            dp = col(prog, f"{tag}_{reg}_d") * 1000
            dn = col(nonp, f"{tag}_{reg}_d") * 1000
            dp = dp[np.isfinite(dp)]; dn = dn[np.isfinite(dn)]
            lo = min(dp.min(), dn.min()); hi = max(dp.max(), dn.max())
            bins = np.linspace(lo, hi, 32)
            ax.hist(dn, bins=bins, density=True, alpha=0.55, color=GRAY,
                    label=f"stable (n={len(dn)})")
            ax.hist(dp, bins=bins, density=True, alpha=0.55, color=RED,
                    label=f"progressor (n={len(dp)})")
            # Threshold at 95% empirical specificity (5th percentile of stable Δ) — the
            # classification threshold actually used in Table 3. SDC95 is NOT drawn:
            # it ignores the stable-knee mean drift and is a variability descriptor only.
            thr = np.percentile(dn, 5)
            sens = np.mean(dp < thr) * 100
            ax.axvline(thr, color="k", ls=":", lw=1.2)
            ax.axvline(0, color="k", lw=0.6, alpha=0.35)
            ax.text(thr, ax.get_ylim()[1] * 0.97,
                    f" 95% spec. threshold {thr:.0f} µm\n sensitivity {sens:.0f}%",
                    fontsize=7.2, color="k", ha="left", va="top", rotation=90, linespacing=1.1)
            ax.set_title(f"{label} — {reg}", fontsize=10.5)
            if row == 1:
                ax.set_xlabel("48-month thickness change (µm)")
            if j == 0:
                ax.set_ylabel("Density")
            ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    out = FIGDIR / "FigureS2_distributions.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def main():
    prog, nonp = load_arms()
    print(f"progressors n={len(prog)}  stable n={len(nonp)} (deduped)")
    print(f"[ok] {fig_roc(prog, nonp)}")
    print(f"[ok] {fig_distributions(prog, nonp)}")


if __name__ == "__main__":
    main()
