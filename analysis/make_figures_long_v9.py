"""v9 longitudinal agreement figures on the ICP-QC-pass PRIMARY progressor cohort.

Reads ../results/v9_prog_qcpass.csv (n=150, already deduped by qc_labelfree_v9.py) and
renders, in the v8b house style:

  Figure3_longitudinal_BA.png       Bland-Altman, PD-auto vs manual DESS (QCart) 48-month Δ,
                                    MFTC / cMF / MT
  Figure4_longitudinal_scatter.png  per-knee scatter with identity + fit line, r

Numbers (bias, LOA, r) are recomputed from the same rows and therefore match
v9_longitudinal_table_qcpass.md (PRIMARY block) to the digit.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_v8b import col
from aggregate_v8 import pearson_r, ba

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE.parent / "manuscript" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["MFTC", "cMF", "MT"]
BLUE = "#2c6fbb"
RED = "#c0392b"


def load():
    return list(csv.DictReader(open(RESULTS / "v9_prog_qcpass.csv", encoding="utf-8")))


def _pair_um(rows, reg):
    pd = col(rows, f"pd_{reg}_d") * 1000.0
    qc = col(rows, f"eck_{reg}_d") * 1000.0
    m = np.isfinite(pd) & np.isfinite(qc)
    return pd[m], qc[m]


PANEL = "abc"  # JMRI: label panels a, b, c in the upper-left, outside the frame


def _panel_letter(ax, i):
    ax.text(-0.22, 1.07, PANEL[i], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def fig_ba(rows):
    # Final print size (JMRI: figures submitted at final size, max 2-column width
    # 17.15 cm = 6.75 in; color minimum 600 dpi).
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45))
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        pd, qc = _pair_um(rows, reg)
        bias, llo, lhi = ba(pd, qc)
        ax.scatter((pd + qc) / 2, pd - qc, s=7, alpha=0.55, edgecolor="none", color=BLUE)
        ax.axhline(bias, color=RED, lw=1.0, label=f"bias {bias:+.0f} µm")
        ax.axhline(lhi, color="gray", ls="--", lw=0.7)
        ax.axhline(llo, color="gray", ls="--", lw=0.7)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"{reg}  (n={len(pd)})", fontsize=8)
        ax.set_xlabel("Mean of PD & manual Δ (µm)", fontsize=7)
        ax.set_ylabel("PD − manual Δ (µm)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(loc="upper right", fontsize=6, frameon=False)
        ax.text(0.03, 0.04, f"95% LOA [{llo:.0f}, {lhi:.0f}]", transform=ax.transAxes,
                fontsize=6, color="gray")
        _panel_letter(ax, i)
    fig.tight_layout()
    out = FIGDIR / "Figure3_longitudinal_BA.png"
    fig.savefig(out, dpi=600); plt.close(fig)
    return out


def fig_scatter(rows):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45))
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        pd, qc = _pair_um(rows, reg)
        r = pearson_r(pd, qc)
        ax.scatter(qc, pd, s=7, alpha=0.55, edgecolor="none", color=BLUE)
        lo = min(pd.min(), qc.min()); hi = max(pd.max(), qc.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.6, label="identity")
        b, a = np.polyfit(qc, pd, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, a + b * xs, color=RED, lw=1.0, label=f"fit (r={r:.2f})")
        ax.set_title(f"{reg}  (n={len(pd)})", fontsize=8)
        ax.set_xlabel("Manual DESS (QCart) Δ (µm)", fontsize=7)
        ax.set_ylabel("PD-auto Δ (µm)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(loc="upper left", fontsize=6, frameon=False)
        _panel_letter(ax, i)
    fig.tight_layout()
    out = FIGDIR / "Figure4_longitudinal_scatter.png"
    fig.savefig(out, dpi=600); plt.close(fig)
    return out


def main():
    rows = load()
    print(f"ICP-QC-pass progressors n={len(rows)}")
    for reg in REGIONS:
        pd, qc = _pair_um(rows, reg)
        bias, llo, lhi = ba(pd, qc)
        print(f"  {reg}: bias {bias:+.0f} µm  LOA [{llo:.0f}, {lhi:.0f}]  r={pearson_r(pd,qc):.2f}  n={len(pd)}")
    print(f"[ok] {fig_ba(rows)}")
    print(f"[ok] {fig_scatter(rows)}")


if __name__ == "__main__":
    main()
