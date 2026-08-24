"""v9 cohort-flow figure (Supplementary Figure S1) — both longitudinal arms with the
predefined label-free QC step, counts computed from the data rather than typed in.

Sources: E:/ cohort file (knees defined per arm), segmentation availability (knees with
both-timepoint segs = processed), ../results/v9_qc_flags.csv (ICP flags), and the
QC-pass deduped CSVs (final n). Cross-sectional (n=61) and manual (n=12) arms are
drawn as fixed side boxes (unchanged from v3.3).

Output: ../manuscript/figures/FigureS1_cohort_flow.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE.parent / "manuscript" / "figures"
COHORT = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/cohort_v33_combined.csv")


def counts():
    coh = pd.read_csv(COHORT)
    defined = {"progressor": int((coh.cohort == "progressor").sum()),
               "stable": int((coh.cohort == "nonprogressor").sum())}
    flags = list(csv.DictReader(open(RESULTS / "v9_qc_flags.csv", encoding="utf-8")))
    out = {}
    for arm in ("progressor", "stable"):
        a = [f for f in flags if f["arm"] == arm]
        out[arm] = {
            "defined": defined[arm],
            "processed": len(a),
            "icp_fail": sum(f["icp_fail"] == "True" for f in a),
            "qc_pass": sum(f["qc_pass_primary"] == "True" for f in a),
        }
    out["progressor"]["final"] = len(list(csv.DictReader(open(RESULTS / "v9_prog_qcpass.csv", encoding="utf-8"))))
    out["stable"]["final"] = len(list(csv.DictReader(open(RESULTS / "v9_nonprog_qcpass.csv", encoding="utf-8"))))
    return out


def box(ax, x, y, w, h, text, fc="#f4f6f9", ec="#444", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.1))


def main():
    c = counts()
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.2); ax.axis("off")

    # header (wide, short lines so nothing can overflow)
    box(ax, 1.0, 7.3, 9.0, 0.7,
        "OAI knees with 48-month follow-up: sagittal PD fat-suppressed FSE at 00m and 48m\n"
        "manual QCart (DESS) readings available for all knees", fc="#e8eef7", bold=True, fs=9)

    cols = {"progressor": 0.5, "stable": 5.9}
    titles = {"progressor": "Radiographic medial-JSW progressors", "stable": "JSW-stable knees"}
    for arm, x in cols.items():
        k = c[arm]
        w = 4.6
        box(ax, x, 6.15, w, 0.65, f"{titles[arm]}\ncohort file: n = {k['defined']}",
            fc="#e8eef7", bold=True, fs=8.5)
        arrow(ax, x + w / 2, 6.15, 5.75)
        box(ax, x, 5.0, w, 0.7,
            f"both timepoints in archive & segmented\nn = {k['processed']}  "
            f"({k['defined'] - k['processed']} missing series)", fs=8.5)
        arrow(ax, x + w / 2, 5.0, 4.6)
        box(ax, x, 3.6, w, 1.0,
            f"label-free QC: registration\nnon-convergence (ASSD > 2 mm or\n"
            f"|t| > 30 mm): excluded {k['icp_fail']}  →  n = {k['qc_pass']}",
            fc="#fff4e6", fs=8.5)
        arrow(ax, x + w / 2, 3.6, 3.15)
        box(ax, x, 2.4, w, 0.75,
            f"one knee per subject (seed 42)\nPRIMARY analysis set: n = {k['final']}",
            fc="#e9f5ec", bold=True, fs=8.5)

    # side arms
    box(ax, 1.0, 1.2, 4.0, 0.8,
        "Cross-sectional PD↔DESS agreement\nn = 61 same-day pairs (template harness)", fc="#f4f6f9")
    box(ax, 6.0, 1.2, 4.0, 0.8,
        "Manual re-segmentation comparison\nn = 12 (sequence-inherent bias)", fc="#f4f6f9")
    ax.text(5.5, 0.55, "Longitudinal arms feed Tables 2–3 and Figures 3–5; side arms feed Table 1 and Figure 2.",
            ha="center", fontsize=8.5, color="gray")

    out = FIGDIR / "FigureS1_cohort_flow.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(c); print(f"[ok] {out}")


if __name__ == "__main__":
    main()
