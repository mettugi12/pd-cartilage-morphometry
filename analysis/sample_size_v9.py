"""Sample-size translation (JMRI_DISCUSSION_PLAN §3.1) on the 150/151 primary frame.

Question: for a hypothetical two-arm 48-month study detecting a 30% / 50% slowing of
cartilage loss, how many knees per arm are needed with PD-auto vs manual QCart?

Two computations, reported side by side:

  NAIVE        delta = slowing x |progressor mean change|. Treats everything the method
               reads in progressors as treatable signal. WRONG for PD, because PD's
               progressor mean includes the ~0.25 mm stable-knee drift, which a
               treatment cannot slow. Shown only to make the error explicit.

  DRIFT-AWARE  delta = slowing x |progressor mean - stable mean| (the between-arm EXCESS
               each method actually reads for true progression). The additive drift
               cancels in this contrast. This is the legitimate calculation for a
               two-arm, same-protocol design.

n per arm = 2 (z_{1-a/2} + z_{1-b})^2 sigma^2 / delta^2, sigma = progressor-arm SD of
change (the trial population), alpha 0.05 two-sided, power 0.80 and 0.90.

Also reports the drift-corrected responsiveness: excess / SD_prog (the SRM with the
stable-knee offset removed) and Cohen's d between arms — the comparison that is NOT
flattered by drift, unlike the progressor-only SRM.

Output: ../results/v9_sample_size.md
"""
from __future__ import annotations

import csv
from math import ceil
from pathlib import Path

import numpy as np
from scipy.stats import norm

from aggregate_v8b import col

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
REGIONS = ["MFTC", "cMF", "MT"]
ALPHA = 0.05


def n_per_arm(sigma, delta, power):
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(power)
    return ceil(2 * (z ** 2) * (sigma ** 2) / (delta ** 2))


def main():
    prog = list(csv.DictReader(open(RESULTS / "v9_prog_qcpass.csv", encoding="utf-8")))
    stab = list(csv.DictReader(open(RESULTS / "v9_nonprog_qcpass.csv", encoding="utf-8")))

    md = ["# v9 — Sample-size translation (two-arm 48-month design, primary frame 150 / 151)",
          "alpha 0.05 two-sided; sigma = progressor-arm SD of change; n = knees per arm.\n",
          "## A. Inputs (μm)\n",
          "| Region | Method | prog mean ± SD | stable mean ± SD | excess (prog − stable) | SRM (prog only) | drift-corrected SRM (excess/SD) | Cohen's d (prog vs stable) |",
          "|---|---|--:|--:|--:|--:|--:|--:|"]
    inputs = {}
    for reg in REGIONS:
        for tag, name in (("pd", "PD-auto"), ("eck", "manual QCart")):
            dp = col(prog, f"{tag}_{reg}_d") * 1000; dn = col(stab, f"{tag}_{reg}_d") * 1000
            dp = dp[np.isfinite(dp)]; dn = dn[np.isfinite(dn)]
            mp, sp = dp.mean(), dp.std(ddof=1); mn, sn = dn.mean(), dn.std(ddof=1)
            excess = mp - mn
            pooled = np.sqrt(((len(dp) - 1) * sp ** 2 + (len(dn) - 1) * sn ** 2) / (len(dp) + len(dn) - 2))
            inputs[(reg, tag)] = dict(mp=mp, sp=sp, mn=mn, excess=excess)
            md.append(f"| {reg} | {name} | {mp:.0f} ± {sp:.0f} | {mn:.0f} ± {sn:.0f} | {excess:.0f} "
                      f"| {mp/sp:.2f} | {excess/sp:.2f} | {excess/pooled:.2f} |")

    md += ["\n## B. Knees per arm — DRIFT-AWARE (legitimate for two-arm, same-protocol designs)\n",
           "delta = slowing × |excess|; drift cancels between arms.\n",
           "| Region | slowing | power | PD-auto n/arm | QCart n/arm | ratio PD/QCart |",
           "|---|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        for slowing in (0.30, 0.50):
            for power in (0.80, 0.90):
                npd = n_per_arm(inputs[(reg, "pd")]["sp"], slowing * abs(inputs[(reg, "pd")]["excess"]), power)
                nqc = n_per_arm(inputs[(reg, "eck")]["sp"], slowing * abs(inputs[(reg, "eck")]["excess"]), power)
                md.append(f"| {reg} | {slowing:.0%} | {power:.0%} | {npd} | {nqc} | {npd/nqc:.2f} |")

    md += ["\n## C. Knees per arm — NAIVE (delta = slowing × |progressor mean|; counts drift as treatable signal — NOT valid for PD)\n",
           "| Region | slowing | power | PD-auto n/arm | QCart n/arm | ratio PD/QCart |",
           "|---|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        for slowing in (0.30, 0.50):
            power = 0.80
            npd = n_per_arm(inputs[(reg, "pd")]["sp"], slowing * abs(inputs[(reg, "pd")]["mp"]), power)
            nqc = n_per_arm(inputs[(reg, "eck")]["sp"], slowing * abs(inputs[(reg, "eck")]["mp"]), power)
            md.append(f"| {reg} | {slowing:.0%} | {power:.0%} | {npd} | {nqc} | {npd/nqc:.2f} |")

    md += ["\n## D. Reading\n",
           "- The NAIVE table makes PD look as efficient as or better than QCart because PD's progressor "
           "mean is inflated by the stable-knee drift; a treatment cannot slow drift, so this is wrong.",
           "- The DRIFT-AWARE table is the honest design number. PD reads a smaller progressor-vs-stable "
           "excess than QCart (slope compression on the between-arm contrast), so it needs MORE knees per "
           "arm for the same power — the ratio in column 6 is the cost of using routine PD instead of "
           "expert DESS reading in a two-arm study.",
           "- Consequence for the manuscript: the progressor-only SRM (Table 2) is flattered by drift for PD; "
           "the drift-corrected SRM and the between-arm Cohen's d (Table A above) are the responsiveness "
           "metrics that are NOT. Report both, and say which is which."]
    out = RESULTS / "v9_sample_size.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
