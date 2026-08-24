"""Progressor vs non-progressor discrimination analysis (JMRI plan sections 3A/3C).

Inputs:  ../results/v8b_merged_deltas.csv      (progressors, n~172 -> dedup 166)
         ../results/v8b_nonprog_deltas.csv     (non-progressors, n<=56)
Outputs: ../results/v8b_discrimination_table.md

Per region, for PD-auto and manual QCart on the SAME knees:
  - non-progressor Δ and SRM (specificity: expectation ~0);
  - progressor vs non-progressor ROC AUC (progression detection), bootstrap 95% CI;
  - smallest detectable change SDC95 = 1.96 * SD(Δ in non-progressors);
  - % of progressors whose Δ exceeds SDC95 (individual-level detection rate).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, fmt
from aggregate_v8b import dedup, col

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
REGIONS = ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]
RNG = np.random.default_rng(42)


def auc_mw(pos, neg):
    """ROC AUC via Mann-Whitney U. 'Positive' = progressor, more-negative delta =
    more progression, so score = -delta."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    sp, sn = -pos, -neg
    order = np.concatenate([sp, sn])
    ranks = np.argsort(np.argsort(order)) + 1.0
    # midranks for ties
    vals, inv, cnt = np.unique(order, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    mid = (cum - (cnt - 1) / 2.0)
    ranks = mid[inv]
    r_pos = ranks[: len(sp)].sum()
    u = r_pos - len(sp) * (len(sp) + 1) / 2.0
    return u / (len(sp) * len(sn))


def auc_boot_ci(pos, neg, n_boot=2000):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    aucs = []
    for _ in range(n_boot):
        bp = pos[RNG.integers(0, len(pos), len(pos))]
        bn = neg[RNG.integers(0, len(neg), len(neg))]
        aucs.append(auc_mw(bp, bn))
    return np.percentile(aucs, [2.5, 97.5])


def mannwhitney_p(a, b):
    """Normal-approximation two-sided p for the Mann-Whitney U statistic."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    u = auc_mw(a, b) * na * nb
    mu = na * nb / 2.0
    sd = np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / sd
    from math import erf
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))


def main():
    prog = dedup([r for r in csv.DictReader(open(RESULTS / "v8b_merged_deltas.csv", encoding="utf-8"))
                  if r.get("cohort") == "progressor"])
    nonp = dedup([r for r in csv.DictReader(open(RESULTS / "v8b_nonprog_deltas.csv", encoding="utf-8"))
                  if r.get("cohort") == "nonprogressor"])

    md = ["# v8b progressor vs non-progressor discrimination",
          f"(progressors n={len(prog)}, non-progressors n={len(nonp)}; both arms same "
          "baseline-grid pipeline; QCart = manual Eckstein readings on the same knees. "
          "AUC: detecting radiographic progressors from 48-month thickness change; "
          "bootstrap 95% CI, 2000 resamples, seed 42. SDC95 = 1.96*SD(Δ) in non-progressors; "
          "'% prog > SDC' = progressors with change beyond the stable-knee detection limit.)\n"]

    for method, tag in (("pd", "PD-auto"), ("eck", "manual QCart")):
        md += [f"\n## {tag}\n",
               "| Region | nonprog Δ (μm) | nonprog SRM | prog Δ (μm) | AUC (95% CI) | MW p | SDC95 (μm) | % prog > SDC |",
               "|--------|--:|--:|--:|--:|--:|--:|--:|"]
        for reg in REGIONS:
            dn = col(nonp, f"{method}_{reg}_d") * 1000
            dp = col(prog, f"{method}_{reg}_d") * 1000
            a = auc_mw(dp, dn)
            lo, hi = auc_boot_ci(dp, dn)
            p = mannwhitney_p(dp, dn)
            sdc = 1.96 * np.nanstd(dn, ddof=1)
            det = np.nanmean((dp < -sdc).astype(float)) * 100
            pstr = "<0.001" if p < 0.001 else fmt(p, 3)
            md.append(f"| {reg} | {fmt(np.nanmean(dn),0)} ± {fmt(np.nanstd(dn,ddof=1),0)} "
                      f"| {fmt(srm(dn))} | {fmt(np.nanmean(dp),0)} "
                      f"| {fmt(a)} ({fmt(lo)}, {fmt(hi)}) | {pstr} "
                      f"| {fmt(sdc,0)} | {fmt(det,0)}% |")

    # paired AUC difference (PD vs QCart), bootstrap over knees
    md += ["\n## AUC difference (PD-auto − QCart), paired bootstrap\n",
           "| Region | ΔAUC (95% CI) |", "|--------|--:|"]
    for reg in REGIONS:
        pn, qn = col(nonp, f"pd_{reg}_d"), col(nonp, f"eck_{reg}_d")
        pp, qp = col(prog, f"pd_{reg}_d"), col(prog, f"eck_{reg}_d")
        diffs = []
        for _ in range(2000):
            bi = RNG.integers(0, len(pp), len(pp)); bj = RNG.integers(0, len(pn), len(pn))
            diffs.append(auc_mw(pp[bi], pn[bj]) - auc_mw(qp[bi], qn[bj]))
        d = auc_mw(pp, pn) - auc_mw(qp, qn)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        md.append(f"| {reg} | {fmt(d)} ({fmt(lo)}, {fmt(hi)}) |")

    out = RESULTS / "v8b_discrimination_table.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
