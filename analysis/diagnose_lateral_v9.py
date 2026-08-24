"""Lateral-compartment discrepancy diagnosis on the v9 PRIMARY frame (ICP-QC-pass
progressors, n=150). Supersedes v8/results/v8b_lateral_diagnosis.md, whose section 2
used the withdrawn reference-dependent QC list.

Three candidate explanations for PD reading more lateral thinning than manual QCart:
  1. Outlier-driven  -> trimmed means / medians.
  2. KL-dependent    -> discrepancy vs KL_00.
  3. Medial->lateral registration cross-talk -> corr(medial PD change, lateral discrepancy).
Plus the stable-arm comparison (n=151): lateral PD drift is also present in stable knees.

Writes ../results/v9_lateral_diagnosis.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, pearson_r, ba, fmt
from aggregate_v8b import col

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return pearson_r(ra, rb)


def main():
    prog = list(csv.DictReader(open(RESULTS / "v9_prog_qcpass.csv", encoding="utf-8")))
    stab = list(csv.DictReader(open(RESULTS / "v9_nonprog_qcpass.csv", encoding="utf-8")))
    md = ["# v9 lateral-compartment discrepancy diagnosis (PRIMARY frame)",
          f"(ICP-QC-pass progressors n={len(prog)}; stable n={len(stab)})\n",
          "## Reference: PD vs QCart by region\n",
          "| Arm | Region | PD Δ (μm) | PD SRM | QCart Δ (μm) | QCart SRM | r | BA bias (μm) |",
          "|-----|--------|--:|--:|--:|--:|--:|--:|"]
    for arm, rows in (("progressor", prog), ("stable", stab)):
        for reg in ("MFTC", "cMF", "MT", "LFTC", "cLF", "LT"):
            p = col(rows, f"pd_{reg}_d"); q = col(rows, f"eck_{reg}_d")
            bias, _, _ = ba(p * 1000, q * 1000)
            md.append(f"| {arm} | {reg} | {fmt(np.nanmean(p)*1000,0)} | {fmt(srm(p))} | "
                      f"{fmt(np.nanmean(q)*1000,0)} | {fmt(srm(q))} | {fmt(pearson_r(p,q))} | {fmt(bias,0)} |")

    md += ["\n## 1. Outlier contribution (progressors)\n",
           "| Region | mean Δ | 10% trimmed mean | median | QCart median |", "|--------|--:|--:|--:|--:|"]
    for reg in ("LFTC", "cLF", "LT", "MFTC"):
        p = col(prog, f"pd_{reg}_d") * 1000; q = col(prog, f"eck_{reg}_d") * 1000
        v = np.sort(p[np.isfinite(p)]); k = max(1, int(round(0.10 * len(v))))
        md.append(f"| {reg} | {fmt(v.mean(),0)} | {fmt(v[k:-k].mean(),0)} | {fmt(np.median(v),0)} | {fmt(np.nanmedian(q),0)} |")

    md += ["\n## 2. Discrepancy (PD−QCart) vs baseline KL (progressors)\n",
           "| Region | Spearman ρ | mean disc KL≤2 (μm) | mean disc KL≥3 (μm) |", "|--------|--:|--:|--:|"]
    kl = col(prog, "KL_00")
    for reg in ("LFTC", "cLF", "LT", "MFTC"):
        disc = (col(prog, f"pd_{reg}_d") - col(prog, f"eck_{reg}_d")) * 1000
        lo = disc[(kl <= 2) & np.isfinite(disc)]; hi = disc[(kl >= 3) & np.isfinite(disc)]
        md.append(f"| {reg} | {fmt(spearman(disc, kl))} | {fmt(lo.mean(),0)} (n={len(lo)}) | {fmt(hi.mean(),0)} (n={len(hi)}) |")

    md += ["\n## 3. Medial→lateral cross-talk (progressors)\n",
           "| Pair | Pearson r | Spearman ρ |", "|------|--:|--:|"]
    m_pd = col(prog, "pd_MFTC_d")
    for reg in ("LFTC", "cLF", "LT"):
        disc = col(prog, f"pd_{reg}_d") - col(prog, f"eck_{reg}_d")
        md.append(f"| pd_MFTC_d vs disc_{reg} | {fmt(pearson_r(m_pd, disc))} | {fmt(spearman(m_pd, disc))} |")

    md += ["\n## 4. Interpretation\n",
           "PD's excess lateral thinning relative to QCart is present in BOTH arms (stable-knee lateral "
           "PD Δ is as large as in progressors), is not outlier-driven (trimmed mean ≈ median ≈ mean), "
           "is not KL-dependent, and is not medial→lateral registration cross-talk. It is the same uniform "
           "longitudinal drift quantified medially in stable knees. Consequence for the discrimination "
           "analysis: lateral PD AUCs fall significantly below 0.5 (inverse association), because the "
           "JSW-stable arm — defined medially — contains lateral progressors while PD adds drift equally "
           "to both arms."]
    out = RESULTS / "v9_lateral_diagnosis.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
