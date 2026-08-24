"""Aggregate v8b (baseline-grid) longitudinal results + 3-way comparison.

Reads ../results/v8b_merged_deltas.csv (from run_baseline_grid_v8b.py): per-knee
PD baseline-grid deltas + manual QCart deltas. Dedups one-knee-per-subject
(seed 42), computes SRM (BCa 95% CI) / Pearson r / CCC / ICC(A,1) / Bland-Altman,
and prints a v3.3 vs v8(template) vs v8b(baseline-grid) comparison.

Outputs ../results/v8b_longitudinal_table.md and v8b_3way_comparison.md.
"""
from __future__ import annotations

import csv
import collections
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, bca_ci_srm, pearson_r, lin_ccc, icc_a1, ba, fmt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
V8B_CSV = RESULTS / "v8b_merged_deltas.csv"

# v3.3 manuscript + v8(template) headline references for the 3-way table.
V33 = {"MFTC": -1.59, "cMF": -1.62, "MT": -1.09,
       "MFTC_r": 0.64, "cMF_r": 0.67, "MT_r": 0.48}
V8 = {"MFTC": -1.40, "cMF": -1.53, "MT": -0.76,
      "MFTC_r": 0.55, "cMF_r": 0.59, "MT_r": 0.38}


def dedup(rows, seed=42):
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["pid"], r["side"])):
        by[r["pid"]].append(r)
    return [ks[0] if len(ks) == 1 else ks[int(rng.integers(0, len(ks)))] for ks in by.values()]


def col(rows, c):
    out = []
    for r in rows:
        v = r.get(c, "")
        out.append(float(v) if v not in ("", "nan", None) else np.nan)
    return np.array(out, float)


def main():
    rows = [r for r in csv.DictReader(open(V8B_CSV, encoding="utf-8"))
            if r.get("cohort") == "progressor"]
    ded = dedup(rows)
    print(f"rows={len(rows)} deduped n={len(ded)}")

    md = ["# v8b Table 4 — Longitudinal change, BASELINE-GRID projection",
          f"(n={len(ded)}; cartilage-morphometry baseline_grid + raycast_2d. "
          "SRM BCa 95% CI seed42; r/CCC/ICC vs manual QCart; BA = PD−manual.)\n",
          "| Region | PD Δ (μm) | PD SRM (95% CI) | QCart Δ (μm) | QCart SRM | r | CCC | ICC | BA bias (μm) | 95% LOA (μm) | n |",
          "|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    res = {}
    for reg in ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]:
        pd = col(ded, f"pd_{reg}_d"); qc = col(ded, f"eck_{reg}_d")
        pdu, qcu = pd * 1000, qc * 1000
        lo, hi = bca_ci_srm(pd)
        bias, llo, lhi = ba(pdu, qcu)
        r = pearson_r(pd, qc)
        res[reg] = {"srm": srm(pd), "r": r}
        n = int((np.isfinite(pd) & np.isfinite(qc)).sum())
        md.append(f"| {reg} | {fmt(np.nanmean(pdu),0)} ± {fmt(np.nanstd(pdu,ddof=1),0)} "
                  f"| {fmt(srm(pd))} ({fmt(lo)}, {fmt(hi)}) "
                  f"| {fmt(np.nanmean(qcu),0)} ± {fmt(np.nanstd(qcu,ddof=1),0)} | {fmt(srm(qc))} "
                  f"| {fmt(r)} | {fmt(lin_ccc(pd,qc))} | {fmt(icc_a1(pd,qc))} "
                  f"| {fmt(bias,0)} | [{fmt(llo,0)}, {fmt(lhi,0)}] | {n} |")
    (RESULTS / "v8b_longitudinal_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # 3-way comparison
    cmp = ["# 3-way: v3.3 (baseline-grid, orig) vs v8 (template) vs v8b (baseline-grid, restored)\n",
           "| Metric | v3.3 | v8 template | v8b grid |", "|--------|--:|--:|--:|"]
    for reg in ["MFTC", "cMF", "MT"]:
        cmp.append(f"| {reg} SRM | {fmt(V33[reg])} | {fmt(V8[reg])} | {fmt(res[reg]['srm'])} |")
    for reg in ["MFTC", "cMF", "MT"]:
        cmp.append(f"| {reg} r vs QCart | {fmt(V33[reg+'_r'])} | {fmt(V8[reg+'_r'])} | {fmt(res[reg]['r'])} |")
    (RESULTS / "v8b_3way_comparison.md").write_text("\n".join(cmp) + "\n", encoding="utf-8")

    print("\n".join(md))
    print("\n".join(cmp))
    print(f"\n[ok] wrote v8b tables to {RESULTS}")


if __name__ == "__main__":
    main()
