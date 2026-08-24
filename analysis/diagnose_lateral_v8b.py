"""Lateral-compartment discrepancy diagnosis for the JMRI reframe (plan section 3B).

Question: why does PD-auto read more lateral thinning than manual QCart in the
progressor cohort (LFTC SRM -1.07 vs -0.49, r 0.43)? Three candidate explanations,
each testable on the existing per-knee v8b deltas:

  1. Outlier-driven: a few failed knees (flip / bad ICP) drag the mean.
     -> trimmed means, medians, QC-suspect exclusion sensitivity.
  2. KL / disease-severity dependent: worse knees fail more.
     -> discrepancy vs KL_00.
  3. Medial->lateral cross-talk: registration drift in knees with large medial
     change smears apparent change into the lateral compartment.
     -> corr(medial PD change, lateral PD-QCart discrepancy).

Writes ../results/v8b_lateral_diagnosis.md. No pipeline reruns; CSV math only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, pearson_r, ba, fmt
from aggregate_v8b import dedup, col

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return pearson_r(ra, rb)


def stats_block(rows, label, regions=("MFTC", "cMF", "MT", "LFTC", "cLF", "LT")):
    out = [f"\n### {label} (n={len(rows)})\n",
           "| Region | PD Δ (μm) | PD SRM | QCart Δ (μm) | QCart SRM | r | BA bias (μm) |",
           "|--------|--:|--:|--:|--:|--:|--:|"]
    for reg in regions:
        pd_ = col(rows, f"pd_{reg}_d"); qc = col(rows, f"eck_{reg}_d")
        bias, _, _ = ba(pd_ * 1000, qc * 1000)
        out.append(f"| {reg} | {fmt(np.nanmean(pd_)*1000,0)} | {fmt(srm(pd_))} "
                   f"| {fmt(np.nanmean(qc)*1000,0)} | {fmt(srm(qc))} "
                   f"| {fmt(pearson_r(pd_, qc))} | {fmt(bias,0)} |")
    return out


def main():
    rows = [r for r in csv.DictReader(open(RESULTS / "v8b_merged_deltas.csv", encoding="utf-8"))
            if r.get("cohort") == "progressor"]
    ded = dedup(rows)
    qc_flag = {(r["pid"], r["side"]) for r in
               csv.DictReader(open(RESULTS / "v8b_qc_suspects.csv", encoding="utf-8"))}

    md = ["# v8b lateral-compartment discrepancy diagnosis",
          f"(progressor cohort, deduped n={len(ded)}; QC-suspect list n={len(qc_flag)})"]

    # --- full cohort reference ---
    md += stats_block(ded, "All knees (reference)")

    # --- 1. outlier contribution ---
    md += ["\n## 1. Outlier contribution\n",
           "| Region | mean Δ | 10% trimmed mean Δ | median Δ | QCart median Δ |",
           "|--------|--:|--:|--:|--:|"]
    for reg in ("LFTC", "cLF", "LT", "MFTC"):
        pd_ = col(ded, f"pd_{reg}_d") * 1000
        qc = col(ded, f"eck_{reg}_d") * 1000
        v = pd_[np.isfinite(pd_)]
        k = max(1, int(round(0.10 * len(v))))
        vs = np.sort(v)
        trim = vs[k:-k].mean() if len(vs) > 2 * k else np.nan
        md.append(f"| {reg} | {fmt(v.mean(),0)} | {fmt(trim,0)} | {fmt(np.median(v),0)} "
                  f"| {fmt(np.nanmedian(qc),0)} |")

    # --- 2. QC-suspect exclusion sensitivity ---
    kept = [r for r in ded if (r["pid"], r["side"]) not in qc_flag]
    md += ["\n## 2. Excluding QC-suspect knees"]
    md += stats_block(kept, f"QC-clean subset ({len(ded)-len(kept)} excluded)")

    # --- 3. KL association ---
    md += ["\n## 3. Discrepancy (PD−QCart) vs baseline KL\n",
           "| Region | Spearman ρ (disc vs KL_00) | mean disc KL≤2 (μm) | mean disc KL≥3 (μm) |",
           "|--------|--:|--:|--:|"]
    kl = col(ded, "KL_00")
    for reg in ("LFTC", "cLF", "LT", "MFTC"):
        disc = (col(ded, f"pd_{reg}_d") - col(ded, f"eck_{reg}_d")) * 1000
        rho = spearman(disc, kl)
        lo = disc[(kl <= 2) & np.isfinite(disc)]
        hi = disc[(kl >= 3) & np.isfinite(disc)]
        md.append(f"| {reg} | {fmt(rho)} | {fmt(lo.mean(),0)} (n={len(lo)}) "
                  f"| {fmt(hi.mean(),0)} (n={len(hi)}) |")

    # --- 4. medial->lateral cross-talk ---
    md += ["\n## 4. Medial→lateral cross-talk\n",
           "corr(PD medial change, lateral PD−QCart discrepancy): if registration drift",
           "smears medial thinning laterally, knees losing more medial cartilage should",
           "show more spurious lateral 'thinning'.\n",
           "| Pair | Pearson r | Spearman ρ |", "|------|--:|--:|"]
    m_pd = col(ded, "pd_MFTC_d")
    for reg in ("LFTC", "cLF", "LT"):
        disc = col(ded, f"pd_{reg}_d") - col(ded, f"eck_{reg}_d")
        md.append(f"| pd_MFTC_d vs disc_{reg} | {fmt(pearson_r(m_pd, disc))} "
                  f"| {fmt(spearman(m_pd, disc))} |")
    qc_lat = col(ded, "eck_LFTC_d")
    md.append(f"| eck_MFTC_d vs eck_LFTC_d (manual, for reference) "
              f"| {fmt(pearson_r(col(ded,'eck_MFTC_d'), qc_lat))} "
              f"| {fmt(spearman(col(ded,'eck_MFTC_d'), qc_lat))} |")

    out = RESULTS / "v8b_lateral_diagnosis.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
