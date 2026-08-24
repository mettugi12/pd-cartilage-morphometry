"""Registration-quality QC for the non-progressor arm (JMRI plan 3A honesty check).

Parses per-knee femur/tibia ICP convergence (final ASSD, |t|) from the run log,
joins with v8b_nonprog_deltas.csv, and re-computes stable-knee drift + SDC after
excluding knees whose 48m->00m registration failed to converge. Tests whether the
PD specificity gap is concentrated in registration failures (fixable by automated
QC) or uniform (a true drift floor).

Writes ../results/v8b_nonprog_icp_qc.md and per-knee ../results/v8b_nonprog_icp.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, fmt
from aggregate_v8b import col

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
LOG = RESULTS / "run_nonprog_v8b.log"

ICP_RE = re.compile(r"ASSD\s+([\d.]+)\S([\d.]+)mm\s+\|t\|=([\d.]+)mm")
KNEE_RE = re.compile(r"\[\d+/\d+\]\s+(\d+)_(LEFT|RIGHT)")


def main():
    icps, per_knee = [], {}
    for line in open(LOG, encoding="utf-8", errors="replace"):
        m = ICP_RE.search(line)
        if m:
            icps.append((float(m.group(2)), float(m.group(3))))  # final ASSD, |t|
            continue
        k = KNEE_RE.search(line)
        if k and icps:
            per_knee[(k.group(1), k.group(2))] = {
                "fem_assd": icps[0][0], "fem_t": icps[0][1],
                "tib_assd": icps[1][0] if len(icps) > 1 else np.nan,
                "tib_t": icps[1][1] if len(icps) > 1 else np.nan}
            icps = []

    rows = list(csv.DictReader(open(RESULTS / "v8b_nonprog_deltas.csv", encoding="utf-8")))
    fa = np.array([per_knee.get((r["pid"], r["side"]), {}).get("fem_assd", np.nan) for r in rows])
    ta = np.array([per_knee.get((r["pid"], r["side"]), {}).get("tib_assd", np.nan) for r in rows])
    ft = np.array([per_knee.get((r["pid"], r["side"]), {}).get("fem_t", np.nan) for r in rows])
    tt = np.array([per_knee.get((r["pid"], r["side"]), {}).get("tib_t", np.nan) for r in rows])

    with open(RESULTS / "v8b_nonprog_icp.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "side", "fem_assd", "fem_t", "tib_assd", "tib_t"])
        for r, a, b, c, d in zip(rows, fa, ft, ta, tt):
            w.writerow([r["pid"], r["side"], a, b, c, d])

    md = [f"# Non-progressor arm — ICP registration QC (n={len(rows)})\n",
          f"final ASSD femur: median {np.nanmedian(fa):.2f} mm, p90 {np.nanpercentile(fa,90):.2f}, max {np.nanmax(fa):.2f}",
          f"final ASSD tibia: median {np.nanmedian(ta):.2f} mm, p90 {np.nanpercentile(ta,90):.2f}, max {np.nanmax(ta):.2f}\n"]

    bad = (fa > 2.0) | (ta > 2.0) | (ft > 30) | (tt > 30)
    md.append(f"QC-fail rule: final ASSD > 2.0 mm (either bone) or |t| > 30 mm -> "
              f"**{int(np.nansum(bad))}/{len(rows)} flagged**\n")

    md += ["| Region | all: Δ (μm), SRM, SDC95 | QC-pass: Δ (μm), SRM, SDC95 |",
           "|--------|--|--|"]
    keep_rows = [r for r, b in zip(rows, bad) if not b]
    for reg in ("MFTC", "cMF", "MT", "LFTC", "cLF", "LT"):
        d_all = col(rows, f"pd_{reg}_d") * 1000
        d_ok = col(keep_rows, f"pd_{reg}_d") * 1000
        md.append(f"| {reg} | {fmt(np.nanmean(d_all),0)}, {fmt(srm(d_all))}, "
                  f"{fmt(1.96*np.nanstd(d_all,ddof=1),0)} "
                  f"| {fmt(np.nanmean(d_ok),0)}, {fmt(srm(d_ok))}, "
                  f"{fmt(1.96*np.nanstd(d_ok,ddof=1),0)} |")

    # is drift magnitude associated with registration quality?
    for reg, a in (("cMF", fa), ("MT", ta)):
        d = col(rows, f"pd_{reg}_d") * 1000
        m = np.isfinite(d) & np.isfinite(a)
        r_ = np.corrcoef(np.abs(d[m]), a[m])[0, 1] if m.sum() > 2 else np.nan
        md.append(f"\ncorr(|Δ {reg}|, final {'femur' if reg=='cMF' else 'tibia'} ASSD): r = {fmt(r_)}")

    out = RESULTS / "v8b_nonprog_icp_qc.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
