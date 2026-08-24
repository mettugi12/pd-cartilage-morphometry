"""v9 PREDEFINED label-free technical QC + QC-pass primary analyses (revised 2026-08-21).

PRIMARY QC = registration non-convergence ONLY, applied symmetrically to both arms:
    icp_fail  - 48m->00m rigid ICP final ASSD > 2.0 mm (either bone) or |t| > 30 mm.
    Pure geometry; reads neither the reference (QCart) nor the PD longitudinal endpoint.

SECONDARY flag (reported separately, NOT used for the primary cohort):
    orient_flag - PD lateral change exceeds medial change by > 0.15 mm in a knee
    radiographically selected for medial JSW progression. Although reference-free, this
    rule reads the PD endpoint under evaluation and is applied only to progressors; the
    decomposition shows it also raises QCart's AUC, i.e. it selects biologically cleaner
    progressors rather than PD technical failures. It is therefore outcome-dependent and
    is reported as an exploratory exclusion pending independent (anatomical) confirmation.

Individual-level detection: sensitivity at FIXED empirical specificity (95% and 90%)
using the stable-arm percentile as threshold. SDC95 = 1.96*SD(stable delta) is retained
only as a variability descriptor; it is NOT a classification threshold (stable knees
have a non-zero mean drift, so +/-SDC cutoffs give method-dependent specificity).

Order: QC first, then one-knee-per-subject dedup (seed 42) among QC-pass knees.

Inputs:  ../results/v9_prog_deltas.csv    + run_prog_v9.log      (progressors)
         ../results/v8b_nonprog_deltas.csv + v8b_nonprog_icp.csv (stable arm)
Outputs: ../results/v9_qc_flags.csv                    per-knee flags, both arms
         ../results/v9_prog_qcpass.csv / v9_nonprog_qcpass.csv   ICP-pass deduped rows
         ../results/v9_longitudinal_table_qcpass.md    responsiveness (primary + full)
         ../results/v9_discrimination_table_qcpass.md  discrimination (primary + full)
         ../results/v9_qc_decomposition.md             none / ICP / orientation / combined
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from aggregate_v8 import srm, bca_ci_srm, pearson_r, lin_ccc, icc_a1, ba, fmt
from aggregate_v8b import dedup, col
from aggregate_discrimination_v8b import auc_mw, auc_boot_ci

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
REGIONS = ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]
ASSD_MAX, T_MAX = 2.0, 30.0
ORIENT_MM = 0.15
RNG = np.random.default_rng(42)

ICP_RE = re.compile(r"ASSD\s+([\d.]+)\S([\d.]+)mm\s+\|t\|=([\d.]+)mm")
KNEE_RE = re.compile(r"\[\d+/\d+\]\s+(\d+)_(LEFT|RIGHT)")


def parse_icp_log(log_path: Path) -> dict:
    out, icps = {}, []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = ICP_RE.search(line)
        if m:
            icps.append((float(m.group(2)), float(m.group(3))))
            continue
        k = KNEE_RE.search(line)
        if k and icps:
            out[(k.group(1), k.group(2))] = {
                "fem_assd": icps[0][0], "fem_t": icps[0][1],
                "tib_assd": icps[1][0] if len(icps) > 1 else np.nan,
                "tib_t": icps[1][1] if len(icps) > 1 else np.nan}
            icps = []
    return out


def icp_fail(m):
    if m is None:
        return None
    vals = (m["fem_assd"], m["tib_assd"], m["fem_t"], m["tib_t"])
    if not all(np.isfinite(v) for v in vals):
        return None
    return (m["fem_assd"] > ASSD_MAX or m["tib_assd"] > ASSD_MAX
            or m["fem_t"] > T_MAX or m["tib_t"] > T_MAX)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def orient_flag(r):
    med = _f(r["pd_cMF_d"]) + _f(r["pd_MT_d"])
    lat = _f(r["pd_cLF_d"]) + _f(r["pd_LT_d"])
    return bool(np.isfinite(med) and np.isfinite(lat) and lat < med - ORIENT_MM)


def sens_at_spec(dp, dn, spec):
    """Sensitivity at fixed empirical specificity: threshold = (1-spec) percentile of
    stable delta; progression = delta below threshold."""
    dp = dp[np.isfinite(dp)]; dn = dn[np.isfinite(dn)]
    thr = np.percentile(dn, (1 - spec) * 100)
    return np.mean(dp < thr) * 100, thr


def disc_block(dp_rows, dn_rows, title):
    md = [f"\n## {title} (prog n={len(dp_rows)}, stable n={len(dn_rows)})\n"]
    for method, tag in (("pd", "PD-auto"), ("eck", "manual QCart")):
        md += [f"### {tag}\n",
               "| Region | stable Δ (μm) | stable SRM | prog Δ (μm) | AUC (95% CI) | SDC95* (μm) | thr@95%spec (μm) | sens@95%spec | sens@90%spec |",
               "|--------|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for reg in REGIONS:
            dn = col(dn_rows, f"{method}_{reg}_d") * 1000
            dp = col(dp_rows, f"{method}_{reg}_d") * 1000
            a = auc_mw(dp, dn); lo, hi = auc_boot_ci(dp, dn)
            s95, t95 = sens_at_spec(dp, dn, 0.95); s90, _ = sens_at_spec(dp, dn, 0.90)
            md.append(f"| {reg} | {fmt(np.nanmean(dn),0)} ± {fmt(np.nanstd(dn,ddof=1),0)} "
                      f"| {fmt(srm(dn))} | {fmt(np.nanmean(dp),0)} "
                      f"| {fmt(a)} ({fmt(lo)}, {fmt(hi)}) | {fmt(1.96*np.nanstd(dn,ddof=1),0)} "
                      f"| {fmt(t95,0)} | {fmt(s95,0)}% | {fmt(s90,0)}% |")
        md.append("")
    md += ["### Paired ΔAUC (PD − QCart), bootstrap 2000×\n", "| Region | ΔAUC (95% CI) |", "|--------|--:|"]
    for reg in REGIONS:
        pn, qn = col(dn_rows, f"pd_{reg}_d"), col(dn_rows, f"eck_{reg}_d")
        pp, qp = col(dp_rows, f"pd_{reg}_d"), col(dp_rows, f"eck_{reg}_d")
        diffs = []
        for _ in range(2000):
            bi = RNG.integers(0, len(pp), len(pp)); bj = RNG.integers(0, len(pn), len(pn))
            diffs.append(auc_mw(pp[bi], pn[bj]) - auc_mw(qp[bi], qn[bj]))
        d = auc_mw(pp, pn) - auc_mw(qp, qn)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        md.append(f"| {reg} | {fmt(d)} ({fmt(lo)}, {fmt(hi)}) |")
    md.append("\n\\*SDC95 = 1.96·SD of stable-knee Δ — a variability descriptor only, not a "
              "classification threshold (stable PD knees carry a non-zero mean drift).")
    return md


def long_block(ded, title):
    md = [f"\n## {title} (n={len(ded)})\n",
          "| Region | PD Δ (μm) | PD SRM (95% CI) | QCart Δ (μm) | QCart SRM | r | CCC | ICC | BA bias (μm) | 95% LOA (μm) |",
          "|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        p = col(ded, f"pd_{reg}_d"); q = col(ded, f"eck_{reg}_d")
        pu, qu = p * 1000, q * 1000
        lo, hi = bca_ci_srm(p); bias, llo, lhi = ba(pu, qu)
        md.append(f"| {reg} | {fmt(np.nanmean(pu),0)} ± {fmt(np.nanstd(pu,ddof=1),0)} "
                  f"| {fmt(srm(p))} ({fmt(lo)}, {fmt(hi)}) "
                  f"| {fmt(np.nanmean(qu),0)} ± {fmt(np.nanstd(qu,ddof=1),0)} | {fmt(srm(q))} "
                  f"| {fmt(pearson_r(p,q))} | {fmt(lin_ccc(p,q))} | {fmt(icc_a1(p,q))} "
                  f"| {fmt(bias,0)} | [{fmt(llo,0)}, {fmt(lhi,0)}] |")
    return md


def main():
    prog = [r for r in csv.DictReader(open(RESULTS / "v9_prog_deltas.csv", encoding="utf-8"))
            if r.get("cohort") == "progressor"]
    nonp = [r for r in csv.DictReader(open(RESULTS / "v8b_nonprog_deltas.csv", encoding="utf-8"))
            if r.get("cohort") == "nonprogressor"]
    icp_p = parse_icp_log(RESULTS / "run_prog_v9.log")
    icp_n = {(r["pid"], r["side"]): {k: float(r[k]) for k in ("fem_assd", "fem_t", "tib_assd", "tib_t")}
             for r in csv.DictReader(open(RESULTS / "v8b_nonprog_icp.csv", encoding="utf-8"))}

    flags = []
    for arm, rows, icp in (("progressor", prog, icp_p), ("stable", nonp, icp_n)):
        for r in rows:
            fi = icp_fail(icp.get((r["pid"], r["side"])))
            fo = orient_flag(r) if arm == "progressor" else False
            flags.append({"arm": arm, "pid": r["pid"], "side": r["side"],
                          "icp_fail": fi, "orient_flag": fo,
                          "qc_pass_primary": fi is not True,
                          "icp_indeterminate": fi is None})
    with open(RESULTS / "v9_qc_flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flags[0].keys())); w.writeheader(); w.writerows(flags)

    key = {(f["arm"], f["pid"], f["side"]): f for f in flags}
    n_ind = sum(1 for f in flags if f["icp_indeterminate"])
    n_icp_p = sum(1 for f in flags if f["arm"] == "progressor" and f["icp_fail"] is True)
    n_icp_n = sum(1 for f in flags if f["arm"] == "stable" and f["icp_fail"] is True)
    n_or = sum(1 for f in flags if f["orient_flag"])
    n_or_and_icp = sum(1 for f in flags if f["orient_flag"] and f["icp_fail"] is True)

    P_icp = [r for r in prog if key[("progressor", r["pid"], r["side"])]["qc_pass_primary"]]
    N_icp = [r for r in nonp if key[("stable", r["pid"], r["side"])]["qc_pass_primary"]]
    P_or = [r for r in prog if not key[("progressor", r["pid"], r["side"])]["orient_flag"]]
    P_both = [r for r in P_icp if not key[("progressor", r["pid"], r["side"])]["orient_flag"]]

    hdr = (f"PRIMARY QC = ICP non-convergence only (ASSD > {ASSD_MAX} mm or |t| > {T_MAX:.0f} mm), both arms: "
           f"progressors {n_icp_p}/{len(prog)} flagged, stable {n_icp_n}/{len(nonp)} flagged; "
           f"ICP metrics indeterminate for {n_ind} knees. Orientation flag (exploratory, progressors only, "
           f"lat < med − {ORIENT_MM} mm): {n_or}/{len(prog)} ({n_or_and_icp} overlap ICP). "
           f"QC precedes one-knee-per-subject dedup (seed 42).")
    print(hdr)

    dP, dN = dedup(P_icp), dedup(N_icp)
    dP_all, dN_all = dedup(prog), dedup(nonp)
    for rows, name in ((dP, "v9_prog_qcpass.csv"), (dN, "v9_nonprog_qcpass.csv")):
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    md = ["# v9 — Longitudinal change vs manual QCart", f"({hdr})"]
    md += long_block(dP, "ICP-QC-pass — PRIMARY")
    md += long_block(dP_all, "Full cohort — sensitivity")
    (RESULTS / "v9_longitudinal_table_qcpass.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    md2 = ["# v9 — Progressor vs stable discrimination", f"({hdr} AUC: Mann-Whitney, bootstrap 95% CI seed 42.)"]
    md2 += disc_block(dP, dN, "ICP-QC-pass — PRIMARY")
    md2 += disc_block(dP_all, dN_all, "Full cohort — sensitivity")
    (RESULTS / "v9_discrimination_table_qcpass.md").write_text("\n".join(md2) + "\n", encoding="utf-8")

    md3 = ["# v9 — QC decomposition (MFTC)",
           "Each row applies one exclusion set. The orientation rule raises QCart's AUC although it never "
           "reads QCart → it selects biologically cleaner progressors (outcome-dependent), so it is excluded "
           "from the primary QC and reported here as exploratory.\n",
           "| QC applied | prog / stable | PD AUC | QCart AUC | PD–QCart r | PD sens@95%spec | QCart sens@95%spec |",
           "|---|--:|--:|--:|--:|--:|--:|"]
    for name, P, N in (("None", prog, nonp), ("ICP only, both arms (PRIMARY)", P_icp, N_icp),
                       ("Orientation only (exploratory)", P_or, nonp), ("ICP + orientation", P_both, N_icp)):
        P, N = dedup(P), dedup(N)
        dp, dn = col(P, "pd_MFTC_d"), col(N, "pd_MFTC_d"); qp, qn = col(P, "eck_MFTC_d"), col(N, "eck_MFTC_d")
        sp, _ = sens_at_spec(dp, dn, 0.95); sq, _ = sens_at_spec(qp, qn, 0.95)
        md3.append(f"| {name} | {len(P)} / {len(N)} | {auc_mw(dp,dn):.3f} | {auc_mw(qp,qn):.3f} "
                   f"| {pearson_r(dp,qp):.3f} | {sp:.0f}% | {sq:.0f}% |")
    (RESULTS / "v9_qc_decomposition.md").write_text("\n".join(md3) + "\n", encoding="utf-8")

    print("\n".join(md)); print("\n".join(md2)); print("\n".join(md3))
    print(f"\n[ok] wrote v9_qc_flags.csv, v9_*_qcpass.csv, 3 tables -> {RESULTS}")


if __name__ == "__main__":
    main()
