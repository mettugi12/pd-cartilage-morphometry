"""v9.3 — QC decomposition (Supplementary Table S4) and full-cohort no-QC sensitivity (Table S5)
recomputed under the v9.3 measurement definitions (reference-matched regions, total-bone-area
means, symmetric follow-up handling, 75 %-definition expert reference).

The earlier supplement carried these two tables from the v9 "original processing" (asymmetric
follow-up rule, grid femoral band), so their 150/151 row (MFTC AUC 0.816) contradicted the 150/151
sensitivity row of the v9.3 main text (0.86). Here every row is computed from the same per-knee
table as the primary analysis (../results/v9_3_long_abs.csv, all 344 processed knees):

  None                         all processed knees, one knee per participant (seed 42)
  Registration only (150/151)  ICP non-convergence flag from v9 (v9_qc_flags.csv), both arms
  Orientation only             exploratory flag recomputed on the v9.3 PD deltas:
                               (cLF + LT) change < (cMF + MT) change − 0.15 mm, progressors only
  Registration + orientation

Outputs: ../results/v9_3_qc_decomposition.json / .md
Usage: python qc_decomposition_v9_3.py [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
V9RES = HERE.parent.parent / "v9" / "results"
sys.path.insert(0, str(HERE.parent.parent / "v9" / "analysis"))
from aggregate_v8 import srm  # noqa: E402
from aggregate_v8b import col, dedup  # noqa: E402
from aggregate_discrimination_v8b import auc_boot_ci, auc_mw  # noqa: E402

ORIENT_MM = 0.15
REGIONS = ["MFTC", "cMF", "MT"]


def sens_at_spec(dp, dn, spec):
    dp = dp[np.isfinite(dp)]; dn = dn[np.isfinite(dn)]
    thr = np.percentile(dn, (1 - spec) * 100)
    return float(np.mean(dp < thr) * 100), float(thr)


def orient_flag(r):
    med = float(r["pd_cMF_d"]) + float(r["pd_MT_d"])
    lat = float(r["pd_cLF_d"]) + float(r["pd_LT_d"])
    return bool(np.isfinite(med) and np.isfinite(lat) and lat < med - ORIENT_MM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    a = ap.parse_args()

    la = pd.read_csv(RES / "v9_3_long_abs.csv")
    la["pid"] = la.pid.astype(str)
    if a.n_run:
        la = la.groupby("cohort").head(a.n_run)
    fl = pd.read_csv(V9RES / "v9_qc_flags.csv", dtype=str)
    fl["arm"] = fl.arm.replace({"stable": "nonprogressor"})
    icp_pass = {(r.arm, r.pid, r.side): r.qc_pass_primary == "True" for r in fl.itertuples()}

    rows = la.to_dict("records")
    prog = [r for r in rows if r["cohort"] == "progressor"]
    nonp = [r for r in rows if r["cohort"] == "nonprogressor"]
    for r in prog:
        r["orient_flag"] = orient_flag(r)
    P_icp = [r for r in prog if icp_pass.get(("progressor", r["pid"], r["side"]), True)]
    N_icp = [r for r in nonp if icp_pass.get(("nonprogressor", r["pid"], r["side"]), True)]
    P_or = [r for r in prog if not r["orient_flag"]]
    P_both = [r for r in P_icp if not r["orient_flag"]]
    n_or = sum(r["orient_flag"] for r in prog)
    print(f"processed {len(prog)}/{len(nonp)}; ICP flagged {len(prog)-len(P_icp)}/{len(nonp)-len(N_icp)}; "
          f"orientation flagged (v9.3 deltas) {n_or}/{len(prog)}")

    # ---- Table S4: decomposition, MFTC
    s4 = []
    for name, P, N in (("None", prog, nonp), ("Registration only (150/151 set)", P_icp, N_icp),
                       ("Orientation only (exploratory)", P_or, nonp), ("Registration + orientation", P_both, N_icp)):
        P, N = dedup(P), dedup(N)
        dp, dn = col(P, "pd_MFTC_d"), col(N, "pd_MFTC_d"); qp, qn = col(P, "eck_MFTC_d"), col(N, "eck_MFTC_d")
        sp, _ = sens_at_spec(dp, dn, 0.95); sq, _ = sens_at_spec(qp, qn, 0.95)
        m = np.isfinite(dp) & np.isfinite(qp)
        s4.append({"qc": name, "n_prog": len(P), "n_stab": len(N), "auc_pd": float(auc_mw(dp, dn)), "auc_q": float(auc_mw(qp, qn)),
                   "r": float(np.corrcoef(dp[m], qp[m])[0, 1]), "sens95_pd": sp, "sens95_q": sq})

    # ---- Table S5: full cohort, no QC, one knee per participant
    dP, dN = dedup(prog), dedup(nonp)
    s5 = {"n_prog": len(dP), "n_stab": len(dN), "regions": {}}
    for reg in REGIONS:
        s5["regions"][reg] = {}
        for tag in ("pd", "eck"):
            dp = col(dP, f"{tag}_{reg}_d") * 1000; dn = col(dN, f"{tag}_{reg}_d") * 1000
            lo, hi = auc_boot_ci(dp, dn); s95, _ = sens_at_spec(dp, dn, 0.95)
            s5["regions"][reg][tag] = {"prog_mean": float(np.nanmean(dp)), "prog_sd": float(np.nanstd(dp, ddof=1)),
                                       "srm": float(srm(dp / 1000)), "stab_mean": float(np.nanmean(dn)), "stab_sd": float(np.nanstd(dn, ddof=1)),
                                       "auc": float(auc_mw(dp, dn)), "auc_ci": [float(lo), float(hi)], "sens95": s95}
    out = {"n_processed": [len(prog), len(nonp)], "n_orient_flagged": int(n_or), "table_s4": s4, "table_s5": s5}

    md = ["# v9.3 — QC decomposition (MFTC) and full-cohort sensitivity, v9.3 definitions", "",
          "| QC applied | prog / stable | PD AUC | Manual DESS AUC | r | PD sens@95% | DESS sens@95% |", "|---|--:|--:|--:|--:|--:|--:|"]
    md += [f"| {r['qc']} | {r['n_prog']} / {r['n_stab']} | {r['auc_pd']:.3f} | {r['auc_q']:.3f} | {r['r']:.3f} | {r['sens95_pd']:.0f}% | {r['sens95_q']:.0f}% |" for r in s4]
    md += ["", f"## Full cohort without QC ({s5['n_prog']} / {s5['n_stab']})", "",
           "| Region | Method | Progressor Δ (μm) | SRM | Stable Δ (μm) | AUC (95% CI) | sens@95% |", "|---|---|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        for tag, name in (("pd", "Automated PD/IW"), ("eck", "Manual DESS")):
            x = s5["regions"][reg][tag]
            md.append(f"| {reg} | {name} | {x['prog_mean']:.0f}±{x['prog_sd']:.0f} | {x['srm']:.2f} | {x['stab_mean']:.0f}±{x['stab_sd']:.0f} "
                      f"| {x['auc']:.2f} ({x['auc_ci'][0]:.2f}, {x['auc_ci'][1]:.2f}) | {x['sens95']:.0f}% |")
    print("\n".join(md))
    if a.dry_run:
        print("[dry_run] not saved"); return
    (RES / "v9_3_qc_decomposition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RES / "v9_3_qc_decomposition.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[ok] wrote", RES / "v9_3_qc_decomposition.{json,md}")


if __name__ == "__main__":
    main()
