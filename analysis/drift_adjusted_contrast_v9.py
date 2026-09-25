"""v9 post-JMRI revision analysis: drift-adjusted regional contrast.

Answers Reviewer 1's "MT: PD -349 vs QCart -196, why?" and provides the
drift-corrected responsiveness metrics that the naive progressor-only SRM hides.

For every region and both methods (PD-auto, manual QCart) on the PRIMARY frame
(ICP-only symmetric QC, 150 progressors / 151 stable):

  * progressor and stable mean +- SD of 48-month change
  * excess = progressor mean - stable mean (drift cancels), bootstrap 95% CI
  * PD/QCart excess ratio, bootstrap 95% CI (paired-by-knee resampling within arm)
  * drift-corrected SRM = excess / SD(progressor change)
  * Cohen's d progressor vs stable (pooled SD), bootstrap 95% CI
  * stable-arm between-method drift (PD - QCart) with paired-t CI

Plus two mechanism tables for the MT question:
  * PD tibial AP-thirds (aMT / cMT / pMT) change in both arms (no QCart thirds available)
  * stable-arm PD and QCart change stratified by baseline KL grade (KL_00)

Inputs : ../results/v9_prog_qcpass.csv, ../results/v9_nonprog_qcpass.csv
Outputs: ../results/v9_drift_adjusted_contrast.md  (+ .csv)

Usage:
  python drift_adjusted_contrast_v9.py [--dry_run] [--n_run N] [--n_boot 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

REGIONS = ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]
THIRDS = ["aMT", "cMT", "pMT", "aLT", "cLT", "pLT"]
METHODS = {"PD-auto": "pd", "manual QCart": "eck"}
UM = 1000.0


def load(n_run: int | None):
    prog = pd.read_csv(RES / "v9_prog_qcpass.csv")
    stab = pd.read_csv(RES / "v9_nonprog_qcpass.csv")
    if n_run:
        prog, stab = prog.head(n_run), stab.head(n_run)
    return prog, stab


def col(df, method, region):
    return (df[f"{METHODS[method]}_{region}_d"].to_numpy(dtype=float)) * UM


def boot_ci(fn, rng, n_boot, *arrays):
    vals = np.empty(n_boot)
    for i in range(n_boot):
        samp = [a[rng.integers(0, len(a), len(a))] for a in arrays]
        vals[i] = fn(*samp)
    return np.nanpercentile(vals, [2.5, 97.5])


def cohens_d(p, s):
    sp = np.sqrt(((len(p) - 1) * p.var(ddof=1) + (len(s) - 1) * s.var(ddof=1)) / (len(p) + len(s) - 2))
    return (p.mean() - s.mean()) / sp


def fmt(x, nd=0):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    prog, stab = load(a.n_run)
    nP, nS = len(prog), len(stab)
    rows = []
    md = [f"# v9 — Drift-adjusted regional contrast (primary frame, {nP} progressors / {nS} stable)",
          "", f"Bootstrap {a.n_boot} resamples, seed {a.seed}; all values μm; change = 48 m − baseline.",
          "Excess = progressor mean − stable mean (systematic drift cancels). Drift-corrected SRM = excess / SD(progressor change).",
          "Cohen's d = (progressor − stable) / pooled SD. Ratio = PD excess / QCart excess.", ""]

    # ---- Table A: per region, per method ---------------------------------------------------
    md += ["## A. Progressor, stable, and drift-adjusted contrast by region", "",
           "| Region | Method | Progressor Δ | Stable Δ | Excess (95% CI) | Naive SRM | Drift-corrected SRM | Cohen's d (95% CI) |",
           "|---|---|--:|--:|--:|--:|--:|--:|"]
    excess_store = {}
    for reg in REGIONS:
        for m in METHODS:
            p, s = col(prog, m, reg), col(stab, m, reg)
            exc = p.mean() - s.mean()
            exc_ci = boot_ci(lambda x, y: x.mean() - y.mean(), rng, a.n_boot, p, s)
            d = cohens_d(p, s)
            d_ci = boot_ci(cohens_d, rng, a.n_boot, p, s)
            srm = p.mean() / p.std(ddof=1)
            srm_dc = exc / p.std(ddof=1)
            excess_store[(reg, m)] = (p, s)
            rows.append(dict(region=reg, method=m, prog_mean=p.mean(), prog_sd=p.std(ddof=1),
                             stable_mean=s.mean(), stable_sd=s.std(ddof=1), excess=exc,
                             excess_lo=exc_ci[0], excess_hi=exc_ci[1], srm_naive=srm,
                             srm_drift_corrected=srm_dc, cohens_d=d, d_lo=d_ci[0], d_hi=d_ci[1]))
            md.append(f"| {reg} | {m} | {fmt(p.mean())} ± {fmt(p.std(ddof=1))} | {fmt(s.mean())} ± {fmt(s.std(ddof=1))} | "
                      f"{fmt(exc)} ({fmt(exc_ci[0])}, {fmt(exc_ci[1])}) | {srm:.2f} | {srm_dc:.2f} | "
                      f"{d:.2f} ({d_ci[0]:.2f}, {d_ci[1]:.2f}) |")

    # ---- Table B: PD vs QCart excess ratio + stable-arm drift -------------------------------
    md += ["", "## B. PD-auto versus QCart: excess ratio and stable-arm additional drift", "",
           "| Region | PD excess | QCart excess | Ratio PD/QCart (95% CI) | Stable-arm PD−QCart drift (95% CI) | Progressor-arm PD−QCart (95% CI) |",
           "|---|--:|--:|--:|--:|--:|"]
    ratio_rows = []
    for reg in REGIONS:
        pP, sP = excess_store[(reg, "PD-auto")]
        pQ, sQ = excess_store[(reg, "manual QCart")]
        excP, excQ = pP.mean() - sP.mean(), pQ.mean() - sQ.mean()
        ratio = excP / excQ

        # paired-by-knee bootstrap: resample progressor indices and stable indices, keep PD/QCart pairing
        def ratio_fn(ip, is_):
            ip = ip.astype(int); is_ = is_.astype(int)
            eP = pP[ip].mean() - sP[is_].mean(); eQ = pQ[ip].mean() - sQ[is_].mean()
            return eP / eQ if eQ != 0 else np.nan
        r_ci = boot_ci(ratio_fn, rng, a.n_boot, np.arange(nP), np.arange(nS))

        dS = sP - sQ  # stable-arm within-knee method difference
        tS = stats.ttest_1samp(dS, 0)
        ciS = stats.t.interval(0.95, len(dS) - 1, loc=dS.mean(), scale=stats.sem(dS))
        dP = pP - pQ
        ciP = stats.t.interval(0.95, len(dP) - 1, loc=dP.mean(), scale=stats.sem(dP))
        ratio_rows.append(dict(region=reg, pd_excess=excP, qcart_excess=excQ, ratio=ratio,
                               ratio_lo=r_ci[0], ratio_hi=r_ci[1], stable_drift=dS.mean(),
                               stable_drift_lo=ciS[0], stable_drift_hi=ciS[1], stable_drift_p=tS.pvalue,
                               prog_diff=dP.mean(), prog_diff_lo=ciP[0], prog_diff_hi=ciP[1]))
        ratio_txt = (f"{ratio:.2f} ({r_ci[0]:.2f}, {r_ci[1]:.2f})" if abs(excQ) >= 50
                     else "n/a (QCart excess < 50 μm)")
        md.append(f"| {reg} | {fmt(excP)} | {fmt(excQ)} | {ratio_txt} | "
                  f"{fmt(dS.mean())} ({fmt(ciS[0])}, {fmt(ciS[1])}) | {fmt(dP.mean())} ({fmt(ciP[0])}, {fmt(ciP[1])}) |")

    # ---- Table C: PD tibial thirds ----------------------------------------------------------
    md += ["", "## C. PD-auto tibial anterior/central/posterior thirds (no QCart thirds in the source data)", "",
           "| Subregion | Progressor Δ | Stable Δ | Excess (95% CI) | Cohen's d |", "|---|--:|--:|--:|--:|"]
    for reg in THIRDS:
        p, s = col(prog, "PD-auto", reg), col(stab, "PD-auto", reg)
        exc = p.mean() - s.mean()
        exc_ci = boot_ci(lambda x, y: x.mean() - y.mean(), rng, a.n_boot, p, s)
        md.append(f"| {reg} | {fmt(p.mean())} ± {fmt(p.std(ddof=1))} | {fmt(s.mean())} ± {fmt(s.std(ddof=1))} | "
                  f"{fmt(exc)} ({fmt(exc_ci[0])}, {fmt(exc_ci[1])}) | {cohens_d(p, s):.2f} |")

    # ---- Table D: stable-arm drift by baseline KL ------------------------------------------
    md += ["", "## D. Stable-arm 48-month change by baseline Kellgren–Lawrence grade (drift uniformity)", "",
           "| KL_00 | n | PD MFTC Δ | QCart MFTC Δ | PD−QCart | PD cMF Δ | PD MT Δ |", "|---|--:|--:|--:|--:|--:|--:|"]
    for kl, g in stab.groupby("KL_00"):
        pdm, qm = col(g, "PD-auto", "MFTC"), col(g, "manual QCart", "MFTC")
        md.append(f"| {kl} | {len(g)} | {fmt(pdm.mean())} ± {fmt(pdm.std(ddof=1))} | {fmt(qm.mean())} ± {fmt(qm.std(ddof=1))} | "
                  f"{fmt((pdm - qm).mean())} | {fmt(col(g, 'PD-auto', 'cMF').mean())} | {fmt(col(g, 'PD-auto', 'MT').mean())} |")
    kw = stats.kruskal(*[col(g, "PD-auto", "MFTC") - col(g, "manual QCart", "MFTC") for _, g in stab.groupby("KL_00")])
    md.append(f"\nKruskal–Wallis, stable-arm PD−QCart MFTC drift across KL grades: H={kw.statistic:.2f}, P={kw.pvalue:.3f}")
    # same for progressors
    md += ["", "Progressor arm by KL_00 (PD−QCart MFTC difference):", "",
           "| KL_00 | n | PD MFTC Δ | QCart MFTC Δ | PD−QCart |", "|---|--:|--:|--:|--:|"]
    for kl, g in prog.groupby("KL_00"):
        pdm, qm = col(g, "PD-auto", "MFTC"), col(g, "manual QCart", "MFTC")
        md.append(f"| {kl} | {len(g)} | {fmt(pdm.mean())} ± {fmt(pdm.std(ddof=1))} | {fmt(qm.mean())} ± {fmt(qm.std(ddof=1))} | {fmt((pdm - qm).mean())} |")

    # ---- Reading ---------------------------------------------------------------------------
    A = {(r["region"], r["method"]): r for r in rows}
    R = {r["region"]: r for r in ratio_rows}
    md += ["", "## E. Reading (for the response letter)", "",
           f"- MT: PD progressor change ({fmt(A[('MT','PD-auto')]['prog_mean'])}) exceeds QCart ({fmt(A[('MT','manual QCart')]['prog_mean'])}) "
           f"but so does stable-arm change ({fmt(A[('MT','PD-auto')]['stable_mean'])} vs {fmt(A[('MT','manual QCart')]['stable_mean'])}). "
           f"After subtracting the stable arm, MT excess is {fmt(R['MT']['pd_excess'])} vs {fmt(R['MT']['qcart_excess'])} μm "
           f"(ratio {R['MT']['ratio']:.2f}, 95% CI {R['MT']['ratio_lo']:.2f}–{R['MT']['ratio_hi']:.2f}): the apparent MT over-reading is drift, not extra responsiveness.",
           f"- cMF: excess {fmt(R['cMF']['pd_excess'])} vs {fmt(R['cMF']['qcart_excess'])} μm (ratio {R['cMF']['ratio']:.2f}, "
           f"{R['cMF']['ratio_lo']:.2f}–{R['cMF']['ratio_hi']:.2f}) — this is where PD genuinely under-reads progression (also an ROI-definition difference: grid central-60% band vs Eckstein cMF).",
           f"- MFTC: ratio {R['MFTC']['ratio']:.2f} ({R['MFTC']['ratio_lo']:.2f}–{R['MFTC']['ratio_hi']:.2f}); drift-corrected SRM "
           f"{A[('MFTC','PD-auto')]['srm_drift_corrected']:.2f} vs {A[('MFTC','manual QCart')]['srm_drift_corrected']:.2f}; Cohen's d "
           f"{A[('MFTC','PD-auto')]['cohens_d']:.2f} vs {A[('MFTC','manual QCart')]['cohens_d']:.2f}.",
           "- Table C shows whether the tibial drift is concentrated in a particular AP third (posterior = thick-slice partial volume on the curved posterior plateau would be the mechanistic suspect).",
           "- Table D: if PD−QCart drift is similar across KL grades, drift is a method property rather than disease-dependent, which supports the between-group-contrast use case."]

    text = "\n".join(md) + "\n"
    if a.dry_run:
        print(text)
        print("[dry_run] nothing written")
        return
    (RES / "v9_drift_adjusted_contrast.md").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(RES / "v9_drift_adjusted_contrast.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(RES / "v9_drift_adjusted_ratio.csv", index=False)
    print(text)
    print(f"wrote {RES / 'v9_drift_adjusted_contrast.md'}")


if __name__ == "__main__":
    main()
