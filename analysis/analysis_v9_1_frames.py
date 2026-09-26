"""v9.1 — All longitudinal endpoints for a given analysis frame, from the per-knee delta CSVs.

Frames
  leakfree : QC-pass knees, one knee per participant, EXCLUDING every participant whose
             DESS label volume was in the depth-densification training pool (PRIMARY in v9.1)
  full     : all QC-pass knees, one knee per participant (150 / 151; the v9 primary, now sensitivity)

For each frame this script writes ../results/v9_1_endpoints_<frame>.json and a markdown
digest with: cohort sizes; Table 2 (responsiveness: Δ, SRM with BCa 95% CI; agreement:
r, CCC, ICC(A,1), Bland–Altman bias / LoA); Table 3 (stable Δ, AUC with bootstrap 95% CI,
paired AUC difference, thresholds and sensitivity at 95% / 90% specificity); exploratory
lateral AUCs; drift-adjusted contrast (excess, ratio, drift-corrected SRM, Cohen's d,
tibial thirds, drift by KL); sample size per arm (drift-aware); participants shared
between arms and the subject-disjoint + participant-clustered check.

Also writes the frame knee lists ../results/v9_1_<prog|nonprog>_<frame>.csv.

Usage: python analysis_v9_1_frames.py [--frame leakfree|full|both] [--n_boot 2000] [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, norm

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
POOL = Path(r"E:/KneeMR/Datasets/segmentation_canonical/legacy/DESS_inhouse_110")
REG_MED = ["MFTC", "cMF", "MT"]
REG_LAT = ["LFTC", "cLF", "LT"]
THIRDS = ["aMT", "cMT", "pMT"]
UM = 1000.0


# ------------------------------------------------------------------ frames
def pool_pids() -> set[int]:
    return {int(re.search(r"(9\d{6})", f).group(1)) for f in glob.glob(str(POOL / "*/segmentations_filtered/*.nii.gz"))}


SOURCE = "v9_1"


def load_frame(frame: str, n_run=None):
    pre = "v9_" if SOURCE == "v9_1" else "v9_2_"
    p = pd.read_csv(RES / f"{pre}prog_qcpass.csv")
    s = pd.read_csv(RES / f"{pre}nonprog_qcpass.csv")
    if frame == "leakfree":
        pool = pool_pids()
        p, s = p[~p.pid.isin(pool)].reset_index(drop=True), s[~s.pid.isin(pool)].reset_index(drop=True)
    elif frame != "full":
        raise SystemExit(frame)
    if n_run:
        p, s = p.head(n_run), s.head(n_run)
    return p, s


# ------------------------------------------------------------------ metrics
def srm(x): return x.mean() / x.std(ddof=1)


def bca_ci(x, fn, n_boot, rng):
    r = stats.bootstrap((x,), fn, n_resamples=n_boot, method="BCa", random_state=rng, vectorized=False)
    return [float(r.confidence_interval.low), float(r.confidence_interval.high)]


def ccc(a, b):
    ma, mb, va, vb = a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)
    cov = np.cov(a, b, ddof=1)[0, 1]
    return 2 * cov / (va + vb + (ma - mb) ** 2)


def icc_a1(a, b):
    """Two-way mixed, absolute agreement, single measures: ICC(A,1) (McGraw & Wong)."""
    Y = np.stack([a, b], axis=1); n, k = Y.shape
    gm = Y.mean(); rm = Y.mean(1); cm = Y.mean(0)
    ssr = k * ((rm - gm) ** 2).sum(); ssc = n * ((cm - gm) ** 2).sum()
    sse = ((Y - rm[:, None] - cm[None, :] + gm) ** 2).sum()
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def auc_mw(prog, stab):
    return mannwhitneyu(-prog, -stab, alternative="two-sided").statistic / (len(prog) * len(stab))


def thr_sens(prog, stab, spec):
    thr = np.quantile(stab, 1 - spec)
    return float(thr), float((prog <= thr).mean())


def cohens_d(p, s):
    sp = np.sqrt(((len(p) - 1) * p.var(ddof=1) + (len(s) - 1) * s.var(ddof=1)) / (len(p) + len(s) - 2))
    return (p.mean() - s.mean()) / sp


def n_per_arm(sigma, delta, power, alpha=0.05):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * (z * sigma / delta) ** 2))


# ------------------------------------------------------------------ main computation
def compute(p: pd.DataFrame, s: pd.DataFrame, n_boot: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out = {"n_prog": int(len(p)), "n_stab": int(len(s)), "n_boot": n_boot, "seed": seed}
    P = {r: p[f"pd_{r}_d"].to_numpy() * UM for r in REG_MED + REG_LAT + THIRDS}
    S = {r: s[f"pd_{r}_d"].to_numpy() * UM for r in REG_MED + REG_LAT + THIRDS}
    Q = {r: p[f"eck_{r}_d"].to_numpy() * UM for r in REG_MED + REG_LAT}
    QS = {r: s[f"eck_{r}_d"].to_numpy() * UM for r in REG_MED + REG_LAT}

    # Table 2
    t2 = {}
    for r in REG_MED:
        x, q = P[r], Q[r]
        d = x - q
        t2[r] = dict(pd_mean=x.mean(), pd_sd=x.std(ddof=1), pd_srm=srm(x), pd_srm_ci=bca_ci(x, srm, n_boot, rng),
                     q_mean=q.mean(), q_sd=q.std(ddof=1), q_srm=srm(q),
                     r=float(np.corrcoef(x, q)[0, 1]), ccc=float(ccc(x, q)), icc=float(icc_a1(x, q)),
                     ba_bias=d.mean(), ba_lo=d.mean() - 1.96 * d.std(ddof=1), ba_hi=d.mean() + 1.96 * d.std(ddof=1))
    out["table2"] = t2

    # Table 3 + lateral
    t3 = {}
    idxP, idxS = np.arange(len(p)), np.arange(len(s))
    for r in REG_MED + REG_LAT:
        rec = {}
        for tag, (pp, ss) in (("pd", (P[r], S[r])), ("q", (Q[r], QS[r]))):
            a = auc_mw(pp, ss)
            boots = np.array([auc_mw(pp[rng.integers(0, len(pp), len(pp))], ss[rng.integers(0, len(ss), len(ss))]) for _ in range(n_boot)])
            thr95, se95 = thr_sens(pp, ss, 0.95); thr90, se90 = thr_sens(pp, ss, 0.90)
            rec[tag] = dict(stab_mean=ss.mean(), stab_sd=ss.std(ddof=1), prog_mean=pp.mean(), prog_sd=pp.std(ddof=1),
                            auc=a, auc_ci=list(np.percentile(boots, [2.5, 97.5])), thr95=thr95, sens95=se95, thr90=thr90, sens90=se90)
        diffs = []
        for _ in range(n_boot):
            ip, is_ = rng.integers(0, len(p), len(p)), rng.integers(0, len(s), len(s))
            diffs.append(auc_mw(P[r][ip], S[r][is_]) - auc_mw(Q[r][ip], QS[r][is_]))
        rec["dauc"] = rec["pd"]["auc"] - rec["q"]["auc"]; rec["dauc_ci"] = list(np.percentile(diffs, [2.5, 97.5]))
        t3[r] = rec
    out["table3"] = t3

    # drift-adjusted contrast
    da = {}
    for r in REG_MED + REG_LAT:
        exP, exQ = P[r].mean() - S[r].mean(), Q[r].mean() - QS[r].mean()
        bP = np.array([P[r][rng.integers(0, len(p), len(p))].mean() - S[r][rng.integers(0, len(s), len(s))].mean() for _ in range(n_boot)])
        bQ = np.array([Q[r][rng.integers(0, len(p), len(p))].mean() - QS[r][rng.integers(0, len(s), len(s))].mean() for _ in range(n_boot)])
        ratios = []
        for _ in range(n_boot):
            ip, is_ = rng.integers(0, len(p), len(p)), rng.integers(0, len(s), len(s))
            eq = Q[r][ip].mean() - QS[r][is_].mean()
            ratios.append((P[r][ip].mean() - S[r][is_].mean()) / eq if eq else np.nan)
        dS = S[r] - QS[r]; ciS = stats.t.interval(0.95, len(dS) - 1, loc=dS.mean(), scale=stats.sem(dS))
        dP = P[r] - Q[r]; ciP = stats.t.interval(0.95, len(dP) - 1, loc=dP.mean(), scale=stats.sem(dP))
        dd = np.array([cohens_d(P[r][rng.integers(0, len(p), len(p))], S[r][rng.integers(0, len(s), len(s))]) for _ in range(n_boot)])
        dq = np.array([cohens_d(Q[r][rng.integers(0, len(p), len(p))], QS[r][rng.integers(0, len(s), len(s))]) for _ in range(n_boot)])
        da[r] = dict(ex_pd=exP, ex_pd_ci=list(np.percentile(bP, [2.5, 97.5])), ex_q=exQ, ex_q_ci=list(np.percentile(bQ, [2.5, 97.5])),
                     ratio=exP / exQ if exQ else np.nan, ratio_ci=list(np.nanpercentile(ratios, [2.5, 97.5])),
                     srm_dc_pd=exP / P[r].std(ddof=1), srm_dc_q=exQ / Q[r].std(ddof=1),
                     d_pd=cohens_d(P[r], S[r]), d_pd_ci=list(np.percentile(dd, [2.5, 97.5])), d_q=cohens_d(Q[r], QS[r]), d_q_ci=list(np.percentile(dq, [2.5, 97.5])),
                     stab_drift=dS.mean(), stab_drift_ci=list(ciS), prog_diff=dP.mean(), prog_diff_ci=list(ciP))
    out["drift"] = da
    out["thirds"] = {r: dict(prog=P[r].mean(), prog_sd=P[r].std(ddof=1), stab=S[r].mean(), stab_sd=S[r].std(ddof=1), excess=P[r].mean() - S[r].mean()) for r in THIRDS}
    kl = {}
    for g, sub in s.groupby("KL_00"):
        kl[int(g)] = dict(n=int(len(sub)), pd=float(sub.pd_MFTC_d.mean() * UM), q=float(sub.eck_MFTC_d.mean() * UM), diff=float((sub.pd_MFTC_d - sub.eck_MFTC_d).mean() * UM))
    out["drift_by_kl"] = kl
    out["drift_by_kl_p"] = float(stats.kruskal(*[(sub.pd_MFTC_d - sub.eck_MFTC_d).to_numpy() for _, sub in s.groupby("KL_00")]).pvalue)

    # sample size (drift-aware)
    ssz = {}
    for r in REG_MED:
        for slow in (0.3, 0.5):
            for pw in (0.8, 0.9):
                nP_ = n_per_arm(P[r].std(ddof=1), slow * abs(da[r]["ex_pd"]), pw)
                nQ_ = n_per_arm(Q[r].std(ddof=1), slow * abs(da[r]["ex_q"]), pw)
                ssz[f"{r}_{int(slow*100)}_{int(pw*100)}"] = dict(pd=nP_, q=nQ_, ratio=nP_ / nQ_)
    out["sample_size"] = ssz

    # shared participants
    shared = sorted(set(p.pid) & set(s.pid))
    out["shared_pids"] = [int(x) for x in shared]
    if shared:
        p2, s2 = p[~p.pid.isin(shared)], s[~s.pid.isin(shared)]
        x = {r: dict(auc_pd=auc_mw(p2[f"pd_{r}_d"].to_numpy(), s2[f"pd_{r}_d"].to_numpy()),
                     auc_q=auc_mw(p2[f"eck_{r}_d"].to_numpy(), s2[f"eck_{r}_d"].to_numpy()),
                     ratio=(p2[f"pd_{r}_d"].mean() - s2[f"pd_{r}_d"].mean()) / (p2[f"eck_{r}_d"].mean() - s2[f"eck_{r}_d"].mean())) for r in REG_MED}
        out["subject_disjoint"] = dict(n_prog=int(len(p2)), n_stab=int(len(s2)), **x)
        # participant-clustered bootstrap for MFTC ΔAUC and ratio
        pids = np.array(sorted(set(p.pid) | set(s.pid))); pb = {k: g for k, g in p.groupby("pid")}; sb = {k: g for k, g in s.groupby("pid")}
        cl = {r: {"dauc": [], "ratio": []} for r in REG_MED}
        for _ in range(n_boot):
            draw = rng.choice(pids, len(pids), replace=True)
            pp = pd.concat([pb[k] for k in draw if k in pb]); ss = pd.concat([sb[k] for k in draw if k in sb])
            for r in REG_MED:
                cl[r]["dauc"].append(auc_mw(pp[f"pd_{r}_d"].to_numpy(), ss[f"pd_{r}_d"].to_numpy()) - auc_mw(pp[f"eck_{r}_d"].to_numpy(), ss[f"eck_{r}_d"].to_numpy()))
                eq = pp[f"eck_{r}_d"].mean() - ss[f"eck_{r}_d"].mean()
                cl[r]["ratio"].append((pp[f"pd_{r}_d"].mean() - ss[f"pd_{r}_d"].mean()) / eq)
        out["clustered"] = {r: dict(dauc_ci=list(np.percentile(cl[r]["dauc"], [2.5, 97.5])), ratio_ci=list(np.nanpercentile(cl[r]["ratio"], [2.5, 97.5]))) for r in REG_MED}
    return out


# ------------------------------------------------------------------ markdown digest
def f0(x): return f"{x:.0f}"


def digest(E: dict, frame: str) -> str:
    L = [f"# v9.1 endpoints — frame `{frame}` ({E['n_prog']} progressors / {E['n_stab']} stable; {E['n_boot']} bootstrap, seed {E['seed']})", ""]
    L += ["## Table 2A responsiveness", "", "| Region | PD Δ (μm) | PD SRM (95% CI) | Manual DESS Δ (μm) | Manual DESS SRM |", "|---|--:|--:|--:|--:|"]
    for r, t in E["table2"].items():
        L.append(f"| {r} | {f0(t['pd_mean'])}±{f0(t['pd_sd'])} | {t['pd_srm']:.2f} ({t['pd_srm_ci'][0]:.2f}, {t['pd_srm_ci'][1]:.2f}) | {f0(t['q_mean'])}±{f0(t['q_sd'])} | {t['q_srm']:.2f} |")
    L += ["", "## Table 2B agreement", "", "| Region | r | CCC | ICC | BA bias (μm) | 95% LOA (μm) |", "|---|--:|--:|--:|--:|--:|"]
    for r, t in E["table2"].items():
        L.append(f"| {r} | {t['r']:.2f} | {t['ccc']:.2f} | {t['icc']:.2f} | {t['ba_bias']:+.0f} | {f0(t['ba_lo'])}, {f0(t['ba_hi'])} |")
    L += ["", "## Table 3 discrimination", "", "| Region | Method | Stable Δ | Progressor Δ | AUC (95% CI) | Thr@95% spec | Sens 95% / 90% |", "|---|---|--:|--:|--:|--:|--:|"]
    for r, t in E["table3"].items():
        for tag, name in (("pd", "PD"), ("q", "Manual DESS")):
            x = t[tag]
            L.append(f"| {r} | {name} | {f0(x['stab_mean'])}±{f0(x['stab_sd'])} | {f0(x['prog_mean'])}±{f0(x['prog_sd'])} | {x['auc']:.2f} ({x['auc_ci'][0]:.2f}, {x['auc_ci'][1]:.2f}) | {f0(x['thr95'])} | {x['sens95']*100:.0f}% / {x['sens90']*100:.0f}% |")
    L.append("\nPaired ΔAUC: " + "; ".join(f"{r} {t['dauc']:.2f} ({t['dauc_ci'][0]:.2f}, {t['dauc_ci'][1]:.2f})" for r, t in E["table3"].items()))
    L += ["", "## Drift-adjusted contrast", "", "| Region | PD excess (95% CI) | Manual excess (95% CI) | Ratio (95% CI) | SRM_dc PD / manual | d PD (CI) | d manual (CI) | Stable drift PD−manual (CI) |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r, t in E["drift"].items():
        L.append(f"| {r} | {f0(t['ex_pd'])} ({f0(t['ex_pd_ci'][0])}, {f0(t['ex_pd_ci'][1])}) | {f0(t['ex_q'])} ({f0(t['ex_q_ci'][0])}, {f0(t['ex_q_ci'][1])}) | {t['ratio']:.2f} ({t['ratio_ci'][0]:.2f}, {t['ratio_ci'][1]:.2f}) | {t['srm_dc_pd']:.2f} / {t['srm_dc_q']:.2f} | {t['d_pd']:.2f} ({t['d_pd_ci'][0]:.2f}, {t['d_pd_ci'][1]:.2f}) | {t['d_q']:.2f} ({t['d_q_ci'][0]:.2f}, {t['d_q_ci'][1]:.2f}) | {f0(t['stab_drift'])} ({f0(t['stab_drift_ci'][0])}, {f0(t['stab_drift_ci'][1])}) |")
    L += ["", "Tibial thirds (PD): " + "; ".join(f"{r} prog {f0(t['prog'])} / stable {f0(t['stab'])} / excess {f0(t['excess'])}" for r, t in E["thirds"].items()),
          "Stable-arm PD−manual MFTC drift by KL: " + "; ".join(f"KL{k} n={v['n']} {f0(v['diff'])}" for k, v in E["drift_by_kl"].items()) + f" (Kruskal–Wallis P={E['drift_by_kl_p']:.2f})"]
    L += ["", "## Sample size per arm (drift-aware)", "", "| Region | Slowing | Power | PD | Manual DESS | Ratio |", "|---|--:|--:|--:|--:|--:|"]
    for k, v in E["sample_size"].items():
        r, sl, pw = k.split("_"); L.append(f"| {r} | {sl}% | {pw}% | {v['pd']} | {v['q']} | {v['ratio']:.2f} |")
    L += ["", f"## Participants in both arms: {len(E['shared_pids'])} {E['shared_pids']}"]
    if E.get("subject_disjoint"):
        sd = E["subject_disjoint"]; L.append(f"Subject-disjoint ({sd['n_prog']}/{sd['n_stab']}): " + "; ".join(f"{r} AUC {sd[r]['auc_pd']:.3f} vs {sd[r]['auc_q']:.3f}, ratio {sd[r]['ratio']:.2f}" for r in REG_MED))
        L.append("Clustered bootstrap: " + "; ".join(f"{r} ΔAUC CI ({E['clustered'][r]['dauc_ci'][0]:.3f}, {E['clustered'][r]['dauc_ci'][1]:.3f}), ratio CI ({E['clustered'][r]['ratio_ci'][0]:.2f}, {E['clustered'][r]['ratio_ci'][1]:.2f})" for r in REG_MED))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="both", choices=["leakfree", "full", "both"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--source", default="v9_1", choices=["v9_1", "v9_2"], help="v9_1 = legacy zero-imputation deltas; v9_2 = symmetric follow-up handling")
    a = ap.parse_args()
    global SOURCE
    SOURCE = a.source
    for frame in (["leakfree", "full"] if a.frame == "both" else [a.frame]):
        p, s = load_frame(frame, a.n_run)
        E = compute(p, s, a.n_boot)
        text = digest(E, frame)
        print(text)
        if not a.dry_run:
            (RES / f"{SOURCE}_endpoints_{frame}.json").write_text(json.dumps(E, indent=1, default=float), encoding="utf-8")
            (RES / f"{SOURCE}_endpoints_{frame}.md").write_text(text, encoding="utf-8")
            p.to_csv(RES / f"{SOURCE}_prog_{frame}.csv", index=False); s.to_csv(RES / f"{SOURCE}_nonprog_{frame}.csv", index=False)
            print("wrote", RES / f"{SOURCE}_endpoints_{frame}.json")


if __name__ == "__main__":
    main()
