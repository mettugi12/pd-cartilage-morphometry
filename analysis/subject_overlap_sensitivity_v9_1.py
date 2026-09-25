"""v9.1 sensitivity analysis: participants represented in both longitudinal arms.

Five OAI participants contribute one knee to the progressor arm and the contralateral
knee to the stable arm. Between-arm bootstrap CIs assume independent knees. Two checks:

  (A) subject-disjoint cohorts: drop the overlapping participants from BOTH arms and
      recompute the primary endpoints (MFTC / cMF / MT: AUC PD and manual DESS, paired
      AUC difference, progressor-minus-stable excess and its PD/reference ratio,
      sensitivity at 95% specificity).
  (B) subject-clustered bootstrap on the full primary frame: resample participants
      (clusters) rather than knees, keeping both knees of a resampled participant.

Inputs : ../results/v9_prog_qcpass.csv, ../results/v9_nonprog_qcpass.csv
Output : ../results/v9_1_subject_overlap_sensitivity.md

Usage: python subject_overlap_sensitivity_v9_1.py [--dry_run] [--n_run N] [--n_boot 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RES = Path(__file__).resolve().parent.parent / "results"
REGIONS = ["MFTC", "cMF", "MT"]
UM = 1000.0


def auc_mw(prog: np.ndarray, stab: np.ndarray) -> float:
    """AUC for 'more negative change = progression' (Mann–Whitney)."""
    u = mannwhitneyu(-prog, -stab, alternative="two-sided").statistic
    return u / (len(prog) * len(stab))


def sens_at_spec(prog, stab, spec=0.95):
    thr = np.quantile(stab, 1 - spec)  # more negative than the 5th percentile of stable
    return float((prog <= thr).mean()), float(thr)


def endpoints(p: pd.DataFrame, s: pd.DataFrame, reg: str):
    pp, ps = p[f"pd_{reg}_d"].to_numpy() * UM, s[f"pd_{reg}_d"].to_numpy() * UM
    qp, qs = p[f"eck_{reg}_d"].to_numpy() * UM, s[f"eck_{reg}_d"].to_numpy() * UM
    a_pd, a_q = auc_mw(pp, ps), auc_mw(qp, qs)
    ex_pd, ex_q = pp.mean() - ps.mean(), qp.mean() - qs.mean()
    return dict(auc_pd=a_pd, auc_q=a_q, dauc=a_pd - a_q, ex_pd=ex_pd, ex_q=ex_q, ratio=ex_pd / ex_q,
                sens_pd=sens_at_spec(pp, ps)[0], sens_q=sens_at_spec(qp, qs)[0])


def cluster_boot(p, s, reg, rng, n_boot):
    """Resample participants. A participant present in both arms is drawn once and
    contributes both knees; arms are then rebuilt from the drawn participants."""
    pids = np.array(sorted(set(p.pid) | set(s.pid)))
    p_by = {k: g for k, g in p.groupby("pid")}
    s_by = {k: g for k, g in s.groupby("pid")}
    out = {k: [] for k in ("dauc", "ratio", "ex_pd", "ex_q", "auc_pd", "auc_q")}
    for _ in range(n_boot):
        draw = rng.choice(pids, len(pids), replace=True)
        pb = pd.concat([p_by[k] for k in draw if k in p_by], ignore_index=True)
        sb = pd.concat([s_by[k] for k in draw if k in s_by], ignore_index=True)
        e = endpoints(pb, sb, reg)
        for k in out:
            out[k].append(e[k])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    p = pd.read_csv(RES / "v9_prog_qcpass.csv")
    s = pd.read_csv(RES / "v9_nonprog_qcpass.csv")
    if a.n_run:
        p, s = p.head(a.n_run), s.head(a.n_run)
    overlap = sorted(set(p.pid) & set(s.pid))
    p2, s2 = p[~p.pid.isin(overlap)], s[~s.pid.isin(overlap)]

    md = [f"# v9.1 — Subject-overlap sensitivity ({len(overlap)} participants in both arms: {overlap})", "",
          f"Primary frame {len(p)} progressors / {len(s)} stable; subject-disjoint frame {len(p2)} / {len(s2)}. "
          f"Cluster bootstrap: {a.n_boot} resamples of participants, seed {a.seed}. Values μm; AUC for more-negative-change = progression.", "",
          "## A. Primary frame versus subject-disjoint frame", "",
          "| Region | Frame | AUC PD | AUC manual DESS | ΔAUC | Excess PD | Excess manual DESS | Ratio | Sens@95% spec PD / manual DESS |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        for name, (pp, ss) in (("primary", (p, s)), ("subject-disjoint", (p2, s2))):
            e = endpoints(pp, ss, reg)
            md.append(f"| {reg} | {name} | {e['auc_pd']:.3f} | {e['auc_q']:.3f} | {e['dauc']:.3f} | {e['ex_pd']:.0f} | {e['ex_q']:.0f} | "
                      f"{e['ratio']:.2f} | {e['sens_pd']*100:.0f}% / {e['sens_q']*100:.0f}% |")
    md += ["", "## B. Subject-clustered bootstrap 95% CIs on the primary frame", "",
           "| Region | AUC PD | AUC manual DESS | ΔAUC (PD − manual DESS) | Excess PD | Excess manual DESS | Ratio |", "|---|--:|--:|--:|--:|--:|--:|"]
    for reg in REGIONS:
        e = endpoints(p, s, reg); ci = cluster_boot(p, s, reg, rng, a.n_boot)
        f = lambda k, nd=3: f"{e[k]:.{nd}f} ({ci[k][0]:.{nd}f}, {ci[k][1]:.{nd}f})"
        md.append(f"| {reg} | {f('auc_pd')} | {f('auc_q')} | {f('dauc')} | {f('ex_pd',0)} | {f('ex_q',0)} | {f('ratio',2)} |")
    md += ["", "Reading: if the subject-disjoint estimates differ from the primary frame by less than the bootstrap CI half-width and the "
           "clustered CIs match the knee-level CIs in the main text (MFTC ΔAUC −0.12, −0.17 to −0.08; ratio 0.58, 0.49–0.67), the five "
           "shared participants do not affect the conclusions."]
    text = "\n".join(md) + "\n"
    print(text)
    if not a.dry_run:
        (RES / "v9_1_subject_overlap_sensitivity.md").write_text(text, encoding="utf-8")
        print("wrote", RES / "v9_1_subject_overlap_sensitivity.md")


if __name__ == "__main__":
    main()
