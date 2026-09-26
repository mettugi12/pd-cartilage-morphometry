"""v9.1 — Leak-free regional cross-sectional validation against the EXPERT reference.

Uses the absolute regional thickness at baseline (00m) and 48 months (48m) from
run_long_abs_v9_1.py (automated PD/IW, patient-specific baseline grid) and the expert
manual DESS thickness (Chondrometrics, OAI kMRI_QCart_Eckstein V00 / V06) for the same
knees. Nothing in this comparison was ever seen by the depth-densification network when
restricted to the leak-free knees (default), and the reference is independent of all
in-house labels.

For each region and visit: n, PD mean, manual mean, bias (PD − manual) ± SD, 95% LoA,
Pearson r, Lin CCC. Also pooled across visits and split by arm.
Figure: Bland–Altman panels (MFTC / cMF / MT × 00m / 48m).

Inputs : ../results/v9_1_long_abs.csv ; pool folder (for leak-free restriction)
Outputs: ../results/v9_1_cross_sectional_abs.{md,json}, ../manuscript/figures/Figure2_regional_abs_BA.png

Usage: python cross_sectional_abs_v9_1.py [--frame leakfree|all] [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "manuscript" / "figures"
POOL = Path(r"E:/KneeMR/Datasets/segmentation_canonical/legacy/DESS_inhouse_110")
REGIONS = ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]
VISITS = [("00m", "Baseline"), ("48m", "48 months")]


def pool_pids():
    return {int(re.search(r"(9\d{6})", f).group(1)) for f in glob.glob(str(POOL / "*/segmentations_filtered/*.nii.gz"))}


def ccc(a, b):
    return 2 * np.cov(a, b, ddof=1)[0, 1] / (a.var(ddof=1) + b.var(ddof=1) + (a.mean() - b.mean()) ** 2)


def stats_block(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = x - y
    return dict(n=int(m.sum()), pd_mean=x.mean(), pd_sd=x.std(ddof=1), q_mean=y.mean(), q_sd=y.std(ddof=1),
                bias=d.mean(), bias_sd=d.std(ddof=1), loa_lo=d.mean() - 1.96 * d.std(ddof=1), loa_hi=d.mean() + 1.96 * d.std(ddof=1),
                r=float(np.corrcoef(x, y)[0, 1]), ccc=float(ccc(x, y)), mae=float(np.abs(d).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="leakfree", choices=["leakfree", "all"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--source", default="v9_1", choices=["v9_1", "v9_2"])
    a = ap.parse_args()

    d = pd.read_csv(RES / f"{a.source}_long_abs.csv")
    d["pid"] = d.pid.astype(int)
    # keep only knees in the QC-pass one-knee-per-participant frames
    keep = pd.concat([pd.read_csv(RES / "v9_prog_qcpass.csv")[["pid", "side"]], pd.read_csv(RES / "v9_nonprog_qcpass.csv")[["pid", "side"]]])
    d = d.merge(keep.assign(_k=1), on=["pid", "side"], how="inner")
    if a.frame == "leakfree":
        d = d[~d.pid.isin(pool_pids())]
    if a.n_run:
        d = d.head(a.n_run)
    d = d.reset_index(drop=True)
    out = {"frame": a.frame, "n_knees": int(len(d)), "n_prog": int((d.cohort == "progressor").sum()), "n_stab": int((d.cohort != "progressor").sum())}

    md = [f"# v9.1 — Regional absolute thickness: automated PD/IW vs expert manual DESS (frame `{a.frame}`: {out['n_knees']} knees, "
          f"{out['n_prog']} progressor / {out['n_stab']} stable; both visits)", "",
          "Values mm. Bias = PD − manual DESS. Reference = OAI kMRI_QCart_Eckstein (V00, V06); automated = baseline-grid regional mean of the same knee/visit.", "",
          "| Region | Visit | n | PD mean ± SD | Manual mean ± SD | Bias ± SD | 95% LoA | r | CCC | MAE |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    out["by_visit"] = {}
    for reg in REGIONS:
        for v, lab in VISITS + [("both", "Pooled")]:
            if v == "both":
                x = np.concatenate([d[f"pd_{reg}_00m"].to_numpy(float), d[f"pd_{reg}_48m"].to_numpy(float)])
                y = np.concatenate([d[f"eck_{reg}_00m"].to_numpy(float), d[f"eck_{reg}_48m"].to_numpy(float)])
            else:
                x, y = d[f"pd_{reg}_{v}"].to_numpy(float), d[f"eck_{reg}_{v}"].to_numpy(float)
            s = stats_block(x, y); out["by_visit"][f"{reg}_{v}"] = s
            md.append(f"| {reg} | {lab} | {s['n']} | {s['pd_mean']:.2f} ± {s['pd_sd']:.2f} | {s['q_mean']:.2f} ± {s['q_sd']:.2f} | {s['bias']:+.2f} ± {s['bias_sd']:.2f} | "
                      f"{s['loa_lo']:+.2f}, {s['loa_hi']:+.2f} | {s['r']:.2f} | {s['ccc']:.2f} | {s['mae']:.2f} |")
    md += ["", "## By arm (MFTC, pooled visits)", "", "| Arm | n knees | Bias ± SD | r | CCC |", "|---|--:|--:|--:|--:|"]
    out["by_arm"] = {}
    for arm, sub in d.groupby("cohort"):
        x = np.concatenate([sub.pd_MFTC_00m.to_numpy(float), sub.pd_MFTC_48m.to_numpy(float)])
        y = np.concatenate([sub.eck_MFTC_00m.to_numpy(float), sub.eck_MFTC_48m.to_numpy(float)])
        s = stats_block(x, y); out["by_arm"][arm] = s
        md.append(f"| {arm} | {len(sub)} | {s['bias']:+.2f} ± {s['bias_sd']:.2f} | {s['r']:.2f} | {s['ccc']:.2f} |")
        for v in ("00m", "48m"):
            sv = stats_block(sub[f"pd_MFTC_{v}"].to_numpy(float), sub[f"eck_MFTC_{v}"].to_numpy(float)); out["by_arm"][f"{arm}_{v}"] = sv
            md.append(f"| {arm} {v} | {sv['n']} | {sv['bias']:+.2f} ± {sv['bias_sd']:.2f} | {sv['r']:.2f} | {sv['ccc']:.2f} |")
    # within-knee change agreement re-derived from the absolute values (consistency check with the delta CSVs)
    dx = d.pd_MFTC_48m - d.pd_MFTC_00m; dy = d.eck_MFTC_48m - d.eck_MFTC_00m
    m = np.isfinite(dx) & np.isfinite(dy)
    out["delta_check_r"] = float(np.corrcoef(dx[m], dy[m])[0, 1])
    md += ["", f"Consistency check: MFTC change (48m − 00m) recomputed from the absolute values, PD vs manual r = {out['delta_check_r']:.2f} "
               f"(all knees in this frame, both arms)."]
    text = "\n".join(md) + "\n"
    print(text)
    if a.dry_run:
        print("[dry_run] nothing written"); return
    (RES / f"{a.source}_cross_sectional_abs{'' if a.frame == 'leakfree' else '_all'}.md").write_text(text, encoding="utf-8")
    (RES / f"{a.source}_cross_sectional_abs{'' if a.frame == 'leakfree' else '_all'}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # ---- Figure: Bland–Altman, 2 rows (visits) x 3 cols (medial regions)
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.6), dpi=600)
    for i, (v, lab) in enumerate(VISITS):
        for j, reg in enumerate(["MFTC", "cMF", "MT"]):
            ax = axes[i, j]
            x, y = d[f"pd_{reg}_{v}"].to_numpy(float), d[f"eck_{reg}_{v}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
            mean, diff = (x + y) / 2, x - y
            s = out["by_visit"][f"{reg}_{v}"]
            col = np.where(d.loc[m, "cohort"].to_numpy() == "progressor", "#1f77b4", "#7f7f7f")
            ax.scatter(mean, diff, s=6, c=col, alpha=0.65, linewidths=0)
            ax.axhline(s["bias"], color="k", lw=0.9); ax.axhline(s["loa_lo"], color="k", lw=0.7, ls="--"); ax.axhline(s["loa_hi"], color="k", lw=0.7, ls="--")
            ax.axhline(0, color="#bbbbbb", lw=0.5)
            ax.set_title(f"{reg}, {lab.lower()} (n={s['n']})", fontsize=7.5)
            ax.text(0.02, 0.04, f"bias {s['bias']:+.2f} mm, r={s['r']:.2f}", transform=ax.transAxes, fontsize=6.5)
            ax.tick_params(labelsize=6.5)
            if i == 1: ax.set_xlabel("Mean of methods (mm)", fontsize=7)
            if j == 0: ax.set_ylabel("PD/IW − manual DESS (mm)", fontsize=7)
    for k, ax in enumerate(axes[0, :]):
        ax.text(-0.12, 1.08, "abc"[k], transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout(w_pad=0.5, h_pad=0.8)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "Figure2_regional_abs_BA.png", dpi=600); plt.close(fig)
    print("wrote", FIG / "Figure2_regional_abs_BA.png")


if __name__ == "__main__":
    main()
