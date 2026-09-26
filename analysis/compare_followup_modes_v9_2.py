"""v9.2 — Compare follow-up handling modes against the expert reference on the validation knees.

Inputs (same knees): legacy zero-imputation rows from ../results/v9_1_long_abs.csv and one or
more symmetric-mode outputs (v9_2_val60_r10.csv, v9_2_val60_r20.csv ...).
For each mode and arm: mean 48-month change (cMF, MT, MFTC) vs expert change; absolute
bias at 00m and 48m vs expert; per-knee change correlation with expert; and the
progressor-minus-stable excess ratio. Writes ../results/v9_2_mode_comparison.md.

Usage: python compare_followup_modes_v9_2.py [--dry_run] [--n_run N] [--files a.csv b.csv ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
REG = ["cMF", "MT", "MFTC"]


def load(path: Path, keys: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["pid"] = d.pid.astype(int)
    return d.merge(keys, on=["pid", "side"], how="inner")


def block(d: pd.DataFrame, label: str):
    rows = []
    for arm, g in d.groupby("cohort"):
        for reg in REG:
            pdd, ed = g[f"pd_{reg}_d"] * 1000, g[f"eck_{reg}_d"] * 1000
            b00 = (g[f"pd_{reg}_00m"] - g[f"eck_{reg}_00m"]).mean()
            b48 = (g[f"pd_{reg}_48m"] - g[f"eck_{reg}_48m"]).mean()
            rows.append(dict(mode=label, arm=arm, region=reg, n=len(g), pd_d=pdd.mean(), pd_sd=pdd.std(ddof=1), eck_d=ed.mean(),
                             r=np.corrcoef(pdd, ed)[0, 1], bias00=b00, bias48=b48))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=["v9_2_val60_r10.csv", "v9_2_val60_r20.csv"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    a = ap.parse_args()
    keys = pd.read_csv(RES / "v9_2_val60_pids.csv")
    keys["pid"] = keys.pid.astype(int)
    if a.n_run:
        keys = keys.head(a.n_run)
    modes = {"legacy zero-imputation (v9.1)": load(RES / "v9_1_long_abs.csv", keys)}
    for f in a.files:
        p = RES / f
        if p.exists():
            modes[f"symmetric {p.stem.split('_')[-1]}"] = load(p, keys)
    rows = []
    for label, d in modes.items():
        rows += block(d, label)
    df = pd.DataFrame(rows)
    md = [f"# v9.2 — Follow-up handling modes vs expert reference (validation knees n={len(keys)})", "",
          "Δ = 48 m − baseline (μm); bias = PD − expert absolute (mm). r = per-knee Δ correlation with expert.", ""]
    for reg in REG:
        md += [f"## {reg}", "", "| Mode | Arm | n | PD Δ ± SD | Expert Δ | Δ difference | r | Bias 00m | Bias 48m |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
        for _, r in df[df.region == reg].iterrows():
            md.append(f"| {r['mode']} | {r['arm']} | {r['n']} | {r['pd_d']:.0f} ± {r['pd_sd']:.0f} | {r['eck_d']:.0f} | {r['pd_d']-r['eck_d']:+.0f} | {r['r']:.2f} | {r['bias00']:+.2f} | {r['bias48']:+.2f} |")
        md.append("")
        md.append("Progressor − stable excess and ratio to expert:")
        for label in modes:
            sub = df[(df.region == reg) & (df["mode"] == label)].set_index("arm")
            if {"progressor", "nonprogressor"} <= set(sub.index):
                exP = sub.loc["progressor", "pd_d"] - sub.loc["nonprogressor", "pd_d"]
                exQ = sub.loc["progressor", "eck_d"] - sub.loc["nonprogressor", "eck_d"]
                md.append(f"- {label}: PD {exP:.0f} vs expert {exQ:.0f} μm → ratio {exP/exQ:.2f}; stable-arm drift (PD−expert Δ) {sub.loc['nonprogressor','pd_d']-sub.loc['nonprogressor','eck_d']:+.0f} μm")
        md.append("")
    text = "\n".join(md) + "\n"
    print(text)
    if not a.dry_run:
        (RES / "v9_2_mode_comparison.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
