"""[v9.3 copy of v9/analysis/make_v9_2_frames.py; paths -> v9.3/results, shared QC lists from v9/results]
v9.2 — Build the QC-pass knee tables from the symmetric-mode rerun.

Takes ../results/v9_2_long_abs.csv (run_long_abs_v9_1.py --mode symmetric) and restricts it
to the knees of the v9 QC-pass, one-knee-per-participant lists (v9_prog_qcpass.csv /
v9_nonprog_qcpass.csv), producing v9_2_prog_qcpass.csv / v9_2_nonprog_qcpass.csv with the
same column layout the downstream scripts expect (pd_<reg>_d ..., eck_<reg>_d, KL_00, cohort).
Also reports how many QC-pass knees are missing from the rerun.

Usage: python make_v9_2_frames.py [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
V9RES = HERE.parent.parent / "v9" / "results"   # shared QC-pass lists / flags (unchanged in v9.3)
REGIONS = ["cMF", "cLF", "MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT", "MFTC", "LFTC"]
ECK = ["cMF", "cLF", "MT", "LT", "MFTC", "LFTC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--source", default="v9_3")
    a = ap.parse_args()
    new = pd.read_csv(RES / f"{a.source}_long_abs.csv")
    new["pid"] = new.pid.astype(int)
    for arm, fn in (("prog", "v9_prog_qcpass.csv"), ("nonprog", "v9_nonprog_qcpass.csv")):
        old = pd.read_csv(V9RES / fn)
        if a.n_run:
            old = old.head(a.n_run)
        keys = old[["pid", "side", "cohort", "KL_00"]]
        m = keys.merge(new.drop(columns=["cohort", "KL_00"]), on=["pid", "side"], how="left")
        missing = m[m["pd_MFTC_d"].isna()]
        cols = ["pid", "side", "cohort", "KL_00"] + [f"pd_{r}_d" for r in REGIONS] + [f"eck_{r}_d" for r in ECK]
        out = m.dropna(subset=["pd_MFTC_d"])[cols]
        # consistency: expert deltas identical to the v9 file?
        chk = out.merge(old[["pid", "side", "eck_MFTC_d"]], on=["pid", "side"], suffixes=("", "_v9"))
        maxdiff = float((chk.eck_MFTC_d - chk.eck_MFTC_d_v9).abs().max()) if len(chk) else float("nan")
        print(f"{arm}: {len(out)}/{len(old)} knees present in v9.3 table (missing {len(missing)}); max |expert Δ difference| vs v9 all-project average = {maxdiff:.4f} mm (expected: 75 %-paired now)")
        if not a.dry_run:
            out.to_csv(RES / f"{a.source}_{arm}_qcpass.csv", index=False)
            print("wrote", RES / f"{a.source}_{arm}_qcpass.csv")


if __name__ == "__main__":
    main()
