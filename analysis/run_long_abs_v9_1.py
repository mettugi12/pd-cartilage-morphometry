"""v9.1 — Longitudinal morphometry rerun capturing ABSOLUTE regional thickness at 00m and 48m.

Identical pipeline to run_prog_icp_v9.py / run_nonprog_v8b.py (cartilage-morphometry
baseline_grid + raycast_2d, RECON disk cache), run on BOTH arms, but keeps the per-visit
regional means that the earlier runs discarded, and joins the expert manual DESS
(Chondrometrics, OAI kMRI_QCart_Eckstein) ABSOLUTE thickness at V00 and V06 so that a
leak-free, regional-level cross-sectional validation against the expert reference is
possible (PD-auto absolute vs manual DESS absolute, both visits).

Region mapping to the OAI QCart columns (as in v3.3 analysis_eckstein_style.py):
  cMF <- V{vv}BMFMTH   cLF <- V{vv}BLFMTH   MT <- V{vv}WMTMTH   LT <- V{vv}WLTMTH
  MFTC = cMF + MT ; LFTC = cLF + LT

Output: ../results/v9_1_long_abs.csv  (RESUMABLE; one row per knee)
  pid, side, cohort, KL_00,
  pd_<reg>_00m, pd_<reg>_48m, pd_<reg>_d          for reg in REGIONS + MFTC, LFTC
  eck_<reg>_00m, eck_<reg>_48m, eck_<reg>_d       for reg in cMF, cLF, MT, LT, MFTC, LFTC

Usage: python run_long_abs_v9_1.py [--n_run N] [--dry_run] [--bones femur,tibia]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_CSV = RESULTS / "v9_1_long_abs.csv"
OAI = Path(r"E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII")
QCART_FILES = {"00": OAI / "kMRI_QCart_Eckstein00.txt", "48": OAI / "kmri_qcart_eckstein06.txt"}
QCOL = {"cMF": "BMFMTH", "cLF": "BLFMTH", "MT": "WMTMTH", "LT": "WLTMTH"}
VIS = {"00": "V00", "48": "V06"}
REGIONS = ["cMF", "cLF", "MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT"]
ECK_REGIONS = ["cMF", "cLF", "MT", "LT", "MFTC", "LFTC"]


def load_qcart_abs():
    """(pid, 'LEFT'|'RIGHT') -> {eck_<reg>_00m/48m/d}.

    The OAI QCart files hold several reading-project rows per knee; as in the v3.3 loader
    (analysis_eckstein_style.load_eckstein_qcart) values are AVERAGED across rows per
    (ID, SIDE) before use, so that the expert values match every previously published number.
    """
    tabs = {}
    for k, f in QCART_FILES.items():
        d = pd.read_csv(f, sep="|", low_memory=False)
        d["pid"] = d["ID"].astype(int)
        d["side"] = d["SIDE"].astype(str).str.contains("1").map({True: "RIGHT", False: "LEFT"})
        cols = {reg: f"{VIS[k]}{col}" for reg, col in QCOL.items() if f"{VIS[k]}{col}" in d.columns}
        for c in cols.values():
            d[c] = pd.to_numeric(d[c], errors="coerce")
        tabs[k] = d.groupby(["pid", "side"])[list(cols.values())].mean()
    keys = set(tabs["00"].index) | set(tabs["48"].index)
    out = {}
    for key in keys:
        rec = {}
        for reg, col in QCOL.items():
            for k in ("00", "48"):
                cname = f"{VIS[k]}{col}"
                v = tabs[k][cname].get(key, np.nan) if cname in tabs[k].columns else np.nan
                rec[f"eck_{reg}_{k}m"] = float(v) if pd.notna(v) else np.nan
        for comp, (a, b) in (("MFTC", ("cMF", "MT")), ("LFTC", ("cLF", "LT"))):
            for k in ("00", "48"):
                rec[f"eck_{comp}_{k}m"] = rec[f"eck_{a}_{k}m"] + rec[f"eck_{b}_{k}m"]
        for reg in ECK_REGIONS:
            rec[f"eck_{reg}_d"] = rec[f"eck_{reg}_48m"] - rec[f"eck_{reg}_00m"]
        out[(str(key[0]), key[1])] = rec
    return out


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["pid"], r["side"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def _write(path, fieldnames, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--bones", default="femur,tibia")
    ap.add_argument("--mode", default="zero_impute", choices=["zero_impute", "symmetric"],
                    help="follow-up handling: legacy zero-imputation (v9.1) or symmetric failure exclusion (v9.2)")
    ap.add_argument("--pids_file", default=None, help="optional CSV with pid,side to restrict the run")
    ap.add_argument("--out", default=None, help="output CSV (default v9_1_long_abs.csv or v9_2_long_abs.csv by mode)")
    ap.add_argument("--idw_radius", type=float, default=2.0, help="symmetric mode: radius (mm) for valid-neighbour IDW; beyond it the vertex is tested for denudation")
    ap.add_argument("--denuded_dist", type=float, default=1.5, help="symmetric mode: no follow-up cartilage voxel within this distance (mm) => denuded (0)")
    args = ap.parse_args()
    global OUT_CSV
    OUT_CSV = Path(args.out) if args.out else (RESULTS / ("v9_2_long_abs.csv" if args.mode == "symmetric" else "v9_1_long_abs.csv"))

    import sys
    REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from cartilage_morphometry import PipelineConfig
    from cartilage_morphometry import pipeline as _pipeline
    from cartilage_morphometry.validation.cohorts import load_v33_cohort, DEFAULT_RECON_CACHE_DIR
    from cartilage_morphometry.validation.shared_mesh import process_long_baseline_grid
    if DEFAULT_RECON_CACHE_DIR.exists():
        from cartilage_morphometry.validation.api import _patch_disk_cache_path_unique
        _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
        _patch_disk_cache_path_unique()
        print(f"[recon] disk cache (path-unique): {DEFAULT_RECON_CACHE_DIR}")

    cfg = PipelineConfig()
    cfg.region_projection = "baseline_grid"
    cfg.femur_unwrap = "per_slice"
    cfg.long_followup_handling = args.mode
    cfg.long_idw_radius_mm = args.idw_radius
    cfg.long_denuded_dist_mm = args.denuded_dist
    bones = [b.strip() for b in args.bones.split(",") if b.strip()]

    qcart = load_qcart_abs()
    cases = list(load_v33_cohort(progressor_only=False))
    if args.pids_file:
        keep = {(str(r.pid), str(r.side)) for r in pd.read_csv(args.pids_file).itertuples()}
        cases = [c for c in cases if (str(c.pid), str(c.side)) in keep]
    print(f"[cohort] {len(cases)} knees with both-timepoint segs "
          f"({sum(c.cohort == 'progressor' for c in cases)} progressor / {sum(c.cohort != 'progressor' for c in cases)} stable)")
    if args.n_run is not None:
        cases = cases[: args.n_run]
    done = load_done(OUT_CSV) if not args.dry_run else set()
    fieldnames = (["pid", "side", "cohort", "KL_00"]
                  + [f"pd_{r}_{k}" for r in REGIONS + ["MFTC", "LFTC"] for k in ("00m", "48m", "d")]
                  + [f"eck_{r}_{k}" for r in ECK_REGIONS for k in ("00m", "48m", "d")]
                  + [f"{b}_{k}" for b in ("femur", "tibia") for k in ("frac_measured", "frac_denuded", "frac_failed", "frac_failed_cells", "frac_denuded_cells")])
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8"))) if (OUT_CSV.exists() and not args.dry_run) else []

    t0 = time.time(); n_new = 0
    for i, c in enumerate(cases, 1):
        if (str(c.pid), str(c.side)) in done:
            continue
        row = {"pid": c.pid, "side": c.side, "cohort": c.cohort, "KL_00": c.KL_00}
        try:
            per = {}
            for bone in bones:
                r = process_long_baseline_grid(c.seg_00m_path, c.seg_48m_path, c.side, bone, cfg)
                per.update(r["baseline_grid_regions"])
                for k_, v_ in r.get("symmetric_info", {}).items():
                    row[f"{bone}_{k_}"] = v_
            for reg in REGIONS:
                for k in ("00m", "48m", "d"):
                    row[f"pd_{reg}_{k}"] = per.get(reg, {}).get(k, np.nan)
            for comp, (a, b) in (("MFTC", ("cMF", "MT")), ("LFTC", ("cLF", "LT"))):
                for k in ("00m", "48m", "d"):
                    va, vb = per.get(a, {}).get(k, np.nan), per.get(b, {}).get(k, np.nan)
                    row[f"pd_{comp}_{k}"] = (va + vb) if np.isfinite(va) and np.isfinite(vb) else np.nan
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {c.pid}_{c.side}: {type(e).__name__}: {e}")
            continue
        finally:
            _pipeline.clear_recon_cache()
        row.update(qcart.get((str(c.pid), str(c.side)), {}))
        rows.append(row); n_new += 1
        rate = n_new / max(1e-6, time.time() - t0)
        print(f"  [{i}/{len(cases)}] {c.pid}_{c.side} {c.cohort[:4]} MFTC 00m={row.get('pd_MFTC_00m', float('nan')):.2f} "
              f"48m={row.get('pd_MFTC_48m', float('nan')):.2f} Δ={row.get('pd_MFTC_d', float('nan')):+.3f} | "
              f"manual 00m={row.get('eck_MFTC_00m', float('nan')):.2f} ({rate*60:.1f}/min)", flush=True)
        if args.dry_run and n_new >= 2:
            print("[dry_run] sample rows:", rows[-2:])
            break
        if not args.dry_run and n_new % 5 == 0:
            _write(OUT_CSV, fieldnames, rows)
    if not args.dry_run:
        _write(OUT_CSV, fieldnames, rows)
        print(f"[ok] wrote {OUT_CSV} ({len(rows)} knees, {n_new} new) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
