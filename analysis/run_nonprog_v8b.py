"""v8b non-progressor arm — same baseline-grid pipeline as run_baseline_grid_v8b.py,
run on the JSW-stable (cohort == "nonprogressor") knees for the JMRI reframe's
specificity / discrimination analysis (see ../manuscript/jmri/JMRI_reframe_plan.md).

Only the 56 non-progressor knees with nnU-Net segs at both 00m and 48m are runnable;
load_v33_cohort(require_segs_present=True) handles the filtering. Manual QCart deltas
join from the same cached v3.3 merged-deltas file (eck_* columns, available for all).

Writes per-knee deltas to ../results/v8b_nonprog_deltas.csv (RESUMABLE).

CLI: --n_run N, --dry_run, --bones femur,tibia
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MERGED_V33 = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/v33_merged_deltas.csv")
OUT_CSV = RESULTS / "v8b_nonprog_deltas.csv"

REGIONS = ["cMF", "cLF", "MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT"]


def load_qcart():
    out = {}
    for r in csv.DictReader(open(MERGED_V33, encoding="utf-8")):
        key = (str(r["pid"]), str(r["side"]))
        d = {}
        for reg in ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC"):
            v = r.get(f"eck_{reg}_d", "")
            d[reg] = float(v) if v not in ("", "nan", None) else np.nan
        out[key] = (str(r.get("cohort", "")), d)
    return out


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["pid"], r["side"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--bones", default="femur,tibia")
    ap.add_argument("--femur_unwrap", choices=["per_slice", "best_fit_circle"], default="per_slice")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    out_csv = OUT_CSV

    import sys
    REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from cartilage_morphometry import PipelineConfig
    from cartilage_morphometry import pipeline as _pipeline
    from cartilage_morphometry.validation.cohorts import load_v33_cohort, DEFAULT_RECON_CACHE_DIR
    from cartilage_morphometry.validation.shared_mesh import process_long_baseline_grid

    # Same disk-cache setup as the progressor run; non-progressor knees are cache
    # misses on first pass (RECON inference runs), cached thereafter.
    if DEFAULT_RECON_CACHE_DIR.exists():
        from cartilage_morphometry.validation.api import _patch_disk_cache_path_unique
        _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
        _patch_disk_cache_path_unique()
        print(f"[recon] disk cache (path-unique): {DEFAULT_RECON_CACHE_DIR}")

    cfg = PipelineConfig()                 # raycast_2d default
    cfg.region_projection = "baseline_grid"
    cfg.femur_unwrap = args.femur_unwrap
    bones = [b.strip() for b in args.bones.split(",") if b.strip()]

    qcart = load_qcart()
    cases = [c for c in load_v33_cohort(progressor_only=False)
             if c.cohort == "nonprogressor"]
    print(f"[cohort] {len(cases)} nonprogressor knees with both-timepoint segs")
    if args.n_run is not None:
        cases = cases[: args.n_run]

    done = load_done(out_csv) if (args.resume and not args.dry_run) else set()
    fieldnames = (["pid", "side", "cohort", "KL_00"]
                  + [f"pd_{r}_d" for r in REGIONS] + ["pd_MFTC_d", "pd_LFTC_d"]
                  + [f"eck_{r}_d" for r in ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC")])
    rows = []
    if out_csv.exists() and not args.dry_run:
        rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))

    t0 = time.time()
    n_new = 0
    for i, c in enumerate(cases, 1):
        if (c.pid, c.side) in done:
            continue
        row = {"pid": c.pid, "side": c.side, "cohort": c.cohort, "KL_00": c.KL_00}
        try:
            per = {}
            for bone in bones:
                r = process_long_baseline_grid(c.seg_00m_path, c.seg_48m_path, c.side, bone, cfg)
                per.update(r["baseline_grid_regions"])
            for reg in REGIONS:
                row[f"pd_{reg}_d"] = per.get(reg, {}).get("d", np.nan)
            cmf = per.get("cMF", {}).get("d", np.nan); mt = per.get("MT", {}).get("d", np.nan)
            clf = per.get("cLF", {}).get("d", np.nan); lt = per.get("LT", {}).get("d", np.nan)
            row["pd_MFTC_d"] = (cmf + mt) if np.isfinite(cmf) and np.isfinite(mt) else np.nan
            row["pd_LFTC_d"] = (clf + lt) if np.isfinite(clf) and np.isfinite(lt) else np.nan
        except Exception as e:
            print(f"  [skip] {c.pid}_{c.side}: {type(e).__name__}: {e}")
            continue
        finally:
            _pipeline.clear_recon_cache()
        _, qd = qcart.get((c.pid, c.side), ("", {}))
        for reg in ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC"):
            row[f"eck_{reg}_d"] = qd.get(reg, np.nan)
        rows.append(row); n_new += 1
        rate = i / max(1e-6, time.time() - t0)
        print(f"  [{i}/{len(cases)}] {c.pid}_{c.side} MFTCΔ={row['pd_MFTC_d']:+.3f} "
              f"cMFΔ={row.get('pd_cMF_d', float('nan')):+.3f} MTΔ={row.get('pd_MT_d', float('nan')):+.3f} "
              f"({rate*60:.1f}/min)")
        if args.dry_run and n_new >= 2:
            break
        if not args.dry_run and n_new % 5 == 0:
            _write(out_csv, fieldnames, rows)

    if args.dry_run:
        print(f"[dry] {n_new} knees computed (not saved).")
        return
    _write(out_csv, fieldnames, rows)
    print(f"[ok] wrote {out_csv} ({len(rows)} knees, {n_new} new) in {(time.time()-t0)/60:.1f} min")


def _write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
