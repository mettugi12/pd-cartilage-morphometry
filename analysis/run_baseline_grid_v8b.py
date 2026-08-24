"""v8b — longitudinal validation using the RESTORED v3.3 baseline-grid projection.

Same cohort/data/pipeline as v8 (cartilage-morphometry @ current HEAD, raycast_2d
thickness) EXCEPT region_projection="baseline_grid" (per-patient 40x40 grid, not
the atlas template). Goal: recover v3.3-level longitudinal QCart agreement,
especially in the medial tibia, while keeping the template path as the default.

For each progressor knee: run process_long_baseline_grid for femur + tibia ->
regional Δ (cMF, MT, etc.); MFTC = cMF + MT. Manual QCart Δ is joined from the
cached v3.3 merged deltas (eck_* columns; same OAI Eckstein QCart release).

Writes per-knee deltas to ../results/v8b_merged_deltas.csv (RESUMABLE — re-run to
fill gaps), then aggregate stats via aggregate_v8 helpers.

CLI: --n_run N (first N knees), --dry_run (compute, print, don't persist),
     --bones femur,tibia
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
OUT_CSV = RESULTS / "v8b_merged_deltas.csv"

REGIONS = ["cMF", "cLF", "MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT"]


def load_qcart():
    """(pid, side) -> {region: qcart_delta_mm} from the cached v3.3 merged deltas."""
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
    out_csv = OUT_CSV if args.femur_unwrap == "per_slice" else \
        OUT_CSV.with_name("v8b_bestfit_merged_deltas.csv")

    import sys
    REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from cartilage_morphometry import PipelineConfig
    from cartilage_morphometry import pipeline as _pipeline
    from cartilage_morphometry.validation.cohorts import load_v33_cohort, DEFAULT_RECON_CACHE_DIR
    from cartilage_morphometry.validation.shared_mesh import process_long_baseline_grid

    # Reuse the shared RECON disk cache (749 prebuilt files) -> fast reruns.
    # MUST apply the harness's path-unique patch: the library default keys the
    # disk cache by filename stem only, which COLLIDES 00m vs 48m (same stem in
    # different dirs) -> 48m loads 00m's RECON -> zero delta. The patch keys by
    # md5(full_path) + flip tag (matches the existing 749 cached files).
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
    cases = [c for c in load_v33_cohort() if c.cohort == "progressor"]
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
            # The in-memory RECON cache grows unbounded across knees (each adds
            # two dense volumes) -> OOM after ~60 knees. We use the DISK cache, so
            # clear in-memory every knee; next knee reloads from disk (fast).
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
