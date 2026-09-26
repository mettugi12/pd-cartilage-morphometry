"""v9.1 — Why does the automated 48-month value under-read only in stable knees?

Hypothesis: the baseline-grid pipeline is asymmetric. The baseline (grid-owner) visit is
binned directly; the follow-up (projected) visit is (i) rigidly registered, (ii) IDW-sampled
(k=3) onto the baseline mesh, where margin vertices average with zero-thickness bone
neighbours, and (iii) zero-imputed in every baseline-footprint cell it fails to cover.
Each step can only lower the projected visit's regional mean.

Test on a subset of leak-free knees (both arms):
  FORWARD  (as in the paper): 00m owns the grid, 48m projected.
  REVERSE  : 48m owns the grid, 00m projected.
For each direction and region record: owner-visit mean, projected-visit mean with the
pipeline's zero imputation, projected-visit mean over covered cells only, fraction of
owner-footprint cells that are uncovered (NaN) or zero at the projected visit, and ICP ASSD.
Join the expert reference (kMRI_QCart_Eckstein) absolute values of both visits.

If the under-reading follows the *projected* role (baseline under-read in REVERSE, 48m
under-read in FORWARD), the drift is methodological, not biological or acquisition-related.

Output: ../results/v9_1_drift_diagnosis.{csv,md}
Usage: python diagnose_drift_v9_1.py [--n_per_arm 30] [--dry_run] [--n_run N] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT_CSV = RES / "v9_1_drift_diagnosis.csv"
REG = {"femur": ["cMF"], "tibia": ["MT"]}


def grid_stats(g0, g4, name: str, bone: str):
    """Owner / projected regional means from the raw 40x40 grids of regional_deltas."""
    from cartilage_morphometry import baseline_grid as bg
    ref = np.isfinite(g0)
    dsl, wsl = {r[0]: (r[1], r[2]) for r in (bg.FC_REGIONS + bg.TC_REGIONS)}[name]
    sub = np.zeros_like(ref); sub[dsl, wsl] = True
    m = ref & sub
    n = int(m.sum())
    owner = float(np.nanmean(g0[m]))
    proj_zi = float(np.where(m & np.isfinite(g4), g4, 0.0)[m].sum() / n)           # pipeline (zero-imputed)
    cov = m & np.isfinite(g4) & (g4 > 0)
    proj_cov = float(g4[cov].mean()) if cov.any() else np.nan                        # covered cells only
    frac_nan = float((m & ~np.isfinite(g4)).sum() / n)
    frac_zero = float((m & np.isfinite(g4) & (g4 <= 0)).sum() / n)
    return dict(owner=owner, proj_zi=proj_zi, proj_cov=proj_cov, frac_nan=frac_nan, frac_zero=frac_zero, n_cells=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_arm", type=int, default=30)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
        _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR); _patch_disk_cache_path_unique()
    cfg = PipelineConfig(); cfg.region_projection = "baseline_grid"; cfg.femur_unwrap = "per_slice"

    rng = np.random.default_rng(args.seed)
    lf = pd.concat([pd.read_csv(RES / "v9_1_prog_leakfree.csv"), pd.read_csv(RES / "v9_1_nonprog_leakfree.csv")])
    pick = pd.concat([g.sample(min(args.n_per_arm, len(g)), random_state=args.seed) for _, g in lf.groupby("cohort")])
    keys = {(str(r.pid), str(r.side)) for r in pick.itertuples()}
    cases = [c for c in load_v33_cohort(progressor_only=False) if (str(c.pid), str(c.side)) in keys]
    if args.n_run:
        cases = cases[: args.n_run]
    absx = pd.read_csv(RES / "v9_1_long_abs.csv"); absx["pid"] = absx.pid.astype(str)
    absx = absx.set_index(["pid", "side"])
    print(f"[diag] {len(cases)} knees ({sum(c.cohort=='progressor' for c in cases)} prog / {sum(c.cohort!='progressor' for c in cases)} stable)")

    done = set()
    rows = []
    if OUT_CSV.exists() and not args.dry_run:
        rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
        done = {(r["pid"], r["side"]) for r in rows}
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        if (str(c.pid), str(c.side)) in done:
            continue
        row = {"pid": c.pid, "side": c.side, "cohort": c.cohort, "KL_00": c.KL_00}
        try:
            for direction, (a, b) in (("fwd", (c.seg_00m_path, c.seg_48m_path)), ("rev", (c.seg_48m_path, c.seg_00m_path))):
                for bone, regs in REG.items():
                    r = process_long_baseline_grid(a, b, c.side, bone, cfg)
                    row[f"{direction}_{bone}_assd"] = r["pair_quality"]["bone_assd_after_mm"]
                    for name in regs:
                        st = grid_stats(r["grid_00m"], r["grid_48m"], name, bone)
                        for k, v in st.items():
                            row[f"{direction}_{name}_{k}"] = v
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {c.pid}_{c.side}: {type(e).__name__}: {e}"); continue
        finally:
            _pipeline.clear_recon_cache()
        key = (str(c.pid), str(c.side))
        if key in absx.index:
            for reg in ("cMF", "MT", "MFTC"):
                for v in ("00m", "48m"):
                    row[f"eck_{reg}_{v}"] = absx.loc[key, f"eck_{reg}_{v}"]
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {c.pid}_{c.side} {c.cohort[:4]} fwd MT owner00={row['fwd_MT_owner']:.2f} proj48={row['fwd_MT_proj_zi']:.2f} "
              f"(cov-only {row['fwd_MT_proj_cov']:.2f}, nan {row['fwd_MT_frac_nan']:.2f}) | rev owner48={row['rev_MT_owner']:.2f} proj00={row['rev_MT_proj_zi']:.2f} "
              f"| expert 00={row.get('eck_MT_00m', float('nan')):.2f} 48={row.get('eck_MT_48m', float('nan')):.2f} ({(i)/(time.time()-t0)*60:.1f}/min)", flush=True)
        if args.dry_run and len(rows) >= 2:
            break
        if not args.dry_run and len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    df = pd.DataFrame(rows)
    if not args.dry_run:
        df.to_csv(OUT_CSV, index=False)
    summarize(df, args.dry_run)


def summarize(df: pd.DataFrame, dry: bool):
    for c in df.columns:
        if c not in ("pid", "side", "cohort"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    md = [f"# v9.1 — Drift diagnosis: owner vs projected visit (n={len(df)}: {int((df.cohort=='progressor').sum())} prog / {int((df.cohort!='progressor').sum())} stable)", "",
          "All values mm (regional mean). FORWARD: 00m owns grid, 48m projected (paper). REVERSE: 48m owns grid, 00m projected.", ""]
    for arm, g in df.groupby("cohort"):
        md += [f"## {arm} (n={len(g)})", "", "| Region | Visit | Role | PD mean | Expert mean | Bias PD−expert | Uncovered frac | Zero frac | Covered-only mean |", "|---|---|---|--:|--:|--:|--:|--:|--:|"]
        for reg in ("cMF", "MT"):
            e0, e4 = g[f"eck_{reg}_00m"].mean(), g[f"eck_{reg}_48m"].mean()
            md.append(f"| {reg} | 00m | owner (FWD) | {g[f'fwd_{reg}_owner'].mean():.2f} | {e0:.2f} | {g[f'fwd_{reg}_owner'].mean()-e0:+.2f} | — | — | — |")
            md.append(f"| {reg} | 00m | projected (REV) | {g[f'rev_{reg}_proj_zi'].mean():.2f} | {e0:.2f} | {g[f'rev_{reg}_proj_zi'].mean()-e0:+.2f} | {g[f'rev_{reg}_frac_nan'].mean():.3f} | {g[f'rev_{reg}_frac_zero'].mean():.3f} | {g[f'rev_{reg}_proj_cov'].mean():.2f} |")
            md.append(f"| {reg} | 48m | owner (REV) | {g[f'rev_{reg}_owner'].mean():.2f} | {e4:.2f} | {g[f'rev_{reg}_owner'].mean()-e4:+.2f} | — | — | — |")
            md.append(f"| {reg} | 48m | projected (FWD) | {g[f'fwd_{reg}_proj_zi'].mean():.2f} | {e4:.2f} | {g[f'fwd_{reg}_proj_zi'].mean()-e4:+.2f} | {g[f'fwd_{reg}_frac_nan'].mean():.3f} | {g[f'fwd_{reg}_frac_zero'].mean():.3f} | {g[f'fwd_{reg}_proj_cov'].mean():.2f} |")
        # change estimates
        for reg in ("cMF", "MT"):
            fwd = (g[f"fwd_{reg}_proj_zi"] - g[f"fwd_{reg}_owner"]).mean()
            rev = (g[f"rev_{reg}_owner"] - g[f"rev_{reg}_proj_zi"]).mean()
            cov = (g[f"fwd_{reg}_proj_cov"] - g[f"fwd_{reg}_owner"]).mean()
            exp = (g[f"eck_{reg}_48m"] - g[f"eck_{reg}_00m"]).mean()
            md.append(f"\n{reg} 48m−00m change: FORWARD {fwd:+.3f} | REVERSE {rev:+.3f} | FORWARD covered-only {cov:+.3f} | expert {exp:+.3f}")
        md.append(f"\nICP ASSD after registration: femur {g.fwd_femur_assd.mean():.2f} / tibia {g.fwd_tibia_assd.mean():.2f} mm (FWD); {g.rev_femur_assd.mean():.2f} / {g.rev_tibia_assd.mean():.2f} (REV)\n")
    md += ["## Reading", "",
           "- If the projected visit is under-read relative to the expert in BOTH directions (00m in REVERSE, 48m in FORWARD) while the owner visit is unbiased, the drift is a projection artefact (IDW margin shrinkage + zero imputation), not acquisition or biology.",
           "- The 'uncovered' and 'zero' fractions quantify how much of the owner footprint the projected visit fails to fill; multiplied by the regional mean they approximate the imputation penalty.",
           "- FORWARD covered-only change removes the imputation component but keeps IDW shrinkage; compare with the expert change."]
    text = "\n".join(md) + "\n"
    print(text)
    if not dry:
        (RES / "v9_1_drift_diagnosis.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
