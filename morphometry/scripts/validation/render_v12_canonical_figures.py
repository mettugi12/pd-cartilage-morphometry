"""Render the full-cohort v1.2 canonical figures from saved thickness arrays.

Reads the per-knee template-thickness .npy arrays written by the v1.2 full
validation run (`thickness_v12_raycast2d/{longitudinal,cross_sectional}/`)
and renders the canonical 2x3 figures (no recompute):
  - longitudinal: `make_long_figure_proportional` (+ QCart bar row +
    per-region cMF/cLF/MT/LT means per timepoint)
  - pairs: `make_pair_figure_proportional` (+ per-vertex PD-vs-DESS scatter)

Usage:
  python -m scripts.render_v12_canonical_figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cartilage_morphometry.validation.cohorts import load_v33_cohort, load_pair_cases
from cartilage_morphometry.validation.eckstein import load_qcart_deltas, lookup_qcart_row
from cartilage_morphometry.validation.viz import (
    make_long_figure_proportional, make_pair_figure_proportional,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thickness_root", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/thickness_v12_raycast2d"))
    p.add_argument("--out_root", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/v1.2_raycast2d_canonical"))
    p.add_argument("--skip_long", action="store_true")
    p.add_argument("--skip_pair", action="store_true")
    args = p.parse_args()

    long_dir = args.thickness_root / "longitudinal"
    pair_dir = args.thickness_root / "cross_sectional"
    (args.out_root / "long").mkdir(parents=True, exist_ok=True)
    (args.out_root / "pairs").mkdir(parents=True, exist_ok=True)

    # ---- longitudinal ----
    if not args.skip_long:
        try:
            qcart_df = load_qcart_deltas()
        except Exception as e:
            print(f"[warn] QCart deltas unavailable ({e}); bar row skipped")
            qcart_df = None
        cohort = list(load_v33_cohort(progressor_only=True))
        print(f"[long] {len(cohort)} cases")
        ok = err = 0
        for c in cohort:
            tag = f"{c.pid}_{c.side}"
            # require all 4 arrays
            need = [long_dir / f"{tag}_{b}_{tp}.npy" for b in ("femur", "tibia")
                    for tp in ("00m", "48m")]
            if not all(n.exists() for n in need):
                print(f"  [skip] {tag}: missing arrays"); continue
            qcart_row = None
            if qcart_df is not None:
                raw = lookup_qcart_row(qcart_df, c.pid, c.side)
                if raw is not None:
                    qcart_row = {"present": True}
                    for r in ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC"):
                        qcart_row[f"{r}_00mm"] = raw.get(f"eck_{r}_00")
                        qcart_row[f"{r}_48mm"] = raw.get(f"eck_{r}_48")
                        qcart_row[f"{r}_d"] = raw.get(f"eck_{r}_d")
            try:
                make_long_figure_proportional(
                    tag, long_dir, args.out_root / "long" / f"long_{tag}.png",
                    qcart_row=qcart_row, pipeline_regions=None)
                ok += 1
            except Exception as e:
                print(f"  [ERR] {tag}: {e}"); err += 1
        print(f"[long] rendered {ok}, errors {err}")

    # ---- cross-sectional pairs ----
    if not args.skip_pair:
        pairs = list(load_pair_cases())
        print(f"[pair] {len(pairs)} cases")
        ok = err = 0
        for pc in pairs:
            cid = pc.case_id
            need = [pair_dir / f"{cid}_{b}_{m}.npy" for b in ("femur", "tibia")
                    for m in ("pd", "dess")]
            if not all(n.exists() for n in need):
                print(f"  [skip] {cid}: missing arrays"); continue
            try:
                make_pair_figure_proportional(
                    cid, pair_dir, args.out_root / "pairs" / f"pair_{cid}.png")
                ok += 1
            except Exception as e:
                print(f"  [ERR] {cid}: {e}"); err += 1
        print(f"[pair] rendered {ok}, errors {err}")


if __name__ == "__main__":
    main()
