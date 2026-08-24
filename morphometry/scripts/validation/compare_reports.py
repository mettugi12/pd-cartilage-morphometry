"""Load N ValidationReport JSONs and print/save a side-by-side comparison.

Usage:
    python -m scripts.compare_reports reports/v1_default.json reports/v1_raycast.json
    python -m scripts.compare_reports reports/*.json --out reports/compare_table.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cartilage_morphometry.validation import ValidationReport

# Rows shown in the comparison table — keep small + manuscript-aligned.
ROWS = [
    ("cross_sectional", "femur", "r_2d_smooth_mean"),
    ("cross_sectional", "femur", "r_2d_smooth_std"),
    ("cross_sectional", "tibia", "r_2d_smooth_mean"),
    ("cross_sectional", "tibia", "r_2d_smooth_std"),
    ("cross_sectional", "femur", "bias_mm"),
    ("cross_sectional", "tibia", "bias_mm"),
    ("longitudinal", "MFTC", "SRM"),
    ("longitudinal", "cMF",  "r_vs_qcart"),
    ("longitudinal", "MT",   "r_vs_qcart"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the table as markdown.")
    args = p.parse_args()

    loaded = [(r.stem, ValidationReport.from_json(r)) for r in args.reports]

    header = "| metric | " + " | ".join(name for name, _ in loaded) + " |"
    sep = "|---" * (1 + len(loaded)) + "|"
    body_rows = []
    for track, key1, key2 in ROWS:
        label = f"{track}.{key1}.{key2}"
        cells = []
        for _, rep in loaded:
            d = getattr(rep, track, {}).get(key1, {})
            v = d.get(key2, None) if isinstance(d, dict) else None
            cells.append("—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        body_rows.append(f"| {label} | " + " | ".join(cells) + " |")

    md = "\n".join([header, sep, *body_rows])
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n", encoding="utf-8")
        print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
