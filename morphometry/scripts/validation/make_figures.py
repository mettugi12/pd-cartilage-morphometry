"""Batch-render PDvDESS-style figures from a saved thickness directory.

Usage:
    # render all cross-sectional pairs + all longitudinal cases under <save_dir>
    python -m scripts.make_figures --save_dir <dir> --out_dir <dir>/figures
    # render only one cohort
    python -m scripts.make_figures --save_dir <dir> --only cross_sectional
    python -m scripts.make_figures --save_dir <dir> --only longitudinal
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cartilage_morphometry.validation.viz import make_long_figure, make_pair_figure


def _load_long_lookup(report_path: Path | None) -> dict[str, dict]:
    """Read ValidationReport JSON and return {case_tag: {qcart_row, pipeline_regions}}."""
    if report_path is None or not report_path.is_file():
        return {}
    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for c in rep.get("longitudinal", {}).get("per_case", []):
        tag = f"{c['pid']}_{c['side']}"
        out[tag] = {
            "qcart_row": c.get("eckstein_qcart"),
            "pipeline_regions": {
                "femur": (c.get("bones", {}).get("femur") or {}).get("regions_delta") or {},
                "tibia": (c.get("bones", {}).get("tibia") or {}).get("regions_delta") or {},
            },
        }
    return out


def _list_cs_cases(d: Path) -> list[str]:
    """Each cross-sectional case writes 4 npys: <case_id>_<bone>_<modality>.npy.
    Recover the unique case_id set."""
    cases: set[str] = set()
    pat = re.compile(r"^(.+)_(femur|tibia)_(pd|dess)\.npy$")
    for f in d.iterdir():
        m = pat.match(f.name)
        if m:
            cases.add(m.group(1))
    return sorted(cases)


def _list_long_cases(d: Path) -> list[str]:
    """Each longitudinal case writes 4 npys: <pid>_<side>_<bone>_<tp>.npy."""
    cases: set[str] = set()
    pat = re.compile(r"^(.+)_(femur|tibia)_(00m|48m)\.npy$")
    for f in d.iterdir():
        m = pat.match(f.name)
        if m:
            cases.add(m.group(1))
    return sorted(cases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=Path, required=True,
                   help="Dir written by validate(save_thickness_dir=...). "
                        "Should contain cross_sectional/ and/or longitudinal/ subdirs.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Where to write PNGs. Default: <save_dir>/figures/.")
    p.add_argument("--only", choices=["cross_sectional", "longitudinal", "both"], default="both")
    p.add_argument("--report", type=Path, default=None,
                   help="ValidationReport JSON. If set, longitudinal figures get a "
                        "5th-row bar comparison: pipeline Δ vs QCart Δ per region.")
    p.add_argument("--vmax_thick", type=float, default=4.0)
    p.add_argument("--vabs_delta", type=float, default=1.5)
    p.add_argument("--smooth_sigma", type=float, default=1.5)
    p.add_argument("--dry_run", action="store_true",
                   help="List cases found; don't render.")
    args = p.parse_args()

    out_dir = args.out_dir or (args.save_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("cross_sectional", "both"):
        cs_dir = args.save_dir / "cross_sectional"
        if cs_dir.is_dir():
            cases = _list_cs_cases(cs_dir)
            print(f"[cs] {len(cases)} cases under {cs_dir}: {cases}")
            if not args.dry_run:
                for case_id in cases:
                    png = out_dir / f"pair_{case_id}.png"
                    try:
                        make_pair_figure(
                            case_id, cs_dir, png,
                            vmax_thick=args.vmax_thick, vabs_delta=args.vabs_delta,
                            smooth_sigma=args.smooth_sigma,
                        )
                        print(f"  [ok] {png}")
                    except Exception as e:
                        print(f"  [ERR] {case_id}: {type(e).__name__}: {e}")

    if args.only in ("longitudinal", "both"):
        long_dir = args.save_dir / "longitudinal"
        if long_dir.is_dir():
            cases = _list_long_cases(long_dir)
            print(f"[long] {len(cases)} cases under {long_dir}: {cases}")
            lookup = _load_long_lookup(args.report)
            if lookup:
                print(f"[long] report lookup: {len(lookup)} cases with QCart/region data")
            if not args.dry_run:
                for case_tag in cases:
                    png = out_dir / f"long_{case_tag}.png"
                    extra = lookup.get(case_tag, {})
                    try:
                        make_long_figure(
                            case_tag, long_dir, png,
                            vmax_thick=args.vmax_thick, vabs_delta=args.vabs_delta,
                            smooth_sigma=args.smooth_sigma,
                            qcart_row=extra.get("qcart_row"),
                            pipeline_regions=extra.get("pipeline_regions"),
                        )
                        print(f"  [ok] {png}")
                    except Exception as e:
                        print(f"  [ERR] {case_tag}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
