"""CLI: run one validate() with a named config and dump reports/<name>.json.

Usage:
    python -m scripts.run_validation --config v1_default --out reports/v1_default.json
    python -m scripts.run_validation --config v1_raycast --out reports/v1_raycast.json
    python -m scripts.run_validation --config v1_default --n_run 5 --dry_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cartilage_morphometry import PipelineConfig

from cartilage_morphometry.validation import validate

# NOTE (2026-06-23): the repo was mirrored to the deployed web-app
# cartilage_morphometry module (web app = canonical; commit 95b8090). That mirror
# REMOVED the validation-only `idw_denudation_gate`, `long_icp_method` and
# `long_trim_fraction` config fields. The old gate/long-ICP/threshold isolation
# matrix below has therefore been dropped — those configs no longer construct.
# The PD-DESS v8 study validates the pure web-app default (`v1_default` =
# PipelineConfig() = raycast_2d + subch_threshold 0.5 + no gate). See the v8
# DATASET.md and Studies/PD-vs-DESS.md.


# Named configs the validation CLI exposes. Add more as needed.
CONFIGS: dict[str, PipelineConfig] = {
    # PD-DESS v8 canonical = the pure web-app default (raycast_2d + tpl_nn +
    # aniso_rigid + subch_threshold 0.5, no denudation gate). This is what the
    # deployed web app runs and what the v8 manuscript is validated against.
    "v1_default": PipelineConfig(),
    # Legacy patient-frame raycast thickness method (pre-2D-slice). Diagnostic.
    "v1_raycast": PipelineConfig(thickness_method="raycast"),
    "v1_raycast_shared_mesh": PipelineConfig(thickness_method="raycast"),
    # v3.3 parity reference — manuscript ran without the cart_cleanup step.
    "v1_no_cart_cleanup": PipelineConfig(cart_cleanup=()),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=sorted(CONFIGS),
                   help="Named config from CONFIGS.")
    p.add_argument("--seg_provider", default="dataset204",
                   choices=["dataset204", "dataset211", "pd_dualplane_fusion"])
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write the JSON ValidationReport.")
    p.add_argument("--no_cross_sectional", action="store_true")
    p.add_argument("--no_longitudinal", action="store_true")
    p.add_argument("--recon_cache_dir", default="default",
                   help="RECON cache dir (default: cohorts.DEFAULT_RECON_CACHE_DIR; "
                        "pass 'none' to disable).")
    p.add_argument("--n_run", type=int, default=None,
                   help="Cap N cases per cohort (fast iteration).")
    p.add_argument("--shared_mesh", dest="shared_mesh", action="store_true", default=True,
                   help="v6 canonical: one bone mesh as the shared sampling surface (default).")
    p.add_argument("--no_shared_mesh", dest="shared_mesh", action="store_false",
                   help="Disable shared-mesh — registers each modality / timepoint independently.")
    p.add_argument("--r_2d_sigma", type=float, default=1.5,
                   help="σ (grid cells) for the border-preserving Gaussian in r_2d_smooth (default 1.5).")
    p.add_argument("--save_thickness_dir", type=Path, default=None,
                   help="If set, persist per-knee template_thickness .npy arrays under "
                        "<dir>/cross_sectional/ and <dir>/longitudinal/ for downstream viz.")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute but do not write the JSON; print a sample.")
    args = p.parse_args()

    if args.recon_cache_dir == "default":
        recon_cache = "default"
    elif args.recon_cache_dir.lower() in ("none", "off", ""):
        recon_cache = None
    else:
        recon_cache = Path(args.recon_cache_dir)

    report = validate(
        CONFIGS[args.config],
        seg_provider=args.seg_provider,
        cross_sectional=not args.no_cross_sectional,
        longitudinal=not args.no_longitudinal,
        n_run=args.n_run,
        recon_cache_dir=recon_cache,
        shared_mesh=args.shared_mesh,
        r_2d_sigma=args.r_2d_sigma,
        save_thickness_dir=args.save_thickness_dir,
    )

    if args.dry_run:
        print(json.dumps(
            {"config_name": args.config,
             "cross_sectional_keys": list(report.cross_sectional),
             "longitudinal_keys": list(report.longitudinal),
             "runtime_s": report.runtime_s},
            indent=2,
        ))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(args.out)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
