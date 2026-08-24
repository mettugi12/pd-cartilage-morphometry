"""3×3 comparison per case: patient-frame vs template tpl_nn vs template cart_prox.

For each case, runs the longitudinal pipeline twice (subch_method="tpl_nn"
vs subch_method="cart_prox"), then renders a single 3×3 figure:

    rows: patient-frame |  template tpl_nn (current canonical) |  template cart_prox (new)
    cols: 00m  |  48m  |  Δ (48m − 00m)

Patient-frame projection uses the 2D grids ported from
cartilage-v1-snuh-test/v1.0/analysis/render_case_variant.py. Template-frame
projection uses the library's `template_thickness_2d_{tibia,femur}` at
grid_size=150 + light σ=1.5 border-preserving smooth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from cartilage_morphometry import (
    PipelineConfig,
    TEMPLATE_PATHS,
    compute_femur_template_subregions,
    template_thickness_2d_femur,
    template_thickness_2d_tibia,
)
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import process_long_shared_mesh
from cartilage_morphometry.validation.viz import _smooth_border_preserving

# Re-use the patient-frame projection functions
from scripts.diagnose_long_patient_2d import _project_tibia_patient, _project_femur_patient


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]

# Template projection cache
_TPL_PROJ: dict[str, dict] = {}
def _tpl_pack(bone):
    if bone in _TPL_PROJ: return _TPL_PROJ[bone]
    m = pv.read(str(TEMPLATE_PATHS[bone]))
    pack = {"mesh": m}
    if bone == "femur":
        pack["sub_femur"] = compute_femur_template_subregions(m)
    _TPL_PROJ[bone] = pack
    return pack


def _project_template(thickness_arr, bone: str, grid_size: int = 150):
    pack = _tpl_pack(bone)
    if bone == "femur":
        g, _, _ = template_thickness_2d_femur(pack["mesh"], thickness_arr,
                                              grid_size=grid_size,
                                              subregions=pack["sub_femur"])
    else:
        g, _, _ = template_thickness_2d_tibia(pack["mesh"], thickness_arr,
                                              grid_size=grid_size)
    return _smooth_border_preserving(g, 1.5)


def _decorate(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                              (0.98, 0.97, "L", "right"),
                              (0.5, 0.97, "P", "center"),
                              (0.5, 0.03, "A", "center")):
        ax.text(tx, ty, label, transform=ax.transAxes, color="white",
                fontsize=8, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))


def _imshow_panel(ax, g, title, kind, vmax_thick, vabs_delta):
    if kind == "delta":
        im = ax.imshow(g, origin="upper", cmap="RdBu",
                       vmin=-vabs_delta, vmax=vabs_delta, interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Δ thickness (mm)", fontsize=9)
        with np.errstate(invalid="ignore"):
            mn = float(np.nanmean(g))
            n_pos = int(np.sum((g > 0.1) & np.isfinite(g)))
            n_neg = int(np.sum((g < -0.1) & np.isfinite(g)))
            n_tot = int(np.sum(np.isfinite(g)))
        ax.text(0.5, -0.06, f"mean Δ={mn:+.3f}mm  +0.1: {n_pos}/{n_tot}  -0.1: {n_neg}/{n_tot}",
                transform=ax.transAxes, fontsize=8, va="top", ha="center")
    else:
        im = ax.imshow(g, origin="upper", cmap="jet_r",
                       vmin=0.0, vmax=vmax_thick, interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("thickness (mm)", fontsize=9)
        with np.errstate(invalid="ignore"):
            mn = float(np.nanmean(g))
        ax.text(0.5, -0.06, f"mean = {mn:.3f}mm",
                transform=ax.transAxes, fontsize=8, va="top", ha="center")
    ax.set_title(title, fontsize=10)
    _decorate(ax)


def _render(case_tag: str, bone: str,
            res_tpl_nn: dict, res_scatter: dict,
            out_png: Path, vmax_thick: float = 4.0, vabs_delta: float = 1.5,
            patient_grid: int = 80):
    """3×3 figure: rows = (patient | template IDW k=3 | template scatter k=1),
    cols = (00m | 48m | Δ)."""
    # Patient-frame projections (from process_long's returned 00m bone + th00 + th48_at_00)
    m00 = res_tpl_nn["bone_mesh_00m"]
    th00_p = res_tpl_nn["th00_patient"]
    th48_p = res_tpl_nn["th48_at_00_patient"]
    proj_p = _project_femur_patient if bone == "femur" else _project_tibia_patient
    pat_00 = _smooth_border_preserving(proj_p(m00, th00_p, patient_grid), 1.0)
    pat_48 = _smooth_border_preserving(proj_p(m00, th48_p, patient_grid), 1.0)
    pat_d = pat_48 - pat_00

    # Template-frame (IDW k=3, current canonical)
    tpl_idw_00 = _project_template(res_tpl_nn["tp00_template_thickness"], bone)
    tpl_idw_48 = _project_template(res_tpl_nn["tp48_template_thickness"], bone)
    tpl_idw_d = tpl_idw_48 - tpl_idw_00

    # Template-frame (scatter k=1, NEW)
    tpl_sc_00 = _project_template(res_scatter["tp00_template_thickness"], bone)
    tpl_sc_48 = _project_template(res_scatter["tp48_template_thickness"], bone)
    tpl_sc_d = tpl_sc_48 - tpl_sc_00

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    rows = [
        ("PATIENT frame  (no template)",              (pat_00, pat_48, pat_d)),
        (f"TEMPLATE  forward IDW k=3  (canonical v1.1)", (tpl_idw_00, tpl_idw_48, tpl_idw_d)),
        (f"TEMPLATE  scatter k=1  (NEW)",                (tpl_sc_00, tpl_sc_48, tpl_sc_d)),
    ]
    cols = ["00m baseline", "48m followup", "Δ (48m − 00m)"]
    for r, (row_label, grids) in enumerate(rows):
        for c, (g, col_label) in enumerate(zip(grids, cols)):
            kind = "delta" if c == 2 else "thickness"
            title = f"{row_label}\n{col_label}"
            _imshow_panel(axes[r, c], g, title, kind, vmax_thick, vabs_delta)
    fig.suptitle(f"{case_tag}  {bone}  — patient-frame vs template tpl_nn vs template cart_prox",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_subch_method"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    print(f"[recon] disk cache: {DEFAULT_RECON_CACHE_DIR}")

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None:
            print(f"[skip] {case_tag} not in cohort"); continue
        for bone in args.bones:
            print(f"\n=== {case_tag} {bone} ===")
            results = {}
            for variant_name, remap in (("tpl_nn", "idw"), ("scatter", "scatter")):
                _pipeline.clear_recon_cache()
                cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn")
                try:
                    results[variant_name] = process_long_shared_mesh(
                        case.seg_00m_path, case.seg_48m_path,
                        side=case.side, bone_name=bone, config=cfg,
                        remap_method=remap,
                    )
                    rd = results[variant_name]
                    th00 = rd["tp00_template_thickness"]
                    tp00_mean = float(th00[np.isfinite(th00)&(th00>0)].mean()) if (np.isfinite(th00)&(th00>0)).any() else float('nan')
                    rdelta = rd["tp48_template_thickness"] - rd["tp00_template_thickness"]
                    print(f"  [{variant_name:9s} remap={remap}] tp00 mean(>0)={tp00_mean:.3f}mm  Δ mean={float(np.nanmean(rdelta)):+.3f}mm")
                except Exception as e:
                    print(f"  [ERR {variant_name}] {e}")
                    continue
            if "tpl_nn" not in results or "scatter" not in results:
                continue
            out = args.out_dir / f"scatter_{case_tag}_{bone}.png"
            _render(case_tag, bone, results["tpl_nn"], results["scatter"], out)
            print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
