"""4-way 00m thickness comparison per case (no longitudinal, no Δ — pure
"what's the cleanest template-frame thickness measurement?" experiment).

Panels (1 row × 4 cols, per case + bone):
  [1] PATIENT 2D — patient raycast on patient bone (reference; no template)
  [2] TEMPLATE — patient raycast → IDW remap (current canonical)
  [3] TEMPLATE — template_raycast (template-inherited normals AND origins;
        no IDW, raycast directly from template vert positions transformed
        back to patient frame)
  [4] TEMPLATE — patient EDT → IDW remap (alternative thickness method)

All panels at the same color scale (jet, 0-4 mm). Open and compare:
  - Are the high-thickness regions the same shape/area as patient (panel 1)?
  - Does template_raycast (panel 3) preserve the patient distribution better
    than IDW remap (panel 2)?
  - Where do "white gaps" appear in template_raycast — true denudation or
    template-normal pointing nowhere?
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
    template_thickness_2d_femur,
    template_thickness_2d_tibia,
    compute_femur_template_subregions,
    process_one_patient,
)
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import _force_ml_flip, _ml_flip_decision
from cartilage_morphometry.validation.viz import _smooth_border_preserving
from scripts.diagnose_long_patient_2d import _project_tibia_patient, _project_femur_patient
from scripts.diagnose_subch_mask import _project_mask_tibia


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]

_TPL_PROJ: dict[str, dict] = {}
def _tpl_pack(bone):
    if bone in _TPL_PROJ: return _TPL_PROJ[bone]
    m = pv.read(str(TEMPLATE_PATHS[bone]))
    pack = {"mesh": m}
    if bone == "femur":
        pack["sub_femur"] = compute_femur_template_subregions(m)
    _TPL_PROJ[bone] = pack
    return pack


def _project_template(thickness, bone, grid_size=150):
    pack = _tpl_pack(bone)
    if bone == "femur":
        g, _, _ = template_thickness_2d_femur(pack["mesh"], thickness,
                                              grid_size=grid_size,
                                              subregions=pack["sub_femur"])
    else:
        g, _, _ = template_thickness_2d_tibia(pack["mesh"], thickness, grid_size=grid_size)
    return _smooth_border_preserving(g, 1.5)


def _decorate(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                              (0.98, 0.97, "L", "right"),
                              (0.5, 0.97, "P", "center"),
                              (0.5, 0.03, "A", "center")):
        ax.text(tx, ty, label, transform=ax.transAxes, color="white",
                fontsize=9, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))


def _imshow(ax, g, title, vmax_thick=4.0, contour_mask=None,
            contour_color="white", contour_label=""):
    """`contour_mask` (optional): 2D bool/float grid (NaN = outside). Will
    draw an outline contour at value=0.5 on top of the colored thickness."""
    im = ax.imshow(g, origin="upper", cmap="jet_r",
                   vmin=0.0, vmax=vmax_thick, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("thickness (mm)", fontsize=8)
    if contour_mask is not None:
        cm = np.where(np.isfinite(contour_mask), contour_mask, 0)
        ax.contour(cm, levels=[0.5], colors=contour_color, linewidths=1.5)
    ax.set_title(title, fontsize=10)
    _decorate(ax)
    with np.errstate(invalid="ignore"):
        mn = float(np.nanmean(g))
        n_valid = int(np.sum(np.isfinite(g)))
        n_pos = int(np.sum(np.isfinite(g) & (g > 0)))
    txt = f"mean = {mn:.3f} mm   valid {n_valid} cells   >0: {n_pos}"
    if contour_label:
        txt = f"{txt}\n{contour_label}"
    ax.text(0.5, -0.05, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="center")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_thickness_method"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None: continue
        for bone in args.bones:
            print(f"\n=== {case_tag} {bone} 00m ===")
            flip = _ml_flip_decision(case.seg_00m_path)
            results = {}
            patient_raycast_thickness = None
            patient_edt_thickness = None
            patient_bone_mesh_raycast = None
            patient_bone_mesh_edt = None
            tpl_subch_field = np.asarray(_tpl_pack(bone)["mesh"].point_data["subch_prob"])
            is_tpl_subch_template = tpl_subch_field >= 0.5
            # 5 variants: patient raycast + 3 raycast+IDW (k=3/5/10) + EDT+IDW (k=3)
            variants = (
                ("raycast_idw_k3",  "raycast", 3),
                ("raycast_idw_k5",  "raycast", 5),
                ("raycast_idw_k10", "raycast", 10),
                ("edt_idw_k3",      "edt",     3),
            )
            for variant, method, k in variants:
                _pipeline.clear_recon_cache()
                cfg = PipelineConfig(thickness_method=method, subch_method="tpl_nn", idw_k=k)
                try:
                    with _force_ml_flip(flip):
                        r = process_one_patient(case.seg_00m_path, case.side, bone,
                                                 "pd", config=cfg)
                    tpl_th = np.asarray(r["template_thickness"])
                    mn = float(np.nanmean(tpl_th))
                    n_valid = int(np.sum(np.isfinite(tpl_th)))
                    n_pos = int(np.sum(np.isfinite(tpl_th) & (tpl_th > 0)))
                    results[variant] = tpl_th
                    print(f"  [{variant:16s}] k={k}  mean={mn:.3f}mm  finite={n_valid}  >0={n_pos}")
                    if variant == "raycast_idw_k3":
                        patient_bone_mesh_raycast = r["bone_mesh"]
                        patient_raycast_thickness = np.asarray(r["bone_mesh"]["thickness_mm"])
                    elif variant == "edt_idw_k3":
                        patient_bone_mesh_edt = r["bone_mesh"]
                        patient_edt_thickness = np.asarray(r["bone_mesh"]["thickness_mm"])
                except Exception as e:
                    print(f"  [ERR {variant}] {e}")
                    continue
            if any(k not in results for k in ("raycast_idw_k3", "raycast_idw_k5",
                                               "raycast_idw_k10", "edt_idw_k3")):
                continue

            # ── 2D patient projections (raycast and EDT)
            proj_p = _project_femur_patient if bone == "femur" else _project_tibia_patient
            g_pat_rc = _smooth_border_preserving(
                proj_p(patient_bone_mesh_raycast, patient_raycast_thickness, 80), 1.0)
            g_pat_edt = _smooth_border_preserving(
                proj_p(patient_bone_mesh_edt, patient_edt_thickness, 80), 1.0)
            # subch_pool mask on PATIENT (tibia only — needs ML midline split projection)
            pat_subch_contour = None
            if bone == "tibia":
                pat_subch_contour = _project_mask_tibia(
                    patient_bone_mesh_raycast,
                    np.ones(patient_bone_mesh_raycast.n_points, dtype=bool),  # all bone verts available
                    80,
                )

            # ── 2D template projections
            g_rc3 = _project_template(results["raycast_idw_k3"], bone)
            g_rc5 = _project_template(results["raycast_idw_k5"], bone)
            g_rc10 = _project_template(results["raycast_idw_k10"], bone)
            g_edt = _project_template(results["edt_idw_k3"], bone)
            tpl_subch_contour = _project_template(
                is_tpl_subch_template.astype(np.float32), bone)

            # ── Render: 2 rows × 5 cols (patient + 3 raycast k variants + EDT)
            fig, axes = plt.subplots(2, 5, figsize=(28, 11),
                                     gridspec_kw={"height_ratios": [2, 1]})
            CONTOUR_COLOR = "black"   # black for visibility on jet colormap

            _imshow(axes[0, 0], g_pat_rc, f"PATIENT frame\nraycast (no template)",
                    contour_mask=pat_subch_contour, contour_color=CONTOUR_COLOR,
                    contour_label="black = patient bone outline")
            _imshow(axes[0, 1], g_rc3, f"TEMPLATE  raycast+IDW k=3\n[v1.1 CANONICAL]",
                    contour_mask=tpl_subch_contour, contour_color=CONTOUR_COLOR,
                    contour_label="black = template subch zone")
            _imshow(axes[0, 2], g_rc5, f"TEMPLATE  raycast+IDW k=5",
                    contour_mask=tpl_subch_contour, contour_color=CONTOUR_COLOR)
            _imshow(axes[0, 3], g_rc10, f"TEMPLATE  raycast+IDW k=10",
                    contour_mask=tpl_subch_contour, contour_color=CONTOUR_COLOR)
            _imshow(axes[0, 4], g_edt, f"TEMPLATE  EDT+IDW k=3",
                    contour_mask=tpl_subch_contour, contour_color=CONTOUR_COLOR)

            def _hist(ax, arrs_labels_colors, title):
                bins = np.linspace(0, 4, 81)
                for arr, label, color in arrs_labels_colors:
                    a = np.asarray(arr, dtype=float).ravel()
                    a = a[np.isfinite(a) & (a > 0)]
                    if not len(a): continue
                    ax.hist(a, bins=bins, alpha=0.55, color=color,
                             label=f"{label} (n={len(a)}, mn={a.mean():.2f})")
                ax.set_xlabel("thickness (mm)", fontsize=8)
                ax.set_ylabel("count", fontsize=8)
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
                ax.tick_params(labelsize=7)

            _hist(axes[1, 0],
                  [(patient_raycast_thickness, "patient raycast", "#1f77b4"),
                   (patient_edt_thickness,     "patient EDT",     "#ff7f0e")],
                  "PATIENT thickness>0  (no remap)")
            _hist(axes[1, 1],
                  [(patient_raycast_thickness, "patient raycast", "#1f77b4"),
                   (results["raycast_idw_k3"], "template k=3",    "#d62728")],
                  "raycast k=3: patient vs template")
            _hist(axes[1, 2],
                  [(patient_raycast_thickness, "patient raycast", "#1f77b4"),
                   (results["raycast_idw_k5"], "template k=5",    "#d62728")],
                  "raycast k=5: patient vs template")
            _hist(axes[1, 3],
                  [(patient_raycast_thickness,  "patient raycast", "#1f77b4"),
                   (results["raycast_idw_k10"], "template k=10",   "#d62728")],
                  "raycast k=10: patient vs template")
            _hist(axes[1, 4],
                  [(patient_edt_thickness,    "patient EDT",   "#ff7f0e"),
                   (results["edt_idw_k3"],    "template k=3",  "#9467bd")],
                  "EDT k=3: patient vs template")

            fig.suptitle(f"{case_tag}  {bone}  00m  — thickness method comparison "
                         f"(with subch contours + value-distribution histograms)",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            out = args.out_dir / f"thickness_methods_{case_tag}_{bone}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
