"""v1.2 verification — compartment-aware tibia remap visual diff.

Per case + timepoint, produces a 1x3 figure using PROPORTIONAL 2D projection
(shared ML/AP bounds across both compartments → real relative sizes, natural
intercondylar gap).

  [PATIENT frame]                 [TEMPLATE v1.1 IDW]            [TEMPLATE v1.2 compartment]
   patient thickness               template thickness              template thickness
   + medial+lateral subch          + medial+lateral subch          + medial+lateral subch
   contours (BLACK, inherited)     contours (BLACK, from atlas)    contours (BLACK, from atlas)

Patient subch contours come from template-NN inheritance: for each patient
bone vert, take the compartment label of the nearest template subch vert
(in template-aligned space). This is bone-based and gap-free — no dependence
on cart_mask staircase artifacts.

Usage:
  python -m scripts.diagnose_compartment_aware \
      --cases 9477205_RIGHT 9235666_LEFT 9567504_RIGHT 9705992_LEFT 9832566_RIGHT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree

from cartilage_morphometry import PipelineConfig, TEMPLATE_PATHS
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import process_long_shared_mesh


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


_TPL_CACHE: dict[str, pv.PolyData] = {}
def _tpl(bone: str) -> pv.PolyData:
    if bone not in _TPL_CACHE:
        _TPL_CACHE[bone] = pv.read(str(TEMPLATE_PATHS[bone]))
    return _TPL_CACHE[bone]


# ---------------------------------------------------------------------------
# Proportional 2D projection — single shared (ML_lo, ML_hi, AP_lo, AP_hi)
# across the entire subch region, so relative sizes are preserved AND the
# intercondylar gap appears naturally where no verts project.
# ---------------------------------------------------------------------------
def _proj_proportional(pts: np.ndarray, scalar: np.ndarray,
                        mask: np.ndarray, grid_size: int = 60,
                        layout: dict | None = None, padding_frac: float = 0.02,
                        agg: str = "mean"):
    """Project (AP, ML) of `pts` onto an isotropic-aspect grid.
    `mask`: bool array of which verts to project (e.g. subch_prob >= 0.5).
    `layout` (optional): reuse a previously computed layout dict so multiple
    overlays (thickness map, subch contours) bin into the SAME grid.
    Returns (grid, layout) where layout = dict with ml_lo/ml_hi/ap_lo/ap_hi/n_rows/n_cols.
    """
    AP = pts[:, 0]; ML = pts[:, 2]
    if layout is None:
        v = mask
        if not v.any():
            return np.full((grid_size, grid_size), np.nan), None
        ml_lo, ml_hi = float(ML[v].min()), float(ML[v].max())
        ap_lo, ap_hi = float(AP[v].min()), float(AP[v].max())
        ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
        ml_lo -= ml_r * padding_frac; ml_hi += ml_r * padding_frac
        ap_lo -= ap_r * padding_frac; ap_hi += ap_r * padding_frac
        ml_r = ml_hi - ml_lo; ap_r = ap_hi - ap_lo
        aspect = ml_r / ap_r
        if aspect >= 1.0:
            n_cols = grid_size
            n_rows = max(int(round(grid_size / aspect)), 1)
        else:
            n_rows = grid_size
            n_cols = max(int(round(grid_size * aspect)), 1)
        layout = {"ml_lo": ml_lo, "ml_hi": ml_hi, "ap_lo": ap_lo, "ap_hi": ap_hi,
                  "ml_r": ml_r, "ap_r": ap_r, "n_rows": n_rows, "n_cols": n_cols}
    n_rows, n_cols = layout["n_rows"], layout["n_cols"]
    ml_r, ap_r = layout["ml_r"], layout["ap_r"]
    ml_lo, ap_lo = layout["ml_lo"], layout["ap_lo"]

    valid = mask & np.isfinite(scalar) if scalar is not None else mask
    if not valid.any():
        return np.full((n_rows, n_cols), np.nan), layout

    ml_n = (ML[valid] - ml_lo) / ml_r
    ap_n = (AP[valid] - ap_lo) / ap_r
    ml_b = np.clip((ml_n * n_cols).astype(int), 0, n_cols - 1)
    ap_b = np.clip((ap_n * n_rows).astype(int), 0, n_rows - 1)

    if agg == "mean":
        sum_g = np.zeros((n_rows, n_cols), dtype=np.float64)
        cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
        np.add.at(sum_g, (ap_b, ml_b), scalar[valid] if scalar is not None else 1.0)
        np.add.at(cnt_g, (ap_b, ml_b), 1)
        grid = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)
    else:  # "any" — binary mask occupancy
        cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
        np.add.at(cnt_g, (ap_b, ml_b), 1)
        grid = np.where(cnt_g > 0, 1.0, np.nan)
    return grid, layout


def _close_filled(g_mask_grid: np.ndarray, iters: int = 1) -> np.ndarray:
    """Binary closing on the in-plateau cells (NaN → 0). Returns float grid
    {0, 1} for cleanest contour rendering."""
    binary = np.where(np.isfinite(g_mask_grid), 1, 0).astype(np.uint8)
    if iters > 0:
        binary = binary_closing(binary, structure=np.ones((3, 3), dtype=bool),
                                 iterations=iters).astype(np.uint8)
    return binary.astype(np.float32)


def _inherit_compartment(aligned_pts: np.ndarray, template_mesh: pv.PolyData,
                          subch_threshold: float = 0.5):
    """Per patient bone vert, return the compartment label inherited from
    the NEAREST template subch vert (in template-aligned space). Also returns
    the NN distance per vert so callers can mask out shaft/non-articular verts.

    Returns (inherited_comp (int8), nn_dist_mm (float32)).
    """
    tpl_pts = np.asarray(template_mesh.points)
    tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    tpl_subch_mask = (np.asarray(template_mesh.point_data["subch_prob"])
                       >= subch_threshold)
    tpl_subch_idx = np.where(tpl_subch_mask)[0]
    tpl_subch_pts = tpl_pts[tpl_subch_idx]
    tpl_subch_comp = tpl_comp[tpl_subch_idx]

    tree = cKDTree(tpl_subch_pts)
    dists, idx = tree.query(np.asarray(aligned_pts), k=1)
    return tpl_subch_comp[idx].astype(np.int8), dists.astype(np.float32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _imshow_with_subch(ax, g_th, g_med_subch, g_lat_subch, title, info,
                        vmax: float = 4.0):
    im = ax.imshow(g_th, origin="upper", cmap="jet_r",
                   vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
        "thickness (mm)", fontsize=8)
    for g in (g_med_subch, g_lat_subch):
        if g is None: continue
        if g.any():
            ax.contour(g, levels=[0.5], colors="black", linewidths=1.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                              (0.98, 0.97, "L", "right"),
                              (0.5, 0.97, "P", "center"),
                              (0.5, 0.03, "A", "center")):
        ax.text(tx, ty, label, transform=ax.transAxes, color="white",
                fontsize=9, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))
    ax.text(0.5, -0.05, info, transform=ax.transAxes,
            fontsize=8, va="top", ha="center")


def _compartment_mean(tpl_th: np.ndarray, bone: str, lbl: int) -> float:
    comp = np.asarray(_tpl(bone).point_data["compartment"])
    sel = (comp == lbl) & np.isfinite(tpl_th)
    return float(tpl_th[sel].mean()) if sel.any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_compartment_aware"))
    p.add_argument("--thickness_method", default="raycast",
                   choices=["raycast", "edt"])
    p.add_argument("--subch_method", default="tpl_nn",
                   choices=["tpl_nn", "cart_prox"])
    p.add_argument("--grid_size", type=int, default=60)
    p.add_argument("--inherit_max_dist_mm", type=float, default=4.0,
                   help="Bone verts whose NN template-subch vert is farther than this are not assigned a compartment (i.e. shaft verts).")
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    cfg = PipelineConfig(thickness_method=args.thickness_method,
                          subch_method=args.subch_method, idw_k=3)

    bone = "tibia"
    tpl = _tpl(bone)
    tpl_pts = np.asarray(tpl.points)
    tpl_compartment = np.asarray(tpl.point_data["compartment"])
    tpl_subch_mask = np.asarray(tpl.point_data["subch_prob"]) >= 0.5
    tpl_med_mask = (tpl_compartment == 1) & tpl_subch_mask
    tpl_lat_mask = (tpl_compartment == 2) & tpl_subch_mask
    tpl_any_subch_mask = tpl_subch_mask

    summary_rows = []
    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None:
            print(f"  [skip] {case_tag}: not in cohort"); continue
        print(f"\n=== {case_tag} {bone} ===")
        try:
            r_off = process_long_shared_mesh(
                case.seg_00m_path, case.seg_48m_path, case.side, bone, cfg,
                compartment_aware=False,
            )
            r_on = process_long_shared_mesh(
                case.seg_00m_path, case.seg_48m_path, case.side, bone, cfg,
                compartment_aware=True,
            )
        except Exception as e:
            print(f"  [ERR] {e}"); continue

        bone_mesh = r_off["bone_mesh_00m"]
        bone_pts = np.asarray(bone_mesh.points)
        aligned_pts = r_off["aligned_pts_00"]   # patient bone in template space
        th00 = r_off["th00_patient"]
        th48 = r_off["th48_at_00_patient"]

        # Template-NN inherited compartment per patient bone vert (in template space)
        inherited_comp, nn_dist = _inherit_compartment(aligned_pts, tpl)
        in_subch = nn_dist < args.inherit_max_dist_mm
        patient_med_mask = (inherited_comp == 1) & in_subch
        patient_lat_mask = (inherited_comp == 2) & in_subch
        patient_any_subch_mask = patient_med_mask | patient_lat_mask

        # === SHARED patient-frame projection layout ===
        # Define layout from the union of medial+lateral inherited subch — so
        # the same grid bins are used for both contour overlays and the
        # thickness map.
        _, layout_pat = _proj_proportional(
            bone_pts, None, patient_any_subch_mask,
            grid_size=args.grid_size, agg="any")
        if layout_pat is None:
            print("  [skip] no patient subch verts inherited"); continue

        # === SHARED template-frame projection layout ===
        _, layout_tpl = _proj_proportional(
            tpl_pts, None, tpl_any_subch_mask,
            grid_size=args.grid_size, agg="any")

        # Pre-project the template subch contours (same for both template panels)
        g_tpl_med_raw, _ = _proj_proportional(
            tpl_pts, None, tpl_med_mask, agg="any", layout=layout_tpl)
        g_tpl_lat_raw, _ = _proj_proportional(
            tpl_pts, None, tpl_lat_mask, agg="any", layout=layout_tpl)
        g_tpl_med = _close_filled(g_tpl_med_raw, iters=1)
        g_tpl_lat = _close_filled(g_tpl_lat_raw, iters=1)

        # Pre-project the patient subch contours (template-NN inherited)
        g_pat_med_raw, _ = _proj_proportional(
            bone_pts, None, patient_med_mask, agg="any", layout=layout_pat)
        g_pat_lat_raw, _ = _proj_proportional(
            bone_pts, None, patient_lat_mask, agg="any", layout=layout_pat)
        g_pat_med = _close_filled(g_pat_med_raw, iters=1)
        g_pat_lat = _close_filled(g_pat_lat_raw, iters=1)

        for tp in ("tp00", "tp48"):
            th_patient = th00 if tp == "tp00" else th48

            # --- PATIENT frame thickness (proportional projection) ---
            patient_th_mask = (th_patient > 0)
            g_pat_th, _ = _proj_proportional(
                bone_pts, th_patient.astype(np.float64), patient_th_mask,
                agg="mean", layout=layout_pat)

            # --- TEMPLATE thickness (v1.1 baseline) ---
            off_th = r_off[f"{tp}_template_thickness"]
            valid_off = np.isfinite(off_th)
            g_off, _ = _proj_proportional(
                tpl_pts, off_th.astype(np.float64), valid_off,
                agg="mean", layout=layout_tpl)

            # --- TEMPLATE thickness (v1.2 compartment-aware) ---
            on_th = r_on[f"{tp}_template_thickness"]
            valid_on = np.isfinite(on_th)
            g_on, _ = _proj_proportional(
                tpl_pts, on_th.astype(np.float64), valid_on,
                agg="mean", layout=layout_tpl)

            # Per-compartment means
            with np.errstate(invalid="ignore"):
                med_pat_m = float(np.mean(th_patient[patient_med_mask & (th_patient > 0)])) if (patient_med_mask & (th_patient > 0)).any() else float("nan")
                lat_pat_m = float(np.mean(th_patient[patient_lat_mask & (th_patient > 0)])) if (patient_lat_mask & (th_patient > 0)).any() else float("nan")
            med_off = _compartment_mean(off_th, bone, 1)
            lat_off = _compartment_mean(off_th, bone, 2)
            med_on = _compartment_mean(on_th, bone, 1)
            lat_on = _compartment_mean(on_th, bone, 2)

            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            _imshow_with_subch(
                axes[0], g_pat_th, g_pat_med, g_pat_lat,
                f"PATIENT frame {tp}  ({args.thickness_method})\n"
                f"black contour = template-NN inherited subch",
                f"med={med_pat_m:.2f}  lat={lat_pat_m:.2f}  "
                f"[{layout_pat['ml_r']:.0f}mm ML x {layout_pat['ap_r']:.0f}mm AP]",
            )
            _imshow_with_subch(
                axes[1], g_off, g_tpl_med, g_tpl_lat,
                "TEMPLATE v1.1 IDW  (no compartment)\n"
                "black contour = template compartment field",
                f"med={med_off:.3f}  lat={lat_off:.3f}  total={np.nanmean(off_th):.3f}",
            )
            _imshow_with_subch(
                axes[2], g_on, g_tpl_med, g_tpl_lat,
                "TEMPLATE v1.2 compartment-aware\n"
                "black contour = template compartment field",
                f"med={med_on:.3f}  lat={lat_on:.3f}  total={np.nanmean(on_th):.3f}   "
                f"d_lat={lat_on-lat_off:+.3f}  d_med={med_on-med_off:+.3f}",
            )
            fig.suptitle(f"{case_tag}  tibia  {tp}  -  proportional projection  "
                          f"[{args.thickness_method} + {args.subch_method}]",
                          fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = args.out_dir / f"compartment_aware_{case_tag}_{bone}_{tp}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"  [ok] {out}")
            summary_rows.append((case_tag, tp, med_pat_m, lat_pat_m,
                                  med_off, lat_off, med_on, lat_on))

    if summary_rows:
        print("\n=== SUMMARY ===")
        print(f"{'case':>20s}  {'tp':>4s}  {'pat_med':>7s}  {'pat_lat':>7s}  "
              f"{'v1.1_med':>8s}  {'v1.1_lat':>8s}  {'v1.2_med':>8s}  {'v1.2_lat':>8s}  "
              f"{'d_med':>6s}  {'d_lat':>6s}")
        for case_tag, tp, mp, lp, mo, lo, mn, ln in summary_rows:
            print(f"{case_tag:>20s}  {tp:>4s}  {mp:7.3f}  {lp:7.3f}  "
                  f"{mo:8.3f}  {lo:8.3f}  {mn:8.3f}  {ln:8.3f}  "
                  f"{mn-mo:+6.3f}  {ln-lo:+6.3f}")


if __name__ == "__main__":
    main()
