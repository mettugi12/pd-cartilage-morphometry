"""Patient-frame 2D projection diagnostic — 00m, 48m, Δ on the SAME 2D grid
extracted directly from the 00m bone mesh in physical mm.

No template, no IDW-to-template. The 48m thickness has already been
IDW-sampled onto 00m bone-verts via the shared-mesh wrapper. We project the
two scalar arrays onto a 2D grid using the same patient-frame convention as
the deliverable's `_patient_tibia_2d_grid`.

For each case + bone, produces a 2×3 figure:
  Row 1 (FINE grid, e.g. 80 cells/side):   00m  |  48m  |  Δ (per-cell)
  Row 2 (COARSE grid, e.g. 20 cells/side): 00m  |  48m  |  Δ (drift-averaged)

The coarse Δ in the bottom-right panel averages many bone-verts per cell,
absorbing the small parallel drift / per-vertex IDW noise that creates
mixed red/blue speckle in the fine-grid Δ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartilage_morphometry import PipelineConfig
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import process_long_shared_mesh
from cartilage_morphometry.validation.viz import _smooth_border_preserving


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


# ---------------------------------------------------------------------------
# Patient-frame projections (ported from cartilage-v1-snuh-test/v1.0/analysis
# /render_case_variant.py — _patient_tibia_2d_grid + _patient_femur_2d_grid)
# ---------------------------------------------------------------------------
def _project_tibia_patient(bone_mesh, thickness, grid_size: int = 80):
    """Patient-frame tibia 2D grid: ML midline split + AP normalize per half.

    Convention matches the library's `template_thickness_2d_tibia`:
        rows  = AP (row 0 = anterior, row N = posterior)
        cols 0..N/2-1     = medial half
        cols N/2..N-1     = lateral half
    Returns the mean thickness per cell, NaN where no vert projected.
    The subch mask used = `thickness > 0` (per-knee patient subch).
    """
    pts = np.asarray(bone_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    th = np.asarray(thickness, dtype=np.float64)
    sub = th > 0
    if not sub.any():
        return np.full((grid_size, grid_size), np.nan)
    ml_lo, ml_hi = ML[sub].min(), ML[sub].max()
    midline = 0.5 * (ml_lo + ml_hi)
    sum_g = np.zeros((grid_size, grid_size), dtype=np.float64)
    cnt_g = np.zeros((grid_size, grid_size), dtype=np.int64)
    for half_lo, half_hi, ml_min, ml_max in (
        (0.0, 0.5, ml_lo, midline),
        (0.5, 1.0, midline, ml_hi),
    ):
        m = sub & (ML >= ml_min) & (ML <= ml_max)
        if not m.any():
            continue
        ml_r = max(ml_max - ml_min, 1e-6)
        ap_min, ap_max = AP[m].min(), AP[m].max()
        ap_r = max(ap_max - ap_min, 1e-6)
        ml_n = half_lo + (ML[m] - ml_min) / ml_r * (half_hi - half_lo)
        ap_n = (AP[m] - ap_min) / ap_r
        ml_b = np.clip((ml_n * grid_size).astype(int), 0, grid_size - 1)
        ap_b = np.clip((ap_n * grid_size).astype(int), 0, grid_size - 1)
        np.add.at(sum_g, (ap_b, ml_b), th[m])
        np.add.at(cnt_g, (ap_b, ml_b), 1)
    return np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)


def _project_femur_patient(bone_mesh, thickness, grid_size: int = 80):
    """Patient-frame femur 2D grid: best-fit circle in (AP, SI) plane → arc
    angle × ML normalize. Same convention as the deliverable's
    `_patient_femur_2d_grid`.
    """
    pts = np.asarray(bone_mesh.points)
    th = np.asarray(thickness, dtype=np.float64)
    sub = th > 0
    if sub.sum() < 100:
        return np.full((grid_size, grid_size), np.nan)
    AP = pts[sub, 0]; SI = pts[sub, 1]; ML = pts[sub, 2]
    thk = th[sub]
    # Best-fit circle in (AP, SI): A·x + B·y + C = x² + y²
    A = np.c_[2 * AP, 2 * SI, np.ones_like(AP)]
    b = AP * AP + SI * SI
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ap_c, si_c = float(sol[0]), float(sol[1])
    theta = np.arctan2(SI - si_c, AP - ap_c)
    theta = (theta + np.pi) % (2 * np.pi)
    # Anchor θ=0 at largest empty arc
    nbins = 180
    hist, edges = np.histogram(theta, bins=nbins, range=(0, 2 * np.pi))
    empty = hist == 0
    if empty.any():
        e2 = np.tile(empty.astype(np.int8), 2)
        d2 = np.diff(e2, prepend=0)
        rs = np.where(d2 == 1)[0]
        re = np.where(d2 == -1)[0]
        if len(rs) > len(re):
            re = np.append(re, len(e2))
        runs = re - rs
        if len(runs):
            best = int(np.argmax(runs))
            theta_start = float(edges[(rs[best] + runs[best]) % nbins])
            theta = (theta - theta_start) % (2 * np.pi)
    theta_range = max(float(theta.max()), 0.01)
    arc_norm = theta / theta_range
    ml_min, ml_max = ML.min(), ML.max()
    ml_norm = (ML - ml_min) / max(ml_max - ml_min, 1e-6)
    ap_b = np.clip(((1.0 - arc_norm) * grid_size).astype(int), 0, grid_size - 1)
    ml_b = np.clip((ml_norm * grid_size).astype(int), 0, grid_size - 1)
    sum_g = np.zeros((grid_size, grid_size), dtype=np.float64)
    cnt_g = np.zeros((grid_size, grid_size), dtype=np.int64)
    np.add.at(sum_g, (ap_b, ml_b), thk)
    np.add.at(cnt_g, (ap_b, ml_b), 1)
    return np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)


def _project(bone_mesh, thickness, bone: str, grid_size: int):
    return (_project_tibia_patient if bone == "tibia"
            else _project_femur_patient)(bone_mesh, thickness, grid_size)


def _decorate_2d(ax):
    """Medial-left convention for both bones (matches our other 2D viz)."""
    ax.set_xticks([]); ax.set_yticks([])
    for tx, ty, label, ha in ((0.02, 0.97, "medial", "left"),
                              (0.98, 0.97, "lateral", "right"),
                              (0.5, 0.97, "posterior", "center"),
                              (0.5, 0.03, "anterior", "center")):
        ax.text(tx, ty, label, transform=ax.transAxes, color="white",
                fontsize=8, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))


def _render_case(case_tag: str, bone_name: str, res: dict, out_png: Path,
                 fine: int = 80, coarse: int = 20,
                 vmax_thick: float = 4.0, vabs_delta: float = 1.5,
                 smooth_sigma_fine: float = 1.0,
                 smooth_sigma_coarse: float = 0.0):
    """2×3 patient-frame figure:
        row1 (fine grid):    00m  |  48m  |  Δ
        row2 (coarse grid):  00m  |  48m  |  Δ (drift-averaged)
    """
    m00 = res["bone_mesh_00m"]
    th00 = res["th00_patient"]
    th48 = res["th48_at_00_patient"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for r, (gsz, sigma, label) in enumerate(((fine, smooth_sigma_fine, f"FINE grid={fine}"),
                                              (coarse, smooth_sigma_coarse, f"COARSE grid={coarse}"))):
        g_00 = _project(m00, th00, bone_name, gsz)
        g_48 = _project(m00, th48, bone_name, gsz)
        if sigma > 0:
            g_00 = _smooth_border_preserving(g_00, sigma)
            g_48 = _smooth_border_preserving(g_48, sigma)
        g_d = g_48 - g_00
        for c, (g, title, cmap, vmin, vmax, cbar_lbl) in enumerate((
            (g_00, f"00m  ({label})",  "jet_r",    0.0, vmax_thick, "thickness (mm)"),
            (g_48, f"48m  ({label})",  "jet_r",    0.0, vmax_thick, "thickness (mm)"),
            (g_d,  f"Δ (48m−00m)",      "RdBu", -vabs_delta, vabs_delta, "Δ thickness (mm)"),
        )):
            ax = axes[r, c]
            im = ax.imshow(g, origin="upper", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(cbar_lbl, fontsize=9)
            ax.set_title(title, fontsize=11)
            _decorate_2d(ax)
            with np.errstate(invalid="ignore"):
                if c == 2:
                    mean = float(np.nanmean(g))
                    n_red = int(np.sum((g > 0.1) & np.isfinite(g)))
                    n_blue = int(np.sum((g < -0.1) & np.isfinite(g)))
                    n_tot = int(np.sum(np.isfinite(g)))
                    ax.text(0.5, -0.06,
                            f"mean Δ={mean:+.3f}mm  thickening>0.1mm: {n_red}/{n_tot} cells  "
                            f"thinning<-0.1mm: {n_blue}/{n_tot} cells",
                            transform=ax.transAxes, fontsize=9, va="top", ha="center")
                else:
                    ax.text(0.5, -0.06, f"mean = {float(np.nanmean(g)):.3f} mm",
                            transform=ax.transAxes, fontsize=9, va="top", ha="center")
    fig.suptitle(f"{case_tag}  {bone_name}  — patient-frame 2D projection (no template)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--bones", nargs="+", choices=["femur", "tibia"], default=["tibia"])
    p.add_argument("--fine", type=int, default=80)
    p.add_argument("--coarse", type=int, default=20)
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_long_patient_2d"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    print(f"[recon] disk cache: {DEFAULT_RECON_CACHE_DIR}")

    config = PipelineConfig(thickness_method="raycast")
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None:
            print(f"[skip] {case_tag} not in cohort"); continue
        for bone in args.bones:
            print(f"\n=== {case_tag} {bone} ===")
            _pipeline.clear_recon_cache()
            try:
                res = process_long_shared_mesh(
                    case.seg_00m_path, case.seg_48m_path,
                    side=case.side, bone_name=bone, config=config,
                )
            except Exception as e:
                print(f"  [ERR] {e}"); continue
            out = args.out_dir / f"patient2d_{case_tag}_{bone}.png"
            _render_case(case_tag, bone, res, out,
                         fine=args.fine, coarse=args.coarse)
            print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
