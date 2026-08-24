"""Visualize what's INCLUDED in each subch_method on the patient 2D map.

Per case + bone, produces a 1×4 figure showing:
  [thickness>0]   [tpl_nn subch]   [cart_prox subch]   [tpl_nn vs cart_prox diff]

All four panels are 2D-projected patient bone in patient frame (no template).
Each shows which patient bone verts get included in the IDW pool that feeds
the template-remap step. If the subch pool extends FAR BEYOND the
thickness>0 region, those extra verts push 0 into the template and dilute
the mean.

Also prints a per-case value-distribution comparison:
  patient_mean_subch_only  vs  template_remap_mean_subch_only
to show how much dilution each method introduces.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartilage_morphometry import PipelineConfig, get_subch
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort
from cartilage_morphometry.validation.shared_mesh import (
    _build_thickness, _force_ml_flip, _ml_flip_decision,
)
from scripts.diagnose_long_patient_2d import _project_tibia_patient


WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _project_mask_tibia(bone_mesh, mask: np.ndarray, grid_size: int = 80):
    """Project a binary mask onto the tibia 2D grid (any cell touched by a
    masked vert → 1, else NaN). Same convention as
    `_project_tibia_patient`."""
    pts = np.asarray(bone_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    sub = mask.astype(bool)
    if not sub.any():
        return np.full((grid_size, grid_size), np.nan)
    ml_lo, ml_hi = ML[sub].min(), ML[sub].max()
    midline = 0.5 * (ml_lo + ml_hi)
    out = np.full((grid_size, grid_size), np.nan)
    for half_lo, half_hi, ml_min, ml_max in (
        (0.0, 0.5, ml_lo, midline),
        (0.5, 1.0, midline, ml_hi),
    ):
        m = sub & (ML >= ml_min) & (ML <= ml_max)
        if not m.any(): continue
        ml_r = max(ml_max - ml_min, 1e-6)
        ap_min, ap_max = AP[m].min(), AP[m].max()
        ap_r = max(ap_max - ap_min, 1e-6)
        ml_n = half_lo + (ML[m] - ml_min) / ml_r * (half_hi - half_lo)
        ap_n = (AP[m] - ap_min) / ap_r
        ml_b = np.clip((ml_n * grid_size).astype(int), 0, grid_size - 1)
        ap_b = np.clip((ap_n * grid_size).astype(int), 0, grid_size - 1)
        out[ap_b, ml_b] = 1.0
    return out


def _render(case_tag: str, bone_mesh, thickness, mask_tpl_nn, mask_cart_prox,
            out_png: Path, grid: int = 80, vmax_thick: float = 4.0):
    g_th = _project_tibia_patient(bone_mesh, thickness, grid)
    g_tpl = _project_mask_tibia(bone_mesh, mask_tpl_nn, grid)
    g_cp = _project_mask_tibia(bone_mesh, mask_cart_prox, grid)
    only_tpl = (np.isfinite(g_tpl) & ~np.isfinite(g_cp)).astype(float)
    only_cp = (np.isfinite(g_cp) & ~np.isfinite(g_tpl)).astype(float)
    both = (np.isfinite(g_tpl) & np.isfinite(g_cp)).astype(float)
    diff_rgb = np.zeros((grid, grid, 3))
    diff_rgb[..., 0] = only_tpl                   # red: only in tpl_nn
    diff_rgb[..., 2] = only_cp                    # blue: only in cart_prox
    diff_rgb[..., 1] = both * 0.7                 # green: both
    bg = np.isnan(g_tpl) & np.isnan(g_cp)
    diff_rgb[bg] = 1.0  # white background

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # Panel 1: thickness > 0
    im0 = axes[0].imshow(g_th, origin="upper", cmap="jet_r",
                          vmin=0.0, vmax=vmax_thick, interpolation="nearest")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02).set_label("thickness (mm)", fontsize=8)
    n_th = int(np.sum(thickness > 0))
    axes[0].set_title(f"thickness>0\n(raycast hits)  n={n_th}", fontsize=10)
    # Panel 2: tpl_nn subch
    im1 = axes[1].imshow(g_tpl, origin="upper", cmap="Greys",
                          vmin=0, vmax=1, interpolation="nearest")
    n_tpl = int(mask_tpl_nn.sum())
    axes[1].set_title(f"tpl_nn subch\n(template-NN ≥ 0.5)  n={n_tpl}", fontsize=10)
    # Panel 3: cart_prox subch
    im2 = axes[2].imshow(g_cp, origin="upper", cmap="Greys",
                          vmin=0, vmax=1, interpolation="nearest")
    n_cp = int(mask_cart_prox.sum())
    axes[2].set_title(f"cart_prox subch\n(within 5mm of cart_mask)  n={n_cp}", fontsize=10)
    # Panel 4: diff red/blue/green
    axes[3].imshow(diff_rgb, origin="upper", interpolation="nearest")
    axes[3].set_title("diff: red=only tpl_nn,\nblue=only cart_prox, green=both", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                                  (0.98, 0.97, "L", "right"),
                                  (0.5, 0.97, "P", "center"),
                                  (0.5, 0.03, "A", "center")):
            ax.text(tx, ty, label, transform=ax.transAxes, color="white" if ax in (axes[0], axes[1], axes[2]) else "black",
                    fontsize=9, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                    bbox=dict(facecolor="black" if ax in (axes[0], axes[1], axes[2]) else "white",
                              alpha=0.5, edgecolor="none", pad=2))
    fig.suptitle(f"{case_tag} tibia 00m — patient verts: thickness vs subch_method pool",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=WORST_PMT_TIBIA)
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/diagnose_subch_mask"))
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Need to run the full registration to get aligned_pts for tpl_nn classification.
    # Load the template tibia mesh for tpl_nn subch.
    from cartilage_morphometry import TEMPLATE_PATHS
    from cartilage_morphometry.pipeline import patient_to_template_align
    import pyvista as pv

    template_mesh = pv.read(str(TEMPLATE_PATHS["tibia"]))

    print(f"{'case':<22s} {'thick>0':>8s} {'tpl_nn':>8s} {'cart_prox':>10s} "
          f"{'patient_mean':>13s} {'tpl_nn_pool_mean':>17s} {'cart_prox_pool_mean':>20s}")
    for case_tag in args.cases:
        case = cohort.get(case_tag)
        if case is None: continue
        _pipeline.clear_recon_cache()
        flip = _ml_flip_decision(case.seg_00m_path)
        cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn")
        try:
            with _force_ml_flip(flip):
                m00, b00, c00, sp00, th00 = _build_thickness(
                    case.seg_00m_path, case.side, "pd", "tibia", cfg)
        except Exception as e:
            print(f"[ERR {case_tag}] {e}"); continue
        # Align to template (needed for tpl_nn)
        aligned, *_ = patient_to_template_align(m00, c00, sp00, template_mesh, "tibia", modality="pd")
        # Classify subch by each method
        sub_tpl_nn_fn = get_subch("tpl_nn")
        sp_tpl, is_tpl = sub_tpl_nn_fn(aligned, template_mesh, c00, sp00, m00, th00, cfg)
        sub_cp_fn = get_subch("cart_prox")
        cfg_cp = PipelineConfig(thickness_method="raycast", subch_method="cart_prox", cart_prox_mm=5.0)
        sp_cp, is_cp = sub_cp_fn(aligned, template_mesh, c00, sp00, m00, th00, cfg_cp)
        # Pool means: average of thickness OVER each subch pool
        patient_mean = float(th00[th00 > 0].mean())
        tpl_pool_mean = float(th00[is_tpl.astype(bool)].mean())
        cp_pool_mean = float(th00[is_cp.astype(bool)].mean())
        print(f"{case_tag:<22s} {int((th00>0).sum()):>8d} {int(is_tpl.sum()):>8d} "
              f"{int(is_cp.sum()):>10d} {patient_mean:>13.3f} {tpl_pool_mean:>17.3f} "
              f"{cp_pool_mean:>20.3f}")
        out = args.out_dir / f"subch_mask_{case_tag}_tibia.png"
        _render(case_tag, m00, th00, is_tpl.astype(bool), is_cp.astype(bool), out)
        print(f"  [ok] {out}")


if __name__ == "__main__":
    main()
