"""
Femur thickness — 2-panel comparison.

LEFT  : traditional algorithm (PD-vs-DESS canonical) — thickness measured at
        cart-OUTER-SURFACE voxels using `dt_bone = distance_transform_edt(~bone)`,
        projected to a 2D unwrapped femoral condyle map IN THE PATIENT'S OWN frame
        (per-ML-slice angular unwrap, anterior gap anchored).
        No template registration involved. This is what `comparison.thickness_map_2d`
        produces for a single patient.

RIGHT : same patient's thickness mapped onto the bone+subchondral template via
        `process_one_patient`, then projected to 2D in the TEMPLATE's frame
        (also angular unwrap; same colourmap so values are directly comparable).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, distance_transform_edt

from cartilage_morphometry import PD_LABELS, TEMPLATE_PATHS, process_one_patient
from cartilage_morphometry.pipeline import get_patient_masks
from cartilage_morphometry.projection_2d import (
    patient_thickness_2d_femur,
    template_thickness_2d_femur,
)


def _label_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    kw = dict(fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.2, -0.06, "← Med", ha="center", va="top", **kw)
    ax.text(0.8, -0.06, "Lat →", ha="center", va="top", **kw)
    ax.text(-0.06, 0.85, "Ant ↑", ha="right", va="center", **kw)
    ax.text(-0.06, 0.15, "Post ↓", ha="right", va="center", **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_path", required=True, type=Path)
    p.add_argument("--side", choices=["RIGHT", "LEFT"], required=True)
    p.add_argument("--modality", choices=["pd", "dess"], required=True)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--out_png", type=Path, default=None)
    args = p.parse_args()

    bone = "femur"
    print(f"=== {bone} ({args.side}, {args.modality}) ===")

    # LEFT panel — traditional patient-frame thickness 2D
    print("[traditional] loading + densify + cart-surface thickness 2D...")
    bone_mask, cart_mask, spacing, seg_native, spacing_native = get_patient_masks(
        args.seg_path, args.side, args.modality, bone,
    )
    grid_pat, ml_pat, arc_pat = patient_thickness_2d_femur(
        bone_mask, cart_mask, spacing, grid_size=args.grid,
    )
    finite_pat = grid_pat[np.isfinite(grid_pat)]
    print(f"  patient 2D: ML={ml_pat:.0f} mm, arc={arc_pat:.0f} mm, "
          f"mean={np.nanmean(grid_pat):.2f}, max={np.nanmax(grid_pat):.2f}")

    # RIGHT panel — template projection
    print("[template] full pipeline...")
    res = process_one_patient(
        seg_path=args.seg_path, side=args.side, bone_name=bone,
        modality=args.modality, out_dir=None,
    )
    template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
    grid_tpl, ml_tpl, arc_tpl = template_thickness_2d_femur(
        template_mesh, res["template_thickness"], grid_size=args.grid,
    )
    finite_tpl = grid_tpl[np.isfinite(grid_tpl)]
    print(f"  template 2D: ML={ml_tpl:.0f} mm, arc={arc_tpl:.0f} mm, "
          f"mean={np.nanmean(grid_tpl):.2f}, max={np.nanmax(grid_tpl):.2f}")

    # Shared color scale
    all_t = np.concatenate([finite_pat, finite_tpl])
    vmax = float(np.percentile(all_t, 98)) if len(all_t) else 4.0
    vmax = max(vmax, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    im = ax.imshow(grid_pat, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
                   extent=[0, ml_pat, arc_pat, 0], aspect="equal",
                   interpolation="bilinear")
    _label_axes(ax)
    ax.set_title(
        f"LEFT — Patient-frame (traditional cart-surface algorithm)\n"
        f"{ml_pat:.0f} × {arc_pat:.0f} mm | mean = {np.nanmean(grid_pat):.2f} mm",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="thickness (mm)")

    ax = axes[1]
    im2 = ax.imshow(grid_tpl, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
                    extent=[0, ml_tpl, arc_tpl, 0], aspect="equal",
                    interpolation="bilinear")
    _label_axes(ax)
    ax.set_title(
        f"RIGHT — Template-projected\n"
        f"{ml_tpl:.0f} × {arc_tpl:.0f} mm | mean = {np.nanmean(grid_tpl):.2f} mm",
        fontsize=11,
    )
    plt.colorbar(im2, ax=ax, fraction=0.04, pad=0.02, label="thickness (mm)")

    fig.suptitle(
        f"Femur cartilage thickness — patient {args.seg_path.stem}  "
        f"({args.side}, {args.modality})",
        fontsize=12,
    )
    plt.tight_layout()
    if args.out_png:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
        print(f"saved {args.out_png}")
    plt.show()


if __name__ == "__main__":
    main()
