"""
2-panel interactive viewer:
  Left  — patient bone in PHYSICAL coords, colored by per-vertex thickness_mm
  Right — TEMPLATE bone, colored by remapped thickness at fixed anatomical locations

Runs the full pipeline (`process_one_patient`) and renders the two outputs
linked into one pyvista window with synchronized cameras.
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from cartilage_morphometry import TEMPLATE_PATHS, process_one_patient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_path", required=True, type=Path)
    p.add_argument("--side", choices=["RIGHT", "LEFT"], required=True)
    p.add_argument("--bone", choices=["femur", "tibia"], default="tibia")
    p.add_argument("--modality", choices=["pd", "dess"], required=True)
    p.add_argument("--max_thickness_mm", type=float, default=6.0)
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()

    print(f"\n=== {args.bone} ({args.side}, {args.modality}) ===")
    result = process_one_patient(
        seg_path=args.seg_path,
        side=args.side,
        bone_name=args.bone,
        modality=args.modality,
        max_thickness_mm=args.max_thickness_mm,
        out_dir=args.out_dir,
    )
    bone_mesh = result["bone_mesh"]
    template_thickness = result["template_thickness"]

    template_mesh = pv.read(str(TEMPLATE_PATHS[args.bone]))
    template_mesh["thickness_mm"] = template_thickness

    # Color limits common to both panels
    p_thick = bone_mesh["thickness_mm"]
    finite_t = template_thickness[np.isfinite(template_thickness)]
    vmax = float(max(p_thick.max(), finite_t.max() if len(finite_t) else 0.0))
    vmax = max(vmax, 0.5)

    pl = pv.Plotter(window_size=(1500, 800), shape=(1, 2))
    pl.set_background("white")

    # Left — patient bone (physical coords)
    pl.subplot(0, 0)
    pl.add_mesh(
        bone_mesh, scalars="thickness_mm", cmap="jet_r",
        clim=[0, vmax], smooth_shading=True,
        scalar_bar_args={"title": "thickness (mm)", "title_font_size": 14},
    )
    pl.add_text(
        f"Patient bone (raw, physical mm)\n"
        f"{Path(args.seg_path).stem} · {args.bone} · {args.side}\n"
        f"verts with cartilage > 0: {int((p_thick > 0).sum())}/{bone_mesh.n_points}\n"
        f"mean(>0) = {p_thick[p_thick > 0].mean():.2f} mm",
        font_size=10, color="black",
    )
    pl.add_axes()

    # Right — template bone (mapped thickness); NaN rendered as gray automatically
    pl.subplot(0, 1)
    template_mesh["thickness_mm"] = template_thickness.astype(np.float32)
    pl.add_mesh(
        template_mesh, scalars="thickness_mm", cmap="jet_r",
        clim=[0, vmax], nan_color="lightgray", nan_opacity=0.45,
        smooth_shading=True,
        scalar_bar_args={"title": "thickness (mm)", "title_font_size": 14},
    )
    n_finite = int(np.isfinite(template_thickness).sum())
    pl.add_text(
        f"Template bone (mapped thickness)\n"
        f"{args.bone} template · {template_mesh.n_points} verts\n"
        f"valid (subch) = {n_finite}, "
        f"mean = {np.nanmean(template_thickness):.2f} mm",
        font_size=10, color="black",
    )
    pl.add_axes()

    pl.link_views()
    pl.show()


if __name__ == "__main__":
    main()
