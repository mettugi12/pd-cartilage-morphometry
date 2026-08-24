"""
3-panel interactive 3D viewer for one patient (femur or tibia).

  Left   — patient bone, RAW per-vertex thickness (everywhere it was found,
           including outside the template's strict subchondral zone)
  Middle — patient bone, thickness MASKED to the template-defined subch region
           (gray outside the subch zone). This is exactly what gets remapped
           to the template — directly comparable to the right panel.
  Right  — TEMPLATE bone, mapped thickness via the IDW remap (zero where the
           patient had no cart inside the subch zone; gray non-subch).

Same colormap and `clim` across all three panels so values are directly comparable.
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
    p.add_argument("--bone", choices=["femur", "tibia"], required=True)
    p.add_argument("--modality", choices=["pd", "dess"], required=True)
    p.add_argument("--max_thickness_mm", type=float, default=6.0)
    p.add_argument("--subch_threshold", type=float, default=0.5)
    args = p.parse_args()

    res = process_one_patient(
        seg_path=args.seg_path, side=args.side, bone_name=args.bone,
        modality=args.modality, max_thickness_mm=args.max_thickness_mm,
        out_dir=None,
    )
    bone_mesh = res["bone_mesh"]              # patient (physical mm)
    template_thickness = res["template_thickness"]   # remapped
    template_mesh = pv.read(str(TEMPLATE_PATHS[args.bone]))

    # Color scale: percentile across both raw patient + template
    raw = np.asarray(bone_mesh["thickness_mm"], dtype=np.float32)
    finite_t = template_thickness[np.isfinite(template_thickness)]
    pool = np.concatenate(
        [raw[raw > 0], finite_t if len(finite_t) else np.array([0.0])]
    )
    vmax = float(np.percentile(pool, 98)) if len(pool) else 4.0
    vmax = max(vmax, 0.5)

    # Panel 1: raw thickness everywhere
    raw_for_render = np.where(raw > 0, raw, np.nan).astype(np.float32)
    bone_raw = bone_mesh.copy()
    bone_raw["thickness_raw"] = raw_for_render

    # Panel 2: thickness masked to template-defined subch region on PATIENT
    subch_prob_p = np.asarray(bone_mesh["subch_prob"], dtype=np.float32)
    in_subch = subch_prob_p >= args.subch_threshold
    masked = np.where(in_subch, raw, np.nan).astype(np.float32)
    bone_masked = bone_mesh.copy()
    bone_masked["thickness_in_subch"] = masked
    n_in_subch = int(in_subch.sum())
    n_in_subch_with_cart = int(((raw > 0) & in_subch).sum())
    n_raw_with_cart = int((raw > 0).sum())

    # Panel 3: template with mapped thickness (NaN outside subch)
    template_mesh["thickness_mm"] = template_thickness.astype(np.float32)
    n_template_subch = int(np.isfinite(template_thickness).sum())

    # Coverage stats
    print(f"\n[{args.bone}] vertex counts:")
    print(f"  raw cart-bearing patient verts (thickness>0): {n_raw_with_cart}")
    print(f"  patient verts in template subch zone:        {n_in_subch}")
    print(f"  patient verts in subch AND with cart:        {n_in_subch_with_cart} "
          f"(zero in subch: {n_in_subch - n_in_subch_with_cart})")
    print(f"  template subch verts (mapped thickness):     {n_template_subch}")

    pl = pv.Plotter(window_size=(1800, 700), shape=(1, 3))
    pl.set_background("white")

    cmap = "jet_r"
    sb_kw = {"title": "thickness (mm)", "title_font_size": 12}

    # Panel 0 — raw on patient
    pl.subplot(0, 0)
    pl.add_mesh(
        bone_raw, scalars="thickness_raw", cmap=cmap,
        clim=[0, vmax], nan_color="lightgray", nan_opacity=0.40,
        smooth_shading=True, scalar_bar_args=sb_kw,
    )
    pl.add_text(
        f"PATIENT bone — RAW thickness\n"
        f"verts with cart: {n_raw_with_cart}/{bone_mesh.n_points}\n"
        f"mean = {raw[raw > 0].mean():.2f} mm  max = {raw.max():.2f} mm",
        font_size=10, color="black",
    )
    pl.add_axes()

    # Panel 1 — masked to template subch region
    pl.subplot(0, 1)
    pl.add_mesh(
        bone_masked, scalars="thickness_in_subch", cmap=cmap,
        clim=[0, vmax], nan_color="lightgray", nan_opacity=0.40,
        smooth_shading=True, scalar_bar_args=sb_kw,
    )
    masked_finite = masked[np.isfinite(masked)]
    pl.add_text(
        f"PATIENT bone — MASKED to template subch\n"
        f"verts in subch zone: {n_in_subch} "
        f"({n_in_subch_with_cart} with cart, "
        f"{n_in_subch - n_in_subch_with_cart} bare)\n"
        f"mean = {masked_finite.mean():.2f} mm",
        font_size=10, color="black",
    )
    pl.add_axes()

    # Panel 2 — template with remapped thickness
    pl.subplot(0, 2)
    pl.add_mesh(
        template_mesh, scalars="thickness_mm", cmap=cmap,
        clim=[0, vmax], nan_color="lightgray", nan_opacity=0.40,
        smooth_shading=True, scalar_bar_args=sb_kw,
    )
    pl.add_text(
        f"TEMPLATE bone — mapped thickness\n"
        f"valid verts: {n_template_subch}/{template_mesh.n_points}\n"
        f"mean = {np.nanmean(template_thickness):.2f} mm  "
        f"max = {np.nanmax(template_thickness):.2f} mm",
        font_size=10, color="black",
    )
    pl.add_axes()

    pl.show()


if __name__ == "__main__":
    main()
