"""
3-panel interactive 3D viewer for the femur DESS template's anatomical subregions.

  Left   — REGION coloring: cMF (medial subch hemisphere) + cLF (lateral subch
           hemisphere) with the central WB band shaded as a saturated overlay.
           Black straight line = best-fit cylinder axis (parallel to ML).
           Orange dots = arc_norm=0 anchor (anterior gap end).
  Middle — arc_norm continuous gradient (HSV) — angular wrap from anterior
           gap end (0) to posterior gap start (1).
  Right  — ml_norm continuous gradient (coolwarm) — medial (0) to lateral (1).

Cameras are linked. Loads the femur template from `TEMPLATE_PATHS["femur"]`.

CLI:
    python -m cartilage_morphometry.viz.viewer_femur_subregions
    python -m cartilage_morphometry.viz.viewer_femur_subregions --no_show
    python -m cartilage_morphometry.viz.viewer_femur_subregions --wb_lo 0.30 --wb_hi 0.70
"""
from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

from cartilage_morphometry import (
    TEMPLATE_PATHS,
    LBL_NON_SUBCH,
    LBL_MED,
    LBL_LAT,
    annotate_femur_template,
    compute_femur_template_subregions,
)


COLOR_NON_SUBCH = (0.85, 0.85, 0.85)
COLOR_MED_PALE  = (0.99, 0.78, 0.74)   # pale red — cMF non-WB tail
COLOR_LAT_PALE  = (0.70, 0.83, 0.91)   # pale blue — cLF non-WB tail
COLOR_MED_WB    = (0.84, 0.18, 0.18)   # solid red — cMF central WB band
COLOR_LAT_WB    = (0.12, 0.47, 0.71)   # solid blue — cLF central WB band


def _line_polydata(pts: np.ndarray) -> pv.PolyData:
    n = len(pts)
    cells = np.hstack([[n], np.arange(n)])
    return pv.PolyData(pts, lines=cells)


def _points_polydata(pts: np.ndarray) -> pv.PolyData:
    return pv.PolyData(pts) if len(pts) else pv.PolyData()


def _extract(mesh: pv.PolyData, mask: np.ndarray) -> pv.PolyData:
    if not mask.any():
        return pv.PolyData()
    return (mesh.extract_points(np.where(mask)[0],
                                adjacent_cells=True, include_cells=True)
            .extract_surface())


def show_femur_subregions(template_mesh, sub):
    annotated = annotate_femur_template(template_mesh, sub)
    pts_all = np.asarray(annotated.points)

    axis_line = _line_polydata(sub.axis_polyline)
    arc_zero_mask = (sub.is_subch & np.isfinite(sub.arc_norm)
                     & (sub.arc_norm <= 0.02))
    arc_zero_pts = _points_polydata(pts_all[arc_zero_mask])

    non_subch_mesh = _extract(annotated, ~sub.is_subch)
    med_pale_mesh  = _extract(annotated, sub.is_subch & (sub.region == LBL_MED) & ~sub.in_wb)
    lat_pale_mesh  = _extract(annotated, sub.is_subch & (sub.region == LBL_LAT) & ~sub.in_wb)
    med_wb_mesh    = _extract(annotated, sub.is_subch & (sub.region == LBL_MED) & sub.in_wb)
    lat_wb_mesh    = _extract(annotated, sub.is_subch & (sub.region == LBL_LAT) & sub.in_wb)

    pl = pv.Plotter(window_size=(1900, 700), shape=(1, 3))
    pl.set_background("white")

    # ---- Panel 0: regions
    pl.subplot(0, 0)
    if non_subch_mesh.n_points:
        pl.add_mesh(non_subch_mesh, color=COLOR_NON_SUBCH, opacity=0.55,
                    smooth_shading=True, label="non-subch (template surface)")
    if med_pale_mesh.n_points:
        pl.add_mesh(med_pale_mesh, color=COLOR_MED_PALE, smooth_shading=True,
                    label="cMF non-WB (anterior + posterior tails)")
    if lat_pale_mesh.n_points:
        pl.add_mesh(lat_pale_mesh, color=COLOR_LAT_PALE, smooth_shading=True,
                    label="cLF non-WB")
    if med_wb_mesh.n_points:
        pl.add_mesh(med_wb_mesh, color=COLOR_MED_WB, smooth_shading=True,
                    label="cMF central WB band (analysis ROI)")
    if lat_wb_mesh.n_points:
        pl.add_mesh(lat_wb_mesh, color=COLOR_LAT_WB, smooth_shading=True,
                    label="cLF central WB band (analysis ROI)")
    if axis_line.n_points >= 2:
        pl.add_mesh(axis_line, color="black", line_width=5,
                    label="angular axis (best-fit circle, parallel to ML)")
    if arc_zero_pts.n_points > 0:
        pl.add_mesh(arc_zero_pts, color="orange", point_size=6,
                    render_points_as_spheres=True,
                    label="arc_norm=0 (anterior gap end)")
    n_med = int((sub.region == LBL_MED).sum())
    n_lat = int((sub.region == LBL_LAT).sum())
    n_med_wb = int(((sub.region == LBL_MED) & sub.in_wb).sum())
    n_lat_wb = int(((sub.region == LBL_LAT) & sub.in_wb).sum())
    pl.add_text(
        f"FEMUR TEMPLATE - anatomical regions\n"
        f"cMF (medial subch): {n_med}  ({n_med_wb} in central WB band, "
        f"{100*n_med_wb/max(n_med,1):.0f}%)\n"
        f"cLF (lateral subch): {n_lat}  ({n_lat_wb} in central WB band, "
        f"{100*n_lat_wb/max(n_lat,1):.0f}%)\n"
        f"WB band = arc_norm in [{sub.wb_band[0]:.2f}, {sub.wb_band[1]:.2f}]\n"
        f"axis = best-fit circle ctr (AP={sub.axis_ap:.1f}, SI={sub.axis_si:.1f})  "
        f"r~={sub.axis_radius:.1f} mm",
        font_size=10, color="black",
    )
    pl.add_legend(bcolor="white", face=None, size=(0.32, 0.18), loc="lower left")
    pl.add_axes()

    # ---- Panel 1: arc_norm
    pl.subplot(0, 1)
    pl.add_mesh(annotated, scalars="arc_norm", cmap="hsv", clim=[0.0, 1.0],
                nan_color="lightgray", nan_opacity=0.40, smooth_shading=True,
                scalar_bar_args={"title": "arc_norm", "title_font_size": 11})
    if axis_line.n_points >= 2:
        pl.add_mesh(axis_line, color="black", line_width=5)
    if arc_zero_pts.n_points > 0:
        pl.add_mesh(arc_zero_pts, color="orange", point_size=6,
                    render_points_as_spheres=True)
    pl.add_text(
        f"angular coord (arc_norm)\n"
        f"0 = anterior gap end -> wraps -> 1 = posterior gap start\n"
        f"theta_start={np.degrees(sub.theta_start):.1f} deg, "
        f"range={np.degrees(sub.theta_range):.1f} deg",
        font_size=10, color="black",
    )
    pl.add_axes()

    # ---- Panel 2: ml_norm
    pl.subplot(0, 2)
    pl.add_mesh(annotated, scalars="ml_norm", cmap="coolwarm", clim=[0.0, 1.0],
                nan_color="lightgray", nan_opacity=0.40, smooth_shading=True,
                scalar_bar_args={"title": "ml_norm", "title_font_size": 11})
    if axis_line.n_points >= 2:
        pl.add_mesh(axis_line, color="black", line_width=5)
    pl.add_text(
        f"ML coord (ml_norm)\n"
        f"0 = medial extreme  ->  1 = lateral extreme\n"
        f"hemispheres split at ml_norm=0.5 (cMF | cLF)",
        font_size=10, color="black",
    )
    pl.add_axes()

    pl.link_views()
    pl.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--wb_lo", type=float, default=0.2)
    ap.add_argument("--wb_hi", type=float, default=0.8)
    args = ap.parse_args()

    template_mesh = pv.read(str(TEMPLATE_PATHS["femur"]))
    sub = compute_femur_template_subregions(
        template_mesh, wb_band=(args.wb_lo, args.wb_hi),
    )
    print(f"  template verts:      {len(sub.region)}")
    print(f"  subch verts:         {int(sub.is_subch.sum())}")
    print(f"  cMF (medial):        {int((sub.region == LBL_MED).sum())}")
    print(f"  cLF (lateral):       {int((sub.region == LBL_LAT).sum())}")
    print(f"  axis (best-fit):     AP={sub.axis_ap:.2f}, SI={sub.axis_si:.2f}, "
          f"r~={sub.axis_radius:.2f}")
    print(f"  theta_start={np.degrees(sub.theta_start):.1f} deg, "
          f"range={np.degrees(sub.theta_range):.1f} deg")
    if args.no_show:
        return
    show_femur_subregions(template_mesh, sub)


if __name__ == "__main__":
    main()
