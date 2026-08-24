"""
3-panel interactive 3D viewer for the tibia DESS template's anatomical subregions.

  Left   — REGION coloring: MT (medial plate, solid red) + LT (lateral plate,
           solid blue). MT and LT are the PRIMARY analysis ROIs (whole-plate,
           matching Eckstein WMTMTH / WLTMTH). White dashed contours mark the
           AP-third boundaries (ap_norm = 1/3 and 2/3) for the optional
           aMT/cMT/pMT / aLT/cLT/pLT sub-region analysis.
  Middle — ap_norm continuous gradient (viridis) — anterior (0) to posterior (1).
  Right  — ml_norm continuous gradient (coolwarm) — medial (0) to lateral (1).

CLI:
    python -m cartilage_morphometry.viz.viewer_tibia_subregions
    python -m cartilage_morphometry.viz.viewer_tibia_subregions --no_show
"""
from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv

from cartilage_morphometry import (
    TEMPLATE_PATHS,
    LBL_NON_SUBCH,
    LBL_MED,
    LBL_LAT,
    LBL_AP_ANTERIOR,
    LBL_AP_CENTRAL,
    LBL_AP_POSTERIOR,
    annotate_tibia_template,
    compute_tibia_template_subregions,
)


COLOR_NON_SUBCH = (0.85, 0.85, 0.85)
COLOR_MT        = (0.84, 0.18, 0.18)   # solid red — whole medial plate (MT)
COLOR_LT        = (0.12, 0.47, 0.71)   # solid blue — whole lateral plate (LT)


def _extract(mesh: pv.PolyData, mask: np.ndarray) -> pv.PolyData:
    if not mask.any():
        return pv.PolyData()
    return (mesh.extract_points(np.where(mask)[0],
                                adjacent_cells=True, include_cells=True)
            .extract_surface())


def _build_third_boundary_curves(sub: TibiaSubregions, template_mesh: pv.PolyData,
                                 tol: float = 0.012) -> pv.PolyData:
    """Return a point cloud at ap_norm ~= 1/3 and ap_norm ~= 2/3 (per compartment)
    so the AP-third boundaries can be drawn on the regions panel as a contour."""
    pts = np.asarray(template_mesh.points)
    near_third = (sub.is_subch
                  & np.isfinite(sub.ap_norm)
                  & ((np.abs(sub.ap_norm - 1.0 / 3.0) < tol)
                     | (np.abs(sub.ap_norm - 2.0 / 3.0) < tol)))
    if not near_third.any():
        return pv.PolyData()
    return pv.PolyData(pts[near_third])


def show_tibia_subregions(template_mesh, sub):
    annotated = annotate_tibia_template(template_mesh, sub)

    non_subch = _extract(annotated, ~sub.is_subch)
    mt_mesh = _extract(annotated, sub.is_subch & (sub.region == LBL_MED))
    lt_mesh = _extract(annotated, sub.is_subch & (sub.region == LBL_LAT))
    third_curves = _build_third_boundary_curves(sub, template_mesh)

    pl = pv.Plotter(window_size=(1900, 700), shape=(1, 3))
    pl.set_background("white")

    # ---- Panel 0: regions
    pl.subplot(0, 0)
    if non_subch.n_points:
        pl.add_mesh(non_subch, color=COLOR_NON_SUBCH, opacity=0.55,
                    smooth_shading=True, label="non-subch")
    if mt_mesh.n_points:
        pl.add_mesh(mt_mesh, color=COLOR_MT, smooth_shading=True,
                    label="MT — whole medial plate (Eckstein WMTMTH)")
    if lt_mesh.n_points:
        pl.add_mesh(lt_mesh, color=COLOR_LT, smooth_shading=True,
                    label="LT — whole lateral plate (Eckstein WLTMTH)")
    if third_curves.n_points:
        pl.add_mesh(third_curves, color="white", point_size=4,
                    render_points_as_spheres=True,
                    label="AP-third boundaries (ap_norm = 1/3, 2/3)")

    n_mt = int((sub.region == LBL_MED).sum())
    n_lt = int((sub.region == LBL_LAT).sum())
    n_cmt = int(((sub.region == LBL_MED) & (sub.ap_third == LBL_AP_CENTRAL)).sum())
    n_clt = int(((sub.region == LBL_LAT) & (sub.ap_third == LBL_AP_CENTRAL)).sum())
    pl.add_text(
        f"TIBIA TEMPLATE - anatomical regions\n"
        f"MT (whole medial, ALL of red): {n_mt}  -- Eckstein WMTMTH\n"
        f"LT (whole lateral, ALL of blue): {n_lt}  -- Eckstein WLTMTH\n"
        f"white dots = AP thirds boundary\n"
        f"sub-regions: aMT/cMT/pMT (cMT n={n_cmt}), aLT/cLT/pLT (cLT n={n_clt})",
        font_size=10, color="black",
    )
    pl.add_legend(bcolor="white", face=None, size=(0.34, 0.18), loc="lower left")
    pl.add_axes()

    # ---- Panel 1: ap_norm
    pl.subplot(0, 1)
    pl.add_mesh(annotated, scalars="ap_norm", cmap="viridis", clim=[0.0, 1.0],
                nan_color="lightgray", nan_opacity=0.40, smooth_shading=True,
                scalar_bar_args={"title": "ap_norm", "title_font_size": 11})
    pl.add_text(
        "AP coord (per-compartment ap_norm)\n"
        "0 = anterior, 1 = posterior\n"
        "thirds: [0,1/3) anterior, [1/3,2/3) central, [2/3,1] posterior",
        font_size=10, color="black",
    )
    pl.add_axes()

    # ---- Panel 2: ml_norm
    pl.subplot(0, 2)
    pl.add_mesh(annotated, scalars="ml_norm", cmap="coolwarm", clim=[0.0, 1.0],
                nan_color="lightgray", nan_opacity=0.40, smooth_shading=True,
                scalar_bar_args={"title": "ml_norm", "title_font_size": 11})
    pl.add_text(
        "ML coord (per-compartment ml_norm)\n"
        "[0, 0.5] = medial half, [0.5, 1.0] = lateral half\n"
        "(intercondylar gap between ml_norm=0.5 in each)",
        font_size=10, color="black",
    )
    pl.add_axes()

    pl.link_views()
    pl.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    template_mesh = pv.read(str(TEMPLATE_PATHS["tibia"]))
    sub = compute_tibia_template_subregions(template_mesh)
    print(f"  template verts:      {len(sub.region)}")
    print(f"  subch verts:         {int(sub.is_subch.sum())}")
    print(f"  MT (medial plate):   {int((sub.region == LBL_MED).sum())}")
    print(f"  LT (lateral plate):  {int((sub.region == LBL_LAT).sum())}")
    if args.no_show:
        return
    show_tibia_subregions(template_mesh, sub)


if __name__ == "__main__":
    main()
