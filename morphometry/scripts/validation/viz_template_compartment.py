"""Visualize the tibia template's compartment field (medial vs lateral subch).

Produces:
  template_compartment_3d.png  — 3D bone, anterior + superior views, with
      medial subch in RED, lateral subch in BLUE, non-articular in light gray.
  template_compartment_2d.png  — standard tibia 2D projection (M/L halves)
      with the same color scheme.

No per-case data — just the static atlas. Confirms the template-side
compartment definition is clean before we use it as the inheritance source
for patient bone verts (template-NN lookup, replacing the cart_mask midline
heuristic).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from cartilage_morphometry import TEMPLATE_PATHS, template_thickness_2d_tibia


def template_thickness_2d_tibia_proportional(template_mesh: pv.PolyData,
                                              scalar_per_vert: np.ndarray,
                                              grid_size: int = 60,
                                              padding_frac: float = 0.02):
    """Proportional alternative to library's `template_thickness_2d_tibia`.

    Uses SHARED (ML_min, ML_max, AP_min, AP_max) over BOTH compartments so
    relative sizes are preserved (the larger plateau renders at more cells)
    AND the intercondylar gap appears naturally (no verts in the eminence
    region → empty cells between the two plateaus).

    Looking down from superior: rows = AP (row 0 = anterior, row N = posterior),
    cols = ML (col 0 = medial, col N = lateral).

    Returns (grid, ml_extent_mm, ap_extent_mm).
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; ML = verts[:, 2]
    subch_prob = np.asarray(template_mesh.point_data["subch_prob"])
    valid = (subch_prob >= 0.5) & np.isfinite(scalar_per_vert)
    if not valid.any():
        return np.full((grid_size, grid_size), np.nan), 0.0, 0.0

    ml_lo, ml_hi = float(ML[valid].min()), float(ML[valid].max())
    ap_lo, ap_hi = float(AP[valid].min()), float(AP[valid].max())
    ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
    # Small padding so edge verts aren't clipped to col/row 0 or grid-1
    ml_lo -= ml_r * padding_frac; ml_hi += ml_r * padding_frac
    ap_lo -= ap_r * padding_frac; ap_hi += ap_r * padding_frac
    ml_r = ml_hi - ml_lo; ap_r = ap_hi - ap_lo

    # Aspect ratio: render with isotropic mm-per-cell. The longer axis gets
    # grid_size cells; the shorter axis gets grid_size * (short/long) cells.
    aspect_ml_per_ap = ml_r / ap_r
    if aspect_ml_per_ap >= 1.0:
        n_cols = grid_size
        n_rows = max(int(round(grid_size / aspect_ml_per_ap)), 1)
    else:
        n_rows = grid_size
        n_cols = max(int(round(grid_size * aspect_ml_per_ap)), 1)

    ml_n = (ML[valid] - ml_lo) / ml_r
    ap_n = (AP[valid] - ap_lo) / ap_r
    ml_b = np.clip((ml_n * n_cols).astype(int), 0, n_cols - 1)
    ap_b = np.clip((ap_n * n_rows).astype(int), 0, n_rows - 1)

    sum_g = np.zeros((n_rows, n_cols), dtype=np.float64)
    cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(sum_g, (ap_b, ml_b), scalar_per_vert[valid])
    np.add.at(cnt_g, (ap_b, ml_b), 1)
    grid = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)
    return grid, float(ml_r), float(ap_r)


def render_3d(mesh: pv.PolyData, out_png: Path):
    compartment = np.asarray(mesh.point_data["compartment"])
    rgb = np.full((mesh.n_points, 3), 0.85, dtype=np.float32)   # gray background
    rgb[compartment == 1] = (1.0, 0.0, 0.0)                     # medial = red
    rgb[compartment == 2] = (0.0, 0.0, 1.0)                     # lateral = blue
    mesh_c = mesh.copy()
    mesh_c["rgb"] = (rgb * 255).astype(np.uint8)

    # 4 views: anterior, posterior, superior (top-down on plateau), oblique
    pl = pv.Plotter(off_screen=True, shape=(1, 4), window_size=(2400, 700))
    cam_views = [
        ("anterior",  [(0, -250, 0), (0, 0, 0), (0, 0, 1)]),
        ("posterior", [(0,  250, 0), (0, 0, 0), (0, 0, 1)]),
        ("superior",  [(0, 0,  250), (0, 0, 0), (0, 1, 0)]),
        ("oblique",   [(150, -150, 150), (0, 0, 0), (0, 0, 1)]),
    ]
    for i, (name, cam) in enumerate(cam_views):
        pl.subplot(0, i)
        pl.add_mesh(mesh_c, scalars="rgb", rgb=True, smooth_shading=True)
        pl.add_text(name, position="upper_left", font_size=10, color="black")
        pl.camera_position = cam
        pl.reset_camera()
    pl.background_color = "white"
    pl.screenshot(str(out_png))
    pl.close()
    print(f"  [ok] {out_png}")


def _insert_mid_gap(grid: np.ndarray, gap_cols: int = 4):
    """Insert `gap_cols` blank columns at the medial/lateral seam so the
    intercondylar separation is visible (the 2D projection normalises each
    half independently, so by default they touch at col grid_size/2)."""
    if gap_cols <= 0:
        return grid
    n_rows, n_cols = grid.shape
    mid = n_cols // 2
    fill_val = np.nan if grid.dtype.kind == "f" else 0
    pad = np.full((n_rows, gap_cols), fill_val, dtype=grid.dtype)
    return np.concatenate([grid[:, :mid], pad, grid[:, mid:]], axis=1)


def render_2d(mesh: pv.PolyData, out_png: Path, grid_size: int = 60):
    """Tibia 2D projection of the compartment field using PROPORTIONAL
    projection (shared ML/AP bounds across both compartments).

    Preserves the real relative size of medial vs lateral plateau and the
    intercondylar gap appears naturally in cells where no subch verts
    project. Small intra-plateau pinholes are filled via binary closing so
    each compartment renders as a single clean filled shape.
    """
    from scipy.ndimage import binary_closing
    compartment = np.asarray(mesh.point_data["compartment"]).astype(np.float32)
    g, ml_mm, ap_mm = template_thickness_2d_tibia_proportional(
        mesh, compartment, grid_size=grid_size)
    g_rounded = np.where(np.isfinite(g), np.round(g), 0).astype(np.int8)

    # Fill 1-2 cell pinholes WITHIN each plateau so contours are single loops.
    # binary_closing acts ONLY on the in-plateau cells, the intercondylar
    # eminence gap stays open.
    structure = np.ones((3, 3), dtype=bool)
    med_filled = binary_closing(g_rounded == 1, structure=structure, iterations=1)
    lat_filled = binary_closing(g_rounded == 2, structure=structure, iterations=1)
    g_rounded = np.zeros_like(g_rounded)
    g_rounded[med_filled] = 1
    g_rounded[lat_filled] = 2

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#ffffff", "#cc0000", "#0033cc"])   # 0=white, 1=red, 2=blue
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: compartment map
    axes[0].imshow(g_rounded, origin="upper", cmap=cmap, norm=norm,
                    interpolation="nearest", aspect="equal")
    axes[0].set_title(f"template compartment field (proportional 2D)\n"
                       f"red = medial subch, blue = lateral subch  "
                       f"[{ml_mm:.0f}mm ML × {ap_mm:.0f}mm AP]", fontsize=11)

    # Right: subch_prob as background, compartment outlines on top (clean closed loops)
    subch_prob = np.asarray(mesh.point_data["subch_prob"]).astype(np.float32)
    g_sub, _, _ = template_thickness_2d_tibia_proportional(
        mesh, subch_prob, grid_size=grid_size)
    im2 = axes[1].imshow(g_sub, origin="upper", cmap="Greys",
                          vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "subch_prob", fontsize=8)
    med_mask = (g_rounded == 1).astype(float)
    lat_mask = (g_rounded == 2).astype(float)
    if med_mask.any():
        axes[1].contour(med_mask, levels=[0.5], colors="red", linewidths=2.0)
    if lat_mask.any():
        axes[1].contour(lat_mask, levels=[0.5], colors="blue", linewidths=2.0)
    axes[1].set_title("template subch_prob + compartment outlines\n"
                       "(red = medial, blue = lateral)", fontsize=11)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for tx, ty, label, ha in ((0.02, 0.97, "M", "left"),
                                  (0.98, 0.97, "L", "right"),
                                  (0.5, 0.97, "P", "center"),
                                  (0.5, 0.03, "A", "center")):
            ax.text(tx, ty, label, transform=ax.transAxes, color="black",
                    fontsize=9, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    # Stats
    n_med = int((compartment == 1).sum())
    n_lat = int((compartment == 2).sum())
    n_bg  = int((compartment == 0).sum())
    fig.suptitle(f"Tibia template ({Path(TEMPLATE_PATHS['tibia']).name})  "
                  f"medial subch={n_med}  lateral subch={n_lat}  "
                  f"non-articular={n_bg}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  [ok] {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/template_compartment_viz"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mesh = pv.read(str(TEMPLATE_PATHS["tibia"]))
    print(f"loaded template: {mesh.n_points} verts, fields={list(mesh.point_data.keys())}")
    compartment = np.asarray(mesh.point_data["compartment"])
    print(f"  compartment uniques: {dict(zip(*[a.tolist() for a in np.unique(compartment, return_counts=True)]))}")

    render_3d(mesh, args.out_dir / "template_compartment_3d.png")
    render_2d(mesh, args.out_dir / "template_compartment_2d.png")


if __name__ == "__main__":
    main()
