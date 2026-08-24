"""
4-panel viewer: 3D template (with mapped thickness) + 2D projection,
for both femur and tibia.

Adapts the PD-vs-DESS 2D projection algorithm
(`01. Repos/knee-mr-pd2dess/comparison/thickness_map_2d.py`):
  Tibia  → split medial/lateral via the template's `compartment` field;
           AP normalised per compartment.
  Femur  → angular unwrapping per ML slice (atan2 around bone centroid),
           anterior gap detection to anchor the arc.

Template coord convention (per `standardize_plateau_anisotropic`):
  vert[0] = AP    vert[1] = SI    vert[2] = ML
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from cartilage_morphometry import TEMPLATE_PATHS, process_one_patient


def project_thickness_to_2d(template_mesh, thickness_per_vert, bone_name,
                             grid_size=40):
    """Project per-template-vertex thickness onto a 2D grid.

    Returns (grid (rows=AP, cols=ML), x_extent_mm, y_extent_mm).
    Cols 0 = medial, cols max = lateral; rows 0 = anterior, rows max = posterior.
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; SI = verts[:, 1]; ML = verts[:, 2]
    subch_prob = np.asarray(template_mesh.point_data["subch_prob"])
    valid = (subch_prob >= 0.5) & np.isfinite(thickness_per_vert)

    grid_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    grid_cnt = np.zeros((grid_size, grid_size), dtype=np.int64)

    if bone_name == "tibia":
        comp = np.asarray(template_mesh.point_data["compartment"])
        med = valid & (comp == 1)
        lat = valid & (comp == 2)

        def _normalize_compartment(mask, ml_lo, ml_hi):
            if not mask.any():
                return None
            ml_min, ml_max = ML[mask].min(), ML[mask].max()
            ap_min, ap_max = AP[mask].min(), AP[mask].max()
            ml_r = max(ml_max - ml_min, 1e-6)
            ap_r = max(ap_max - ap_min, 1e-6)
            ml_n = ml_lo + (ML[mask] - ml_min) / ml_r * (ml_hi - ml_lo)
            ap_n = (AP[mask] - ap_min) / ap_r
            return ml_n, ap_n, thickness_per_vert[mask], ml_r, ap_r

        # Compartment-extent estimates for the imshow physical scale
        ml_extent_mm, ap_extent_mm = 0.0, 0.0
        for mask, lo, hi in [(med, 0.0, 0.5), (lat, 0.5, 1.0)]:
            res = _normalize_compartment(mask, lo, hi)
            if res is None:
                continue
            ml_n, ap_n, t, ml_r, ap_r = res
            ml_extent_mm += ml_r
            ap_extent_mm = max(ap_extent_mm, ap_r)
            ml_b = np.clip((ml_n * grid_size).astype(int), 0, grid_size - 1)
            ap_b = np.clip((ap_n * grid_size).astype(int), 0, grid_size - 1)
            np.add.at(grid_sum, (ap_b, ml_b), t)
            np.add.at(grid_cnt, (ap_b, ml_b), 1)

    elif bone_name == "femur":
        ml_min, ml_max = ML[valid].min(), ML[valid].max()
        ml_r = max(ml_max - ml_min, 1e-6)
        # Per-ML-slice bone centroid (use ALL bone verts, not just subch)
        n_slices = 60
        slice_bin_all = np.clip(
            ((ML - ml_min) / ml_r * n_slices).astype(int), 0, n_slices - 1
        )
        ap_c = np.full(n_slices, np.nan); si_c = np.full(n_slices, np.nan)
        for s in range(n_slices):
            ms = slice_bin_all == s
            if ms.sum() > 5:
                ap_c[s] = AP[ms].mean(); si_c[s] = SI[ms].mean()
        idxs = np.arange(n_slices)
        v = np.isfinite(ap_c)
        if v.sum() > 1:
            ap_c = np.interp(idxs, idxs[v], ap_c[v])
            si_c = np.interp(idxs, idxs[v], si_c[v])

        # θ for valid verts
        v_idx = np.where(valid)[0]
        ml_b_v = slice_bin_all[v_idx]
        theta = np.arctan2(SI[v_idx] - si_c[ml_b_v],
                           AP[v_idx] - ap_c[ml_b_v]) % (2 * np.pi)

        # Largest empty histogram run = anterior shaft gap → anchor arc
        n_angle_bins = 360
        hist, edges = np.histogram(theta, bins=n_angle_bins, range=(0.0, 2 * np.pi))
        is_empty = (hist == 0)
        if is_empty.any():
            ie2 = np.tile(is_empty.astype(np.int8), 2)
            d2 = np.diff(ie2, prepend=0)
            run_starts = np.where(d2 == 1)[0]
            run_ends = np.where(d2 == -1)[0]
            if len(run_starts) > len(run_ends):
                run_ends = np.append(run_ends, len(ie2))
            run_lens = run_ends - run_starts
            best = int(np.argmax(run_lens))
            arc_start_bin = int((run_starts[best] + run_lens[best]) % n_angle_bins)
            theta_start = float(edges[arc_start_bin])
        else:
            theta_start = 0.0

        theta_shifted = (theta - theta_start) % (2 * np.pi)
        theta_range = max(float(theta_shifted.max()), 0.01)
        arc_norm = theta_shifted / theta_range
        ml_norm = (ML[v_idx] - ml_min) / ml_r
        # Flip so anterior at top
        ap_norm = 1.0 - arc_norm

        ml_b = np.clip((ml_norm * grid_size).astype(int), 0, grid_size - 1)
        ap_b = np.clip((ap_norm * grid_size).astype(int), 0, grid_size - 1)
        t_v = thickness_per_vert[v_idx]
        np.add.at(grid_sum, (ap_b, ml_b), t_v)
        np.add.at(grid_cnt, (ap_b, ml_b), 1)
        # Estimated extents for imshow aspect
        ml_extent_mm = ml_r
        # arc length ~ θ_range × mean radius
        r_pts = np.sqrt((SI[v_idx] - si_c[ml_b_v]) ** 2 +
                        (AP[v_idx] - ap_c[ml_b_v]) ** 2)
        ap_extent_mm = theta_range * float(np.mean(r_pts))

    else:
        raise ValueError(f"unknown bone: {bone_name}")

    grid_mean = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)
    return grid_mean, float(ml_extent_mm), float(ap_extent_mm)


def render_3d_screenshot(template_mesh, thickness, vmax, title, size=(700, 700)):
    """Off-screen pyvista render of template colored by thickness; returns PNG array."""
    m = template_mesh.copy()
    m["thickness_mm"] = thickness.astype(np.float32)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.add_mesh(
        m, scalars="thickness_mm", cmap="jet_r",
        clim=[0, vmax], nan_color="lightgray", nan_opacity=0.45,
        smooth_shading=True,
        scalar_bar_args={"title": "thickness (mm)"},
    )
    pl.add_text(title, font_size=11, color="black")
    pl.add_axes()
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_path", required=True, type=Path)
    p.add_argument("--side", choices=["RIGHT", "LEFT"], required=True)
    p.add_argument("--modality", choices=["pd", "dess"], required=True)
    p.add_argument("--max_thickness_mm", type=float, default=6.0)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--out_png", type=Path, default=None)
    args = p.parse_args()

    results = {}
    for bone in ("femur", "tibia"):
        print(f"\n=== {bone} ===")
        results[bone] = process_one_patient(
            seg_path=args.seg_path, side=args.side, bone_name=bone,
            modality=args.modality, max_thickness_mm=args.max_thickness_mm,
            out_dir=None,
        )

    # Shared color scale across all panels
    all_thick = np.concatenate([
        results[b]["template_thickness"][np.isfinite(results[b]["template_thickness"])]
        for b in ("femur", "tibia")
    ])
    vmax = float(np.percentile(all_thick, 98)) if len(all_thick) else 4.0
    vmax = max(vmax, 0.5)

    # 4-panel matplotlib: rows = bone, cols = (3D, 2D)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")

    for row, bone in enumerate(("femur", "tibia")):
        template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
        thickness = results[bone]["template_thickness"]

        # 3D screenshot
        title3d = (f"{bone.capitalize()} template — 3D\n"
                   f"mean = {np.nanmean(thickness):.2f} mm, "
                   f"max = {np.nanmax(thickness):.2f} mm")
        img3d = render_3d_screenshot(template_mesh, thickness, vmax, title3d)
        axes[row, 0].imshow(img3d)
        axes[row, 0].axis("off")

        # 2D projection
        grid, ml_mm, ap_mm = project_thickness_to_2d(
            template_mesh, thickness, bone, grid_size=args.grid,
        )
        ax2d = axes[row, 1]
        im = ax2d.imshow(
            grid, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
            extent=[0, ml_mm, ap_mm, 0], aspect="equal",
            interpolation="bilinear",
        )
        ax2d.set_xticks([]); ax2d.set_yticks([])
        ax2d.text(0.2, -0.05, "← Med", ha="center", va="top",
                  transform=ax2d.transAxes, fontsize=10, fontweight="bold")
        ax2d.text(0.8, -0.05, "Lat →", ha="center", va="top",
                  transform=ax2d.transAxes, fontsize=10, fontweight="bold")
        if bone == "femur":
            ax2d.text(-0.05, 0.85, "Ant ↑", ha="right", va="center",
                      transform=ax2d.transAxes, fontsize=10, fontweight="bold")
            ax2d.text(-0.05, 0.15, "Post ↓", ha="right", va="center",
                      transform=ax2d.transAxes, fontsize=10, fontweight="bold")
        else:
            ax2d.text(-0.05, 0.85, "Ant ↑", ha="right", va="center",
                      transform=ax2d.transAxes, fontsize=10, fontweight="bold")
            ax2d.text(-0.05, 0.15, "Post ↓", ha="right", va="center",
                      transform=ax2d.transAxes, fontsize=10, fontweight="bold")
        ax2d.set_title(
            f"{bone.capitalize()} template — 2D unwrapped\n"
            f"({ml_mm:.0f} × {ap_mm:.0f} mm, grid {args.grid}×{args.grid})",
            fontsize=11,
        )
        if bone == "tibia":
            # Vertical line marking medial/lateral split
            ax2d.axvline(ml_mm / 2, color="white", lw=1.0, ls="--", alpha=0.7)
        plt.colorbar(im, ax=ax2d, fraction=0.04, pad=0.02, label="thickness (mm)")

    suptitle = (f"Cartilage thickness — patient {args.seg_path.stem}  "
                f"({args.side}, {args.modality})")
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    plt.tight_layout()

    if args.out_png:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
        print(f"\nsaved {args.out_png}")
    plt.show()


if __name__ == "__main__":
    main()
