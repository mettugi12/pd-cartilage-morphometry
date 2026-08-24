"""Interactive viewer of a bone template colored by subchondral label."""
import argparse
from pathlib import Path

import pyvista as pv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk",
                   default=r"MOAKS_seg/standardization/femur_template_with_subch.vtk")
    p.add_argument("--scalar", default="subchondral",
                   help="Field to color by (subchondral or subchondral_dist_mm)")
    args = p.parse_args()

    m = pv.read(str(Path(args.vtk)))
    print(f"{args.vtk}: {m.n_points} verts, {m.n_cells} faces")
    print(f"point_data fields: {list(m.point_data.keys())}")
    if args.scalar in m.point_data:
        s = m.point_data[args.scalar]
        print(f"  '{args.scalar}': min={s.min():.3f}, max={s.max():.3f}, "
              f"mean={s.mean():.3f}, n>0={(s > 0).sum()}")

    pl = pv.Plotter(window_size=(1200, 900))
    pl.set_background("white")
    if args.scalar in ("subchondral", "subch_prob"):
        pl.add_mesh(m, scalars=args.scalar, cmap="Reds",
                    clim=[0, 1], smooth_shading=True,
                    scalar_bar_args={"title": args.scalar})
    elif args.scalar == "compartment":
        # 0 = non-articular (gray), 1 = medial (blue), 2 = lateral (red)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#cccccc", "#3060e0", "#e03030"])
        pl.add_mesh(m, scalars=args.scalar, cmap=cmap,
                    clim=[0, 2], smooth_shading=True,
                    annotations={0: "non-art", 1: "medial", 2: "lateral"},
                    scalar_bar_args={"title": "compartment", "n_labels": 0})
    else:
        pl.add_mesh(m, scalars=args.scalar, cmap="viridis_r",
                    clim=[0, 5], smooth_shading=True,
                    scalar_bar_args={"title": args.scalar})
    pl.add_axes()
    pl.show_grid()
    pl.add_text(f"{Path(args.vtk).name} | colored by {args.scalar}",
                font_size=10, color="black")
    print("Press 'q' to close.")
    pl.show()


if __name__ == "__main__":
    main()
