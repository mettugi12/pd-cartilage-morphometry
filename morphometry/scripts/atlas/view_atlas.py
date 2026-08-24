"""
Interactive PyVista viewer for the knee-mr-atlas.

13 toggle-able layers (checkbox widgets stacked bottom-left):
  bones (meshes, default ON):    femur bone, tibia bone
  subch ROIs (point clouds):     femur subch ROI, tibia subch ROI
  prob volumes (volume render):  femur cart, tibia cart,
                                  patella, patella cart,
                                  medial meniscus, lateral meniscus,
                                  ACL, PCL, patellar tendon

The 9 probability classes use vtk volume rendering — soft cloud, no
isosurface "gap-at-the-bone-edge" artifact from probability dilution at
the cart/bone interface.

Keyboard:
  q   quit
  s   screenshot
  drag-rotate / wheel-zoom / shift-drag-pan
"""
from __future__ import annotations
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import LinearSegmentedColormap, to_rgb


CLASS_COLORS = {
    # Note: patella is now a MESH (loaded from patella_template_full_with_subch.vtk)
    # so it's removed from this prob-volume color map.  patella_cart stays as a
    # volume since we don't have a separate cart-mesh layer for it.
    "patella_cart":      "#FF8C00",   # orange
    "femur_cart":        "#FFB0B0",   # pale pink
    "tibia_cart":        "#B0FFB0",   # pale green
    "medial_meniscus":   "#E03030",   # red
    "lateral_meniscus":  "#3060E0",   # blue
    "acl":               "#30C030",   # green
    "pcl":               "#30C0C0",   # cyan
    "patellar_tendon":   "#C030C0",   # magenta
}

SUBCH_COLORS = {
    "femur_subch":    "#FF6060",
    "tibia_subch":    "#60FF60",
    "patella_subch":  "#FFD060",
}

BONE_COLORS = {
    "femur_bone":   "white",
    "tibia_bone":   "#DDDDFF",
    "patella_bone": "#FFE8AA",
}

DEFAULT_DIR    = r"E:/KneeMR/Datasets/Bone-Atlas/soft_tissue"
TIBIA_VTK      = r"E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk"
FEMUR_VTK      = r"E:/KneeMR/Datasets/Bone-Atlas/femur_template_full_with_subch.vtk"
PATELLA_VTK    = r"E:/KneeMR/Datasets/Bone-Atlas/patella_template_full_with_subch.vtk"


def to_image_data(nii_path: Path) -> pv.ImageData:
    img = nib.load(str(nii_path))
    arr = img.get_fdata().astype(np.float32)
    aff = img.affine
    spacing = (float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2]))
    origin  = tuple(float(v) for v in aff[:3, 3])
    grid = pv.ImageData(dimensions=arr.shape, spacing=spacing, origin=origin)
    grid.point_data["p"] = arr.ravel(order="F")
    return grid


def add_toggle(pl: pv.Plotter, actors: list, label: str,
                position_y: int, color: str, default_on: bool,
                button_size: int = 20):
    """Add a checkbox widget at bottom-left + a text label next to it.
    `actors` is bound as a parameter so each callback closes over its own list.
    """
    # initial visibility
    for a in actors:
        a.SetVisibility(1 if default_on else 0)

    def _cb(value, _actors=actors, _label=label):
        v = 1 if value else 0
        for a in _actors:
            a.SetVisibility(v)
        print(f"  toggle [{_label:18s}] -> {bool(value)}  ({len(_actors)} actors)")
        pl.render()

    pl.add_checkbox_button_widget(
        callback=_cb, value=default_on, position=(10, position_y),
        size=button_size, border_size=1,
        color_on=color, color_off="grey", background_color="black",
    )
    pl.add_text(label, position=(10 + button_size + 8, position_y + 2),
                  font_size=10, color="white")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dir",     default=DEFAULT_DIR)
    ap_.add_argument("--tibia",   default=TIBIA_VTK)
    ap_.add_argument("--femur",   default=FEMUR_VTK)
    ap_.add_argument("--patella", default=PATELLA_VTK)
    ap_.add_argument("--p_min", type=float, default=0.02,
                      help="probability noise floor; voxels below fully transparent")
    ap_.add_argument("--saturate", type=float, default=0.30,
                      help="fraction of per-class p_max above which voxels render fully opaque "
                           "(lower = bigger / more solid blob)")
    ap_.add_argument("--subch_thresh", type=float, default=0.5,
                      help="threshold on subch_prob to define subchondral ROI on bone meshes")
    args = ap_.parse_args()

    in_dir = Path(args.dir)
    print(f"[viewer] dir={in_dir}")
    print(f"[viewer] volume-render p_min={args.p_min}")
    print(f"[viewer] subch threshold: {args.subch_thresh}")

    pl = pv.Plotter(window_size=(1400, 1000), title="knee-mr-atlas viewer")
    pl.set_background("black")

    # --- bone shells (toggle-able, default ON) ------------------------------
    femur_mesh = pv.read(args.femur)
    tibia_mesh = pv.read(args.tibia)
    patella_mesh = pv.read(args.patella) if Path(args.patella).exists() else None
    bone_actors = {}
    bone_actors["femur_bone"] = [pl.add_mesh(
        femur_mesh, color=BONE_COLORS["femur_bone"], opacity=0.18,
        smooth_shading=True, name="femur_bone")]
    bone_actors["tibia_bone"] = [pl.add_mesh(
        tibia_mesh, color=BONE_COLORS["tibia_bone"], opacity=0.18,
        smooth_shading=True, name="tibia_bone")]
    if patella_mesh is not None:
        bone_actors["patella_bone"] = [pl.add_mesh(
            patella_mesh, color=BONE_COLORS["patella_bone"], opacity=0.25,
            smooth_shading=True, name="patella_bone")]
    else:
        print(f"  [skip] patella mesh: {args.patella} not found")

    # --- subchondral ROI layers (toggle) -------------------------------------
    subch_actors = {}
    subch_sources = [
        ("femur_subch",   femur_mesh,   SUBCH_COLORS["femur_subch"]),
        ("tibia_subch",   tibia_mesh,   SUBCH_COLORS["tibia_subch"]),
    ]
    if patella_mesh is not None:
        subch_sources.append(("patella_subch", patella_mesh, SUBCH_COLORS["patella_subch"]))
    for tag, mesh, color in subch_sources:
        if "subch_prob" not in mesh.point_data:
            print(f"  [skip] {tag}: subch_prob not on mesh")
            continue
        subch_pts = mesh.points[mesh["subch_prob"] >= args.subch_thresh]
        if len(subch_pts) == 0:
            print(f"  [skip] {tag}: 0 verts above thresh")
            continue
        # tubelet/sphere glyph cloud — visible regardless of camera
        cloud = pv.PolyData(subch_pts)
        a = pl.add_mesh(cloud, color=color, point_size=4.0,
                          render_points_as_spheres=True,
                          name=f"{tag}_actor")
        subch_actors[tag] = [a]
        print(f"  {tag:18s}  {len(subch_pts):6d} verts  color={color}")

    # --- soft-tissue layers: mesh if {class}_mesh.vtk exists, else volume ----
    # build_softtissue_meshes.py drops {class}_mesh.vtk alongside the NIfTIs;
    # if found we render it as a semi-transparent mesh (same style as bones).
    # Classes without a mesh fall back to probability volume rendering.
    SOFT_MESH_OPACITY = 0.40

    soft_actors = {}
    soft_meta = {}   # name -> (actor, p_max, rgb) for live slider (volumes only)

    def _make_opacity_fn(p_lo: float, p_hi: float) -> vtk.vtkPiecewiseFunction:
        """Opacity ramp: 0 below p_lo, smooth ramp to ~0.6 by p_hi, then capped."""
        f = vtk.vtkPiecewiseFunction()
        eps = max(1e-4, (p_hi - p_lo) * 0.05)
        f.AddPoint(0.0,                              0.0)
        f.AddPoint(max(0.0, p_lo - eps),             0.0)
        f.AddPoint(p_lo + (p_hi - p_lo) * 0.20,      0.20)
        f.AddPoint(p_lo + (p_hi - p_lo) * 0.50,      0.45)
        f.AddPoint(p_hi,                             0.65)
        f.AddPoint(1.0,                              0.65)
        return f

    def _make_color_fn(rgb) -> vtk.vtkColorTransferFunction:
        f = vtk.vtkColorTransferFunction()
        f.AddRGBPoint(0.0, *rgb)
        f.AddRGBPoint(1.0, *rgb)
        return f

    for name, color in CLASS_COLORS.items():
        vtk_path = in_dir / f"{name}_mesh.vtk"
        nii_path = in_dir / f"{name}.nii.gz"

        if vtk_path.exists():
            mesh = pv.read(str(vtk_path))
            actor = pl.add_mesh(
                mesh, color=color, opacity=SOFT_MESH_OPACITY,
                smooth_shading=True, name=f"{name}_mesh",
            )
            soft_actors[name] = [actor]
            print(f"  {name:18s}  mesh  verts={mesh.n_points:6d}  color={color}")

        elif nii_path.exists():
            grid = to_image_data(nii_path)
            p_max = float(grid["p"].max())
            if p_max < args.p_min:
                print(f"  [skip] {name}: pmax={p_max:.2f} < p_min={args.p_min}")
                continue
            rgb = to_rgb(color)
            cmap = LinearSegmentedColormap.from_list(f"c_{name}", [rgb, rgb], N=8)
            upper = max(args.p_min + 1e-3, p_max * args.saturate)
            actor = pl.add_volume(
                grid, scalars="p",
                cmap=cmap, clim=(0.0, 1.0),
                shade=True,
                ambient=0.25, diffuse=0.85,
                specular=0.30, specular_power=12,
                mapper="smart",
                name=f"{name}_volume",
            )
            prop = actor.GetProperty() if hasattr(actor, "GetProperty") else actor.prop
            prop.SetColor(_make_color_fn(rgb))
            prop.SetScalarOpacity(_make_opacity_fn(args.p_min, upper))
            prop.SetScalarOpacityUnitDistance(3.0)
            soft_actors[name] = [actor]
            soft_meta[name] = (actor, p_max, rgb)
            print(f"  {name:18s}  pmax={p_max:.2f}  vol-render solid up to {upper:.2f}  color={color}")

        else:
            print(f"  [skip] {name}: neither mesh nor nii.gz found")

    # --- live threshold slider (controls all soft-tissue volumes) ----------
    def _on_slider(value: float):
        """value in [0, 1] — the lower-cutoff fraction of each class's p_max.
        Voxels below this are fully transparent; above ramp to opaque."""
        for nm, (act, pmx, rgb_) in soft_meta.items():
            lo = float(value) * pmx
            hi = max(lo + 1e-3, pmx * args.saturate)
            if hi <= lo:
                hi = lo + 1e-3
            prop = act.GetProperty() if hasattr(act, "GetProperty") else act.prop
            prop.SetScalarOpacity(_make_opacity_fn(lo, hi))
        pl.render()

    if soft_meta:
        pl.add_slider_widget(
            callback=_on_slider,
            rng=(0.0, 1.0), value=args.p_min,
            title="prob threshold  (fraction of each class peak)",
            pointa=(0.55, 0.92), pointb=(0.95, 0.92),
            style="modern",
        )

    # --- toggle widgets (vertical stack from bottom-up) ---------------------
    # 4-tuple: (label, actors, color, default_on)
    all_layers = [
        # Bones (meshes) — default ON
        ("femur bone",           bone_actors.get("femur_bone", []),        BONE_COLORS["femur_bone"], True),
        ("tibia bone",           bone_actors.get("tibia_bone", []),        BONE_COLORS["tibia_bone"], True),
        ("patella bone",         bone_actors.get("patella_bone", []),      BONE_COLORS["patella_bone"], True),
        # Subch ROIs (per-vertex point cloud on bone meshes)
        ("femur subch ROI",      subch_actors.get("femur_subch", []),      SUBCH_COLORS["femur_subch"], False),
        ("tibia subch ROI",      subch_actors.get("tibia_subch", []),      SUBCH_COLORS["tibia_subch"], False),
        ("patella subch ROI",    subch_actors.get("patella_subch", []),    SUBCH_COLORS["patella_subch"], False),
        # Cart prob volumes (327 cohort, build_cart_atlas.py)
        ("femur cart",           soft_actors.get("femur_cart", []),        CLASS_COLORS["femur_cart"], False),
        ("tibia cart",           soft_actors.get("tibia_cart", []),        CLASS_COLORS["tibia_cart"], False),
        ("patella cart",         soft_actors.get("patella_cart", []),      CLASS_COLORS["patella_cart"], False),
        # Soft tissue prob volumes (110 cohort, build_softtissue_atlas.py)
        ("medial meniscus",      soft_actors.get("medial_meniscus", []),   CLASS_COLORS["medial_meniscus"], False),
        ("lateral meniscus",     soft_actors.get("lateral_meniscus", []),  CLASS_COLORS["lateral_meniscus"], False),
        ("ACL",                  soft_actors.get("acl", []),               CLASS_COLORS["acl"], False),
        ("PCL",                  soft_actors.get("pcl", []),               CLASS_COLORS["pcl"], False),
        ("patellar tendon",      soft_actors.get("patellar_tendon", []),   CLASS_COLORS["patellar_tendon"], False),
    ]
    # Stack buttons from BOTTOM up so they're always visible regardless of
    # window size / DPI scaling.  First entry in all_layers ends up at the top
    # of the column.
    visible = [(l, a, c, d) for (l, a, c, d) in all_layers if a]
    n = len(visible)
    spacing = 28
    margin_bottom = 14
    print(f"\n[viewer] {n} toggle layers -- visible at y in [{margin_bottom}, "
          f"{margin_bottom + (n - 1) * spacing}]")
    for i, (lbl, actors, color, default_on) in enumerate(visible):
        y = margin_bottom + (n - 1 - i) * spacing
        add_toggle(pl, actors, lbl, position_y=y, color=color, default_on=default_on)

    pl.add_text("knee-mr-atlas  |  bones always on  |  axes: AP / SI / ML",
                  font_size=9, position="upper_edge", color="white")
    pl.add_axes(line_width=2)
    pl.show()


if __name__ == "__main__":
    main()
