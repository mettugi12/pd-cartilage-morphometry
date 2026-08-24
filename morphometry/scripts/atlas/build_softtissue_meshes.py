"""
Build VTK mesh templates for soft-tissue probability classes by running
marching cubes on the NIfTI probability volumes produced by
build_softtissue_atlas.py and build_cart_atlas.py.

For each class the pipeline is:
  1. Load {class}.nii.gz from --in_dir
  2. Gaussian pre-smooth (sigma=0.8 vox) to reduce staircase artifacts
  3. Marching cubes at threshold (default: 50% of per-class pmax, floor 0.10)
     -- the "mean-shape" isosurface: voxels present in the majority of cases
  4. Translate vertices from array-local mm to physical-mm (NIfTI affine origin)
  5. PyVista clean (merge duplicate verts) + Taubin smooth (n=30)
  6. Save as {class}_mesh.vtk in the same directory

Skips `patella` (already has patella_template_full_with_subch.vtk).
Skips any class whose pmax < --min_pmax (default 0.10).

Outputs land next to the .nii.gz files so the viewer auto-detects them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage import measure


CLASSES = [
    "femur_cart",
    "tibia_cart",
    "patella_cart",
    "medial_meniscus",
    "lateral_meniscus",
    "acl",
    "pcl",
    "patellar_tendon",
]

DEFAULT_IN_DIR = r"E:/KneeMR/Datasets/Bone-Atlas/soft_tissue"


def prob_to_mesh(nii_path: Path, threshold: float, sigma: float,
                 taubin_iters: int) -> pv.PolyData:
    img = nib.load(str(nii_path))
    arr = img.get_fdata().astype(np.float32)
    aff = img.affine
    spacing = (float(abs(aff[0, 0])), float(abs(aff[1, 1])), float(abs(aff[2, 2])))
    origin = aff[:3, 3].astype(np.float64)

    smoothed = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr
    verts, faces, _, _ = measure.marching_cubes(
        smoothed, level=threshold, spacing=spacing, step_size=1,
    )
    verts = verts + origin  # array-local mm -> physical mm (NIfTI frame)

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv).clean()
    if taubin_iters > 0:
        mesh = mesh.smooth_taubin(n_iter=taubin_iters, pass_band=0.1)
    return mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",       default=DEFAULT_IN_DIR)
    p.add_argument("--frac",         type=float, default=0.4,
                   help="isosurface as fraction of per-class pmax (default 0.4, matches viewer slider)")
    p.add_argument("--threshold",    type=float, default=None,
                   help="override: absolute prob threshold for all classes (ignores --frac)")
    p.add_argument("--sigma",        type=float, default=0.8,
                   help="Gaussian pre-smooth sigma in voxels (default 0.8)")
    p.add_argument("--taubin_iters", type=int,   default=30,
                   help="Taubin smooth iterations (default 30)")
    p.add_argument("--min_pmax",     type=float, default=0.10,
                   help="skip class if pmax < this (default 0.10)")
    p.add_argument("--dry_run",      action="store_true")
    p.add_argument("--n_run",        type=int,   default=None)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    classes = CLASSES[: args.n_run] if args.n_run else CLASSES

    for name in classes:
        nii_path = in_dir / f"{name}.nii.gz"
        if not nii_path.exists():
            print(f"  [skip] {name}: {nii_path} not found")
            continue

        img = nib.load(str(nii_path))
        arr = img.get_fdata().astype(np.float32)
        pmax = float(arr.max())

        if pmax < args.min_pmax:
            print(f"  [skip] {name}: pmax={pmax:.2f} < min_pmax={args.min_pmax}")
            continue

        thresh = args.threshold if args.threshold is not None else max(0.05, args.frac * pmax)
        if thresh >= pmax:
            print(f"  [skip] {name}: thresh={thresh:.2f} >= pmax={pmax:.2f}")
            continue
        print(f"  {name:18s}  pmax={pmax:.2f}  thresh={thresh:.2f}", end="  ")

        try:
            mesh = prob_to_mesh(nii_path, threshold=thresh,
                                sigma=args.sigma, taubin_iters=args.taubin_iters)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        print(f"verts={mesh.n_points:6d}  faces={mesh.n_cells:6d}")

        if args.dry_run:
            continue

        out = in_dir / f"{name}_mesh.vtk"
        mesh.save(str(out))
        print(f"    -> {out}")

    if not args.dry_run:
        print("\n[done] meshes saved to", in_dir)


if __name__ == "__main__":
    main()
