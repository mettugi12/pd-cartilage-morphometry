"""
Build the patella bone template (mesh + per-vertex subchondral field) for the
11-class anatomy atlas.  Mirrors the existing femur/tibia template build but
on the 110 manually-labeled DESS cohort, since the 327 OAI-MOAKS cohort
doesn't have patella labeled.

Why a separate template (not a tibia-anchored prob volume): patella moves
with the femur; tibia-anchoring it produces a wide smear (peak prob 0.93 but
P>=0.1 halo of 41k voxels).  Mesh-style template with mean-centroid
re-centering recovers the sharp shape AND places it correctly in the tibia
template frame.

Per case
--------
1. Load DESS seg, mirror LEFT axis-2 to match RIGHT.
2. Compute the tibia-anchor transform: (s_ap, 1, s_ml) + translation that
   takes this case's tibia bone into the existing tibia template (same as
   build_softtissue_atlas.py).
3. Build patella bone mesh via marching_cubes; per-vertex subch_dist_mm =
   distance to nearest patella-cart voxel; subch = (dist <= 1.0 mm).
4. Apply the tibia-anchor transform to the patella mesh.

Cross-case
----------
5. Compute mean patella centroid in tibia-template frame.
6. Re-center each case's patella to this mean (cancels femur-tibia
   pose variation that would otherwise smear the average).
7. KDTree-correspondence average (first valid case = topology reference)
   for both vertex positions and the per-vertex scalars.
8. Save  patella_template_full_with_subch.vtk  with point_data fields
   subch_dist_mm + subch_prob.

Output (default):  E:/KneeMR/Datasets/Bone-Atlas/patella_template_full_with_subch.vtk
"""
from __future__ import annotations

import argparse
import sys
import time
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
from tqdm import tqdm

from knee_mr_seg.postproc import (
    load_seg_oriented, per_case_transform,
    TIBIA_BONE_LABEL_DEFAULT,
)


PATELLA_BONE_LABEL = 1
PATELLA_CART_LABEL = 4
SUBCH_THRESH_MM    = 1.0

DEFAULT_DESS_GLOB = r"E:/KneeMR/Datasets/DESS18seg/3D_DESS_labels/*.nii.gz"
DEFAULT_TIBIA_VTK = r"E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk"
DEFAULT_OUT       = r"E:/KneeMR/Datasets/Bone-Atlas/patella_template_full_with_subch.vtk"


def build_patella_mesh_with_subch(mask_bone: np.ndarray,
                                    mask_cart: np.ndarray,
                                    spacing: tuple) -> pv.PolyData:
    closed = binary_closing(mask_bone, structure=np.ones((3, 3, 3)), iterations=2)
    smooth = gaussian_filter(closed.astype(np.float32), sigma=0.5)
    verts, faces, _, _ = measure.marching_cubes(
        volume=smooth, level=0.5, spacing=spacing, step_size=1,
    )
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)

    if mask_cart.sum() == 0:
        mesh["subch_dist_mm"] = np.full(len(verts), 999.0, dtype=np.float32)
        mesh["subch"]         = np.zeros(len(verts), dtype=np.uint8)
        return mesh
    cart_mm = np.argwhere(mask_cart) * np.asarray(spacing)
    tree = cKDTree(cart_mm)
    dists, _ = tree.query(verts, k=1)
    mesh["subch_dist_mm"] = dists.astype(np.float32)
    mesh["subch"]         = (dists <= SUBCH_THRESH_MM).astype(np.uint8)
    return mesh


def apply_aniso_then_translate(mesh: pv.PolyData,
                                 scale: tuple, translation: np.ndarray) -> pv.PolyData:
    pts = mesh.points * np.asarray(scale) + translation
    out = pv.PolyData(pts, mesh.faces)
    for n in mesh.array_names:
        out[n] = mesh[n]
    return out


def average_meshes_with_scalars(meshes: list, scalar_names: list, smooth: bool = True):
    """Correspondence-based mean (KDTree NN to first-case topology)."""
    ref = meshes[0]
    n = ref.n_points
    pos_sum   = np.zeros((n, 3), dtype=np.float64)
    scalar_sum = {nm: np.zeros(n, dtype=np.float64) for nm in scalar_names}
    used = 0
    for m in meshes:
        if m.n_points == n:
            pos_sum += m.points
            for nm in scalar_names:
                scalar_sum[nm] += m[nm].astype(np.float64)
        else:
            tree = cKDTree(m.points)
            _, idx = tree.query(ref.points, k=1)
            pos_sum += m.points[idx]
            for nm in scalar_names:
                scalar_sum[nm] += np.take(m[nm].astype(np.float64), idx)
        used += 1
    avg_pts = pos_sum / used
    avg = pv.PolyData(avg_pts, ref.faces)
    if smooth:
        avg = avg.smooth(n_iter=50, relaxation_factor=0.1)
    for nm in scalar_names:
        avg[nm] = (scalar_sum[nm] / used).astype(np.float32)
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dess_glob",       default=DEFAULT_DESS_GLOB)
    p.add_argument("--tibia_template",  default=DEFAULT_TIBIA_VTK)
    p.add_argument("--out",             default=DEFAULT_OUT)
    p.add_argument("--n_run",           type=int, default=None)
    p.add_argument("--dry_run",         action="store_true")
    args = p.parse_args()

    files = sorted(glob(args.dess_glob))
    if args.n_run is not None:
        files = files[: args.n_run]
    print(f"[input] {len(files)} cases from {args.dess_glob}")

    target_tibia = pv.read(args.tibia_template)
    print(f"[anchor] tibia = {Path(args.tibia_template).name}  "
          f"({target_tibia.n_points} verts)")

    meshes_in_frame: list[pv.PolyData] = []
    centroids:       list[np.ndarray] = []
    fail = []
    t0 = time.time()

    pbar = tqdm(files, desc="pass1: tibia-anchor + patella mesh")
    for path in pbar:
        try:
            seg, spacing, side = load_seg_oriented(path)
            scale, translation = per_case_transform(
                seg, spacing, target_tibia,
                tibia_bone_label=TIBIA_BONE_LABEL_DEFAULT,
            )
            mask_pat_b = (seg == PATELLA_BONE_LABEL)
            mask_pat_c = (seg == PATELLA_CART_LABEL)
            if mask_pat_b.sum() < 500:
                fail.append((Path(path).name, "tiny patella")); continue
            pat = build_patella_mesh_with_subch(mask_pat_b, mask_pat_c, spacing)
            pat_in_frame = apply_aniso_then_translate(pat, scale, translation)
            meshes_in_frame.append(pat_in_frame)
            centroids.append(np.asarray(pat_in_frame.center))
            pbar.set_postfix(side=side, n=len(meshes_in_frame),
                              t=f"{(time.time()-t0)/max(len(meshes_in_frame),1):.1f}s/case")
        except Exception as e:  # noqa: BLE001
            fail.append((Path(path).name, repr(e))); continue

    n_used = len(meshes_in_frame)
    print(f"\n[done pass1] used={n_used} / failed={len(fail)}")
    for nm, r in fail[:10]:
        print(f"  fail: {nm}  {r}")
    if n_used == 0:
        return

    # Mean centroid in tibia frame  -> the position the template should sit at
    mean_centroid = np.mean(np.stack(centroids), axis=0)
    spread = np.std(np.stack(centroids), axis=0)
    print(f"[centroid] mean = ({mean_centroid[0]:.1f}, {mean_centroid[1]:.1f}, "
          f"{mean_centroid[2]:.1f}) mm  | per-axis std = "
          f"({spread[0]:.1f}, {spread[1]:.1f}, {spread[2]:.1f}) mm")

    # Re-center each case to the mean (cancels femur-tibia pose variation)
    recentered = []
    for m, c in zip(meshes_in_frame, centroids):
        shifted = m.points + (mean_centroid - c)
        new = pv.PolyData(shifted, m.faces)
        for nm in m.array_names:
            new[nm] = m[nm]
        recentered.append(new)

    print(f"[avg] correspondence-average over {n_used} re-centered meshes...")
    avg = average_meshes_with_scalars(
        recentered, scalar_names=["subch_dist_mm", "subch"], smooth=True,
    )
    avg["subch_prob"] = avg["subch"].astype(np.float32)
    del avg.point_data["subch"]

    print(f"[result] verts={avg.n_points}  "
          f"bounds={tuple(round(x,1) for x in avg.bounds)}")
    print(f"  subch_dist_mm  min={float(avg['subch_dist_mm'].min()):.2f}  "
          f"mean={float(avg['subch_dist_mm'].mean()):.2f}  "
          f"max={float(avg['subch_dist_mm'].max()):.2f}")
    print(f"  subch_prob     >=0.5 verts = "
          f"{int((avg['subch_prob'] >= 0.5).sum())}")

    if args.dry_run:
        print("[dry_run] not saving."); return
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    avg.save(str(out))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
