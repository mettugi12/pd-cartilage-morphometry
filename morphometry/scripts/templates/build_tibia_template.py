"""
Build a unified tibia template (full bone + per-vertex subchondral label).
Mirrors the unified femur pipeline. Articular = subset of bone vertices by
construction (per-vertex distance to cart, threshold).

Per case:
  1. Load DESS seg → bone (label 3), cart (label 4), spacing
  2. Densify slice axis if spacing > 1.5mm (uses interpolate_segmentation_filled)
  3. Marching-cubes the bone mask (closed + smoothed) → full tibia mesh
  4. Per-vertex Euclidean distance (mm) to nearest cart voxel → `subch_dist_mm`
     and `subch` (binary, ≤ subch_thresh_mm)
  5. Anisotropic scale by bone bounds → target ML/AP

Cross-case:
  6. Translation-register all bones to the first case
  7. Correspondence-average vertex positions and scalars

Output: `tibia_template_full_with_subch.vtk` (point_data: subch_dist_mm, subch_prob)
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from new_standard import create_pv_mesh, interpolate_segmentation_filled
from new_standard_femur import register_femur_translation_only


DESS_GLOB = (r"E:\OAI_all\OAI_MOAKS\48M_PRED_MINI"
             r"\predicted_3d_good_moaks_score_RIGHT\*.nii.gz")


def apply_anisotropic_scale(mesh, scale_factors):
    ml, ap, z = scale_factors
    transform = np.array([[ap, 0, 0, 0], [0, z, 0, 0], [0, 0, ml, 0], [0, 0, 0, 1]])
    homog = np.hstack([mesh.points, np.ones((len(mesh.points), 1))])
    scaled = (homog @ transform.T)[:, :3]
    out = pv.PolyData(scaled, mesh.faces)
    for name in mesh.array_names:
        out[name] = mesh[name]
    return out


def build_bone_mesh_with_subch(mask_bone, mask_cart, spacing, subch_thresh_mm=1.0):
    closed = binary_closing(mask_bone, structure=np.ones((3, 3, 3)), iterations=2)
    smooth = gaussian_filter(closed.astype(float), sigma=0.5)
    verts, faces, _, _ = measure.marching_cubes(
        volume=smooth, level=0.5, spacing=spacing, step_size=1,
    )
    mesh = create_pv_mesh(verts, faces)

    cart_idx = np.argwhere(mask_cart)
    if len(cart_idx) == 0:
        mesh["subch_dist_mm"] = np.full(len(verts), 999.0, dtype=np.float32)
        mesh["subch"] = np.zeros(len(verts), dtype=np.uint8)
        return mesh
    cart_mm = cart_idx * np.array(spacing)
    tree = cKDTree(cart_mm)
    dists, _ = tree.query(verts, k=1)
    mesh["subch_dist_mm"] = dists.astype(np.float32)
    mesh["subch"] = (dists <= subch_thresh_mm).astype(np.uint8)
    return mesh


def average_meshes_with_scalars(registered_meshes, scalar_names, smooth=True):
    ref = registered_meshes[0]
    n = ref.n_points
    pos_sum = np.zeros((n, 3), dtype=np.float64)
    scalar_sum = {name: np.zeros(n, dtype=np.float64) for name in scalar_names}
    used = 0
    for mesh in registered_meshes:
        if mesh.n_points == n:
            pos_sum += mesh.points
            for name in scalar_names:
                scalar_sum[name] += mesh[name].astype(np.float64)
        else:
            tree = cKDTree(mesh.points)
            _, idx = tree.query(ref.points, k=1)
            pos_sum += mesh.points[idx]
            for name in scalar_names:
                scalar_sum[name] += np.take(mesh[name].astype(np.float64), idx)
        used += 1
    avg_pts = pos_sum / used
    avg_mesh = pv.PolyData(avg_pts, ref.faces)
    if smooth:
        avg_mesh = avg_mesh.smooth(n_iter=50, relaxation_factor=0.1)
    for name in scalar_names:
        avg_mesh[name] = (scalar_sum[name] / used).astype(np.float32)
    return avg_mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dess_glob", default=DESS_GLOB)
    p.add_argument("--n_run", type=int, default=None)
    p.add_argument("--bone_label", type=int, default=3)
    p.add_argument("--cart_label", type=int, default=4)
    p.add_argument("--target_ml", type=float, default=75.0)
    p.add_argument("--target_ap", type=float, default=55.0)
    p.add_argument("--subch_thresh_mm", type=float, default=1.0)
    p.add_argument("--out",
                   default="MOAKS_seg/standardization/tibia_template_full_with_subch.vtk")
    args = p.parse_args()

    files = sorted(glob(args.dess_glob))
    if args.n_run:
        files = files[: args.n_run]
    print(f"processing {len(files)} cases (tibia: bone={args.bone_label}, cart={args.cart_label})")

    bone_meshes_scaled = []
    fail = 0
    for path in tqdm(files, desc="extract+scale"):
        try:
            nii = nib.load(path)
            seg = nii.get_fdata().astype(int)
            spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
            if spacing[2] > 1.5:
                seg, spacing = interpolate_segmentation_filled(
                    seg, spacing, args.bone_label, args.cart_label,
                    slices_between=4,
                )
            mask_bone = (seg == args.bone_label)
            mask_cart = (seg == args.cart_label)
            if mask_bone.sum() < 1000:
                fail += 1; continue

            bone_mesh = build_bone_mesh_with_subch(
                mask_bone, mask_cart, spacing, args.subch_thresh_mm,
            )

            # Anisotropic scale based on FULL BONE bounds
            b = bone_mesh.bounds
            current_ml = b[5] - b[4]
            current_ap = b[1] - b[0]
            ml_scale = args.target_ml / current_ml if current_ml > 0 else 1.0
            ap_scale = args.target_ap / current_ap if current_ap > 0 else 1.0
            scale_factors = (ml_scale, ap_scale, 1.0)

            bone_scaled = apply_anisotropic_scale(bone_mesh, scale_factors)
            bone_meshes_scaled.append(bone_scaled)
        except Exception as e:
            print(f"\n  [skip] {Path(path).name}: {e}")
            fail += 1
            continue

    print(f"succeeded {len(bone_meshes_scaled)}, failed {fail}")
    if not bone_meshes_scaled:
        return

    ref = bone_meshes_scaled[0]
    print(f"reference bone: {ref.n_points} verts | "
          f"bounds={[round(v,1) for v in ref.bounds]}")

    registered = [ref]
    for i in tqdm(range(1, len(bone_meshes_scaled)), desc="register"):
        _, translation = register_femur_translation_only(
            bone_meshes_scaled[i], ref,
        )
        translated_points = bone_meshes_scaled[i].points + translation
        translated_mesh = pv.PolyData(translated_points, bone_meshes_scaled[i].faces)
        for name in bone_meshes_scaled[i].array_names:
            translated_mesh[name] = bone_meshes_scaled[i][name]
        registered.append(translated_mesh)

    print("averaging mesh + scalars...")
    avg = average_meshes_with_scalars(
        registered, scalar_names=["subch_dist_mm", "subch"], smooth=True,
    )
    avg["subch_prob"] = avg["subch"].astype(np.float32)
    del avg.point_data["subch"]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    avg.save(str(out))
    print(f"saved {out}")
    print(f"  bone verts: {avg.n_points}")
    print(f"  subch_dist_mm: min={avg['subch_dist_mm'].min():.2f}, "
          f"mean={avg['subch_dist_mm'].mean():.2f}, max={avg['subch_dist_mm'].max():.2f}")
    print(f"  subch_prob: max={avg['subch_prob'].max():.2f}, "
          f"verts >= 0.5: {int((avg['subch_prob'] >= 0.5).sum())}")


if __name__ == "__main__":
    main()
