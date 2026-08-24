"""
Build a single femur template where the articular surface is a SUBSET of the
full bone surface — by definition.

Per case:
  1. Load DESS seg → bone mask + cart mask + spacing
  2. Marching-cubes the bone mask → full bone mesh (uses the same closing/smoothing
     as the existing articular pipeline so surfaces match)
  3. For each bone vertex, compute Euclidean distance (mm) to nearest cart voxel
     → per-vertex `subch_dist_mm`
  4. Anisotropic scale (using interface bounds → ML=75, AP=65, matches existing
     articular template's coordinate frame)

Cross-case:
  5. Translation-register each case's interface to first case's interface; apply
     same translation to the full bone mesh + scalars (consistent with how the
     existing articular template was built)
  6. Correspondence average vertex positions AND per-vertex scalars

Outputs:
  `femur_template_full_with_subch.vtk` — full femur mesh with point_data:
      `subch_dist_mm` — average distance to cart (mm)
      `subch_prob`    — fraction of cases where vertex was subchondral (binary < threshold)
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from new_standard import (
    create_pv_mesh,
    standardize_plateau_anisotropic,
)
from new_standard_femur import (
    create_mesh_from_points,
    extract_femur_cartilage_interface_points,
    register_femur_translation_only,
)


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
    """Marching-cubes full bone mesh + per-vertex distance-to-cart scalars."""
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
    """Correspondence-based mean of vertex positions + scalars (NN to ref topology)."""
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
    p.add_argument("--bone_label", type=int, default=1)
    p.add_argument("--cart_label", type=int, default=2)
    p.add_argument("--target_ml", type=float, default=75.0)
    p.add_argument("--target_ap", type=float, default=65.0)
    p.add_argument("--subch_thresh_mm", type=float, default=1.0)
    p.add_argument("--out",
                   default="MOAKS_seg/standardization/femur_template_full_with_subch.vtk")
    args = p.parse_args()

    files = sorted(glob(args.dess_glob))
    if args.n_run:
        files = files[: args.n_run]
    print(f"processing {len(files)} cases")

    bone_meshes_scaled = []
    interface_meshes_for_reg = []
    fail = 0
    for path in tqdm(files, desc="extract+scale"):
        try:
            interface_points, _, mask_bone, mask_bone_cart, spacing = (
                extract_femur_cartilage_interface_points(
                    path, bone_label=args.bone_label,
                    cartilage_label=args.cart_label,
                )
            )
            if len(interface_points) < 100:
                fail += 1; continue
            mask_cart = mask_bone_cart & ~mask_bone

            bone_mesh = build_bone_mesh_with_subch(
                mask_bone, mask_cart, spacing, args.subch_thresh_mm,
            )

            # Same anisotropic scaling as existing articular template
            interface_mesh = create_mesh_from_points(
                interface_points, smoothing_strength="light",
            )
            interface_scaled, scale_factors = standardize_plateau_anisotropic(
                interface_mesh, target_ml=args.target_ml, target_ap=args.target_ap,
            )

            bone_scaled = apply_anisotropic_scale(bone_mesh, scale_factors)
            bone_meshes_scaled.append(bone_scaled)
            interface_meshes_for_reg.append(interface_scaled)
        except Exception as e:
            print(f"\n  [skip] {Path(path).name}: {e}")
            fail += 1
            continue

    print(f"succeeded {len(bone_meshes_scaled)}, failed {fail}")
    if not bone_meshes_scaled:
        return

    # Translation register using interface (matches existing pipeline)
    ref_interface = interface_meshes_for_reg[0]
    registered = [bone_meshes_scaled[0]]
    for i in tqdm(range(1, len(bone_meshes_scaled)), desc="register"):
        _, translation = register_femur_translation_only(
            interface_meshes_for_reg[i], ref_interface,
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
    print(f"  subch_prob (binary thresh={args.subch_thresh_mm}mm avg): "
          f"max={avg['subch_prob'].max():.2f}, "
          f"verts with prob>=0.5: {int((avg['subch_prob'] >= 0.5).sum())}")


if __name__ == "__main__":
    main()
