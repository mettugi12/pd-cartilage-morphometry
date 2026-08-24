"""
Shared atlas-build helpers.  Used by both
  scripts/build_softtissue_atlas.py  (110 manually-labeled DESS, 11-class)
  scripts/build_cart_atlas.py        (327 KLG=0 OAI-MOAKS, 4-class DESS preds)

The pipeline is identical for both cohorts:
  per case:
      load NIfTI seg (apply LEFT axis-2 flip if needed)
      build tibia bone mesh in physical mm via marching_cubes
      anisotropic scale to (ML=75, AP=55)
      translation-only register to existing tibia template VTK
  cross-case:
      compose total transform  T(p) = (s_ap, 1, s_ml) * p_native_mm + t
      apply T to every requested class mask via affine_transform(order=0)
      voxel-wise mean across cases  ->  float32 probability volume
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pyvista as pv
from pathlib import Path
from scipy.ndimage import affine_transform, binary_closing, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure


TIBIA_BONE_LABEL_DEFAULT = 3   # both PD nnU-Net (11-class) and DESS preds (4-class) use 3 = tibia bone
TARGET_ML = 75.0   # tibia template build constants — must match the existing
TARGET_AP = 55.0   # tibia_template_full_with_subch.vtk anchor


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def side_from_path(p) -> str:
    name = Path(p).name.upper()
    if "LEFT" in name:
        return "LEFT"
    if "RIGHT" in name:
        return "RIGHT"
    return "UNKNOWN"


def load_seg_oriented(path):
    """Load NIfTI; apply LEFT -> RIGHT axis-2 flip so the result matches
    the convention of the existing tibia template (axcodes ('P','I','R'))."""
    nii = nib.load(str(path))
    seg = nii.get_fdata().astype(np.int16)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    side = side_from_path(path)
    if side == "LEFT":
        seg = seg[:, :, ::-1].copy()
    return seg, spacing, side


# ---------------------------------------------------------------------------
# Mesh + registration to existing tibia template
# ---------------------------------------------------------------------------

def build_tibia_mesh(mask_bone: np.ndarray, spacing: tuple) -> pv.PolyData:
    closed = binary_closing(mask_bone, structure=np.ones((3, 3, 3)), iterations=2)
    smooth = gaussian_filter(closed.astype(np.float32), sigma=0.5)
    verts, faces, _, _ = measure.marching_cubes(
        volume=smooth, level=0.5, spacing=spacing, step_size=1,
    )
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    return pv.PolyData(verts, faces_pv)


def compute_aniso_scale(mesh: pv.PolyData) -> tuple[float, float, float]:
    """Returns (s_ap, 1.0, s_ml) — matches the diag of the transform in
    knee-mr-cartilage/scripts/templates/build_tibia_template.apply_anisotropic_scale.
    """
    b = mesh.bounds
    cur_ap = b[1] - b[0]
    cur_ml = b[5] - b[4]
    s_ap = TARGET_AP / cur_ap if cur_ap > 0 else 1.0
    s_ml = TARGET_ML / cur_ml if cur_ml > 0 else 1.0
    return (float(s_ap), 1.0, float(s_ml))


def apply_aniso_scale_mesh(mesh: pv.PolyData, scale: tuple) -> pv.PolyData:
    return pv.PolyData(mesh.points * np.asarray(scale), mesh.faces)


def register_translation_only(source: pv.PolyData, target: pv.PolyData,
                                max_iter: int = 20, tol: float = 1e-2) -> np.ndarray:
    t = np.asarray(target.center) - np.asarray(source.center)
    target_tree = cKDTree(target.points)
    for _ in range(max_iter):
        moved = source.points + t
        _, idx = target_tree.query(moved)
        update = (target.points[idx] - moved).mean(axis=0)
        t += update * 0.5
        if np.linalg.norm(update) < tol:
            break
    return t


# ---------------------------------------------------------------------------
# Voxelization onto shared grid
# ---------------------------------------------------------------------------

def template_grid(femur_vtk, tibia_vtk, spacing_mm: float, margin_mm: float):
    """Returns (origin_mm, spacing_mm, dims) for a fixed grid that encloses
    the union of (femur template, tibia template) + margin_mm on each side."""
    f_b = pv.read(str(femur_vtk)).bounds
    t_b = pv.read(str(tibia_vtk)).bounds
    lo = np.array([min(f_b[0], t_b[0]), min(f_b[2], t_b[2]), min(f_b[4], t_b[4])])
    hi = np.array([max(f_b[1], t_b[1]), max(f_b[3], t_b[3]), max(f_b[5], t_b[5])])
    lo -= margin_mm; hi += margin_mm
    dims = tuple(int(np.ceil((hi[a] - lo[a]) / spacing_mm)) for a in range(3))
    return lo.astype(np.float64), float(spacing_mm), dims


def affine_for_case(scale, translation, native_spacing,
                     grid_origin, grid_spacing):
    """Inverse transform for scipy.ndimage.affine_transform.

    Forward: template_mm[a] = scale[a] * native_spacing[a] * native_idx[a] + translation[a]
    Inverse: native_idx[a]  = (grid_origin[a] + g_idx[a]*grid_spacing - translation[a])
                              / (scale[a] * native_spacing[a])
    """
    sa = np.asarray(scale,            dtype=np.float64)
    ns = np.asarray(native_spacing,   dtype=np.float64)
    denom = sa * ns
    mat = np.diag(grid_spacing / denom)
    off = (grid_origin - translation) / denom
    return mat, off


def voxelize_mask_to_grid(mask, native_spacing, scale, translation,
                           grid_origin, grid_spacing, grid_dims):
    if mask.sum() == 0:
        return np.zeros(grid_dims, dtype=np.uint8)
    mat, off = affine_for_case(scale, translation, native_spacing,
                                grid_origin, grid_spacing)
    out = affine_transform(
        mask.astype(np.uint8),
        matrix=mat, offset=off,
        output_shape=grid_dims, order=0, mode="constant", cval=0,
    )
    return (out > 0).astype(np.uint8)


def make_grid_affine(origin: np.ndarray, spacing: float) -> np.ndarray:
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = spacing
    aff[1, 1] = spacing
    aff[2, 2] = spacing
    aff[:3, 3] = origin
    return aff


# ---------------------------------------------------------------------------
# Per-case driver — used by both build scripts
# ---------------------------------------------------------------------------

def per_case_transform(seg: np.ndarray, spacing: tuple,
                        target_tibia: pv.PolyData,
                        tibia_bone_label: int = TIBIA_BONE_LABEL_DEFAULT):
    """Returns (scale, translation) or raises ValueError on tiny tibia."""
    mask_tibia = (seg == tibia_bone_label)
    if mask_tibia.sum() < 1000:
        raise ValueError("tiny tibia bone mask")
    mesh = build_tibia_mesh(mask_tibia, spacing)
    scale = compute_aniso_scale(mesh)
    scaled = apply_aniso_scale_mesh(mesh, scale)
    translation = register_translation_only(scaled, target_tibia)
    return scale, translation
