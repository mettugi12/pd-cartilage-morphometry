"""
Atlas-based postprocessing filter for the 7 soft-tissue classes of the
PD nnU-Net Dataset204_Knee3D segmentation.

Per class, runs 3D connected components on the predicted mask.  For each
component, transforms its voxels into the atlas grid frame using the
existing per-case (anisotropic-scale + translation) tibia-anchor transform,
samples the per-class atlas probability at those locations, and drops the
component if the mean probability falls below `p_min`.

Soft-tissue classes covered (atlas labels = PD nnU-Net labels):
    1=patella, 4=patella_cart, 7=medial_meniscus, 8=lateral_meniscus,
    9=acl, 10=pcl, 11=patellar_tendon

Bones (2, 3) and cart 5/6 are filtered by `keep_largest_bone_components` /
`drop_small_cart_components` / `filter_cart_near_bone_per_slice` in
knee_mr_cartilage.pipeline — atlas filtering is redundant for those.

Side detection: filename heuristic first ("Lt"/"Rt"/"Left"/"Right"); on
miss, runs the registration both as-is and ML-flipped and picks whichever
gives the smaller post-translation centroid residual (LEFT NIfTIs need an
axis-2 flip to match the RIGHT-convention tibia template).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

from .atlas_core import (
    TIBIA_BONE_LABEL_DEFAULT,
    build_tibia_mesh, compute_aniso_scale, apply_aniso_scale_mesh,
    register_translation_only,
)


# Soft-tissue label IDs (PD nnU-Net Dataset204_Knee3D)
SOFT_TISSUE_CLASS_NAMES: dict[int, str] = {
    1:  "patella",
    4:  "patella_cart",
    7:  "medial_meniscus",
    8:  "lateral_meniscus",
    9:  "acl",
    10: "pcl",
    11: "patellar_tendon",
}

DEFAULT_ATLAS_DIR  = r"E:/KneeMR/Datasets/Bone-Atlas/soft_tissue"
DEFAULT_TIBIA_VTK  = r"E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk"


# ---------------------------------------------------------------------------
# Side detection
# ---------------------------------------------------------------------------

_SIDE_RE_LEFT  = re.compile(r"(?<![A-Za-z])(Lt|Left|LEFT)(?![A-Za-z])", re.IGNORECASE)
_SIDE_RE_RIGHT = re.compile(r"(?<![A-Za-z])(Rt|Right|RIGHT)(?![A-Za-z])", re.IGNORECASE)


def detect_side_from_name(name: str) -> str:
    """Returns 'LEFT', 'RIGHT', or 'UNKNOWN' from the filename."""
    n = Path(name).name
    has_l = bool(_SIDE_RE_LEFT.search(n))
    has_r = bool(_SIDE_RE_RIGHT.search(n))
    if has_l and not has_r:
        return "LEFT"
    if has_r and not has_l:
        return "RIGHT"
    return "UNKNOWN"


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return the binary mask of the largest connected component."""
    from scipy.ndimage import label as cc_label
    if mask.sum() == 0:
        return mask
    cc, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(cc.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    return cc == keep


def _clean_tibia_mask(seg: np.ndarray, tibia_label: int) -> np.ndarray:
    """Return the largest tibia connected component as a binary mask.

    Spurious tibia islands (typical in raw nnU-Net output) corrupt the
    aniso-scale computation (bbox includes the islands), giving wrong
    scale factors and a registration that lands soft-tissue CCs outside
    the atlas grid — which drops them all.  Clean the mask before
    building the tibia mesh."""
    return _largest_cc(seg == tibia_label)


def _registration_residual(seg: np.ndarray, spacing: tuple,
                            target_tibia: pv.PolyData,
                            tibia_label: int = TIBIA_BONE_LABEL_DEFAULT) -> float:
    """Anisotropic-scale + translation-register the patient tibia to the
    template.  Returns the mean nearest-neighbour residual (mm) after
    registration.  Lower = better fit."""
    mask = _clean_tibia_mask(seg, tibia_label)
    if mask.sum() < 1000:
        return float("inf")
    try:
        mesh = build_tibia_mesh(mask, spacing)
        scale = compute_aniso_scale(mesh)
        scaled = apply_aniso_scale_mesh(mesh, scale)
        t = register_translation_only(scaled, target_tibia)
        moved = scaled.points + t
        target_tree = cKDTree(target_tibia.points)
        dists, _ = target_tree.query(moved)
        return float(dists.mean())
    except Exception:
        return float("inf")


def detect_side_via_flip(seg: np.ndarray, spacing: tuple,
                          target_tibia: pv.PolyData,
                          tibia_label: int = TIBIA_BONE_LABEL_DEFAULT,
                          ) -> tuple[str, float, float]:
    """ML-flip check.  Tries the seg both as-is (assume RIGHT) and flipped
    on axis 2 (assume LEFT), runs the tibia registration both ways, and
    returns the side whose post-fit residual is smaller.

    Returns (side, residual_as_is_mm, residual_flipped_mm).
    """
    res_R = _registration_residual(seg, spacing, target_tibia, tibia_label)
    seg_flipped = seg[:, :, ::-1]
    res_L = _registration_residual(seg_flipped, spacing, target_tibia, tibia_label)
    side = "RIGHT" if res_R <= res_L else "LEFT"
    return side, res_R, res_L


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------

@dataclass
class AtlasBundle:
    """Loaded soft-tissue atlas — probability volumes + grid metadata."""
    grid_origin: np.ndarray            # (3,) physical mm
    grid_spacing: float                # mm
    grid_dims: tuple[int, int, int]
    prob_by_label: dict[int, np.ndarray] = field(default_factory=dict)
    target_tibia: pv.PolyData = None
    tibia_target_ml: float = 75.0
    tibia_target_ap: float = 55.0


def load_atlas_bundle(atlas_dir: str | Path = DEFAULT_ATLAS_DIR,
                       tibia_vtk: str | Path = DEFAULT_TIBIA_VTK,
                       labels: tuple[int, ...] = tuple(SOFT_TISSUE_CLASS_NAMES),
                       ) -> AtlasBundle:
    atlas_dir = Path(atlas_dir)
    meta_path = atlas_dir / "_grid_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    grid_origin = np.asarray(meta["grid_origin_mm"], dtype=np.float64)
    grid_spacing = float(meta["grid_spacing_mm"])
    grid_dims = tuple(int(d) for d in meta["grid_dims"])

    bundle = AtlasBundle(
        grid_origin=grid_origin,
        grid_spacing=grid_spacing,
        grid_dims=grid_dims,
        target_tibia=pv.read(str(tibia_vtk)),
        tibia_target_ml=float(meta.get("tibia_target_ml_mm", 75.0)),
        tibia_target_ap=float(meta.get("tibia_target_ap_mm", 55.0)),
    )
    for lbl in labels:
        cls_name = SOFT_TISSUE_CLASS_NAMES[lbl]
        nii_path = atlas_dir / f"{cls_name}.nii.gz"
        if not nii_path.exists():
            continue
        bundle.prob_by_label[lbl] = nib.load(str(nii_path)).get_fdata().astype(np.float32)
    return bundle


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _apply_left_flip_if_needed(seg: np.ndarray, side: str) -> np.ndarray:
    """LEFT seg → flip axis 2 to match RIGHT-convention template frame."""
    if side == "LEFT":
        return seg[:, :, ::-1]
    return seg


def filter_by_atlas(seg: np.ndarray,
                     spacing: tuple,
                     side: str,
                     atlas: AtlasBundle,
                     p_min: float = 0.05,
                     tibia_label: int = TIBIA_BONE_LABEL_DEFAULT,
                     verbose: bool = False,
                     ) -> tuple[np.ndarray, dict]:
    """Atlas-based CC filtering for the 7 soft-tissue classes.

    Args:
        seg          (H, W, D) int label volume in patient-native voxel space.
        spacing      native voxel spacing (3-tuple, mm).
        side         "LEFT" / "RIGHT" / "UNKNOWN" — controls the axis-2 flip.
                     If "UNKNOWN", caller should run `detect_side_via_flip`
                     beforehand.
        atlas        loaded AtlasBundle.
        p_min        drop a CC if mean atlas prob over its voxels < p_min.
        tibia_label  PD label for the tibia bone (default 3).

    Returns:
        seg_filtered   (H, W, D) int — same shape as input, with dropped CCs
                       zeroed out.
        log            {label: {"n_cc": int, "n_kept": int, "n_dropped": int,
                                "vox_dropped": int}}
    """
    # Flip LEFT to RIGHT convention so the patient → template transform applies
    seg_oriented = _apply_left_flip_if_needed(seg, side)

    # Per-case scale + translation (anisotropic + translation-only ICP).
    # IMPORTANT: clean the tibia to its largest CC before building the mesh —
    # spurious tibia islands break the bbox-based aniso scale.
    mask_tibia = _clean_tibia_mask(seg_oriented, tibia_label)
    if mask_tibia.sum() < 1000:
        # No tibia → can't register → return seg unchanged
        return seg.copy(), {"error": "tibia_mask_too_small"}

    mesh = build_tibia_mesh(mask_tibia, spacing)
    scale = compute_aniso_scale(mesh)               # (s_ap, 1.0, s_ml)
    scaled = apply_aniso_scale_mesh(mesh, scale)
    translation = register_translation_only(scaled, atlas.target_tibia)

    scale_arr   = np.asarray(scale,   dtype=np.float64)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    grid_origin = atlas.grid_origin
    grid_spacing = atlas.grid_spacing
    G = np.asarray(atlas.grid_dims)

    # Operate on the (possibly flipped) seg, then unflip at the end if needed.
    seg_out_oriented = seg_oriented.copy()
    log: dict = {}

    for lbl, cls_name in SOFT_TISSUE_CLASS_NAMES.items():
        prob_vol = atlas.prob_by_label.get(lbl)
        if prob_vol is None:
            continue
        cls_mask = (seg_oriented == lbl)
        if cls_mask.sum() == 0:
            log[lbl] = {"n_cc": 0, "n_kept": 0, "n_dropped": 0, "vox_dropped": 0}
            continue

        cc, n_cc = cc_label(cls_mask)
        n_kept = n_dropped = vox_dropped = 0

        for k in range(1, n_cc + 1):
            cc_idx = np.argwhere(cc == k)             # (N, 3) native voxel indices
            if len(cc_idx) == 0:
                continue
            # native_mm = native_spacing * native_idx
            # template_mm = scale * native_mm + translation
            native_mm   = spacing_arr * cc_idx
            template_mm = scale_arr * native_mm + translation
            atlas_idx   = ((template_mm - grid_origin) / grid_spacing).round().astype(np.int64)

            in_bounds = ((atlas_idx >= 0) & (atlas_idx < G)).all(axis=1)
            n_total = len(cc_idx)
            if in_bounds.any():
                valid = atlas_idx[in_bounds]
                probs = prob_vol[valid[:, 0], valid[:, 1], valid[:, 2]]
                # OOB voxels contribute 0 to the mean — denom is full CC size
                mean_prob = float(probs.sum() / n_total)
            else:
                mean_prob = 0.0

            if mean_prob < p_min:
                seg_out_oriented[cc == k] = 0
                n_dropped += 1
                vox_dropped += int(n_total)
                if verbose:
                    print(f"  drop {cls_name} CC{k}: {n_total}vox  mean_p={mean_prob:.3f}")
            else:
                n_kept += 1

        log[lbl] = {
            "name": cls_name,
            "n_cc": int(n_cc),
            "n_kept": int(n_kept),
            "n_dropped": int(n_dropped),
            "vox_dropped": int(vox_dropped),
        }

    # Unflip to patient-native orientation if we flipped at entry
    if side == "LEFT":
        seg_out = seg_out_oriented[:, :, ::-1]
    else:
        seg_out = seg_out_oriented

    log["_meta"] = {
        "side": side,
        "scale": list(scale),
        "translation": [float(x) for x in translation],
        "p_min": p_min,
    }
    return seg_out, log
