"""Per-case drivers used by validate().

Two variants:

- `process_pair_independent` / `process_long_independent` — call
  `cartilage_morphometry.process_one_patient` per modality / timepoint
  independently. Quick to run; doesn't isolate registration noise.

- `process_pair_shared_mesh` / `process_long_shared_mesh` — the v6 canonical
  trick. One bone mesh acts as the shared sampling surface for both modality /
  timepoint, so per-template-vertex differences carry pure thickness-value
  differences, not bone-mesh or registration drift.

  Steps (PD↔DESS):
    1. Compute thickness on each modality's own bone mesh.
    2. Rigid PD→DESS bone-bone ICP in patient frame.
    3. IDW-sample PD thickness onto DESS bone-vert positions.
    4. Register DESS to template via patient_to_template_align (aniso_rigid).
    5. Subch_prob via tpl_nn on the DESS-aligned verts.
    6. IDW-remap both arrays to template using the SAME source-vert set + subch
       zone → only thickness values differ at each template vertex.

  Steps (00m↔48m) are identical with 00m as the canonical surface.

Ported from:
  KneeMR/Studies/PD-vs-DESS/v6/section/analysis/pipeline_pair_shared_mesh_v6.py
  KneeMR/Studies/PD-vs-DESS/v6/longitudinal/analysis/pipeline_long_shared_mesh_v6.py
without the v6 monkeypatch layer — we call library strategies directly. The
library exposes `raycast` as a thickness method (PipelineConfig.thickness_method
="raycast"); we read that off the config and invoke it via `get_thickness()`.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from cartilage_morphometry import (
    PipelineConfig,
    TEMPLATE_PATHS,
    get_thickness,
    patient_to_template_align,
    get_subch,
    pull_subchondral_from_template,
    remap_thickness_to_template,
    rigid_icp,
)
from cartilage_morphometry import pipeline as _pipeline


# DESS GT (chae_pairs + v32_seg) is 11-class; library default is 4-class.
_DESS_LABELS_11CLASS = {
    "femur": {"bone": 2, "cart": 5},
    "tibia": {"bone": 3, "cart": 6},
}


@contextlib.contextmanager
def _dess_labels_11class():
    original = _pipeline.DESS_LABELS
    _pipeline.DESS_LABELS = _DESS_LABELS_11CLASS
    try:
        yield
    finally:
        _pipeline.DESS_LABELS = original


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _idw_sample(query_pts: np.ndarray, source_pts: np.ndarray,
                source_values: np.ndarray, k: int = 3,
                eps: float = 1e-6) -> np.ndarray:
    tree = cKDTree(source_pts)
    dists, idxs = tree.query(query_pts, k=k)
    if k == 1:
        return source_values[idxs]
    weights = 1.0 / (dists + eps)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * source_values[idxs]).sum(axis=1)


def scatter_thickness_to_template(template_mesh, aligned_points: np.ndarray,
                                    patient_thickness: np.ndarray,
                                    subch_prob: np.ndarray,
                                    subch_threshold: float = 0.5):
    """Reverse-direction remap: each patient subch vert pushes its thickness
    to its single nearest template subch vert. Template verts then get
    arithmetic mean of the patient values pushed to them.

    Why over `remap_thickness_to_template` (forward IDW k=3):
      - Forward IDW averages each template vert's k=3 nearest patient verts.
        A small red region in patient frame (~5-10 verts) gets mixed with
        surrounding blue at the template-vert level → sign flip artifact.
      - Scatter k=1 has no cross-region averaging. Each patient vert picks
        ONE template vert (the closest); patient verts in a small red region
        almost always pick the SAME template vert; their thicknesses get
        averaged among themselves but not with verts from outside the
        region. Spatial pattern preserved.

    Returns (template_thickness, quality_dict).
    """
    template_pts = np.asarray(template_mesh.points)
    template_subch_field = np.asarray(template_mesh.point_data["subch_prob"])
    template_subch_mask = template_subch_field >= subch_threshold
    template_subch_idx = np.where(template_subch_mask)[0]
    template_subch_pts = template_pts[template_subch_idx]
    n_template_total = template_mesh.n_points

    # Filter patient verts by subch_prob
    patient_mask = subch_prob >= subch_threshold
    patient_pts = aligned_points[patient_mask]
    patient_th = patient_thickness[patient_mask]

    if len(patient_pts) == 0 or len(template_subch_pts) == 0:
        out = np.full(n_template_total, np.nan, dtype=np.float32)
        return out, {"n_patient": 0, "n_template_subch": int(len(template_subch_pts)),
                     "n_template_touched": 0, "frac_template_touched": 0.0}

    # For each patient vert, find nearest template subch vert (k=1)
    tree = cKDTree(template_subch_pts)
    dists, idxs = tree.query(patient_pts, k=1)
    # Aggregate (mean) patient thickness per template vert
    sum_arr = np.zeros(len(template_subch_pts), dtype=np.float64)
    cnt_arr = np.zeros(len(template_subch_pts), dtype=np.int64)
    np.add.at(sum_arr, idxs, patient_th)
    np.add.at(cnt_arr, idxs, 1)
    out_subch = np.where(cnt_arr > 0, sum_arr / np.maximum(cnt_arr, 1), np.nan)
    out = np.full(n_template_total, np.nan, dtype=np.float32)
    out[template_subch_idx] = out_subch.astype(np.float32)

    quality = {
        "n_patient": int(len(patient_pts)),
        "n_template_subch": int(len(template_subch_pts)),
        "n_template_touched": int((cnt_arr > 0).sum()),
        "frac_template_touched": float((cnt_arr > 0).mean()),
        "mean_patient_per_template_touched": float(cnt_arr[cnt_arr > 0].mean()) if (cnt_arr > 0).any() else 0.0,
        "mean_mapping_dist_mm_subch_only": float(dists.mean()),
        "median_mapping_dist_mm_subch_only": float(np.median(dists)),
        "max_mapping_dist_mm_subch_only": float(dists.max()),
        # Field names mirroring remap_thickness_to_template for swap-in compat
        "mean_mapping_dist_mm_all_template": float("nan"),
        "frac_mutual": 0.0,
        "assd_subch_mm": float(dists.mean()),
    }
    return out, quality


def _bone_assd(src_pts: np.ndarray, tgt_pts: np.ndarray) -> float:
    s_to_t, _ = cKDTree(tgt_pts).query(src_pts, k=1)
    t_to_s, _ = cKDTree(src_pts).query(tgt_pts, k=1)
    return float(0.5 * (s_to_t.mean() + t_to_s.mean()))


def _trimmed_rigid_icp(src: np.ndarray, tgt: np.ndarray,
                        trim_fraction: float = 0.8, max_iter: int = 50,
                        tol: float = 1e-5):
    """Trimmed rigid ICP — handles PARTIAL OVERLAP between src and tgt.

    Standard ICP matches every src point to its NN in tgt; when one cloud has
    extra points with no true correspondence (e.g. a larger cart ROI or a new
    osteophyte at one timepoint), those points still get matched and bias the
    rigid fit. Trimmed ICP keeps only the closest `trim_fraction` of
    correspondences each iteration (by residual), so non-overlapping points
    are rejected and the fit converges to the COMMON region.

    Returns (src_aligned, R_3x3, t_3). Transform maps the ORIGINAL src.
    """
    src_orig = src.copy()
    src_cur = src.copy()
    tgt_tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev = float("inf")
    k_keep = max(int(round(trim_fraction * len(src_cur))), 3)
    for _ in range(max_iter):
        d, idx = tgt_tree.query(src_cur, k=1)
        # Keep the closest k_keep correspondences (trim the worst residuals)
        keep = np.argpartition(d, k_keep - 1)[:k_keep]
        S = src_cur[keep]; T = tgt[idx[keep]]
        mu_s = S.mean(axis=0); mu_t = T.mean(axis=0)
        H = (S - mu_s).T @ (T - mu_t)
        U, _, Vt = np.linalg.svd(H)
        d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d_sign])
        R = Vt.T @ D @ U.T
        t = mu_t - R @ mu_s
        # Compose onto the running transform (applied to src_cur)
        R_total = R @ R_total
        t_total = R @ t_total + t
        src_cur = src_orig @ R_total.T + t_total
        trimmed_assd = float(d[keep].mean())
        if abs(prev - trimmed_assd) < tol:
            break
        prev = trimmed_assd
    return src_cur, R_total, t_total


def _articular_mask(bone_pts: np.ndarray, cart_mask: np.ndarray,
                    spacing: tuple, rad_mm: float) -> np.ndarray:
    """Boolean mask over bone_pts: True for verts within `rad_mm` of any
    cart-mask voxel (in physical mm). Used to restrict cross-modality /
    cross-timepoint rigid ICP to the cart-bearing band — kills the tibia
    plateau's rotational ambiguity. From v6.1 longitudinal pipeline."""
    idx = np.argwhere(cart_mask.astype(bool))
    if len(idx) == 0:
        return np.ones(len(bone_pts), dtype=bool)
    cart_mm = idx * np.asarray(spacing, dtype=np.float64)
    d, _ = cKDTree(cart_mm).query(bone_pts, k=1)
    return d <= rad_mm


def _bounded_affine_icp(src: np.ndarray, tgt: np.ndarray,
                         max_iter: int = 50, tol: float = 1e-5,
                         s_min: float = 0.7, s_max: float = 1.3
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded-affine ICP (12-DOF with singular-value clamp to ±30 % per axis).

    Ported from v6.1 `pipeline_long_shared_mesh_v6` companion module
    `affine_icp.py`. Designed for the 48 m → 00 m cross-timepoint step:
    rigid ICP is sufficient to align bone vertebra-by-vertebra but cannot
    absorb segmentation-driven shape differences (small osteophyte / boundary
    drift between PD time-points). Bounded affine adds 6 more DOF (axial
    scale + shear) but the SVD clamp prevents catastrophic axis collapse.

    Returns (src_aligned, A_3x3, t_3,).
    """
    tgt_tree = cKDTree(tgt)
    src_orig = src.copy()
    src_cur = src.copy()
    A_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_assd = float("inf")
    mu_s = src_orig.mean(axis=0)
    Sc = src_orig - mu_s
    for _ in range(max_iter):
        _, idxs = tgt_tree.query(src_cur, k=1)
        matched_tgt = tgt[idxs]
        mu_t = matched_tgt.mean(axis=0)
        Tc = matched_tgt - mu_t
        A_lsq, *_ = np.linalg.lstsq(Sc, Tc, rcond=None)
        A_new = A_lsq.T
        U, sv, Vt = np.linalg.svd(A_new)
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        sv_c = np.clip(sv, s_min, s_max)
        A_total = U @ np.diag(sv_c) @ Vt
        t_total = mu_t - A_total @ mu_s
        src_cur = src_orig @ A_total.T + t_total
        new_dists, _ = tgt_tree.query(src_cur, k=1)
        assd = float(new_dists.mean())
        if abs(prev_assd - assd) < tol:
            break
        prev_assd = assd
    return src_cur, A_total, t_total


def _articular_icp(src_pts: np.ndarray, src_cart: np.ndarray, src_spacing,
                   tgt_pts: np.ndarray, tgt_cart: np.ndarray, tgt_spacing,
                   articular_radius_mm: float | None,
                   min_verts: int = 100, max_iter: int = 30,
                   method: str = "rigid",
                   centroid_prealign: bool = True,
                   trim_fraction: float = 0.8):
    """ICP src → tgt, restricted to the articular subset (verts within
    `articular_radius_mm` of each side's cart_mask).

    `method`:
      - "rigid"          : 6-DOF rigid (canonical, v6.1 default)
      - "trimmed_rigid"  : 6-DOF rigid with `trim_fraction` outlier rejection
                           — robust to PARTIAL OVERLAP (one timepoint has a
                           larger ROI / extra osteophyte). Converges to the
                           common region.
      - "bounded_affine" : 12-DOF affine with ±30 % per-axis SV clamp.

    `centroid_prealign`: if True (default), translate src so its articular
    centroid coincides with tgt's articular centroid before running ICP.
    This eliminates the catastrophic-divergence failure mode when the two
    bones start 20–80 mm apart in physical space (NIfTI affines differ
    between 00m and 48m scans by patient-positioning offsets).

    Returns (src_aligned, A_3x3, t_3, used_articular: bool, n_art_src,
    n_art_tgt, method_used). For rigid, `A_3x3` is the rotation matrix R;
    t includes the centroid-prealign translation if used.
    """
    def _run(src_in, tgt_in, src_pts_to_apply):
        """Run the chosen ICP method on `src_in → tgt_in` and apply the
        resulting (A,t) to `src_pts_to_apply`. Returns (aligned_full, A, t)."""
        if method == "bounded_affine":
            _, A, t = _bounded_affine_icp(src_in, tgt_in, max_iter=max_iter)
        elif method == "trimmed_rigid":
            _, A, t = _trimmed_rigid_icp(src_in, tgt_in,
                                          trim_fraction=trim_fraction,
                                          max_iter=max(max_iter, 50))
        else:
            _, A, t = rigid_icp(src_in, tgt_in, max_iter=max_iter)
        aligned = src_pts_to_apply @ A.T + t
        return aligned, A, t

    if articular_radius_mm is None or articular_radius_mm <= 0:
        # Full-bone ICP path. Centroid prealign on full bone if requested.
        if centroid_prealign:
            offset = tgt_pts.mean(axis=0) - src_pts.mean(axis=0)
            src_pre = src_pts + offset
        else:
            offset = np.zeros(3)
            src_pre = src_pts
        aligned, A, t_icp = _run(src_pre, tgt_pts, src_pre)
        t_total = A @ offset + t_icp if method == "bounded_affine" else offset + t_icp
        # Re-derive aligned via composed (A, t_total) on ORIGINAL src
        # (numerically equivalent up to floating-point; explicit for safety)
        aligned = src_pts @ A.T + t_total
        return aligned, A, t_total, False, len(src_pts), len(tgt_pts), method

    art_src = _articular_mask(src_pts, src_cart, src_spacing, articular_radius_mm)
    art_tgt = _articular_mask(tgt_pts, tgt_cart, tgt_spacing, articular_radius_mm)
    n_src, n_tgt = int(art_src.sum()), int(art_tgt.sum())
    if n_src < min_verts or n_tgt < min_verts:
        # Articular subset too small — fall back to full-bone (with prealign)
        return _articular_icp(src_pts, src_cart, src_spacing, tgt_pts, tgt_cart, tgt_spacing,
                              articular_radius_mm=None, min_verts=min_verts,
                              max_iter=max_iter, method=method,
                              centroid_prealign=centroid_prealign)
    # Centroid-prealign the articular subset
    if centroid_prealign:
        src_art_pts = src_pts[art_src]
        tgt_art_pts = tgt_pts[art_tgt]
        offset = tgt_art_pts.mean(axis=0) - src_art_pts.mean(axis=0)
        src_pre_art = src_art_pts + offset
    else:
        offset = np.zeros(3)
        src_pre_art = src_pts[art_src]
    # Fit ICP on the centroid-pre-aligned articular subset
    if method == "bounded_affine":
        _, A, t_icp = _bounded_affine_icp(src_pre_art, tgt_pts[art_tgt], max_iter=max_iter)
        # Composed transform: src_orig @ A.T + (A @ offset + t_icp)
        t_total = A @ offset + t_icp
    elif method == "trimmed_rigid":
        _, A, t_icp = _trimmed_rigid_icp(src_pre_art, tgt_pts[art_tgt],
                                          trim_fraction=trim_fraction,
                                          max_iter=max(max_iter, 50))
        t_total = A @ offset + t_icp
    else:
        _, A, t_icp = rigid_icp(src_pre_art, tgt_pts[art_tgt], max_iter=max_iter)
        t_total = A @ offset + t_icp
    aligned = src_pts @ A.T + t_total
    return aligned, A, t_total, True, n_src, n_tgt, method


# ---------------------------------------------------------------------------
# v1.2 — compartment-aware tibia processing
# ---------------------------------------------------------------------------
# The tibia is anatomically two separate cart-bearing plateaus (medial /
# lateral) separated by the intercondylar eminence (no cartilage). The library
# treats subch as a single connected region — IDW pulls k=3 NN across the
# eminence, mixing medial template verts with lateral patient verts (and vice
# versa) wherever they are geometrically close in template space. Subch verts
# near the eminence with thickness=0 also dilute the IDW pool.
#
# Fix: split the patient subch pool by compartment, run the library remap
# once per compartment, and assemble template output by template `compartment`
# field. See HANDOFF_v1.2_tibia_compartments.md.


def _tibia_compartment_label(bone_pts: np.ndarray, cart_mask: np.ndarray,
                              spacing: tuple, ml_axis: int = 2) -> np.ndarray:
    """Returns int8 array (n_bone_verts,) with 1=medial, 2=lateral, 0=unassigned.

    Midline = mean ML coordinate of cart_mask voxels in physical mm. For RIGHT
    knees (and LEFT after canonical right-flip), medial = smaller ML value.
    The library applies the canonical right-knee flip before reaching here, so
    `medial = bone_pts[:, ml_axis] < midline` holds for both sides.

    `ml_axis`: index of the ML axis in patient-frame mm (default 2, matching
    `scripts/templates/add_tibia_compartment.py`).
    """
    cart_idx = np.argwhere(cart_mask.astype(bool))
    if len(cart_idx) == 0:
        return np.zeros(len(bone_pts), dtype=np.int8)
    cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
    midline = float(cart_mm[:, ml_axis].mean())
    labels = np.where(bone_pts[:, ml_axis] < midline, 1, 2).astype(np.int8)
    return labels


def compartment_aware_remap(template_mesh, aligned_points: np.ndarray,
                             patient_thickness: np.ndarray,
                             subch_prob: np.ndarray,
                             patient_compartment: np.ndarray,
                             k: int = 3, mutual_nn: bool = False,
                             eps: float = 1e-6, subch_threshold: float = 0.5):
    """Per-compartment IDW remap wrapper around `remap_thickness_to_template`.

    Calls the library remap twice (compartment 1=medial, 2=lateral). Each call
    sees only the patient subch verts belonging to that compartment (others
    masked to subch_prob=0 → excluded by the library's subch_threshold filter).
    Template output is assembled by the template's own `compartment` field —
    medial template verts get the medial-call result, lateral verts the
    lateral-call result. Verts with template `compartment==0` (non-articular,
    or anywhere outside subch zone) are left NaN.

    Requires `template_mesh.point_data["compartment"]` (uint8: 0/1/2). Raises
    KeyError if missing — caller should gate on bone_name == "tibia" and
    confirm the template has been augmented via
    `scripts/templates/add_tibia_compartment.py`.

    Returns (template_thickness, quality_dict) — same shape and dtype as the
    underlying library call. Quality dict contains per-compartment sub-dicts
    under keys "comp_1" and "comp_2".
    """
    tpl_compartment = np.asarray(template_mesh.point_data["compartment"])
    n_t = template_mesh.n_points
    out = np.full(n_t, np.nan, dtype=np.float32)
    quality: dict[str, Any] = {"compartment_aware": True}
    for lbl in (1, 2):
        in_lbl = (patient_compartment == lbl)
        sp_lbl = np.where(in_lbl, subch_prob, 0.0).astype(np.float32, copy=False)
        tpl_lbl, q_lbl = remap_thickness_to_template(
            template_mesh, aligned_points, patient_thickness, sp_lbl,
            k=k, mutual_nn=mutual_nn, eps=eps, subch_threshold=subch_threshold,
        )
        is_lbl_template = (tpl_compartment == lbl)
        out[is_lbl_template] = tpl_lbl[is_lbl_template]
        quality[f"comp_{lbl}"] = q_lbl
        quality[f"n_patient_subch_comp_{lbl}"] = int((in_lbl & (subch_prob >= subch_threshold)).sum())
        quality[f"n_template_comp_{lbl}"] = int(is_lbl_template.sum())
    # Aggregate top-level mirrors of remap_thickness_to_template keys so that
    # downstream code reading e.g. q["assd_subch_mm"] keeps working.
    quality["n_template_total"] = int(n_t)
    quality["n_template_subch"] = int(
        (np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold).sum()
    )
    quality["n_patient_subch"] = int(
        quality["n_patient_subch_comp_1"] + quality["n_patient_subch_comp_2"]
    )
    # Mean ASSD over the two compartments (each restricted to its own pool).
    assd1 = quality["comp_1"].get("assd_subch_mm", float("nan"))
    assd2 = quality["comp_2"].get("assd_subch_mm", float("nan"))
    finite = [v for v in (assd1, assd2) if np.isfinite(v)]
    quality["assd_subch_mm"] = float(np.mean(finite)) if finite else float("nan")
    quality["frac_mutual"] = float("nan")
    quality["mean_mapping_dist_mm_all_template"] = float("nan")
    quality["mean_mapping_dist_mm_subch_only"] = float("nan")
    return out, quality


def _tibia_template_has_compartment(template_mesh) -> bool:
    """True if the loaded tibia template has the `compartment` point_data
    field (added via scripts/templates/add_tibia_compartment.py)."""
    return "compartment" in template_mesh.point_data


# Module-level: current flip override state. Read by the RECON disk-cache key
# patcher in api.py so the cached file slot is partitioned by the actual flip
# applied (otherwise a cache hit under a different override returns the wrong
# orientation's densified volume).
ACTIVE_FLIP_OVERRIDE: bool | None = None


@contextlib.contextmanager
def _force_ml_flip(decision: bool | None):
    """Override the library's empirical ML-flip check with a fixed decision.

    Used by `process_long_shared_mesh` to force the same ML-flip across both
    timepoints — the empirical check is noisy (OA shape distortion throws
    off the tibia-mesh-vs-template fit) and can give *opposite answers* on
    00m vs 48m of the SAME patient, mirroring 48m and breaking the cross-
    timepoint ICP catastrophically.

    Also sets `ACTIVE_FLIP_OVERRIDE` so the RECON disk-cache key gets a
    `_flipT` / `_flipF` tag while this context is active — prevents stale
    cache hits when the override differs from the per-timepoint empirical
    decision that was used the first time RECON ran on this seg.

    `decision=None` → no override (use empirical check as before).
    """
    global ACTIVE_FLIP_OVERRIDE
    if decision is None:
        yield
        return
    original = _pipeline._empirical_flip_needed
    _pipeline._empirical_flip_needed = lambda *a, **k: bool(decision)
    prev_override = ACTIVE_FLIP_OVERRIDE
    ACTIVE_FLIP_OVERRIDE = bool(decision)
    try:
        yield
    finally:
        _pipeline._empirical_flip_needed = original
        ACTIVE_FLIP_OVERRIDE = prev_override


def _ml_flip_decision(seg_path: Path) -> bool:
    """Run the library's empirical ML-flip check once on a single seg file,
    without modifying anything else. Returns True if the seg needs flipping
    to match the right-knee canonical voxel layout."""
    import nibabel as nib
    nii = nib.load(str(seg_path))
    seg = nii.get_fdata().astype(np.int16)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return bool(_pipeline._empirical_flip_needed(seg, spacing, bone_label=3))


def _build_thickness(seg_path: Path, side: str, modality: str, bone_name: str,
                     config: PipelineConfig, dess_is_11class: bool = False):
    """Run the library's seg→mask→mesh→thickness chain for one input."""
    cm = _dess_labels_11class() if (modality == "dess" and dess_is_11class) else contextlib.nullcontext()
    with cm:
        bone_mask, cart_mask, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, modality, bone_name, config=config,
        )
    # Bone-mesh smoothing — raycast needs normals so smooth a few iters
    smooth_iters = 0 if config.thickness_method in ("edt", "template_raycast") else 5
    bone_mesh, _ = _pipeline.build_meshes(
        bone_mask, cart_mask, spacing, bone_smooth_iters=smooth_iters,
    )
    thickness_fn = get_thickness(config.thickness_method)
    thickness = thickness_fn(bone_mesh, bone_mask, cart_mask, spacing, config)
    return bone_mesh, bone_mask, cart_mask, spacing, thickness


# ---------------------------------------------------------------------------
# Public — independent registration (v1 baseline)
# ---------------------------------------------------------------------------
def process_pair_independent(pd_seg_path: Path, dess_seg_path: Path,
                             side: str, bone_name: str, config: PipelineConfig,
                             dess_is_11class: bool = True,
                             pd_is_dess_equivalent: bool = False) -> dict[str, Any]:
    """Both modalities registered independently to template via process_one_patient.

    `pd_is_dess_equivalent=True`: PD arm holds a DESS-equivalent 5-class label
    NIfTI (e.g. from a Dataset211 inference) — process via the dess code path
    using the default `DESS_LABELS`.
    """
    pd_modality = "dess" if pd_is_dess_equivalent else "pd"
    pd_res = _pipeline.process_one_patient(pd_seg_path, side, bone_name, pd_modality, config=config)
    cm = _dess_labels_11class() if dess_is_11class else contextlib.nullcontext()
    with cm:
        dess_res = _pipeline.process_one_patient(dess_seg_path, side, bone_name, "dess", config=config)
    return {
        "pd_template_thickness": np.asarray(pd_res["template_thickness"]),
        "dess_template_thickness": np.asarray(dess_res["template_thickness"]),
        "pd_quality": pd_res["quality"], "dess_quality": dess_res["quality"],
        "pd_transform": pd_res["transform_meta"], "dess_transform": dess_res["transform_meta"],
    }


def process_long_independent(seg_00m_path: Path, seg_48m_path: Path,
                             side: str, bone_name: str, config: PipelineConfig,
                             pd_is_dess_equivalent: bool = False) -> dict[str, Any]:
    pd_modality = "dess" if pd_is_dess_equivalent else "pd"
    r00 = _pipeline.process_one_patient(seg_00m_path, side, bone_name, pd_modality, config=config)
    r48 = _pipeline.process_one_patient(seg_48m_path, side, bone_name, pd_modality, config=config)
    return {
        "tp00_template_thickness": np.asarray(r00["template_thickness"]),
        "tp48_template_thickness": np.asarray(r48["template_thickness"]),
        "tp00_quality": r00["quality"], "tp48_quality": r48["quality"],
    }


# ---------------------------------------------------------------------------
# Public — shared-mesh (v6 canonical)
# ---------------------------------------------------------------------------
def process_pair_shared_mesh(pd_seg_path: Path, dess_seg_path: Path,
                             side: str, bone_name: str, config: PipelineConfig,
                             dess_is_11class: bool = True,
                             pd_is_dess_equivalent: bool = False,
                             out_vtk_dir: Path | None = None,
                             case_id: str | None = None,
                             articular_radius_mm: float | None = 5.0,
                             compartment_aware: bool = True) -> dict[str, Any]:
    """v6 canonical shared-mesh PD↔DESS pair. DESS is the canonical sampling surface.

    `articular_radius_mm` (default 5.0, v6.1 canonical): restrict the
    PD→DESS rigid ICP to bone verts within this distance of either side's
    cart_mask. Eliminates the tibia-plate rotational ambiguity that
    inflates per-knee Δ noise. Pass None to disable (full-bone ICP).

    `pd_is_dess_equivalent=True`: PD arm holds a DESS-equivalent 5-class label
    NIfTI (e.g. from Dataset211 inference, on the PD-SAG voxel grid). The PD
    arm is then processed via the dess code path using the default 5-class
    `DESS_LABELS` (femur 1/2, tibia 3/4) — RECON densification is skipped (the
    masks are already DESS-quality), and through-plane interpolation kicks in
    if the PD slab thickness exceeds 1.5 mm.

    `out_vtk_dir` + `case_id`: if both given, persist patient + template VTKs
    for 3D viz:
      - `<case_id>_<bone>_pd_patient.vtk`   PD bone mesh in patient mm, thickness_mm scalar
      - `<case_id>_<bone>_dess_patient.vtk` DESS bone mesh in patient mm, thickness_mm scalar
      - `<case_id>_<bone>_template.vtk`     template mesh with pd_thickness_mm + dess_thickness_mm + subch_prob
    """
    template_mesh = pv.read(str(TEMPLATE_PATHS[bone_name]))
    if "subch_prob" not in template_mesh.point_data:
        raise RuntimeError(f"Template {TEMPLATE_PATHS[bone_name]} missing 'subch_prob'")
    t0 = time.time()

    # 1) DESS (canonical surface)
    print(f"  [DESS] {dess_seg_path.name}")
    de_mesh, de_bone, de_cart, de_sp, de_thick = _build_thickness(
        dess_seg_path, side, "dess", bone_name, config, dess_is_11class=dess_is_11class,
    )
    print(f"    bone={int(de_bone.sum())} cart={int(de_cart.sum())} mesh={de_mesh.n_points}v "
          f"thick>0={int((de_thick > 0).sum())} mean={de_thick[de_thick > 0].mean():.2f}mm")

    # 2) PD in its own frame
    pd_modality = "dess" if pd_is_dess_equivalent else "pd"
    print(f"  [PD]   {pd_seg_path.name}  (modality={pd_modality}"
          + (f", 5-class DESS-equivalent" if pd_is_dess_equivalent else "") + ")")
    pd_mesh, pd_bone, pd_cart, pd_sp, pd_thick = _build_thickness(
        pd_seg_path, side, pd_modality, bone_name, config,
    )
    print(f"    bone={int(pd_bone.sum())} cart={int(pd_cart.sum())} mesh={pd_mesh.n_points}v "
          f"thick>0={int((pd_thick > 0).sum())} mean={pd_thick[pd_thick > 0].mean():.2f}mm")

    # 3) Rigid PD→DESS in patient frame (articular-only by default, v6.1 fix).
    #    v1.2 — config may select trimmed_rigid (partial-overlap robust) via
    #    `long_icp_method`; defaults to plain rigid for backward-compat.
    _icp_method = getattr(config, "long_icp_method", "rigid")
    _trim_frac = float(getattr(config, "long_trim_fraction", 0.8))
    pd_pts = np.asarray(pd_mesh.points); de_pts = np.asarray(de_mesh.points)
    assd_before = _bone_assd(pd_pts, de_pts)
    pd_aligned, R, t, used_art, n_pd_art, n_de_art, _ = _articular_icp(
        pd_pts, pd_cart, pd_sp, de_pts, de_cart, de_sp,
        articular_radius_mm=articular_radius_mm,
        method=_icp_method, trim_fraction=_trim_frac,
    )
    assd_after = _bone_assd(pd_aligned, de_pts)
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    art_tag = f"art-ICP rad={articular_radius_mm}mm PD={n_pd_art} DESS={n_de_art}" if used_art else "full-bone-ICP (art fallback)"
    print(f"  PD→DESS [{art_tag}]: ASSD {assd_before:.2f}→{assd_after:.2f}mm rot={rot_deg:.1f}° |t|={np.linalg.norm(t):.2f}mm")

    # 4) IDW: sample PD thickness onto DESS bone-verts
    pd_thick_at_de = _idw_sample(de_pts, pd_aligned, pd_thick, k=3).astype(np.float32)

    # 5) DESS bone → template
    de_aligned, _sf, _tr, _Ri, _ti = patient_to_template_align(
        de_mesh, de_cart, de_sp, template_mesh, bone_name, modality="dess",
    )

    # 6) Subch via config.subch_method on the DESS-aligned verts (same source for both)
    #    `tpl_nn` (legacy/canonical): template-NN lookup, anchor-dependent
    #    `cart_prox`: subch ≡ patient vert within cfg.cart_prox_mm of any cart voxel
    #                 — anchor-INVARIANT, excludes peripheral over-expansion
    #    `cart_hit`: subch ≡ thickness > 0 — drops denudation, diagnostic only
    subch_fn = get_subch(config.subch_method)
    de_subch_prob, _ = subch_fn(de_aligned, template_mesh, de_cart, de_sp,
                                 de_mesh, de_thick, config)

    # 7) IDW remap both to template using the shared source verts + subch zone.
    #    v1.2 — for tibia (when the template has a `compartment` field), split
    #    the IDW into medial/lateral passes; see compartment_aware_remap.
    use_compartment = (compartment_aware and bone_name == "tibia"
                        and _tibia_template_has_compartment(template_mesh))
    if use_compartment:
        de_compartment = _tibia_compartment_label(de_pts, de_cart, de_sp)
        print(f"  compartment-aware remap: med={int((de_compartment==1).sum())} "
              f"lat={int((de_compartment==2).sum())} (patient bone verts)")
        de_tpl, q_de = compartment_aware_remap(
            template_mesh, de_aligned, de_thick, de_subch_prob, de_compartment,
            k=config.idw_k, mutual_nn=config.idw_mutual_nn,
            subch_threshold=config.subch_threshold,
        )
        pd_tpl, q_pd = compartment_aware_remap(
            template_mesh, de_aligned, pd_thick_at_de, de_subch_prob, de_compartment,
            k=config.idw_k, mutual_nn=config.idw_mutual_nn,
            subch_threshold=config.subch_threshold,
        )
    else:
        de_tpl, q_de = remap_thickness_to_template(
            template_mesh, de_aligned, de_thick, de_subch_prob, k=config.idw_k,
            mutual_nn=config.idw_mutual_nn, subch_threshold=config.subch_threshold,
        )
        pd_tpl, q_pd = remap_thickness_to_template(
            template_mesh, de_aligned, pd_thick_at_de, de_subch_prob, k=config.idw_k,
            mutual_nn=config.idw_mutual_nn, subch_threshold=config.subch_threshold,
        )
    template_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= config.subch_threshold
    de_tpl_m = de_tpl.copy(); de_tpl_m[~template_subch] = np.nan
    pd_tpl_m = pd_tpl.copy(); pd_tpl_m[~template_subch] = np.nan

    if out_vtk_dir is not None and case_id is not None:
        out = Path(out_vtk_dir); out.mkdir(parents=True, exist_ok=True)
        # Patient bone meshes in patient mm with per-vert thickness
        pd_save = pd_mesh.copy(); pd_save["thickness_mm"] = pd_thick.astype(np.float32)
        de_save = de_mesh.copy(); de_save["thickness_mm"] = de_thick.astype(np.float32)
        pd_save.save(str(out / f"{case_id}_{bone_name}_pd_patient.vtk"))
        de_save.save(str(out / f"{case_id}_{bone_name}_dess_patient.vtk"))
        tpl_save = template_mesh.copy()
        tpl_save["pd_thickness_mm"] = pd_tpl_m.astype(np.float32)
        tpl_save["dess_thickness_mm"] = de_tpl_m.astype(np.float32)
        tpl_save["delta_mm"] = (pd_tpl_m - de_tpl_m).astype(np.float32)
        tpl_save.save(str(out / f"{case_id}_{bone_name}_template.vtk"))

    print(f"  shared-mesh pair done in {time.time() - t0:.1f}s")
    return {
        "pd_template_thickness": pd_tpl_m,
        "dess_template_thickness": de_tpl_m,
        "pd_quality": q_pd, "dess_quality": q_de,
        "pair_quality": {
            "bone_assd_before_mm": assd_before, "bone_assd_after_mm": assd_after,
            "pd2dess_rot_deg": rot_deg, "pd2dess_trans_mm": float(np.linalg.norm(t)),
            "articular_icp": used_art, "articular_radius_mm": articular_radius_mm,
            "n_pd_articular": n_pd_art, "n_dess_articular": n_de_art,
        },
    }


def process_long_shared_mesh(seg_00m_path: Path, seg_48m_path: Path,
                             side: str, bone_name: str, config: PipelineConfig,
                             pd_is_dess_equivalent: bool = False,
                             out_vtk_dir: Path | None = None,
                             case_id: str | None = None,
                             articular_radius_mm: float | None = 5.0,
                             icp_method: str = "rigid",
                             consistent_flip: bool = True,
                             remap_method: str = "idw",
                             compartment_aware: bool = True) -> dict[str, Any]:
    """v6 canonical shared-mesh longitudinal pair. 00m is the canonical surface.

    `articular_radius_mm` (default 5.0, v6.1 canonical): restrict the 48m→00m
    rigid ICP to bone verts within this distance of each timepoint's
    cart_mask. Eliminates the tibia-plate rotational ambiguity.

    `consistent_flip` (default True, v1.2 fix): determine the ML-flip
    empirically on 00m ONCE, then force the same decision on 48m. The
    library's per-timepoint empirical check is noisy (depends on tibia
    mesh-vs-template fit, which is OA-shape-dependent) and can give
    opposite answers on the SAME patient — mirroring 48m and breaking the
    cross-timepoint ICP (rot=40°, |t|=80 mm in the worst case 9477205).

    `pd_is_dess_equivalent=True`: both timepoints hold DESS-equivalent 5-class
    label NIfTIs (e.g. Dataset211 inference outputs). Same code path as the
    pair-mode flag — `modality='dess'` with default `DESS_LABELS`.

    `out_vtk_dir` + `case_id`: if both given, persist VTKs for 3D viz:
      - `<case_id>_<bone>_00m_patient.vtk`  00m bone mesh in patient mm, thickness_mm
      - `<case_id>_<bone>_48m_patient.vtk`  48m bone mesh in patient mm, thickness_mm
      - `<case_id>_<bone>_template.vtk`     template mesh with tp00/tp48/delta scalars
    """
    template_mesh = pv.read(str(TEMPLATE_PATHS[bone_name]))
    if "subch_prob" not in template_mesh.point_data:
        raise RuntimeError(f"Template {TEMPLATE_PATHS[bone_name]} missing 'subch_prob'")
    t0 = time.time()

    pd_modality = "dess" if pd_is_dess_equivalent else "pd"
    tag = f"modality={pd_modality}" + (" 5-class DESS-eq" if pd_is_dess_equivalent else "")

    # v1.2 fix — determine ML-flip from 00m once, force the same on 48m
    flip_override = None
    if consistent_flip and not pd_is_dess_equivalent:
        flip_override = _ml_flip_decision(seg_00m_path)
        print(f"  [flip] consistent decision from 00m: flip={flip_override}")

    print(f"  [00m] {seg_00m_path.name}  ({tag})")
    with _force_ml_flip(flip_override):
        m00, b00, c00, sp00, th00 = _build_thickness(seg_00m_path, side, pd_modality, bone_name, config)
    print(f"    bone={int(b00.sum())} cart={int(c00.sum())} mesh={m00.n_points}v "
          f"thick>0={int((th00 > 0).sum())} mean={th00[th00 > 0].mean():.2f}mm")
    print(f"  [48m] {seg_48m_path.name}  ({tag})")
    with _force_ml_flip(flip_override):
        m48, b48, c48, sp48, th48 = _build_thickness(seg_48m_path, side, pd_modality, bone_name, config)
    print(f"    bone={int(b48.sum())} cart={int(c48.sum())} mesh={m48.n_points}v "
          f"thick>0={int((th48 > 0).sum())} mean={th48[th48 > 0].mean():.2f}mm")

    p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
    assd_before = _bone_assd(p48, p00)
    # v1.2 — config.long_icp_method (e.g. "trimmed_rigid") overrides the
    # function-arg default so api.validate can drive it via PipelineConfig.
    _icp_method = getattr(config, "long_icp_method", None) or icp_method
    _trim_frac = float(getattr(config, "long_trim_fraction", 0.8))
    p48_aligned, A, t, used_art, n_48_art, n_00_art, method_used = _articular_icp(
        p48, c48, sp48, p00, c00, sp00,
        articular_radius_mm=articular_radius_mm,
        method=_icp_method, trim_fraction=_trim_frac,
    )
    assd_after = _bone_assd(p48_aligned, p00)
    if method_used == "rigid":
        rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
        rot_tag = f"rot={rot_deg:.1f}°"
    else:
        sv = np.linalg.svd(A, compute_uv=False)
        rot_deg = float("nan")
        rot_tag = f"sv=[{sv[0]:.2f},{sv[1]:.2f},{sv[2]:.2f}]"
    art_tag = f"art-{method_used}-ICP rad={articular_radius_mm}mm 48m={n_48_art} 00m={n_00_art}" if used_art else f"full-bone-{method_used}-ICP (art fallback)"
    print(f"  48m→00m [{art_tag}]: ASSD {assd_before:.2f}→{assd_after:.2f}mm {rot_tag} |t|={np.linalg.norm(t):.2f}mm")

    th48_at_00 = _idw_sample(p00, p48_aligned, th48, k=3).astype(np.float32)

    aligned_00, _sf, _tr, _Ri, _ti = patient_to_template_align(
        m00, c00, sp00, template_mesh, bone_name, modality=pd_modality,
    )
    # Subch via config.subch_method (see process_pair_shared_mesh §6 for rationale)
    subch_fn = get_subch(config.subch_method)
    subch_prob_00, _ = subch_fn(aligned_00, template_mesh, c00, sp00,
                                  m00, th00, config)

    # Template remap: forward IDW k=3 (idw, library canonical) or scatter k=1
    # (reverse direction, preserves local spatial Δ pattern — see
    # scatter_thickness_to_template docstring).
    #
    # v1.2 — for tibia (when the template has a `compartment` field), the
    # forward IDW path splits into medial/lateral passes. Scatter path is
    # unchanged for now (experimental switch; compartment fix can be added
    # later if needed).
    use_compartment = (compartment_aware and bone_name == "tibia"
                        and remap_method != "scatter"
                        and _tibia_template_has_compartment(template_mesh))
    if remap_method == "scatter":
        tpl00, q00 = scatter_thickness_to_template(
            template_mesh, aligned_00, th00, subch_prob_00,
            subch_threshold=config.subch_threshold,
        )
        tpl48, q48 = scatter_thickness_to_template(
            template_mesh, aligned_00, th48_at_00, subch_prob_00,
            subch_threshold=config.subch_threshold,
        )
        print(f"  remap=scatter k=1  touched_frac={q00['frac_template_touched']:.1%}  "
              f"mean_patient_per_vert={q00['mean_patient_per_template_touched']:.1f}")
    elif use_compartment:
        compartment_00 = _tibia_compartment_label(p00, c00, sp00)
        print(f"  compartment-aware remap: med={int((compartment_00==1).sum())} "
              f"lat={int((compartment_00==2).sum())} (patient bone verts)")
        tpl00, q00 = compartment_aware_remap(
            template_mesh, aligned_00, th00, subch_prob_00, compartment_00,
            k=config.idw_k, mutual_nn=config.idw_mutual_nn,
            subch_threshold=config.subch_threshold,
        )
        tpl48, q48 = compartment_aware_remap(
            template_mesh, aligned_00, th48_at_00, subch_prob_00, compartment_00,
            k=config.idw_k, mutual_nn=config.idw_mutual_nn,
            subch_threshold=config.subch_threshold,
        )
    else:
        tpl00, q00 = remap_thickness_to_template(
            template_mesh, aligned_00, th00, subch_prob_00, k=config.idw_k,
            mutual_nn=config.idw_mutual_nn, subch_threshold=config.subch_threshold,
        )
        tpl48, q48 = remap_thickness_to_template(
            template_mesh, aligned_00, th48_at_00, subch_prob_00, k=config.idw_k,
            mutual_nn=config.idw_mutual_nn, subch_threshold=config.subch_threshold,
        )
    template_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= config.subch_threshold
    tpl00_m = tpl00.copy(); tpl00_m[~template_subch] = np.nan
    tpl48_m = tpl48.copy(); tpl48_m[~template_subch] = np.nan

    if out_vtk_dir is not None and case_id is not None:
        out = Path(out_vtk_dir); out.mkdir(parents=True, exist_ok=True)
        # 00m bone mesh carries BOTH timepoints' thickness (48m IDW'd onto 00m
        # verts) — this is the canonical patient-frame artefact for the no-
        # template longitudinal diagnostic.
        m00_save = m00.copy()
        m00_save["thickness_mm"] = th00.astype(np.float32)   # alias for backward compat
        m00_save["tp00_thickness_mm"] = th00.astype(np.float32)
        m00_save["tp48_thickness_mm"] = th48_at_00.astype(np.float32)
        m00_save["delta_mm"] = (th48_at_00 - th00).astype(np.float32)
        m48_save = m48.copy(); m48_save["thickness_mm"] = th48.astype(np.float32)
        m00_save.save(str(out / f"{case_id}_{bone_name}_00m_patient.vtk"))
        m48_save.save(str(out / f"{case_id}_{bone_name}_48m_patient.vtk"))
        tpl_save = template_mesh.copy()
        tpl_save["tp00_thickness_mm"] = tpl00_m.astype(np.float32)
        tpl_save["tp48_thickness_mm"] = tpl48_m.astype(np.float32)
        tpl_save["delta_mm"] = (tpl48_m - tpl00_m).astype(np.float32)
        tpl_save.save(str(out / f"{case_id}_{bone_name}_template.vtk"))

    print(f"  shared-mesh long done in {time.time() - t0:.1f}s")
    return {
        "tp00_template_thickness": tpl00_m,
        "tp48_template_thickness": tpl48_m,
        # Patient-frame intermediates (00m bone mesh as the shared sampling surface)
        "bone_mesh_00m": m00,
        "th00_patient": np.asarray(th00, dtype=np.float32),
        "th48_at_00_patient": np.asarray(th48_at_00, dtype=np.float32),
        # v1.2 — bone-based subch_prob per 00m bone vert (template-NN lookup,
        # gap-free vs the raycast `thickness > 0` mask). Use this for clean
        # compartment-zone visualization on the patient frame.
        "subch_prob_00_patient": np.asarray(subch_prob_00, dtype=np.float32),
        # 00m bone verts in TEMPLATE space (post patient_to_template_align).
        # Index-aligned with bone_mesh_00m.points. Lets downstream diagnostics
        # inherit per-vert template fields (e.g. compartment) via template-NN.
        "aligned_pts_00": np.asarray(aligned_00, dtype=np.float32),
        "tp00_quality": q00, "tp48_quality": q48,
        "pair_quality": {
            "bone_assd_before_mm": assd_before, "bone_assd_after_mm": assd_after,
            "long_rot_deg": rot_deg, "long_trans_mm": float(np.linalg.norm(t)),
            "articular_icp": used_art, "articular_radius_mm": articular_radius_mm,
            "n_48m_articular": n_48_art, "n_00m_articular": n_00_art,
        },
    }


def process_long_baseline_grid(seg_00m_path: Path, seg_48m_path: Path,
                               side: str, bone_name: str, config: PipelineConfig,
                               pd_is_dess_equivalent: bool = False,
                               articular_radius_mm: float | None = 5.0,
                               consistent_flip: bool = True) -> dict[str, Any]:
    """OPT-IN longitudinal path using the v3.3 per-patient BASELINE-GRID
    projection instead of the atlas template (config.region_projection ==
    "baseline_grid"). The load + densify + 48m->00m registration are identical to
    `process_long_shared_mesh`; only the final regional projection differs — both
    timepoints' thickness (th00 and 48m-IDW'd-onto-00m) are binned onto a 40x40
    grid built from the patient's OWN 00m bone geometry. Template path untouched.

    Returns {"baseline_grid_regions": {region: {"00m","48m","d"}}, ...quality}.
    """
    from .. import baseline_grid as _bg

    t0 = time.time()
    pd_modality = "dess" if pd_is_dess_equivalent else "pd"

    flip_override = None
    if consistent_flip and not pd_is_dess_equivalent:
        flip_override = _ml_flip_decision(seg_00m_path)
        print(f"  [flip] consistent decision from 00m: flip={flip_override}")

    print(f"  [00m] {seg_00m_path.name}  (baseline_grid, modality={pd_modality})")
    with _force_ml_flip(flip_override):
        m00, b00, c00, sp00, th00 = _build_thickness(seg_00m_path, side, pd_modality, bone_name, config)
    print(f"  [48m] {seg_48m_path.name}  (baseline_grid, modality={pd_modality})")
    with _force_ml_flip(flip_override):
        m48, b48, c48, sp48, th48 = _build_thickness(seg_48m_path, side, pd_modality, bone_name, config)

    p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
    assd_before = _bone_assd(p48, p00)
    _icp_method = getattr(config, "long_icp_method", None) or "rigid"
    _trim_frac = float(getattr(config, "long_trim_fraction", 0.8))
    p48_aligned, A, t, used_art, n_48_art, n_00_art, method_used = _articular_icp(
        p48, c48, sp48, p00, c00, sp00,
        articular_radius_mm=articular_radius_mm,
        method=_icp_method, trim_fraction=_trim_frac,
    )
    assd_after = _bone_assd(p48_aligned, p00)
    print(f"  48m→00m [art-{method_used}-ICP rad={articular_radius_mm}mm]: "
          f"ASSD {assd_before:.2f}→{assd_after:.2f}mm |t|={np.linalg.norm(t):.2f}mm")

    # Both timepoints' thickness now live on the SAME 00m bone-mesh vertices p00.
    th48_at_00 = _idw_sample(p00, p48_aligned, th48, k=3).astype(np.float32)

    # AXIS CONVENTION: this pipeline's mesh/mask order is (AP, SI, ML) — see
    # pipeline.py `ml_r=ptp(verts[:,2])`, `ap_r=ptp(verts[:,0])`. The v3.3 grid
    # expects (ML, SI, AP) — axis0=ML (medial/lateral split), axis2=AP (a-p arc /
    # width). Reorder both masks and points by swapping AP<->ML (axes 2,1,0)
    # before gridding. All knees are flipped to right-knee orientation upstream,
    # so v3.3's "right_oriented" D-axis convention (low ML index = medial) holds.
    b00_g = np.ascontiguousarray(np.transpose(b00, (2, 1, 0)))
    c00_g = np.ascontiguousarray(np.transpose(c00, (2, 1, 0)))
    sp00_g = (sp00[2], sp00[1], sp00[0])
    p00_g = np.ascontiguousarray(p00[:, [2, 1, 0]])
    regions = _bg.regional_deltas(
        bone_name, p00_g, np.asarray(th00, np.float32), th48_at_00,
        b00_g, c00_g, sp00_g, laterality="right_oriented",
        femur_unwrap=getattr(config, "femur_unwrap", "per_slice"),
    )
    fp = regions.pop("_footprint_bins", 0)
    g00 = regions.pop("_grid_00m", None); g48 = regions.pop("_grid_48m", None)
    verts = regions.pop("_verts", None)
    md = ", ".join(f"{k}Δ={v['d']:+.3f}" for k, v in regions.items() if k in ("cMF", "MT", "cLF", "LT"))
    print(f"  baseline-grid done in {time.time()-t0:.1f}s  footprint_bins={fp}  {md}")
    return {
        "baseline_grid_regions": {k: v for k, v in regions.items()},
        "grid_00m": g00, "grid_48m": g48, "verts": verts,
        "pair_quality": {
            "bone_assd_before_mm": assd_before, "bone_assd_after_mm": assd_after,
            "long_trans_mm": float(np.linalg.norm(t)),
            "articular_icp": used_art, "footprint_bins": fp,
        },
    }


# Backwards-compat thin aliases for api.py (default = shared-mesh)
def process_pair(*args, shared_mesh: bool = True, **kwargs):
    fn = process_pair_shared_mesh if shared_mesh else process_pair_independent
    return fn(*args, **kwargs)


def process_long(*args, shared_mesh: bool = True, **kwargs):
    fn = process_long_shared_mesh if shared_mesh else process_long_independent
    return fn(*args, **kwargs)
