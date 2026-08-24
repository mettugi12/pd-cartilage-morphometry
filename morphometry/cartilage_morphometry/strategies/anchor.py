"""Registration-anchor strategies: patient bone mesh → DESS template frame.

Each anchor takes (bone_mesh, cart_mask, spacing, template_mesh, bone_name,
modality, cfg) and returns (aligned_points, transform_meta_dict).

transform_meta is a dict logging what each strategy actually did. Always
includes:
    rot_deg        — final rotation angle (axis-angle, degrees)
    trans_mm       — final translation magnitude (mm)
And optionally (per strategy):
    scale_factors  — (ml, ap, z) for aniso variants
    iso_scale      — single scalar for rigid_only_iso_scale
    sv_min/sv_max  — singular value bounds applied (bounded affine)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import register_anchor


# ---------------------------------------------------------------------------
# Target ML/AP dimensions (mm) used by aniso variants — matches template build
# ---------------------------------------------------------------------------
TARGET_SCALE_MM = {
    "femur": (75.0, 65.0),  # (target_ml, target_ap)
    "tibia": (75.0, 55.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _articular_source_mask(bone_mesh, cart_mask, spacing, articular_radius_mm):
    """Patient bone-mesh verts within `articular_radius_mm` of any cart voxel."""
    cart_idx = np.argwhere(cart_mask.astype(bool))
    if len(cart_idx) == 0:
        return np.ones(bone_mesh.n_points, dtype=bool)
    cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
    cart_tree = cKDTree(cart_mm)
    d_to_cart, _ = cart_tree.query(np.asarray(bone_mesh.points), k=1)
    return d_to_cart <= articular_radius_mm


def _template_subch_points(template_mesh, subch_threshold=0.5):
    pts = np.asarray(template_mesh.points)
    mask = np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold
    return pts[mask], pts


def _apply_aniso(points, scale_factors):
    ml, ap, z = scale_factors
    M = np.array([[ap, 0, 0, 0], [0, z, 0, 0], [0, 0, ml, 0], [0, 0, 0, 1]])
    homog = np.hstack([points, np.ones((len(points), 1))])
    return (homog @ M.T)[:, :3]


def _femur_scale_factors(cart_mask, spacing, target_ml, target_ap):
    cart_idx = np.argwhere(cart_mask.astype(bool))
    if len(cart_idx) < 100:
        raise ValueError(f"Too few cart voxels ({len(cart_idx)}) for femur aniso scale")
    cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
    ml_r = max(float(np.ptp(cart_mm[:, 2])), 1e-6)
    ap_r = max(float(np.ptp(cart_mm[:, 0])), 1e-6)
    return (target_ml / ml_r, target_ap / ap_r, 1.0)


def _tibia_scale_factors(bone_mesh, target_ml, target_ap):
    b = bone_mesh.bounds
    ml = b[5] - b[4]
    ap = b[1] - b[0]
    if ml <= 0 or ap <= 0:
        raise ValueError("Tibia bone mesh has degenerate bounds")
    return (target_ml / ml, target_ap / ap, 1.0)


def _bone_diagonal_iso_scale(bone_mesh, template_mesh):
    """Single isotropic scale matching patient-bone diagonal to template-bone diagonal."""
    pb = bone_mesh.bounds  # xmin,xmax,ymin,ymax,zmin,zmax
    pd = np.hypot.reduce([pb[1] - pb[0], pb[3] - pb[2], pb[5] - pb[4]])
    tb = template_mesh.bounds
    td = np.hypot.reduce([tb[1] - tb[0], tb[3] - tb[2], tb[5] - tb[4]])
    if pd <= 0 or td <= 0:
        raise ValueError("Degenerate bone diagonal for iso-scale")
    return td / pd


def _rigid_icp(src, tgt, max_iter, tol):
    """SVD-based rigid ICP — aligns src to tgt. Returns (aligned, R, t)."""
    s = src.copy()
    target_tree = cKDTree(tgt)
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_err = np.inf
    for _ in range(max_iter):
        _, idx = target_tree.query(s, k=1)
        match = tgt[idx]
        sc = s.mean(0)
        mc = match.mean(0)
        A = s - sc
        B = match - mc
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step = Vt.T @ U.T
        t_step = mc - R_step @ sc
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step
        s = (R_step @ s.T).T + t_step
        err = float(np.linalg.norm(s - match, axis=1).mean())
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return s, R_total, t_total


def _bounded_affine_icp(src, tgt, max_iter, tol, sv_min, sv_max):
    """Affine ICP with singular values clamped to [sv_min, sv_max] every iter.

    Prevents the unconstrained-affine collapse documented in PDvDESS 2026-05-21
    (sv 0.10, anisotropy 10× on 9117066_LEFT femur), while still allowing
    modest non-uniform scaling that helps on outlier knees.
    Returns (aligned, A, t)  where aligned = src @ A.T + t.
    """
    s = src.copy()
    target_tree = cKDTree(tgt)
    A_total = np.eye(3)
    t_total = np.zeros(3)
    prev_err = np.inf
    sv_log = (1.0, 1.0, 1.0)
    for _ in range(max_iter):
        _, idx = target_tree.query(s, k=1)
        match = tgt[idx]
        sc = s.mean(0)
        mc = match.mean(0)
        # Least-squares 3x3 linear that maps centered src → centered match
        A_step, *_ = np.linalg.lstsq(s - sc, match - mc, rcond=None)
        A_step = A_step.T   # so that match - mc ≈ A_step @ (s - sc)
        # Clamp singular values
        U, S, Vt = np.linalg.svd(A_step)
        S_clamped = np.clip(S, sv_min, sv_max)
        sv_log = tuple(float(v) for v in S_clamped)
        A_step = U @ np.diag(S_clamped) @ Vt
        if np.linalg.det(A_step) < 0:
            # Flip last sv sign to avoid reflection
            S_clamped[-1] *= -1
            A_step = U @ np.diag(S_clamped) @ Vt
        t_step = mc - A_step @ sc
        A_total = A_step @ A_total
        t_total = A_step @ t_total + t_step
        s = (A_step @ s.T).T + t_step
        err = float(np.linalg.norm(s - match, axis=1).mean())
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return s, A_total, t_total, sv_log


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Common pre-ICP step: build articular subsets in patient (post-pre-scale) frame
# ---------------------------------------------------------------------------
def _prepare_articular(scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg):
    """Returns (articular_src_mask, src_for_icp, target_for_icp, translation,
    translated_pts) — common across anchor variants. `scaled_pts` may be the
    raw bone_mesh.points (no scale) or scale-applied pts.
    """
    articular_src_mask = _articular_source_mask(
        bone_mesh, cart_mask, spacing, cfg.articular_radius_mm
    )
    n_src = int(articular_src_mask.sum())
    template_articular, template_all = _template_subch_points(template_mesh, cfg.subch_threshold)

    if n_src > 100 and len(template_articular) > 100:
        src_art_scaled = scaled_pts[articular_src_mask]
        translation = template_articular.mean(0) - src_art_scaled.mean(0)
        target_for_icp = template_articular
        translated_pts = scaled_pts + translation
        src_for_icp = translated_pts[articular_src_mask]
    else:
        translation = template_all.mean(0) - scaled_pts.mean(0)
        target_for_icp = template_all
        translated_pts = scaled_pts + translation
        src_for_icp = translated_pts
    return articular_src_mask, src_for_icp, target_for_icp, translation, translated_pts


# ---------------------------------------------------------------------------
# Anchor strategies
# ---------------------------------------------------------------------------
@register_anchor("aniso_rigid")
def aniso_rigid(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
    """Current default: aniso scale (cart-bbox or bone-bbox) + translation +
    articular-subset rigid ICP. Sensitive to far-from-bone cart artifacts
    (use with cart_cleanup in cfg).
    """
    target_ml, target_ap = TARGET_SCALE_MM[bone_name]
    if bone_name == "femur":
        scale_factors = _femur_scale_factors(cart_mask, spacing, target_ml, target_ap)
    else:
        scale_factors = _tibia_scale_factors(bone_mesh, target_ml, target_ap)
    scaled_pts = _apply_aniso(bone_mesh.points, scale_factors)
    _, src_for_icp, target_for_icp, translation, translated_pts = _prepare_articular(
        scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg
    )
    _, R_icp, t_icp = _rigid_icp(
        src_for_icp, target_for_icp, cfg.rigid_icp_max_iter, cfg.rigid_icp_tol
    )
    aligned_pts = (R_icp @ translated_pts.T).T + t_icp
    return aligned_pts, {
        "anchor": "aniso_rigid",
        "scale_factors": tuple(float(s) for s in scale_factors),
        "translation_mm": [float(x) for x in translation],
        "rot_deg": _rot_deg(R_icp),
        "trans_mm": float(np.linalg.norm(t_icp)),
    }


@register_anchor("rigid_only")
def rigid_only(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
    """No scale at all — just translation (articular centroid match) + rigid ICP.
    Tests the hypothesis that rigid ICP alone suffices when registration is
    not corrupted by artifact-driven aniso scale. PDvDESS 2026-05-21:
    identity-scale ICP on 9226021_RIGHT_femur → ASSD 1.77 mm vs 6.67 mm with
    artifact-corrupted aniso+rigid.
    """
    scaled_pts = np.asarray(bone_mesh.points)
    _, src_for_icp, target_for_icp, translation, translated_pts = _prepare_articular(
        scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg
    )
    _, R_icp, t_icp = _rigid_icp(
        src_for_icp, target_for_icp, cfg.rigid_icp_max_iter, cfg.rigid_icp_tol
    )
    aligned_pts = (R_icp @ translated_pts.T).T + t_icp
    return aligned_pts, {
        "anchor": "rigid_only",
        "scale_factors": None,
        "translation_mm": [float(x) for x in translation],
        "rot_deg": _rot_deg(R_icp),
        "trans_mm": float(np.linalg.norm(t_icp)),
    }


@register_anchor("rigid_only_iso_scale")
def rigid_only_iso_scale(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
    """Single isotropic scale (patient-bone diagonal → template-bone diagonal)
    + translation + rigid ICP. A middle ground if pure rigid fails on bones
    very different in size from the OAI template.
    """
    iso = _bone_diagonal_iso_scale(bone_mesh, template_mesh)
    scaled_pts = np.asarray(bone_mesh.points) * iso
    _, src_for_icp, target_for_icp, translation, translated_pts = _prepare_articular(
        scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg
    )
    _, R_icp, t_icp = _rigid_icp(
        src_for_icp, target_for_icp, cfg.rigid_icp_max_iter, cfg.rigid_icp_tol
    )
    aligned_pts = (R_icp @ translated_pts.T).T + t_icp
    return aligned_pts, {
        "anchor": "rigid_only_iso_scale",
        "iso_scale": float(iso),
        "scale_factors": None,
        "translation_mm": [float(x) for x in translation],
        "rot_deg": _rot_deg(R_icp),
        "trans_mm": float(np.linalg.norm(t_icp)),
    }


@register_anchor("bounded_affine")
def bounded_affine(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
    """Articular-subset bounded-affine ICP (sv clamped to cfg.bounded_affine_sv_range,
    default [0.7, 1.3]). PDvDESS 2026-05-21: never collapses, modest help on
    outliers (9634187 femur rigid 3.43 → bounded 1.33 mm ASSD).
    """
    scaled_pts = np.asarray(bone_mesh.points)
    _, src_for_icp, target_for_icp, translation, translated_pts = _prepare_articular(
        scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg
    )
    sv_min, sv_max = cfg.bounded_affine_sv_range
    _, A_total, t_total, sv_log = _bounded_affine_icp(
        src_for_icp, target_for_icp, cfg.rigid_icp_max_iter, cfg.rigid_icp_tol,
        sv_min=sv_min, sv_max=sv_max,
    )
    aligned_pts = (A_total @ translated_pts.T).T + t_total
    # Estimate "rotation" from polar decomposition for QA logging
    U, _, Vt = np.linalg.svd(A_total)
    R_eff = U @ Vt
    if np.linalg.det(R_eff) < 0:
        Vt[-1] *= -1
        R_eff = U @ Vt
    return aligned_pts, {
        "anchor": "bounded_affine",
        "sv_clamp": (float(sv_min), float(sv_max)),
        "sv_final": sv_log,
        "scale_factors": None,
        "translation_mm": [float(x) for x in translation],
        "rot_deg": _rot_deg(R_eff),
        "trans_mm": float(np.linalg.norm(t_total)),
    }


@register_anchor("rigid_then_bounded_affine")
def rigid_then_bounded_affine(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
    """Cascade: rigid first (gross alignment), then bounded-affine refinement.
    PDvDESS 2026-05-21: helped most on rotational outlier 9226021_RIGHT_femur
    (pure bounded 6.15 → cascade 4.79 mm).
    """
    scaled_pts = np.asarray(bone_mesh.points)
    _, src_for_icp, target_for_icp, translation, translated_pts = _prepare_articular(
        scaled_pts, bone_mesh, cart_mask, spacing, template_mesh, cfg
    )
    _, R_icp, t_icp = _rigid_icp(
        src_for_icp, target_for_icp, cfg.rigid_icp_max_iter, cfg.rigid_icp_tol
    )
    rigid_pts = (R_icp @ translated_pts.T).T + t_icp
    sv_min, sv_max = cfg.bounded_affine_sv_range
    _, A_total, t_total, sv_log = _bounded_affine_icp(
        rigid_pts[_articular_source_mask(bone_mesh, cart_mask, spacing, cfg.articular_radius_mm)],
        target_for_icp,
        cfg.rigid_icp_max_iter, cfg.rigid_icp_tol,
        sv_min=sv_min, sv_max=sv_max,
    )
    aligned_pts = (A_total @ rigid_pts.T).T + t_total
    U, _, Vt = np.linalg.svd(A_total @ R_icp)
    R_eff = U @ Vt
    if np.linalg.det(R_eff) < 0:
        Vt[-1] *= -1
        R_eff = U @ Vt
    return aligned_pts, {
        "anchor": "rigid_then_bounded_affine",
        "sv_clamp": (float(sv_min), float(sv_max)),
        "sv_final": sv_log,
        "scale_factors": None,
        "translation_mm": [float(x) for x in translation],
        "rot_deg": _rot_deg(R_eff),
        "trans_mm": float(np.linalg.norm(R_icp @ t_total + t_icp)),
    }
