"""IDW remap of patient per-vertex thickness onto the DESS template.

K-NN IDW from patient SUBCHONDRAL verts (incl. thickness=0 verts in the
subch zone, so the template inherits 0 where cart is anatomically expected
but missing). `mutual_nn=False` is the canonical default (PDvDESS 2026-04-27:
mutual NN inflates femur means by ~8%).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def remap_thickness_to_template_radius(template_mesh, aligned_points,
                                       patient_thickness, patient_subch_prob,
                                       radius: float = 4.0, k_fallback: int = 3,
                                       eps: float = 1e-6,
                                       subch_threshold: float = 0.5,
                                       target_mask=None):
    """VIEWER-ONLY radius-ball IDW remap (the study pipeline keeps the k-NN
    `remap_thickness_to_template` for validated metrics).

    Averages ALL patient subchondral verts within `radius` mm of each template
    vertex (distance-weighted). The patient mask is dense in-plane but sampled at
    ~3 mm in the slice (ML) direction, so a k=3 nearest remap snaps every fine
    template vertex to a SINGLE patient slice → a 3 mm staircase on the template.
    A ball of radius ≈ the slice spacing always spans neighbouring slices, so the
    values blend across slices (smooth, like the patient frame) while in-plane
    detail — where sampling is dense — is preserved. Falls back to k-nearest
    where the ball is empty (template verts beyond the patient cart extent).

    Returns a (n_template,) float32 array (NaN outside `target_mask`).
    """
    tp = np.asarray(template_mesh.points)
    n_t = len(tp)
    out = np.full(n_t, np.nan, dtype=np.float64)
    is_subch_p = np.asarray(patient_subch_prob) >= subch_threshold
    if not is_subch_p.any():
        return out.astype(np.float32)
    src = np.asarray(aligned_points)[is_subch_p]
    sth = np.asarray(patient_thickness)[is_subch_p]
    tree = cKDTree(src)
    targets = np.where(target_mask)[0] if target_mask is not None else np.arange(n_t)
    tp_t = tp[targets]
    nb = tree.query_ball_point(tp_t, radius)
    kf = min(k_fallback, len(src))
    dk, ik = tree.query(tp_t, k=kf)
    if kf == 1:
        dk = dk[:, None]; ik = ik[:, None]
    for j, lst in enumerate(nb):
        i = targets[j]
        if lst:
            d = np.linalg.norm(src[lst] - tp_t[j], axis=1)
            w = 1.0 / (d + eps)
            out[i] = float((w * sth[lst]).sum() / w.sum())
        else:
            w = 1.0 / (dk[j] + eps)
            out[i] = float((w * sth[ik[j]]).sum() / w.sum())
    return out.astype(np.float32)


def remap_thickness_to_template(template_mesh, aligned_points, patient_thickness,
                                patient_subch_prob,
                                k: int = 3,
                                mutual_nn: bool = False,
                                eps: float = 1e-6,
                                subch_threshold: float = 0.5):
    """Returns (template_thickness, quality_dict).

    `template_thickness` is shape (n_template,). Quality dict includes
    ASSD_subch_mm (the primary registration QA metric).
    """
    n_t = template_mesh.n_points
    template_pts = np.asarray(template_mesh.points)
    aligned_points = np.asarray(aligned_points)

    is_subch_p = patient_subch_prob >= subch_threshold
    n_p_subch = int(is_subch_p.sum())
    if n_p_subch == 0:
        return np.full(n_t, np.nan, dtype=np.float32), {
            "n_template_total": int(n_t),
            "n_template_subch": int((np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold).sum()),
            "n_patient_subch": 0,
            "n_mutual": 0, "frac_mutual": 0.0,
            "mean_mapping_dist_mm_all_template": float("nan"),
            "mean_mapping_dist_mm_subch_only": float("nan"),
            "median_mapping_dist_mm_subch_only": float("nan"),
            "max_mapping_dist_mm_subch_only": float("nan"),
            "assd_subch_mm": float("nan"),
        }

    patient_subch_pts = aligned_points[is_subch_p]
    patient_subch_thick = patient_thickness[is_subch_p]

    patient_tree = cKDTree(patient_subch_pts)
    template_tree = cKDTree(template_pts)

    k_eff = min(k, n_p_subch)
    dists_t2p, idx_t2p = patient_tree.query(template_pts, k=k_eff)
    if k_eff == 1:
        dists_t2p = dists_t2p[:, None]
        idx_t2p = idx_t2p[:, None]

    _, idx_p2t = template_tree.query(patient_subch_pts, k=1)
    nearest_patient_for_template = idx_t2p[:, 0]
    mapped_back = idx_p2t[nearest_patient_for_template]
    is_mutual = mapped_back == np.arange(n_t)

    weights = 1.0 / (dists_t2p + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)
    template_thickness = (weights * patient_subch_thick[idx_t2p]).sum(axis=1)

    if mutual_nn and (~is_mutual).any() and is_mutual.any():
        mutual_idx = np.where(is_mutual)[0]
        mutual_tree = cKDTree(template_pts[mutual_idx])
        nm = ~is_mutual
        k_mut = min(k, len(mutual_idx))
        _, nearest_mutual = mutual_tree.query(template_pts[nm], k=k_mut)
        if k_mut == 1:
            template_thickness[nm] = template_thickness[mutual_idx[nearest_mutual]]
        else:
            template_thickness[nm] = template_thickness[mutual_idx[nearest_mutual]].mean(axis=1)

    template_subch_mask = np.asarray(
        template_mesh.point_data["subch_prob"]
    ) >= subch_threshold
    d_subch_t2p = dists_t2p[template_subch_mask, 0]
    template_subch_pts = template_pts[template_subch_mask]
    if len(template_subch_pts) > 0:
        tmpl_subch_tree = cKDTree(template_subch_pts)
        d_p2tsubch, _ = tmpl_subch_tree.query(patient_subch_pts, k=1)
    else:
        d_p2tsubch = np.array([], dtype=np.float32)
    assd_subch_mm = float(
        (d_subch_t2p.sum() + d_p2tsubch.sum())
        / max(len(d_subch_t2p) + len(d_p2tsubch), 1)
    )

    quality = {
        "n_template_total": int(n_t),
        "n_template_subch": int(template_subch_mask.sum()),
        "n_patient_subch": int(n_p_subch),
        "n_mutual": int(is_mutual.sum()),
        "frac_mutual": float(is_mutual.mean()),
        "mean_mapping_dist_mm_all_template": float(dists_t2p[:, 0].mean()),
        "mean_mapping_dist_mm_subch_only": float(d_subch_t2p.mean()) if len(d_subch_t2p) else float("nan"),
        "median_mapping_dist_mm_subch_only": float(np.median(d_subch_t2p)) if len(d_subch_t2p) else float("nan"),
        "max_mapping_dist_mm_subch_only": float(d_subch_t2p.max()) if len(d_subch_t2p) else float("nan"),
        "assd_subch_mm": assd_subch_mm,
    }
    return template_thickness.astype(np.float32), quality
