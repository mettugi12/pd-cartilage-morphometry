"""
Anatomical subregions on the bone+subchondral templates.

Defines per-template-vertex coordinates and region labels that downstream
analysis (regional means, ROC, LOA) and viz (3D + 2D projections) consume.

============================================================================
TEMPLATE COORDINATE ORIENTATION (CRITICAL for any subregion / ROI definition)
============================================================================

Both the femur and tibia template VTKs use the same axis convention, verified
by the diagnostic viewer (KneeMR/Studies/PD-vs-DESS/v4.1/analysis/viewer_femur_axis_check.py)
on the v0.2.0 templates from `E:/KneeMR/Datasets/Bone-Atlas/`:

  axis 0 — AP (anterior-posterior)
            MIN AP  =  anterior (front of the knee)
            MAX AP  =  posterior (back of the knee)

  axis 1 — SI (superior-inferior)
            MIN SI  =  superior / proximal (toward femoral shaft / hip)
            MAX SI  =  inferior / distal (toward joint surface / foot)

  axis 2 — ML (medial-lateral)
            MIN ML  =  medial
            MAX ML  =  lateral

Note this differs from the casual mental model "AP increases means anterior".
DO NOT assume AP_max is anterior; it's posterior. This caught us during the
v4.1 cMF/cLF analysis — see `subregions.find_femur_eckstein_landmarks` for
the reasoning and the corrected algorithm.

============================================================================
Femur subregion approaches (this module provides BOTH):
============================================================================

(A) Arc-fraction (best-fit cylinder axis, parallel to ML, anchored at the
    best-fit circle center of subchondral verts in (AP, SI)). Used for the
    smooth 2D-projection visualization (`projection_2d.template_thickness_2d_femur`).
    Each sagittal slice is treated as a sub-circle around this axis. Produces
    `arc_norm in [0, 1]` along the angular wrap.

(B) AP-distance (Eckstein anatomical convention, per OAI Eckstein doc + FNIH
    Chondrometrics doc). cMF/pMF split is a vertical plane (constant AP)
    perpendicular to the femoral shaft. Anchors:
      START (trochlear notch) = max AP among femur subch verts AT THE CENTRAL
                                 ML SLICE (= ML midpoint between med/lat condyles).
                                 Anatomically the back of the intercondylar fossa
                                 in that central slice — IS at relatively HIGH AP
                                 because central-slice bone is mostly intercondylar.
      END (posterior condyle tip) = max AP among femur subch verts overall, at
                                     the LOWER intersection (max SI = most distal)
                                     of the AP_max plane with the femur mesh.
    Per-side END uses max-AP within that compartment's subch verts.
    AP_split @ fraction f = AP_notch + f * (AP_end - AP_notch)
    cMF (anterior to split) = medial subch with AP_notch <= AP <= AP_split
    pMF (posterior to split) = medial subch with AP_split < AP <= AP_end
    Pre-notch cartilage (medial subch with AP < AP_notch) = trochlear cartilage
    proper, NOT part of cMF in the Eckstein convention.

(B) is what Eckstein/Chondrometrics actually measure (per their figures and
documentation). (A) is convenient for visualization but its arc_norm fraction
is NOT geometrically equivalent to Eckstein's AP fraction — using (A)'s 60%
band as a stand-in for Eckstein cMF60 introduces a systematic ROI mismatch.
For Eckstein-comparable analyses use (B).

For v4.1: arc-fraction is the visualization, AP-distance is the canonical
analysis.

============================================================================
ARC-FRACTION (visualization, smooth 2D projection):
============================================================================

Femur (this module currently):
  Angular axis = a SINGLE STRAIGHT line parallel to ML, anchored at the
  best-fit circle center of the subchondral verts in the (AP, SI) plane.
  This sits proximal of the cartilage-bearing surface, where the femoral
  condyle's natural cylindrical rotation axis lies. Each sagittal slice is
  treated as a sub-circle around this single axis.

  Per subch vert:
    ml_norm  : 0 (medial extreme) .. 1 (lateral extreme)
    arc_norm : 0 (anterior gap end) .. 1 (posterior gap start)
               — measured as theta from the single global axis,
                 normalised by the global theta_range so adjacent ML
                 slices stay anatomically consistent.

  Region labels (LBL_*):
    LBL_NON_SUBCH  template surface outside the subchondral zone
    LBL_MED        all medial subch  (ml_norm <= 0.5)  — cMF anatomical hemisphere
    LBL_LAT        all lateral subch (ml_norm >  0.5)  — cLF anatomical hemisphere

  WB band (`in_wb` boolean) is kept separate from the region label so the
  central weight-bearing analysis ROI can be shaded inside cMF/cLF without
  fragmenting the hemisphere coloring. Default `wb_band=(0.2, 0.8)` matches
  the v3.3/Eckstein "central 60% weight-bearing" convention.

  Why this design (vs the older per-ML-slice centroid in `projection_2d`):
    The old projection computed (AP, SI) centroid per ML slice — the centroid
    wobbled across slices (especially through the intercondylar notch) and the
    per-slice theta_range varied, producing ragged borders on the 2D unwrap.
    With a single straight axis at the best-fit cylinder centre:
      - no centroid wobble → smooth arc_norm transitions across ML
      - one global theta_range → consistent vertical anatomy across columns
      - ragged 2D borders go away
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv


# Region label IDs (used as int8 scalars on the template mesh)
LBL_NON_SUBCH = 0
LBL_MED       = 1
LBL_LAT       = 2

REGION_NAMES = {
    LBL_NON_SUBCH: "non-subch",
    LBL_MED:       "medial subch (cMF hemisphere)",
    LBL_LAT:       "lateral subch (cLF hemisphere)",
}


@dataclass
class FemurSubregions:
    ml_norm: np.ndarray            # (V,) in [0,1] for subch verts; NaN otherwise
    arc_norm: np.ndarray           # (V,) in [0,1] for subch verts; NaN otherwise
    region: np.ndarray             # (V,) int label
    in_wb: np.ndarray              # (V,) bool — vert in central WB band
    is_subch: np.ndarray           # (V,) bool
    axis_polyline: np.ndarray      # (n_ml_slices, 3) physical mm — straight axis line
    axis_ap: float                 # AP coord of the axis (best-fit circle center)
    axis_si: float                 # SI coord of the axis
    axis_radius: float             # mean fit radius (informational)
    theta_start: float             # radians, anterior gap anchor
    theta_range: float             # radians, total span used for normalisation
    wb_band: tuple                 # (lo, hi) used for WB filtering


def fit_circle_2d(x: np.ndarray, y: np.ndarray) -> tuple:
    """Algebraic best-fit circle to 2D points (x, y). Returns (a, b, r).

    Solves the linearised problem
        x^2 + y^2 = 2 a x + 2 b y + (r^2 - a^2 - b^2)
    via ordinary least squares — robust on partial arcs (femoral subch wraps
    only ~270° of the cylinder, so direct geometric fit is well-conditioned).
    """
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = sol
    r = float(np.sqrt(max(c + a ** 2 + b ** 2, 0.0)))
    return float(a), float(b), r


def compute_femur_template_subregions(
    template_mesh: pv.PolyData,
    subch_thresh: float = 0.5,
    n_ml_slices: int = 60,
    wb_band: tuple = (0.2, 0.8),
    n_theta_bins: int = 360,
) -> FemurSubregions:
    """Per-template-vertex (ml_norm, arc_norm, region) for the femur.

    See module docstring for the algorithm. Uses the same axis convention as
    the rest of the pipeline (vert[0]=AP, vert[1]=SI, vert[2]=ML).
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; SI = verts[:, 1]; ML = verts[:, 2]
    sp = np.asarray(template_mesh.point_data["subch_prob"])
    is_subch = sp >= subch_thresh

    n = len(verts)
    ml_norm  = np.full(n, np.nan, dtype=np.float64)
    arc_norm = np.full(n, np.nan, dtype=np.float64)
    region   = np.full(n, LBL_NON_SUBCH, dtype=np.int8)
    in_wb    = np.zeros(n, dtype=bool)

    if not is_subch.any():
        return FemurSubregions(
            ml_norm=ml_norm, arc_norm=arc_norm, region=region,
            in_wb=in_wb, is_subch=is_subch,
            axis_polyline=np.zeros((n_ml_slices, 3)),
            axis_ap=0.0, axis_si=0.0, axis_radius=0.0,
            theta_start=0.0, theta_range=0.0, wb_band=wb_band,
        )

    ml_min, ml_max = float(ML[is_subch].min()), float(ML[is_subch].max())
    ml_r = max(ml_max - ml_min, 1e-6)

    ap_axis, si_axis, axis_radius = fit_circle_2d(AP[is_subch], SI[is_subch])

    pad = 0.15 * ml_r
    ml_line = np.linspace(ml_min - pad, ml_max + pad, n_ml_slices)
    axis_polyline = np.column_stack([
        np.full(n_ml_slices, ap_axis),
        np.full(n_ml_slices, si_axis),
        ml_line,
    ])

    sub_idx = np.where(is_subch)[0]
    theta = np.arctan2(SI[sub_idx] - si_axis,
                       AP[sub_idx] - ap_axis) % (2 * np.pi)

    hist, edges = np.histogram(theta, bins=n_theta_bins, range=(0.0, 2 * np.pi))
    is_empty = hist == 0
    if is_empty.any():
        ie2 = np.tile(is_empty.astype(np.int8), 2)
        d2 = np.diff(ie2, prepend=0)
        rs = np.where(d2 == 1)[0]; re = np.where(d2 == -1)[0]
        if len(rs) > len(re):
            re = np.append(re, len(ie2))
        rl = re - rs
        best = int(np.argmax(rl))
        arc_start_bin = int((rs[best] + rl[best]) % n_theta_bins)
        theta_start = float(edges[arc_start_bin])
    else:
        theta_start = 0.0

    theta_shift = (theta - theta_start) % (2 * np.pi)
    theta_range = max(float(theta_shift.max()), 0.01)
    arc_norm[sub_idx] = theta_shift / theta_range
    ml_norm[sub_idx]  = (ML[sub_idx] - ml_min) / ml_r

    wb_lo, wb_hi = wb_band
    mln = ml_norm[sub_idx]
    arc = arc_norm[sub_idx]
    is_med = mln <= 0.5
    sub_region = np.where(is_med, LBL_MED, LBL_LAT).astype(np.int8)
    region[sub_idx] = sub_region
    in_wb[sub_idx] = (arc >= wb_lo) & (arc <= wb_hi)

    return FemurSubregions(
        ml_norm=ml_norm, arc_norm=arc_norm, region=region,
        in_wb=in_wb, is_subch=is_subch,
        axis_polyline=axis_polyline,
        axis_ap=ap_axis, axis_si=si_axis, axis_radius=axis_radius,
        theta_start=theta_start, theta_range=theta_range, wb_band=wb_band,
    )


def annotate_femur_template(template_mesh: pv.PolyData,
                            sub: FemurSubregions) -> pv.PolyData:
    """Return a copy of `template_mesh` with ml_norm / arc_norm / region / in_wb
    attached as point_data — convenient for viz code that wants scalars."""
    out = template_mesh.copy()
    out["ml_norm"]  = sub.ml_norm.astype(np.float32)
    out["arc_norm"] = sub.arc_norm.astype(np.float32)
    out["region"]   = sub.region.astype(np.int8)
    out["in_wb"]    = sub.in_wb.astype(np.uint8)
    return out


def femur_region_means_3d(thickness_per_vert: np.ndarray,
                          sub: FemurSubregions,
                          zero_impute: bool = True) -> dict:
    """Compute cMF / cLF region means directly on the 3D template (no 2D binning).

    cMF / cLF means use the central WB band intersected with the medial /
    lateral hemisphere — this matches Eckstein's `BMFMTH` / `BLFMTH` ROI.

    `zero_impute=True` (default): treat thickness == 0 within the ROI as 0
    (matches the cartilage repo's IDW-with-zeros design — bare zones in the
    template subch carry 0 so the means reflect total subch coverage). Set
    `False` to compute mean over thickness > 0 only (ThCtAB-style).

    Returns: {'cMF': float, 'cLF': float}
    """
    out = {}
    for region_label, name in ((LBL_MED, "cMF"), (LBL_LAT, "cLF")):
        mask = sub.is_subch & (sub.region == region_label) & sub.in_wb
        if not mask.any():
            out[name] = float("nan"); continue
        vals = thickness_per_vert[mask]
        if zero_impute:
            vals = np.where(np.isfinite(vals), vals, 0.0)
            out[name] = float(vals.mean())
        else:
            vals = vals[np.isfinite(vals) & (vals > 0)]
            out[name] = float(vals.mean()) if len(vals) else float("nan")
    return out


# ===========================================================================
# Tibia
# ===========================================================================
# AP-third labels (anterior / central / posterior of each plateau)
LBL_AP_NA       = -1
LBL_AP_ANTERIOR =  0
LBL_AP_CENTRAL  =  1
LBL_AP_POSTERIOR = 2


@dataclass
class TibiaSubregions:
    """Tibia template anatomical coordinates + region labels.

    The tibia plateau is roughly flat — no angular unwrap needed. The template's
    built-in `compartment` point_data field gives the medial/lateral split
    (1=medial, 2=lateral, 0=intercondylar non-cartilage gap). Per-compartment
    AP normalisation lets us define anterior / central / posterior thirds
    (aMT/cMT/pMT and aLT/cLT/pLT in Eckstein nomenclature).
    """
    ml_norm: np.ndarray       # (V,) per-compartment: medial in [0, 0.5], lateral in [0.5, 1.0]
    ap_norm: np.ndarray       # (V,) per-compartment AP in [0, 1] (anterior = 0)
    region: np.ndarray        # (V,) int: LBL_NON_SUBCH / LBL_MED (MT) / LBL_LAT (LT)
    ap_third: np.ndarray      # (V,) int: LBL_AP_ANTERIOR/CENTRAL/POSTERIOR or LBL_AP_NA
    is_subch: np.ndarray      # (V,) bool


def compute_tibia_template_subregions(
    template_mesh: pv.PolyData,
    subch_thresh: float = 0.5,
) -> TibiaSubregions:
    """Per-template-vertex (ml_norm, ap_norm, region, ap_third) for the tibia.

    Uses the template's `compartment` field (must be present — set up by
    `add_tibia_compartment.py` when building the template).

    Convention (matches `projection_2d.template_thickness_2d_tibia`):
      ML normalisation is per-compartment so columns 0-0.5 = medial half,
      0.5-1.0 = lateral half on the 2D grid.
      AP normalisation is per-compartment so each plateau's anterior-posterior
      extent fills [0, 1] independently.

    AP thirds: 0..1/3 = anterior, 1/3..2/3 = central, 2/3..1 = posterior.
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; ML = verts[:, 2]
    sp = np.asarray(template_mesh.point_data["subch_prob"])
    comp = np.asarray(template_mesh.point_data["compartment"])
    is_subch = sp >= subch_thresh

    n = len(verts)
    ml_norm  = np.full(n, np.nan, dtype=np.float64)
    ap_norm  = np.full(n, np.nan, dtype=np.float64)
    region   = np.full(n, LBL_NON_SUBCH, dtype=np.int8)
    ap_third = np.full(n, LBL_AP_NA, dtype=np.int8)

    if not is_subch.any():
        return TibiaSubregions(ml_norm, ap_norm, region, ap_third, is_subch)

    for label, lo, hi, region_lbl in ((1, 0.0, 0.5, LBL_MED),
                                      (2, 0.5, 1.0, LBL_LAT)):
        mask = is_subch & (comp == label)
        if not mask.any():
            continue
        ml_min = float(ML[mask].min()); ml_max = float(ML[mask].max())
        ap_min = float(AP[mask].min()); ap_max = float(AP[mask].max())
        ml_r = max(ml_max - ml_min, 1e-6)
        ap_r = max(ap_max - ap_min, 1e-6)
        ml_norm[mask] = lo + (ML[mask] - ml_min) / ml_r * (hi - lo)
        ap_norm[mask] = (AP[mask] - ap_min) / ap_r
        region[mask] = region_lbl

        a = ap_norm[mask]
        third_local = np.where(a < 1 / 3, LBL_AP_ANTERIOR,
                       np.where(a < 2 / 3, LBL_AP_CENTRAL, LBL_AP_POSTERIOR)).astype(np.int8)
        ap_third[mask] = third_local

    return TibiaSubregions(
        ml_norm=ml_norm, ap_norm=ap_norm, region=region,
        ap_third=ap_third, is_subch=is_subch,
    )


def annotate_tibia_template(template_mesh: pv.PolyData,
                            sub: TibiaSubregions) -> pv.PolyData:
    """Return a copy of `template_mesh` with ml_norm / ap_norm / region /
    ap_third attached as point_data — convenient for viz code."""
    out = template_mesh.copy()
    out["ml_norm"]  = sub.ml_norm.astype(np.float32)
    out["ap_norm"]  = sub.ap_norm.astype(np.float32)
    out["region"]   = sub.region.astype(np.int8)
    out["ap_third"] = sub.ap_third.astype(np.int8)
    return out


def tibia_region_means_3d(thickness_per_vert: np.ndarray,
                          sub: TibiaSubregions,
                          zero_impute: bool = True) -> dict:
    """Compute MT / LT + aMT/cMT/pMT + aLT/cLT/pLT means directly on the 3D
    template (no 2D binning).

    MT, LT match Eckstein's `WMTMTH` / `WLTMTH` (whole plate mean).
    aMT/cMT/pMT and aLT/cLT/pLT match `AMTMTH`, `CMTMTH`, `PMTMTH`, etc.

    `zero_impute` semantics — same as `femur_region_means_3d`. Default True.

    Returns: {'MT', 'LT', 'aMT', 'cMT', 'pMT', 'aLT', 'cLT', 'pLT'}
    """
    out = {}
    for region_label, name in ((LBL_MED, "MT"), (LBL_LAT, "LT")):
        mask = sub.is_subch & (sub.region == region_label)
        out[name] = _mean_or_nan(thickness_per_vert, mask, zero_impute)

    third_prefix = {LBL_MED: "M", LBL_LAT: "L"}
    third_letter = {LBL_AP_ANTERIOR: "a", LBL_AP_CENTRAL: "c", LBL_AP_POSTERIOR: "p"}
    for region_label in (LBL_MED, LBL_LAT):
        for third_label, lt in third_letter.items():
            name = f"{lt}{third_prefix[region_label]}T"  # aMT/cMT/pMT/aLT/cLT/pLT
            mask = (sub.is_subch
                    & (sub.region == region_label)
                    & (sub.ap_third == third_label))
            out[name] = _mean_or_nan(thickness_per_vert, mask, zero_impute)
    return out


def _mean_or_nan(thk: np.ndarray, mask: np.ndarray, zero_impute: bool) -> float:
    if not mask.any():
        return float("nan")
    vals = thk[mask]
    if zero_impute:
        vals = np.where(np.isfinite(vals), vals, 0.0)
        return float(vals.mean())
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(vals.mean()) if len(vals) else float("nan")


# ===========================================================================
# Femur — Eckstein AP-distance subregions (anatomically correct cMF/cLF)
# ===========================================================================
# This block implements the Eckstein/Chondrometrics convention for the cMF
# and cLF regions: a vertical plane perpendicular to the femoral shaft (i.e.,
# constant AP) splits the femoral cartilage at a fraction of the AP distance
# from the trochlear notch (anterior boundary of cMF) to the posterior
# condyle tip (posterior boundary).
#
# Per the OAI Eckstein_Descrip 2017 doc and FNIH-Chondrometrics_Descrip 2015
# doc (both with Figure 2 / Figure 1 bottom row showing the magenta/turquoise/
# blue line construction):
#   - Magenta line: parallel to femoral shaft, through trochlear notch in a
#                   CENTRAL ML slice between MF and LF.
#   - Blue line:    parallel to femoral shaft, through the posterior end of
#                   the femoral condyle.
#   - Turquoise line at fraction f (0.6 or 0.75) of the distance between
#                   magenta and blue, dividing MF cartilage into cMF (anterior
#                   side) and pMF (posterior side).
#
# In template coords (corrected orientation: MIN AP = anterior, MAX AP = posterior):
#   - Trochlear notch ≈ subch vert with MAX AP at the central ML slice
#                       (anatomically: the back of the intercondylar fossa,
#                       which is the deepest concavity at the front of the
#                       distal femur — IS at high AP within that central slice
#                       where central-slice bone is mostly intercondylar fossa).
#   - Posterior condyle tip ≈ subch vert with MAX AP overall, taking the LOWER
#                              intersection (max SI = most distal) of the
#                              AP_max plane with the femur mesh.
#
# AP_split @ f = AP_notch + f * (AP_end - AP_notch)
# cMF = medial subch with AP_notch <= AP <= AP_split
# pMF = medial subch with AP_split < AP <= AP_end
# Pre-notch cartilage (medial subch with AP < AP_notch) = trochlear cartilage
# proper, NOT part of cMF.
# Same for cLF / pLF on the lateral compartment.

LBL_NOTCH_PRE  =  3  # cartilage anterior to the trochlear notch (trochlear cart)
LBL_CMF_REGION =  4  # cMF (medial, anterior of AP_split)
LBL_PMF_REGION =  5  # pMF (medial, posterior of AP_split)
LBL_CLF_REGION =  6  # cLF (lateral, anterior of AP_split)
LBL_PLF_REGION =  7  # pLF (lateral, posterior of AP_split)


@dataclass
class FemurEcksteinSubregions:
    """Eckstein-convention AP-distance based femur subregions."""
    region: np.ndarray            # (V,) int label per LBL_*_REGION above
    is_subch: np.ndarray          # (V,) bool
    ml_mid: float                 # ML midpoint between med/lat used to find central slice
    notch_pt: np.ndarray          # (3,) physical mm — START anchor (trochlear notch)
    end_pt_global: np.ndarray     # (3,) — END anchor (posterior condyle tip, lower intersection)
    end_pt_med: np.ndarray        # (3,) — per-side END for medial condyle (max AP among medial subch)
    end_pt_lat: np.ndarray        # (3,) — per-side END for lateral condyle
    ap_notch: float
    ap_end_global: float
    ap_end_med: float
    ap_end_lat: float
    ap_split_med: float           # AP plane @ fraction for medial side
    ap_split_lat: float
    fraction: float               # the fraction used (e.g., 0.60 or 0.75)


def find_femur_eckstein_landmarks(template_mesh: pv.PolyData,
                                  subch_thresh: float = 0.5,
                                  ml_slice_tol_mm: float = 2.0) -> dict:
    """Find the trochlear notch (START) and posterior condyle tip (END) anchors
    on the femur template per the Eckstein convention.

    See module-level "TEMPLATE COORDINATE ORIENTATION" doc for axis convention.

    Returns dict with: notch_pt, end_pt_global (= max AP global, max SI lower
    intersection), end_pt_med, end_pt_lat, ml_mid, plus the AP scalars.
    """
    verts = np.asarray(template_mesh.points)
    AP = verts[:, 0]; SI = verts[:, 1]; ML = verts[:, 2]
    sp = np.asarray(template_mesh.point_data["subch_prob"])
    is_subch = sp >= subch_thresh

    if not is_subch.any():
        raise ValueError("Template has no subchondral verts (subch_prob >= 0.5)")

    ml_mid = float((ML[is_subch].min() + ML[is_subch].max()) / 2.0)

    # Central ML slice (with adaptive tolerance if too few verts)
    in_central = is_subch & (np.abs(ML - ml_mid) <= ml_slice_tol_mm)
    if in_central.sum() < 5:
        in_central = is_subch & (np.abs(ML - ml_mid) <= ml_slice_tol_mm * 3)
    sub_idx_central = np.where(in_central)[0]
    notch_idx = sub_idx_central[np.argmax(AP[sub_idx_central])]
    notch_pt = verts[notch_idx]

    # END global = max AP, lower intersection (= max SI at that AP)
    ap_max_global = float(AP[is_subch].max())
    near_apmax = is_subch & (AP >= ap_max_global - 1.0)
    sub_idx_apmax = np.where(near_apmax)[0]
    end_idx_global = sub_idx_apmax[np.argmax(SI[sub_idx_apmax])]
    end_pt_global = verts[end_idx_global]

    # Per-compartment END = max AP within hemisphere
    is_med = is_subch & (ML <= ml_mid)
    is_lat = is_subch & (ML >  ml_mid)
    end_idx_med = np.where(is_med)[0][np.argmax(AP[is_med])]
    end_idx_lat = np.where(is_lat)[0][np.argmax(AP[is_lat])]
    end_pt_med = verts[end_idx_med]
    end_pt_lat = verts[end_idx_lat]

    return {
        "ml_mid": ml_mid,
        "notch_pt": notch_pt,
        "end_pt_global": end_pt_global,
        "end_pt_med": end_pt_med,
        "end_pt_lat": end_pt_lat,
        "ap_notch": float(notch_pt[0]),
        "ap_end_global": float(end_pt_global[0]),
        "ap_end_med": float(end_pt_med[0]),
        "ap_end_lat": float(end_pt_lat[0]),
        "is_subch": is_subch,
        "is_medial": is_med,
        "is_lateral": is_lat,
    }


def compute_femur_eckstein_apsplit_subregions(
    template_mesh: pv.PolyData,
    fraction: float = 0.60,
    subch_thresh: float = 0.5,
    ml_slice_tol_mm: float = 2.0,
) -> FemurEcksteinSubregions:
    """Eckstein AP-distance cMF/pMF and cLF/pLF subregions on the femur template.

    `fraction` is the cMF/pMF split fraction (0.60 = Eckstein 9A, 0.75 = 9B/22/66).
    Returns a `FemurEcksteinSubregions` dataclass with per-vertex region labels:
      LBL_NON_SUBCH, LBL_NOTCH_PRE, LBL_CMF_REGION, LBL_PMF_REGION,
      LBL_CLF_REGION, LBL_PLF_REGION.
    """
    L = find_femur_eckstein_landmarks(template_mesh, subch_thresh=subch_thresh,
                                      ml_slice_tol_mm=ml_slice_tol_mm)
    AP = np.asarray(template_mesh.points)[:, 0]
    is_subch = L["is_subch"]
    is_med   = L["is_medial"]
    is_lat   = L["is_lateral"]
    ap_notch = L["ap_notch"]

    ap_split_med = ap_notch + fraction * (L["ap_end_med"] - ap_notch)
    ap_split_lat = ap_notch + fraction * (L["ap_end_lat"] - ap_notch)

    region = np.full(len(AP), LBL_NON_SUBCH, dtype=np.int8)
    # Pre-notch cartilage (anterior of notch)
    region[is_med & (AP < ap_notch)] = LBL_NOTCH_PRE
    region[is_lat & (AP < ap_notch)] = LBL_NOTCH_PRE
    # cMF (medial, between notch and split)
    region[is_med & (AP >= ap_notch) & (AP <= ap_split_med)] = LBL_CMF_REGION
    # pMF (medial, between split and end)
    region[is_med & (AP > ap_split_med) & (AP <= L["ap_end_med"])] = LBL_PMF_REGION
    # cLF (lateral, between notch and split)
    region[is_lat & (AP >= ap_notch) & (AP <= ap_split_lat)] = LBL_CLF_REGION
    # pLF (lateral, between split and end)
    region[is_lat & (AP > ap_split_lat) & (AP <= L["ap_end_lat"])] = LBL_PLF_REGION

    return FemurEcksteinSubregions(
        region=region, is_subch=is_subch, ml_mid=L["ml_mid"],
        notch_pt=L["notch_pt"], end_pt_global=L["end_pt_global"],
        end_pt_med=L["end_pt_med"], end_pt_lat=L["end_pt_lat"],
        ap_notch=ap_notch, ap_end_global=L["ap_end_global"],
        ap_end_med=L["ap_end_med"], ap_end_lat=L["ap_end_lat"],
        ap_split_med=ap_split_med, ap_split_lat=ap_split_lat,
        fraction=fraction,
    )


def femur_eckstein_region_means_3d(thickness_per_vert: np.ndarray,
                                   sub: FemurEcksteinSubregions,
                                   zero_impute: bool = True) -> dict:
    """Compute cMF/pMF/cLF/pLF means directly on 3D template verts (no 2D binning),
    using the Eckstein AP-distance subregions.

    Returns: {'cMF', 'pMF', 'cLF', 'pLF'} keyed by region name.
    """
    out = {}
    for lbl, name in ((LBL_CMF_REGION, "cMF"), (LBL_PMF_REGION, "pMF"),
                      (LBL_CLF_REGION, "cLF"), (LBL_PLF_REGION, "pLF")):
        mask = sub.region == lbl
        out[name] = _mean_or_nan(thickness_per_vert, mask, zero_impute)
    return out


def annotate_femur_eckstein_template(template_mesh: pv.PolyData,
                                     sub: FemurEcksteinSubregions) -> pv.PolyData:
    """Attach the Eckstein region labels as point_data on a copy of the template."""
    out = template_mesh.copy()
    out["region_eckstein"] = sub.region.astype(np.int8)
    return out
