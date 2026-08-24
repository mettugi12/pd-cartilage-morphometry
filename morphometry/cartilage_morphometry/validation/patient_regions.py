"""Patient-frame region means — no template, no IDW-remap-to-template.

Diagnostic: the template-frame longitudinal driver (`shared_mesh.process_long_
shared_mesh`) ends with a `patient_to_template_align` + IDW remap onto the
canonical template subch zone. If implausible thickening originates in that
template-anchoring step rather than in the segmentation, computing region
means DIRECTLY on the patient's own bone mesh — using only the
00m↔48m IDW (which the shared-mesh wrapper already does for us) — should
recover cleaner Δ values and a better cohort r vs Eckstein QCart.

Conventions (approximate but close to QCart):
- subch verts ≡ bone verts with thickness > 0 on either timepoint.
- Tibia: split by ML midline of subch verts.
    MT = mean over medial half, LT = mean over lateral half.
    AP thirds (per half) → aMT/cMT/pMT, aLT/cLT/pLT.
- Femur: split by ML midline of subch verts.
    cMF / cLF = mean over the *central 60 % AP band* of each condyle's subch
    (matches Eckstein BMFMTH/BLFMTH's "central weight-bearing 75 %" intent;
    we use 60 % because we don't have notch detection here and the central
    60 % is robust to the anterior trochlear groove not being clipped).
- Composites: MFTC = cMF + MT; LFTC = cLF + LT.

The output dict for each timepoint has the same keys as the template-frame
`tibia_region_means_3d` / `femur_eckstein_region_means_3d` so the api driver
can apply identical aggregation logic to both.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv


def _mean_or_nan(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(vals.mean()) if vals.size else float("nan")


def tibia_patient_regions(bone_mesh: pv.PolyData,
                          thickness: np.ndarray,
                          subch_mask: np.ndarray | None = None) -> dict:
    """{MT, LT, aMT, cMT, pMT, aLT, cLT, pLT} on the patient tibia bone mesh.

    subch ≡ `thickness > 0` if `subch_mask` is None. Pass an explicit mask if
    you want the same subch set across both timepoints.
    """
    pts = np.asarray(bone_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    th = np.asarray(thickness, dtype=np.float64)
    if subch_mask is None:
        sub = th > 0
    else:
        sub = np.asarray(subch_mask, dtype=bool)
    if sub.sum() < 50:
        return {k: float("nan") for k in
                ("MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT")}
    ml_lo, ml_hi = ML[sub].min(), ML[sub].max()
    ml_mid = 0.5 * (ml_lo + ml_hi)
    is_med = sub & (ML <= ml_mid)
    is_lat = sub & (ML >  ml_mid)
    out: dict = {
        "MT": _mean_or_nan(th[is_med]),
        "LT": _mean_or_nan(th[is_lat]),
    }
    for is_half, prefix in ((is_med, "M"), (is_lat, "L")):
        if is_half.sum() < 10:
            for letter in ("a", "c", "p"):
                out[f"{letter}{prefix}T"] = float("nan")
            continue
        ap_lo, ap_hi = AP[is_half].min(), AP[is_half].max()
        ap_r = max(ap_hi - ap_lo, 1e-6)
        ap_norm = (AP - ap_lo) / ap_r       # only meaningful where is_half
        # AP thirds: 0..1/3 anterior, 1/3..2/3 central, 2/3..1 posterior
        for lo, hi, letter in ((0.0, 1/3, "a"), (1/3, 2/3, "c"), (2/3, 1.0, "p")):
            m = is_half & (ap_norm >= lo) & (ap_norm <= hi)
            out[f"{letter}{prefix}T"] = _mean_or_nan(th[m])
    return out


def femur_patient_regions(bone_mesh: pv.PolyData,
                          thickness: np.ndarray,
                          central_ap_frac: float = 0.60,
                          subch_mask: np.ndarray | None = None) -> dict:
    """{cMF, cLF, pMF, pLF} on the patient femur bone mesh.

    Subch verts split medial/lateral by ML midline. Per condyle: AP normalised
    to [0,1] over the condyle's own subch AP range. `central_ap_frac` (default
    0.60) defines the central weight-bearing band centred on AP=0.5 → cMF/cLF
    means. Posterior third (top 1 − central_ap_frac/2 of AP) → pMF/pLF.
    """
    pts = np.asarray(bone_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    th = np.asarray(thickness, dtype=np.float64)
    if subch_mask is None:
        sub = th > 0
    else:
        sub = np.asarray(subch_mask, dtype=bool)
    out: dict = {k: float("nan") for k in ("cMF", "cLF", "pMF", "pLF")}
    if sub.sum() < 50:
        return out
    ml_lo, ml_hi = ML[sub].min(), ML[sub].max()
    ml_mid = 0.5 * (ml_lo + ml_hi)
    band_lo = 0.5 - central_ap_frac / 2.0   # e.g. 0.20
    band_hi = 0.5 + central_ap_frac / 2.0   # e.g. 0.80
    for is_half, name_c, name_p in (
        (sub & (ML <= ml_mid), "cMF", "pMF"),
        (sub & (ML >  ml_mid), "cLF", "pLF"),
    ):
        if is_half.sum() < 10:
            continue
        ap_lo, ap_hi = AP[is_half].min(), AP[is_half].max()
        ap_r = max(ap_hi - ap_lo, 1e-6)
        ap_norm = (AP - ap_lo) / ap_r
        central = is_half & (ap_norm >= band_lo) & (ap_norm <= band_hi)
        posterior = is_half & (ap_norm > band_hi)
        out[name_c] = _mean_or_nan(th[central])
        out[name_p] = _mean_or_nan(th[posterior])
    return out
