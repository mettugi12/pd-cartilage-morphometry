"""v9.3 — Web-app template-frame 2D panels for patient values and region labels.

Reproduces how the web application draws its 2D cartilage maps: the patient bone is registered to the atlas
template (library `aniso_rigid` anchor), patient values are remapped onto the template's cartilage-bearing zone
(subch_prob >= 0.5) with the viewer's radius-ball inverse-distance remap (3 mm), and the template is projected
with `web_export.project_by_proj` / `_project_labels` (intrinsic arc x ML, 150 x 150, gap fill, border-preserving
smoothing). The outline is therefore the template's healthy cartilage zone, not the patient's raw footprint.

Values are measured on the patient (analysis grid); the template is only the display frame.
"""
from __future__ import annotations

import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from cartilage_morphometry import PipelineConfig, TEMPLATE_PATHS  # noqa: E402
from cartilage_morphometry import web_export as we  # noqa: E402
from cartilage_morphometry.remap import remap_thickness_to_template_radius  # noqa: E402
from cartilage_morphometry.strategies import get_anchor  # noqa: E402

SUBCH_T = 0.5
RADIUS = 3.0


def align_to_template(z, bone):
    """Patient bone vertices (captured grid order ML,SI,AP) -> template frame. Returns (template, aligned, zone_t)."""
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    cart_g = np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool)
    cart_p = np.ascontiguousarray(np.transpose(cart_g, (2, 1, 0)))           # pipeline order (AP, SI, ML)
    sp_p = tuple(z["spacing"])[::-1]
    pts_p = np.ascontiguousarray(z["points"][:, [2, 1, 0]], dtype=float)
    tmpl = pv.read(str(TEMPLATE_PATHS[bone]))
    aligned, _ = get_anchor("aniso_rigid")(pv.PolyData(pts_p), cart_p, sp_p, tmpl, bone, "pd", PipelineConfig())
    zone_t = np.asarray(tmpl.point_data["subch_prob"]) >= SUBCH_T
    return tmpl, np.asarray(aligned), zone_t


def remap_values(tmpl, aligned, zone_t, src_mask, values):
    """Radius-ball IDW of patient `values` (on `src_mask` vertices) onto the template zone."""
    return remap_thickness_to_template_radius(tmpl, aligned, np.where(src_mask, values, 0.0), src_mask.astype(float),
                                              radius=RADIUS, subch_threshold=0.5, target_mask=zone_t)


def remap_labels(tmpl, aligned, zone_t, src_mask, labels):
    """Nearest patient source vertex (within 2 x radius) -> integer label on the template zone (0 = none)."""
    out = np.zeros(tmpl.n_points, int)
    src = np.where(src_mask)[0]
    if not len(src):
        return out
    tree = cKDTree(aligned[src])
    tz = np.where(zone_t)[0]
    d, i = tree.query(np.asarray(tmpl.points)[tz], k=1)
    ok = d <= 2 * RADIUS
    out[tz[ok]] = np.asarray(labels)[src[i[ok]]]
    return out


def panel_grids(bone, tmpl, zone_t, values_t, labels_t, n_labels):
    g = we.project_by_proj(bone, tmpl, values_t, zone_t & np.isfinite(values_t), zone_t)
    gl = we._project_labels(bone, tmpl, labels_t, zone_t, n_labels)
    return g, gl
