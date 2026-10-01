"""v9.3 — Where cMF / cLF are measured, drawn in the web-app 2D panel style.

Rendering reuses cartilage-morphometry `web_export` (the web app's quant panel): `project_by_proj` for the
thickness grid (intrinsic femur coordinates: best-fit-circle arc x normalised ML, lateral left, anterior at
the bottom, 150x150, gap fill + border-preserving smoothing) and `_project_labels` for majority-vote region
labels; drawing mirrors `render_quant_panel_png` (jet_r 0-3 mm, grey empty cells, smoothed black region
outlines with name tags, L / M corner tags).

Per knee, 00m thickness on the patient's own femur mesh, two panels:
  legacy (v9.2)  grid "cMF" / "cLF" = medial / lateral half x 20-80 % of the whole cartilage arc
  v9.3           Chondrometrics first-crossing cMF75 / cLF75 (`baseline_grid.eckstein_first_crossing_masks`),
                 plus the rest of the condyles: anterior to the notch plane (trochlea) and pMF / pLF
Region membership is decided on the analysis grid (per-slice baseline grid, exactly as measured) and
carried to each vertex through its grid cell; only the display layout is the web-app one.

Inputs: captured baseline-grid inputs E:/KneeMR/Studies/PD-vs-DESS/v9.2/femur_region_diag/<knee>.npz
Output: ../results/figures/v9_3_femur_regions_webstyle.png
Usage: python viz_femur_regions_webstyle_v9_3.py [--knees 9xxxxxx_RIGHT,...] [--dry_run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyvista as pv  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
CAP = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v9.2/femur_region_diag")
REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from cartilage_morphometry import baseline_grid as bg  # noqa: E402
from cartilage_morphometry import web_export as we  # noqa: E402

G = bg.GRID
LEGACY = ["cMF (grid)", "cLF (grid)"]
V93 = ["cMF75", "cLF75", "trochlea", "pMF", "pLF"]


def knee_labels(z):
    """Per-vertex labels for both definitions (0 = unlabelled) + mesh + valid mask + thickness."""
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    bone = np.unpackbits(z["bone"])[:n].reshape(shape).astype(bool)
    cart = np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool)
    pts, th00, sp = z["points"], z["th00"], tuple(z["spacing"])
    geo = bg.compute_ref_geometry(cart, bone, "FC", "right_oriented", femur_unwrap="per_slice")
    valid = (th00 > bg.MIN_THICK_MM) & (th00 < bg.MAX_THICK_MM)
    # v9.3 measurement: total subchondral bone area of the baseline plate (denuded bone = 0 mm)
    rr = bg.regional_deltas("femur", pts, th00, z["th48"], bone, cart, sp, th48_status=z["status"],
                            femur_region="eckstein75", total_bone_area=True)
    den00 = rr["_tab_info"]["vertex_den00"]
    shown = valid | den00                                   # bone vertices that enter the baseline mean
    dn, wn, _ = bg.vertex_norm_coords(pts, np.ones(len(pts)), geo, sp)   # every vertex
    db_all = np.clip((dn * G).astype(int), 0, G - 1); wb_all = np.clip((wn * G).astype(int), 0, G - 1)
    db, wb = db_all[valid], wb_all[valid]

    # legacy grid regions (cell masks)
    leg = np.zeros((G, G), int)
    leg[bg.D_MED, bg.W_WB] = 1; leg[bg.D_LAT, bg.W_WB] = 2
    # v9.3 regions + the rest of the condyles
    ref = np.isfinite(rr["_grid_00m"])
    mk_med, mk_lat, info = bg.eckstein_first_crossing_masks(pts, th00, geo, sp, fraction=0.75, zone=ref)
    s = np.zeros((G, G)); c = np.zeros((G, G))
    np.add.at(s, (db, wb), pts[valid][:, 2]); np.add.at(c, (db, wb), 1)
    cell_ap = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    _, inds = distance_transform_edt(~np.isfinite(cell_ap), return_indices=True)
    fill = ref & ~np.isfinite(cell_ap); cell_ap[fill] = cell_ap[tuple(inds[:, fill])]
    rows_med = np.zeros((G, G), bool); rows_med[bg.D_MED, :] = True
    new = np.zeros((G, G), int)
    covered = np.isfinite(cell_ap)
    new[covered & (cell_ap < info["ap_notch"])] = 3
    new[covered & ~(cell_ap < info["ap_notch"]) & rows_med] = 4
    new[covered & ~(cell_ap < info["ap_notch"]) & ~rows_med] = 5
    new[mk_med] = 1; new[mk_lat] = 2

    lab_leg = np.zeros(len(pts), int); lab_new = np.zeros(len(pts), int)
    idx = np.where(shown)[0]
    lab_leg[idx] = leg[db_all[idx], wb_all[idx]]; lab_new[idx] = new[db_all[idx], wb_all[idx]]
    mesh = pv.PolyData(np.ascontiguousarray(pts[:, [2, 1, 0]], dtype=float))   # pipeline order (AP, SI, ML)
    thick = np.where(valid, np.asarray(th00, float), 0.0)    # denuded bone drawn as 0 mm
    info["den00_frac"] = float(den00.sum() / max(shown.sum(), 1))
    return mesh, valid, shown, thick, lab_leg, lab_new, info


def draw(ax, g, lab, names, title, colors):
    """Mirror of web_export.render_quant_panel_png for one axis."""
    cmap = plt.get_cmap("jet_r").copy(); cmap.set_bad(color=(0.92, 0.92, 0.92))
    ax.imshow(g, origin="upper", cmap=cmap, vmin=0.0, vmax=3.0, interpolation="bilinear", aspect="auto")
    zone = np.isfinite(g); present = lab > 0
    if present.any():
        _, inds = distance_transform_edt(~present, return_indices=True)
        lab = np.where(zone, lab[tuple(inds)], 0)
    for ri, name in enumerate(names, start=1):
        mask = lab == ri
        if mask.sum() < 4:
            continue
        ax.contour(gaussian_filter(mask.astype(float), sigma=1.0), levels=[0.5], colors=colors[ri - 1],
                   linewidths=2.0 if name.startswith("c") else 1.2)
        rr, cc = np.where(mask)
        ax.text(cc.mean(), rr.mean(), name, fontsize=9, fontweight="bold", ha="center", va="center", color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for x, t, ha in ((0.02, "L", "left"), (0.98, "M", "right")):
        ax.text(x, 0.97, t, transform=ax.transAxes, fontsize=10, color="white", va="top", ha=ha, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
    ax.text(0.5, 0.01, "anterior (trochlea)", transform=ax.transAxes, fontsize=8, ha="center", va="bottom", color="0.3")
    ax.text(0.5, 0.99, "posterior", transform=ax.transAxes, fontsize=8, ha="center", va="top", color="0.3")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knees", default=None, help="comma list pid_SIDE (default: 2 progressors + 2 stable, median notch distance)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    la = pd.read_csv(RES / "v9_3_long_abs.csv"); la["k"] = la.pid.astype(str) + "_" + la.side
    lf = pd.concat([pd.read_csv(RES / "v9_3_prog_leakfree.csv"), pd.read_csv(RES / "v9_3_nonprog_leakfree.csv")])
    lf["k"] = lf.pid.astype(str) + "_" + lf.side
    la = la[la.k.isin(lf.k)]
    v92 = pd.read_csv(HERE.parent.parent / "v9" / "results" / "v9_2_long_abs.csv"); v92["k"] = v92.pid.astype(str) + "_" + v92.side
    if a.knees:
        knees = a.knees.split(",")
    else:
        knees = []
        for arm in ("progressor", "nonprogressor"):
            g = la[la.cohort == arm].copy()
            g["z"] = (g.notch_to_post_med_mm - g.notch_to_post_med_mm.median()).abs()
            knees += list(g.sort_values("z").k.head(2))
    print("[knees]", knees)

    fig, axes = plt.subplots(len(knees), 2, figsize=(12, 5.6 * len(knees)))
    axes = np.atleast_2d(axes)
    col_leg = ["black", "black"]
    col_new = ["black", "black", "0.35", "0.35", "0.35"]
    for i, k in enumerate(knees):
        z = np.load(CAP / f"{k}.npz", allow_pickle=True)
        mesh, valid, shown, thick, lab_leg, lab_new, info = knee_labels(z)
        g_leg = we.project_by_proj("femur", mesh, thick, valid, shown)      # legacy: cartilage-bearing vertices only
        g = we.project_by_proj("femur", mesh, thick, shown, shown)          # v9.3: + denuded plate bone as 0 mm
        gl = we._project_labels("femur", mesh, lab_leg, shown, len(LEGACY))
        gn = we._project_labels("femur", mesh, lab_new, shown, len(V93))
        u, v, base = we.panel_coords("femur", mesh, shown)
        side = "medial on the right (M)" if u[lab_new == 1].mean() > u[lab_new == 2].mean() else "!! cMF LEFT of cLF — orientation check"
        r = la[la.k == k].iloc[0]
        tag = f"{k}  {r.cohort}, KL{int(r.KL_00)}"
        r2 = v92[v92.k == k].iloc[0]
        draw(axes[i, 0], g_leg, gl, LEGACY, f"{tag}\nlegacy (v9.2): grid cMF / cLF = medial / lateral half × 20–80 % of the arc\n"
             f"00m: PD cMF {r2.pd_cMF_00m:.2f} vs expert {r2.eck_cMF_00m:.2f} mm; PD cLF {r2.pd_cLF_00m:.2f} vs {r2.eck_cLF_00m:.2f} mm", col_leg)
        draw(axes[i, 1], g, gn, V93, f"{tag}\nv9.3: Chondrometrics first-crossing cMF75 / cLF75 "
             f"(notch → posterior end {r.notch_to_post_med_mm:.0f} mm); denuded plate bone = 0 mm ({100 * info['den00_frac']:.1f} %)\n"
             f"00m: PD cMF {r.pd_cMF_00m:.2f} vs expert {r.eck_cMF_00m:.2f} mm; PD cLF {r.pd_cLF_00m:.2f} vs {r.eck_cLF_00m:.2f} mm", col_new)
        print(f"  {k}: {side}; v9.3 cMF cells {int((lab_new == 1).sum())} verts, cLF {int((lab_new == 2).sum())}")
    fig.suptitle("Where cMF / cLF are measured — femoral cartilage thickness at baseline (web-app 2D panel layout)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    if a.dry_run:
        print("[dry_run] not saved"); return
    out = RES / "figures" / "v9_3_femur_regions_webstyle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)
    print("[ok]", out)


if __name__ == "__main__":
    main()
