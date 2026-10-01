"""v9.3 — Figure 2: the regions in which automated thickness was compared with the expert reference.

One representative leak-free progressor knee (baseline), three panels:
  a  side view of the medial femoral condyle: trochlear-notch plane, 75 % divider and posterior end
     (Chondrometrics construction), cartilage coloured by region (trochlea / cMF / pMF). cMF is the cartilage
     continuous from the notch up to the first crossing of the divider.
  b  femoral cartilage map in the web-app 2D layout (lateral left, anterior/trochlea at the top): thickness over the total
     subchondral bone area (denuded plate bone = 0 mm), cMF / cLF outlined, trochlea and pMF / pLF labelled
  c  tibial cartilage map (lateral left, anterior top): MT / LT outlined
Region membership is taken from the analysis grid exactly as measured and carried to the vertices; only the
display layout is the web-app one (`web_export.project_by_proj`, `_project_labels`).

Inputs: captured baseline-grid inputs (E:/KneeMR/Studies/PD-vs-DESS/v9.2/{femur,tibia}_region_diag/)
Output: ../manuscript/figures/Figure2_regions.png (600 dpi)
Usage: python make_figure_regions_v9_3.py [--knee 9xxxxxx_RIGHT] [--dry_run]
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
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from viz_femur_regions_webstyle_v9_3 import knee_labels, draw, bg, we, G  # noqa: E402

RES = HERE.parent / "results"
FIG = HERE.parent / "manuscript" / "figures"
CAP_F = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v9.2/femur_region_diag")
CAP_T = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v9.2/tibia_region_diag")
C_TRO, C_CMF, C_PMF = "#bdbdbd", "#1f4e9c", "#8e5ea2"
C_NOTCH, C_DIV, C_POST = "#d6249f", "#17becf", "#1f3fbf"


def tibia_labels(z):
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    bone = np.unpackbits(z["bone"])[:n].reshape(shape).astype(bool)
    cart = np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool)
    pts, th00, sp = z["points"], z["th00"], tuple(z["spacing"])
    geo = bg.compute_ref_geometry(cart, bone, "TC", "right_oriented")
    valid = (th00 > bg.MIN_THICK_MM) & (th00 < bg.MAX_THICK_MM)
    rr = bg.regional_deltas("tibia", pts, th00, z["th48"], bone, cart, sp, th48_status=z["status"], total_bone_area=True)
    den00 = rr["_tab_info"]["vertex_den00"]
    shown = valid | den00
    dn, wn, _ = bg.vertex_norm_coords(pts, np.ones(len(pts)), geo, sp)
    db = np.clip((dn * G).astype(int), 0, G - 1)
    lab = np.zeros(len(pts), int)
    lab[shown & (db < bg.D_MED.stop)] = 1
    lab[shown & (db >= bg.D_MED.stop)] = 2
    mesh = pv.PolyData(np.ascontiguousarray(pts[:, [2, 1, 0]], dtype=float))
    return mesh, shown, np.where(valid, np.asarray(th00, float), 0.0), lab, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knee", default="9010370_RIGHT")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    k = a.knee
    la = pd.read_csv(RES / "v9_3_long_abs.csv"); la["k"] = la.pid.astype(str) + "_" + la.side
    r = la[la.k == k].iloc[0]

    zf = np.load(CAP_F / f"{k}.npz", allow_pickle=True)
    mesh_f, valid_f, shown_f, thick_f, _, lab_f, info = knee_labels(zf)
    zt = np.load(CAP_T / f"{k}.npz", allow_pickle=True)
    mesh_t, shown_t, thick_t, lab_t, rr_t = tibia_labels(zt)

    fig = plt.figure(figsize=(17.15 / 2.54 * 2, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.18)

    # a — side view of the medial condyle
    ax = fig.add_subplot(gs[0, 0])
    pts = zf["points"]
    med = shown_f & np.isin(lab_f, [1, 3, 4])
    cols = np.array([C_TRO, C_CMF, C_TRO, C_TRO, C_PMF, C_TRO])   # label 0..5 -> colour (1 cMF, 3 trochlea, 4 pMF)
    ax.scatter(pts[med, 2], pts[med, 1], c=cols[lab_f[med]], s=0.5, lw=0, rasterized=True)
    ap0, ape = info["ap_notch"], info["ap_end_med"]
    for x, c, ls in ((ap0, C_NOTCH, "-"), (ap0 + 0.75 * (ape - ap0), C_DIV, "--"), (ape, C_POST, "-")):
        ax.axvline(x, color=c, lw=1.4, ls=ls)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("anterior → posterior (mm)"); ax.set_ylabel("superior → inferior (mm)")
    ax.legend(handles=[Patch(color=C_TRO, label="trochlea (anterior to notch)"), Patch(color=C_CMF, label="cMF"),
                       Patch(color=C_PMF, label="pMF (incl. posterior curl)"),
                       plt.Line2D([], [], color=C_NOTCH, label="trochlear-notch plane"),
                       plt.Line2D([], [], color=C_DIV, ls="--", label="75 % divider"),
                       plt.Line2D([], [], color=C_POST, label="posterior end")], fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)   # below the axes: the lower-left corner holds the trochlear points
    ax.set_title("a  Medial femoral condyle, side view", fontsize=10, loc="left")

    # b — femur map
    ax = fig.add_subplot(gs[0, 1])
    g = we.project_by_proj("femur", mesh_f, thick_f, shown_f, shown_f)
    gl = we._project_labels("femur", mesh_f, lab_f, shown_f, 5)
    draw(ax, g, gl, ["cMF", "cLF", "trochlea", "pMF", "pLF"], "", ["black", "black", "0.4", "0.4", "0.4"])
    # the projection puts the trochlea at the top and the posterior condyles (notch opening) at the bottom;
    # the template edge text inherited from draw() states the opposite, so replace it
    for t in ax.texts[:]:
        if t.get_text() in ("anterior (trochlea)", "posterior"):
            t.remove()
    ax.text(0.5, 0.99, "anterior (trochlea)", transform=ax.transAxes, fontsize=8, ha="center", va="top", color="0.3")
    ax.text(0.5, 0.01, "posterior", transform=ax.transAxes, fontsize=8, ha="center", va="bottom", color="0.3")
    ax.set_title("b  Femoral cartilage map (baseline)", fontsize=10, loc="left")

    # c — tibia map
    ax = fig.add_subplot(gs[0, 2])
    gt = we.project_by_proj("tibia", mesh_t, thick_t, shown_t, shown_t)
    glt = we._project_labels("tibia", mesh_t, lab_t, shown_t, 2)
    draw(ax, gt, glt, ["MT", "LT"], "", ["black", "black"])
    for t in ax.texts[:]:
        if t.get_text() in ("anterior (trochlea)", "posterior"):
            t.remove()
    ax.text(0.5, 0.99, "anterior", transform=ax.transAxes, fontsize=8, ha="center", va="top", color="0.3")
    ax.text(0.5, 0.01, "posterior", transform=ax.transAxes, fontsize=8, ha="center", va="bottom", color="0.3")
    ax.set_title("c  Tibial cartilage map (baseline)", fontsize=10, loc="left")
    sm = plt.cm.ScalarMappable(cmap="jet_r", norm=plt.Normalize(0, 3))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02); cb.set_label("cartilage thickness (mm)", fontsize=8)

    print(f"[{k}] {r.cohort} KL{r.KL_00}: PD cMF {r.pd_cMF_00m:.2f} (expert {r.eck_cMF_00m:.2f}), MT {r.pd_MT_00m:.2f} (expert {r.eck_MT_00m:.2f}); "
          f"notch→end {ape - ap0:.1f} mm; femur denuded {100 * info['den00_frac']:.1f} %")
    if a.dry_run:
        print("[dry_run] not saved"); return
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "Figure2_regions.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


if __name__ == "__main__":
    main()
