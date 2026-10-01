"""v9.3 — Figure 2: how cartilage thickness was measured and where it was compared with the expert reference.

One primary-set progressor knee at baseline (automated PD/IW segmentation after depth densification):
  a  thickness computation on one sagittal slice through the medial condyle, drawn with the library's own
     raycast_2d helpers (bone mask smoothed, outward in-plane normal, ray through the smoothed cartilage from
     entry to exit, 0.05-mm steps, 6-mm cap); rays coloured by thickness; bone-surface points within 4 mm of
     cartilage whose ray crosses no cartilage are marked
  b  per-vertex treatment over the femoral plate (web-app layout): measured, denuded (no cartilage within
     1.5 mm; counted as 0 mm) and failed ray casts (cartilage nearby but no crossing; excluded from both visits);
     bone outside the baseline plate outline is not part of any region
  c  side view of the medial condyle with the Chondrometrics construction (notch plane, 75 % divider,
     posterior end) and the regions (trochlea / cMF / pMF)
  d  femoral and e tibial thickness maps with the analysed regions outlined (denuded bone = 0 mm)

Inputs: captured baseline-grid inputs (E:/KneeMR/Studies/PD-vs-DESS/v9.2/{femur,tibia}_region_diag/)
Output: ../manuscript/figures/Figure2_measurement.png (600 dpi)
Usage: python make_figure_measurement_v9_3.py [--knee 9733288_RIGHT] [--dry_run]
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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from viz_femur_regions_webstyle_v9_3 import knee_labels, draw, bg, we, G  # noqa: E402
from template_panels_v9_3 import align_to_template, remap_values, remap_labels, panel_grids  # noqa: E402
from make_figure_regions_v9_3 import tibia_labels, C_TRO, C_CMF, C_PMF, C_NOTCH, C_DIV, C_POST, CAP_F, CAP_T  # noqa: E402

RES = HERE.parent / "results"
FIG = HERE.parent / "manuscript" / "figures"
REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from cartilage_morphometry.strategies.thickness import _smooth_bone_normals_2d, _raycast_2d_batch  # noqa: E402

C_MEAS, C_DEN, C_FAIL = "#4c9be8", "#d62728", "#7f7f7f"


def unpack(z):
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    return (np.unpackbits(z["bone"])[:n].reshape(shape).astype(bool),
            np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool), tuple(z["spacing"]))


def panel_raycast(ax, z):
    """One sagittal slice through the medial condyle; masks are in grid order (ML, SI, AP)."""
    bone, cart, sp = unpack(z)
    geo = bg.compute_ref_geometry(cart, bone, "FC", "right_oriented")
    zi = int(geo["d_min"] + 0.25 * geo["d_range"])                         # centre of the medial condyle
    b2d, c2d = bone[zi].T, cart[zi].T                                       # (AP, SI) as in the library slice
    spy, spx = sp[2], sp[1]
    soft, boundary, ny, nx = _smooth_bone_normals_2d(b2d, sigma=1.8)
    cart_soft = gaussian_filter(c2d.astype(np.float32), sigma=0.8)
    ys, xs = np.where(boundary)
    dt = distance_transform_edt(~c2d, sampling=(spy, spx))
    keep = dt[ys, xs] <= 4.0
    ys, xs = ys[keep], xs[keep]
    th = _raycast_2d_batch(ys.astype(float), xs.astype(float), ny[ys, xs], nx[ys, xs], cart_soft, (spy, spx), max_mm=6.0)
    # entry point along each ray (for drawing): first step inside the cartilage
    step = 0.05; ts = np.arange(1, 121) * step
    from scipy.ndimage import map_coordinates
    yy = ys[None, :] + ts[:, None] * (ny[ys, xs] / spy)[None, :]
    xx = xs[None, :] + ts[:, None] * (nx[ys, xs] / spx)[None, :]
    inside = map_coordinates(cart_soft, np.stack([yy.ravel(), xx.ravel()]), order=1).reshape(len(ts), -1) > 0.5
    first = np.argmax(inside, axis=0)
    ent = ts[first]
    # crop to the articular region and draw (AP horizontal, SI vertical; mm)
    rows, cols = np.where(c2d | (dt <= 4.0) & b2d)
    r0, r1 = max(rows.min() - 15, 0), min(rows.max() + 15, b2d.shape[0])
    c0, c1 = max(cols.min() - 15, 0), min(cols.max() + 15, b2d.shape[1])
    img = np.zeros(b2d.shape + (3,)) + 1.0
    img[b2d] = (0.80, 0.80, 0.80); img[c2d] = (0.72, 0.86, 0.95)
    ext = [c0 * spx, c1 * spx, r1 * spy, r0 * spy]
    ax.imshow(img[r0:r1, c0:c1].transpose(1, 0, 2), origin="upper",
              extent=[r0 * spy, r1 * spy, c1 * spx, c0 * spx], interpolation="nearest")
    hit = th > 0
    idx = np.where(hit)[0][::3]
    segs = []
    for i in idx:
        p0 = np.array([ys[i] + ent[i] * ny[ys[i], xs[i]] / spy, xs[i] + ent[i] * nx[ys[i], xs[i]] / spx])
        p1 = p0 + th[i] * np.array([ny[ys[i], xs[i]] / spy, nx[ys[i], xs[i]] / spx])
        segs.append([(p0[0] * spy, p0[1] * spx), (p1[0] * spy, p1[1] * spx)])
    lc = LineCollection(segs, cmap="jet_r", norm=plt.Normalize(0, 3), linewidths=1.0)
    lc.set_array(th[idx]); ax.add_collection(lc)
    miss = ~hit
    ax.scatter(ys[miss] * spy, xs[miss] * spx, s=3, c=C_DEN, lw=0, label="no cartilage crossing")
    ax.set_xlim(r0 * spy, r1 * spy); ax.set_ylim(c1 * spx, c0 * spx); ax.set_aspect("equal")
    ax.set_xlabel("anterior → posterior (mm)"); ax.set_ylabel("superior → inferior (mm)")
    ax.legend(handles=[Patch(color=(0.80, 0.80, 0.80), label="bone"), Patch(color=(0.72, 0.86, 0.95), label="cartilage"),
                       plt.Line2D([], [], color="#1f77b4", label="ray: entry → exit (colour = thickness)"),
                       plt.Line2D([], [], marker="o", ls="", color=C_DEN, ms=3, label="bone point without a crossing")],
              fontsize=6.5, loc="lower left", frameon=True, framealpha=0.85)
    return lc, float(np.median(th[hit])), int(hit.sum()), int(miss.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knee", default="9733288_RIGHT")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    k = a.knee
    la = pd.read_csv(RES / "v9_3_long_abs.csv"); la["k"] = la.pid.astype(str) + "_" + la.side
    r = la[la.k == k].iloc[0]
    zf = np.load(CAP_F / f"{k}.npz", allow_pickle=True)
    zt = np.load(CAP_T / f"{k}.npz", allow_pickle=True)
    mesh_f, valid_f, shown_f, thick_f, _, lab_f, info = knee_labels(zf)
    mesh_t, shown_t, thick_t, lab_t, _ = tibia_labels(zt)

    # per-vertex treatment classes over the femoral plate
    bone, cart, sp = unpack(zf)
    geo = bg.compute_ref_geometry(cart, bone, "FC", "right_oriented")
    g00, g48, ref, tab = bg.total_bone_area_grids(zf["points"], zf["th00"], zf["th48"], zf["status"], cart, geo, sp)
    zone, den, fail = tab["vertex_zone"], tab["vertex_den00"], tab["vertex_fail00"]
    cls = np.zeros(len(den), int)
    cls[zone & valid_f] = 1; cls[den] = 2; cls[fail] = 3

    # web-app template frame for the 2D maps (patient values remapped onto the atlas cartilage zone)
    tf, alf, ztf = align_to_template(zf, "femur")
    tt, alt, ztt = align_to_template(zt, "tibia")
    src_cls = cls > 0
    cls_t = remap_labels(tf, alf, ztf, src_cls, cls)
    thick_ft = remap_values(tf, alf, ztf, shown_f, thick_f)
    lab_ft = remap_labels(tf, alf, ztf, shown_f, lab_f)
    thick_tt = remap_values(tt, alt, ztt, shown_t, thick_t)
    lab_tt = remap_labels(tt, alt, ztt, shown_t, lab_t)

    fig = plt.figure(figsize=(7.2 * 1.9, 8.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.28, wspace=0.55)

    ax = fig.add_subplot(gs[0, :3])
    lc, med_th, nhit, nmiss = panel_raycast(ax, zf)
    ax.set_title("a  Thickness by in-plane ray cast from the bone surface (one sagittal slice)", fontsize=9.5, loc="left")
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02); cb.set_label("thickness (mm)", fontsize=8)

    ax = fig.add_subplot(gs[0, 3:])
    gl = we._project_labels("femur", tf, cls_t, ztf, 3)
    zdisp = np.isfinite(we.project_by_proj("femur", tf, np.ones(tf.n_points), ztf, ztf))
    _, inds = distance_transform_edt(gl == 0, return_indices=True)
    gl = np.where(zdisp, gl[tuple(inds)], 0)
    cm = ListedColormap([C_MEAS, C_DEN, C_FAIL])
    ax.imshow(np.where(gl > 0, gl, np.nan), origin="upper", cmap=cm, vmin=0.5, vmax=3.5, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for x_, t_, ha in ((0.02, "L", "left"), (0.98, "M", "right")):
        ax.text(x_, 0.97, t_, transform=ax.transAxes, fontsize=10, color="white", va="top", ha=ha, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
    ax.legend(handles=[Patch(color=C_MEAS, label="measured"), Patch(color=C_DEN, label="denuded: no cartilage within 1.5 mm → 0 mm"),
                       Patch(color=C_FAIL, label="failed ray cast (cartilage nearby) → excluded at both visits")],
              fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    ax.set_title("b  Treatment of each bone-surface point inside the plate (femur)", fontsize=9.5, loc="left")

    # c side view
    ax = fig.add_subplot(gs[1, :2])
    pts = zf["points"]
    med = shown_f & np.isin(lab_f, [1, 3, 4])
    cols = np.array([C_TRO, C_CMF, C_TRO, C_TRO, C_PMF, C_TRO])
    ax.scatter(pts[med, 2], pts[med, 1], c=cols[lab_f[med]], s=0.5, lw=0, rasterized=True)
    ap0, ape = info["ap_notch"], info["ap_end_med"]
    for x, c, ls in ((ap0, C_NOTCH, "-"), (ap0 + 0.75 * (ape - ap0), C_DIV, "--"), (ape, C_POST, "-")):
        ax.axvline(x, color=c, lw=1.3, ls=ls)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("anterior → posterior (mm)"); ax.set_ylabel("superior → inferior (mm)")
    ax.legend(handles=[Patch(color=C_TRO, label="trochlea"), Patch(color=C_CMF, label="cMF"), Patch(color=C_PMF, label="pMF"),
                       plt.Line2D([], [], color=C_NOTCH, label="notch plane"), plt.Line2D([], [], color=C_DIV, ls="--", label="75 % divider"),
                       plt.Line2D([], [], color=C_POST, label="posterior end")], fontsize=6.3, loc="lower left", frameon=False)
    ax.set_title("c  Medial condyle, side view", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 2:4])
    g, glab = panel_grids("femur", tf, ztf, thick_ft, lab_ft, 5)
    draw(ax, g, glab, ["cMF", "cLF", "trochlea", "pMF", "pLF"], "", ["black", "black", "0.4", "0.4", "0.4"])
    ax.set_title("d  Femoral thickness and regions", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 4:])
    gt, glt = panel_grids("tibia", tt, ztt, thick_tt, lab_tt, 2)
    draw(ax, gt, glt, ["MT", "LT"], "", ["black", "black"])
    for t in ax.texts[:]:
        if t.get_text() in ("anterior (trochlea)", "posterior"):
            t.remove()
    ax.text(0.5, 0.99, "anterior", transform=ax.transAxes, fontsize=8, ha="center", va="top", color="0.3")
    ax.text(0.5, 0.01, "posterior", transform=ax.transAxes, fontsize=8, ha="center", va="bottom", color="0.3")
    ax.set_title("e  Tibial thickness and regions", fontsize=9.5, loc="left")
    sm = plt.cm.ScalarMappable(cmap="jet_r", norm=plt.Normalize(0, 3))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02); cb.set_label("thickness (mm)", fontsize=8)

    n_meas, n_den = int((cls == 1).sum()), int((cls == 2).sum())
    print(f"[{k}] {r.cohort} KL{r.KL_00}: slice rays {nhit} hit / {nmiss} without crossing, median {med_th:.2f} mm; "
          f"femur plate vertices measured {n_meas}, denuded {n_den} ({100 * info['den00_frac']:.1f} %), failed {int((cls == 3).sum())}; "
          f"cMF {r.pd_cMF_00m:.2f} vs expert {r.eck_cMF_00m:.2f}; MT {r.pd_MT_00m:.2f} vs {r.eck_MT_00m:.2f}")
    if a.dry_run:
        print("[dry_run] not saved"); return
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "Figure2_measurement.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


if __name__ == "__main__":
    main()
