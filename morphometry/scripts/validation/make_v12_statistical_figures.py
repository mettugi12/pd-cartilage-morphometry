"""v1.2 cohort statistical figures:

  stat_BA_cross_sectional.png   — Bland-Altman of per-knee MEAN thickness,
      PD vs DESS, per bone (femur, tibia). x=(PD+DESS)/2, y=PD-DESS, with
      bias line + 95% LoA (bias +/- 1.96 SD). 61 knees/bone.
  stat_scatter_longitudinal_delta.png — pipeline Δ vs OAI Eckstein QCart Δ
      scatter, per region (cMF, cLF, MT, LT) + composites (MFTC, LFTC),
      with identity line, Pearson r, and the worst outliers labelled.

Reads the v1.2 report JSON for the longitudinal Δ pairs and the saved
cross-sectional thickness arrays for the per-knee BA means.

Usage:
  python -m scripts.make_v12_statistical_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _per_knee_mean(arr_path: Path) -> float:
    """nanmean over the subch-masked template thickness array (NaN outside)."""
    a = np.load(arr_path)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def bland_altman(ax, pd_vals, de_vals, title):
    pd_vals = np.asarray(pd_vals); de_vals = np.asarray(de_vals)
    m = np.isfinite(pd_vals) & np.isfinite(de_vals)
    pd_vals, de_vals = pd_vals[m], de_vals[m]
    mean_ = 0.5 * (pd_vals + de_vals)
    diff = pd_vals - de_vals
    bias = diff.mean(); sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    ax.scatter(mean_, diff, s=22, alpha=0.6, c="#1f77b4", edgecolors="none")
    ax.axhline(bias, color="k", lw=1.3, label=f"bias {bias:+.3f}")
    ax.axhline(lo, color="r", ls="--", lw=1.0, label=f"95% LoA [{lo:+.2f}, {hi:+.2f}]")
    ax.axhline(hi, color="r", ls="--", lw=1.0)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("mean thickness (PD+DESS)/2  (mm)", fontsize=9)
    ax.set_ylabel("PD - DESS  (mm)", fontsize=9)
    ax.set_title(f"{title}  (n={m.sum()})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls=":", alpha=0.4)


def scatter_delta(ax, pipe, qcart, tags, kl, title):
    pipe = np.asarray(pipe); qcart = np.asarray(qcart)
    r = np.corrcoef(pipe, qcart)[0, 1] if len(pipe) >= 3 else float("nan")
    ax.scatter(qcart, pipe, s=22, alpha=0.55, c="#1f77b4", edgecolors="none")
    lim = [min(pipe.min(), qcart.min()) - 0.1, max(pipe.max(), qcart.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="identity")
    # least-squares fit line
    if len(pipe) >= 3:
        b1, b0 = np.polyfit(qcart, pipe, 1)
        xs = np.array(lim)
        ax.plot(xs, b1 * xs + b0, color="#d62728", lw=1.2, label=f"fit (slope {b1:.2f})")
    # label the 3 worst residuals
    resid = np.abs(pipe - qcart)
    for i in np.argsort(-resid)[:3]:
        ax.annotate(tags[i].replace("_", " "), (qcart[i], pipe[i]),
                    fontsize=6.5, color="#555", xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(0, color="0.7", lw=0.6, zorder=0); ax.axvline(0, color="0.7", lw=0.6, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("QCart Δ (mm)", fontsize=9)
    ax.set_ylabel("pipeline Δ (mm)", fontsize=9)
    ax.set_title(f"{title}   r = {r:.3f}  (n={len(pipe)})", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path,
                   default=Path(r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Studies/cartilage-validation-pipeline/v1.2/reports/v1.2_raycast2d_FULL.json"))
    p.add_argument("--cross_thickness_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/thickness_v12_raycast2d/cross_sectional"))
    p.add_argument("--out_dir", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/v1.2_raycast2d_canonical/long/statistical_figures"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = json.load(open(args.report, encoding="utf-8"))

    # ---------- Bland-Altman (cross-sectional, per-knee mean thickness) ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, bone in zip(axes, ("femur", "tibia")):
        pd_means, de_means = [], []
        for c in d["cross_sectional"]["per_case"]:
            cid = c["case_id"]
            pd_p = args.cross_thickness_dir / f"{cid}_{bone}_pd.npy"
            de_p = args.cross_thickness_dir / f"{cid}_{bone}_dess.npy"
            if pd_p.exists() and de_p.exists():
                pd_means.append(_per_knee_mean(pd_p))
                de_means.append(_per_knee_mean(de_p))
        bland_altman(ax, pd_means, de_means,
                     f"{bone.title()} — PD vs DESS per-knee mean thickness")
    fig.suptitle("v1.2 cross-sectional Bland-Altman (per-knee mean cartilage thickness)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_ba = args.out_dir / "stat_BA_cross_sectional.png"
    fig.savefig(out_ba, dpi=140); plt.close(fig)
    print(f"[ok] {out_ba}")

    # ---------- Longitudinal Δ scatter (pipeline vs QCart) ----------
    pc = d["longitudinal"]["per_case"]
    def pull(region, bone, composite=False):
        tags, pipe, qc, kl = [], [], [], []
        for c in pc:
            q = c["eckstein_qcart"]
            if not q or not q.get("present"):
                continue
            qd = q.get(f"{region}_d")
            if composite:
                pv = c["composites"].get(f"{region}_d")
            else:
                pv = c["bones"][bone]["regions_delta"].get(region)
            if qd is None or pv is None or not np.isfinite(qd) or not np.isfinite(pv):
                continue
            tags.append(f"{c['pid']}_{c['side']}"); pipe.append(float(pv))
            qc.append(float(qd)); kl.append(c.get("KL_00"))
        return pipe, qc, tags, kl

    panels = [
        ("cMF", "femur", False, "femur.cMF"),
        ("cLF", "femur", False, "femur.cLF"),
        ("MFTC", None, True, "MFTC (cMF+MT)"),
        ("MT", "tibia", False, "tibia.MT"),
        ("LT", "tibia", False, "tibia.LT"),
        ("LFTC", None, True, "LFTC (cLF+LT)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    for ax, (region, bone, comp, title) in zip(axes.ravel(), panels):
        pipe, qc, tags, kl = pull(region, bone, comp)
        scatter_delta(ax, pipe, qc, tags, kl, title)
    fig.suptitle("v1.2 longitudinal Δ: pipeline vs OAI Eckstein QCart  "
                 "(48m − 00m; identity dashed, LS fit red; worst-3 residuals labelled)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_sc = args.out_dir / "stat_scatter_longitudinal_delta.png"
    fig.savefig(out_sc, dpi=140); plt.close(fig)
    print(f"[ok] {out_sc}")


if __name__ == "__main__":
    main()
