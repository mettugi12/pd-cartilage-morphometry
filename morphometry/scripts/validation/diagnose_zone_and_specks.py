"""Diagnose two template-projection artifacts on a handful of PD cases:

  1. WIDE SUBCH ZONE -> red rim. The canonical template mask is subch_prob>=0.5
     (a 50%-prevalence envelope, wider than any single patient's plate), so the
     periphery inherits near-zero (red) thickness. Raising the threshold narrows
     the zone.
  2. DENUDATION SPECKS. k=3 IDW bleeds cart-bearing values across a denudation
     boundary, painting small nonzero specks where the patient is fully denuded.
     The `idw_denudation_gate` (NN decides presence, IDW decides magnitude)
     suppresses them.

For each (seg, bone) we run the pipeline ONCE (RECON is the expensive step,
cached), then re-run only the cheap IDW remap under four variants:

    A  thr=0.5  gate=off   (canonical baseline)
    B  thr=0.5  gate=on    (isolates the gate effect -> specks)
    C  thr=0.8  gate=off   (isolates the threshold effect -> rim)
    D  thr=0.8  gate=on    (combined fix)

Output: one PNG per (seg, bone) with a 2x5 grid (3D row / 2D row; columns =
patient + the 4 variants), annotated with rim% and gated-speck counts.

Honors --dry_run (skip figure save, print metrics only) and --n_run N.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from cartilage_morphometry import PipelineConfig, TEMPLATE_PATHS
from cartilage_morphometry import pipeline as _pipeline
from cartilage_morphometry.remap import remap_thickness_to_template
from cartilage_morphometry.validation.viz import (
    project_thickness_2d_canonical,
    screenshot_3d_thickness,
)

REPO = Path(__file__).resolve().parents[2]
SEGS = [
    REPO / "Rt_PD SPIR SAG_20180409_mask.nii.gz",
    REPO / "Rt_SAG FSE PD FS_20180409_mask.nii.gz",
]
SIDE = "RIGHT"
MODALITY = "pd"
RECON_CACHE = Path(r"E:/KneeMR/Studies/_diagnostics/recon_cache")

# (label, subch_threshold, denudation_gate)
VARIANTS = [
    ("A thr0.5 gate-off", 0.5, False),
    ("B thr0.5 gate-on", 0.5, True),
    ("C thr0.8 gate-off", 0.8, False),
    ("D thr0.8 gate-on", 0.8, True),
]
RIM_MM = 0.5   # thickness below this counts as "thin/red rim"
VMAX = 4.0


def _rim_frac(masked_thickness: np.ndarray) -> tuple[int, int, float]:
    """(n_valid, n_thin, thin_fraction) over finite (in-zone) template verts."""
    finite = np.isfinite(masked_thickness)
    n_valid = int(finite.sum())
    if n_valid == 0:
        return 0, 0, float("nan")
    n_thin = int((finite & (masked_thickness < RIM_MM)).sum())
    return n_valid, n_thin, n_thin / n_valid


def run_case(seg_path: Path, bone: str, out_dir: Path, dry_run: bool):
    print(f"\n=== {seg_path.name} | {bone} ===")
    cfg = PipelineConfig()  # canonical defaults for the single expensive pass
    res = _pipeline.process_one_patient(
        seg_path=seg_path, side=SIDE, bone_name=bone, modality=MODALITY,
        config=cfg,
    )
    bone_mesh = res["bone_mesh"]
    aligned_pts = np.asarray(res["aligned_pts"])
    patient_thickness = np.asarray(bone_mesh["thickness_mm"], dtype=np.float32)
    patient_subch_prob = np.asarray(bone_mesh["subch_prob"], dtype=np.float32)

    template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
    tpl_prob = np.asarray(template_mesh.point_data["subch_prob"])

    # Re-remap cheaply per variant
    variant_out = []
    for label, thr, gate in VARIANTS:
        tt, q = remap_thickness_to_template(
            template_mesh, aligned_pts, patient_thickness, patient_subch_prob,
            k=cfg.idw_k, mutual_nn=cfg.idw_mutual_nn,
            subch_threshold=thr, denudation_gate=gate,
        )
        masked = tt.copy()
        masked[~(tpl_prob >= thr)] = np.nan
        n_valid, n_thin, rim = _rim_frac(masked)
        variant_out.append({
            "label": label, "thr": thr, "gate": gate,
            "masked": masked, "n_valid": n_valid, "n_thin": n_thin,
            "rim_frac": rim, "n_gated": q["n_denudation_gated"],
            "mean": float(np.nanmean(masked)) if n_valid else float("nan"),
        })
        print(f"  {label:18s} n_zone={n_valid:5d}  rim(<{RIM_MM}mm)={rim:5.1%}  "
              f"specks_gated={q['n_denudation_gated']:4d}  mean={variant_out[-1]['mean']:.2f}mm")

    if dry_run:
        print("  [dry_run] skipping figure")
        return

    # ---- figure: 2 rows (3D / 2D) x 5 cols (patient + 4 variants) ----
    fig, axes = plt.subplots(2, 5, figsize=(26, 11))

    # Patient column
    img_pat = screenshot_3d_thickness(bone_mesh, patient_thickness, bone, vmax=VMAX)
    axes[0, 0].imshow(img_pat)
    axes[0, 0].set_title("PATIENT 3D (patient frame)", fontsize=11)
    g_pat, _ = project_thickness_2d_canonical(
        bone, bone_mesh, patient_thickness, subch_prob=patient_subch_prob)
    axes[1, 0].imshow(g_pat, origin="upper", cmap="jet_r", vmin=0, vmax=VMAX,
                      interpolation="bilinear", aspect="equal")
    n_den = int((patient_subch_prob >= 0.5).sum() - (patient_thickness > 0).sum())
    axes[1, 0].set_title(f"PATIENT 2D\n(subch verts denuded ~{max(n_den,0)})", fontsize=10)

    for j, v in enumerate(variant_out, start=1):
        img = screenshot_3d_thickness(template_mesh, v["masked"], bone, vmax=VMAX,
                                      zeros_as_denuded=True)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"TEMPLATE 3D  {v['label']}", fontsize=10)
        g, _ = project_thickness_2d_canonical(bone, template_mesh, v["masked"])
        axes[1, j].imshow(g, origin="upper", cmap="jet_r", vmin=0, vmax=VMAX,
                          interpolation="bilinear", aspect="equal")
        axes[1, j].set_title(
            f"{v['label']}\nrim<{RIM_MM}mm={v['rim_frac']:.0%}  "
            f"specks_killed={v['n_gated']}\nmean={v['mean']:.2f}mm  n={v['n_valid']}",
            fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    cid = seg_path.name.replace("_mask.nii.gz", "").replace(" ", "_")
    fig.suptitle(f"{seg_path.name}  |  {bone}  |  zone-width (rim) & denudation specks",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{cid}_{bone}_zone_specks.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=REPO / "diagnostics" / "zone_specks")
    ap.add_argument("--bones", default="femur,tibia")
    ap.add_argument("--n_run", type=int, default=None, help="limit number of segs")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    _pipeline.set_recon_disk_cache_dir(RECON_CACHE)
    bones = [b for b in args.bones.split(",") if b]
    segs = SEGS[: args.n_run] if args.n_run else SEGS
    for seg in segs:
        if not seg.exists():
            print(f"[skip] missing {seg}")
            continue
        for bone in bones:
            run_case(seg, bone, args.out_dir, args.dry_run)
        _pipeline.clear_recon_cache()  # free the densified volume between segs


if __name__ == "__main__":
    main()
