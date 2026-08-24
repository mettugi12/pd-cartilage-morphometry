"""
Evaluate per-class Dice on the PD nnU-Net Dataset204_Knee3D internal test set
under five postprocessing variants:

    baseline    raw nnU-Net output, no postproc
    +bone       keep_largest_bone_components               (cartilage_morphometry)
    +cart       drop_small_cart_components +
                filter_cart_near_bone_per_slice (native)   (cartilage_morphometry)
    +atlas      filter_by_atlas (7 soft-tissue classes)    (knee_mr_atlas)
    +all        bone + cart + atlas

Inputs
------
--preds_dir   Dir of {stem}_seg.nii.gz predictions
--labels_dir  Dir of {stem}.nii.gz GT labels
--atlas_dir   Atlas dir (default E:/.../Bone-Atlas/soft_tissue)
--out_csv     Output CSV path
--p_min       Atlas filter threshold (default 0.05)

Side detection:
- filename Lt/Rt/Left/Right tag → use directly
- otherwise → ML-flip check (per-case tibia ICP residual, smaller wins)

Cart filter runs on the NATIVE (non-RECON) prediction — production pipeline
uses RECON 4× upsample but for measuring postproc effect on Dice we want
the cleaner head-to-head comparison.

Usage:
    python scripts/eval_postproc_dice.py \
        --preds_dir  E:/KneeMR/eval/PD_seg_test/predsTs \
        --labels_dir E:/KneeMR/eval/PD_seg_test/labelsTs \
        --out_csv    E:/KneeMR/eval/PD_seg_test/dice_results.csv \
        --n_run 5     # pilot
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# Imports from sibling repos
_KMC_REPO = Path(r"C:\Users\mettu\OneDrive\바탕 화면\Connecteve_Research\KneeMR\Repos\knee-mr-cartilage")
if str(_KMC_REPO) not in sys.path:
    sys.path.insert(0, str(_KMC_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knee_mr_seg.postproc.atlas_filter import (
    SOFT_TISSUE_CLASS_NAMES, AtlasBundle, load_atlas_bundle,
    detect_side_from_name, detect_side_via_flip,
    filter_by_atlas, DEFAULT_ATLAS_DIR, DEFAULT_TIBIA_VTK,
)
from cartilage_morphometry.pipeline import (
    keep_largest_bone_components,
    drop_small_cart_components,
    filter_cart_near_bone_per_slice,
)


PD_LABELS = {
    1: "patella",       2: "femur",         3: "tibia",
    4: "patella_cart",  5: "femur_cart",    6: "tibia_cart",
    7: "medial_meniscus", 8: "lateral_meniscus",
    9: "acl", 10: "pcl", 11: "patellar_tendon",
}

CART_LABELS = (4, 5, 6)


# ---------------------------------------------------------------------------
def dice_per_label(pred: np.ndarray, gt: np.ndarray, labels) -> dict:
    out = {}
    for lbl in labels:
        p = (pred == lbl)
        g = (gt   == lbl)
        np_ = int(p.sum()); ng = int(g.sum())
        if np_ == 0 and ng == 0:
            out[lbl] = float("nan")
            continue
        inter = int((p & g).sum())
        out[lbl] = 2.0 * inter / (np_ + ng)
    return out


def apply_bone_postproc(seg: np.ndarray) -> np.ndarray:
    out, _ = keep_largest_bone_components(seg)
    return out


def apply_cart_postproc(seg: np.ndarray, spacing: tuple,
                        min_size_vox: int = 100,
                        r_mm: float = 6.0) -> np.ndarray:
    """Apply drop_small + filter_near_bone per cart class.

    Cart label 4 (PC) is anchored to patella (label 1).
    Cart label 5 (FC) is anchored to femur (label 2).
    Cart label 6 (TC) is anchored to tibia (label 3).
    """
    out = seg.copy()
    cart_to_bone = {4: 1, 5: 2, 6: 3}
    for cart_lbl, bone_lbl in cart_to_bone.items():
        cart = (out == cart_lbl)
        bone = (out == bone_lbl)
        if cart.sum() == 0:
            continue
        # 1) drop small CC
        cart_ds, _ = drop_small_cart_components(cart, min_size_vox=min_size_vox)
        # 2) drop voxels far from bone (per sagittal slice)
        if bone.sum() > 0:
            cart_ds, _ = filter_cart_near_bone_per_slice(
                cart_ds, bone, spacing, r_mm=r_mm, slice_axis=2,
            )
        # zero out cart voxels that were dropped
        out[cart & ~cart_ds] = 0
    return out


def apply_atlas_postproc(seg: np.ndarray, spacing: tuple, side: str,
                          atlas: AtlasBundle, p_min: float) -> tuple[np.ndarray, dict]:
    return filter_by_atlas(seg, spacing, side, atlas, p_min=p_min)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir",  required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--atlas_dir",  default=DEFAULT_ATLAS_DIR)
    ap.add_argument("--tibia_vtk",  default=DEFAULT_TIBIA_VTK)
    ap.add_argument("--out_csv",    required=True)
    ap.add_argument("--p_min",      type=float, default=0.05)
    ap.add_argument("--cart_min_vox", type=int, default=100)
    ap.add_argument("--cart_r_mm",   type=float, default=6.0)
    ap.add_argument("--n_run",      type=int, default=None)
    ap.add_argument("--dry_run",    action="store_true")
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    preds_dir  = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    out_csv    = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] atlas from {args.atlas_dir}")
    atlas = load_atlas_bundle(args.atlas_dir, args.tibia_vtk)
    print(f"[load] atlas OK — {len(atlas.prob_by_label)} class prob volumes, "
          f"grid_dims={atlas.grid_dims}")

    pred_files = sorted(preds_dir.glob("*_seg.nii.gz")) or sorted(preds_dir.glob("*.nii.gz"))
    if args.n_run is not None:
        pred_files = pred_files[: args.n_run]
    print(f"[load] {len(pred_files)} predictions found in {preds_dir}")

    if args.dry_run:
        for p in pred_files[:5]:
            print(f"  sample: {p.name}")
        return

    rows = []
    variants = ["baseline", "+bone", "+cart", "+atlas", "+all"]
    eval_labels = sorted(PD_LABELS.keys())

    for pi, pred_path in enumerate(pred_files):
        # Match GT label by stripping _seg suffix from prediction
        stem = pred_path.name.replace("_seg.nii.gz", "").replace(".nii.gz", "")
        # GT files don't have _0000 suffix (they're labels), and don't have _seg
        # The pred filename pattern: <case>_<series>_<...>_<date>_seg.nii.gz
        # Or:                       <case>_<series>_<...>_<date>_0000_seg.nii.gz (if Modal kept _0000)
        # Strip trailing _0000 if present
        if stem.endswith("_0000"):
            stem = stem[:-5]
        # GT may match either with or without an extra trailing tag
        gt_path = labels_dir / f"{stem}.nii.gz"
        if not gt_path.exists():
            # Try without the extra suffixes
            print(f"  [skip] no GT for {pred_path.name} (looked for {gt_path.name})")
            continue

        t0 = time.time()
        pred_nii = nib.load(str(pred_path))
        gt_nii   = nib.load(str(gt_path))
        pred = pred_nii.get_fdata().astype(np.int16)
        gt   = gt_nii.get_fdata().astype(np.int16)
        spacing = tuple(float(s) for s in pred_nii.header.get_zooms()[:3])

        if pred.shape != gt.shape:
            print(f"  [skip] shape mismatch {pred_path.name}: {pred.shape} vs {gt.shape}")
            continue

        # --- side detection
        side = detect_side_from_name(pred_path.name)
        if side == "UNKNOWN":
            side, res_R, res_L = detect_side_via_flip(pred, spacing, atlas.target_tibia)
            tag = f"flip-check ({res_R:.2f} vs {res_L:.2f}mm)"
        else:
            tag = "filename"

        # --- variants
        seg_bone  = apply_bone_postproc(pred)
        seg_cart  = apply_cart_postproc(pred, spacing,
                                        args.cart_min_vox, args.cart_r_mm)
        seg_atlas, atlas_log = apply_atlas_postproc(pred, spacing, side, atlas, args.p_min)
        # all = compose: bone first, then cart, then atlas
        seg_all = apply_bone_postproc(pred)
        seg_all = apply_cart_postproc(seg_all, spacing,
                                       args.cart_min_vox, args.cart_r_mm)
        seg_all, _ = apply_atlas_postproc(seg_all, spacing, side, atlas, args.p_min)

        all_segs = {
            "baseline": pred,
            "+bone":    seg_bone,
            "+cart":    seg_cart,
            "+atlas":   seg_atlas,
            "+all":     seg_all,
        }

        for variant in variants:
            d = dice_per_label(all_segs[variant], gt, eval_labels)
            rows.append({
                "case": stem,
                "side": side,
                "side_tag": tag,
                "variant": variant,
                **{f"dice_{lbl:02d}_{PD_LABELS[lbl]}": d[lbl] for lbl in eval_labels},
            })

        elapsed = time.time() - t0
        # Compact per-case summary
        d_base = dice_per_label(pred,    gt, eval_labels)
        d_all  = dice_per_label(seg_all, gt, eval_labels)
        deltas = {lbl: (d_all[lbl] - d_base[lbl]) for lbl in eval_labels
                  if not (np.isnan(d_all[lbl]) or np.isnan(d_base[lbl]))}
        print(f"[{pi+1}/{len(pred_files)}] {stem}  side={side}({tag})  "
              f"{elapsed:.1f}s  Δ_all-base "
              + " ".join(f"{PD_LABELS[lbl][:3]}{deltas[lbl]:+.3f}"
                          for lbl in sorted(deltas)))

    # --- write CSV
    if not rows:
        print("\nNo cases evaluated — exiting.")
        return

    fieldnames = ["case", "side", "side_tag", "variant"] + \
                 [f"dice_{lbl:02d}_{PD_LABELS[lbl]}" for lbl in eval_labels]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {len(rows)} rows -> {out_csv}")

    # --- per-class summary
    print("\n" + "=" * 86)
    print(f"{'class':<20} " + " ".join(f"{v:>11}" for v in variants))
    print("-" * 86)
    by_var = {v: {lbl: [] for lbl in eval_labels} for v in variants}
    for r in rows:
        for lbl in eval_labels:
            d = r[f"dice_{lbl:02d}_{PD_LABELS[lbl]}"]
            if not (d is None or (isinstance(d, float) and np.isnan(d))):
                by_var[r["variant"]][lbl].append(d)
    for lbl in eval_labels:
        line = f"{lbl:>2} {PD_LABELS[lbl][:17]:<17} "
        means = []
        for v in variants:
            arr = by_var[v][lbl]
            mean = float(np.mean(arr)) if arr else float("nan")
            means.append(mean)
        for i, m in enumerate(means):
            if i == 0:
                line += f" {m:>10.4f} "
            else:
                delta = m - means[0]
                marker = "▲" if delta > 1e-4 else ("▼" if delta < -1e-4 else " ")
                line += f"{m:>7.4f}{marker}{delta:+.3f}".rjust(12)
        print(line)
    print("=" * 86)


if __name__ == "__main__":
    main()
