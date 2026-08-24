"""
Build femur-cart and tibia-cart probability volumes from the 327 KLG=0
OAI-MOAKS cohort, in the same tibia-template space as the soft-tissue atlas.

Cohort:  E:/KneeMR/Datasets/OAI/Legacy/OAI_MOAKS/48M_PRED_MINI/
                predicted_3d_good_moaks_score_RIGHT/  (327 cases, all RIGHT)
Anchor:  E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk

DESS 4-class label scheme (matches existing tibia_template build):
    1 = femur bone   2 = femur cart   3 = tibia bone   4 = tibia cart

This script extracts the cart probability VOLUMES — different and
complementary to the per-vertex `subch_prob` field already on the bone
templates (which is the bone-side cartilage *imprint*, not the cart layer
itself).

Outputs go to the same dir as build_softtissue_atlas.py, sharing the same
grid:
    femur_cart.nii.gz, tibia_cart.nii.gz
The `_grid_meta.json` is updated under `builds.cart` (preserves any prior
`builds.soft_tissue` entry).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from tqdm import tqdm

from knee_mr_seg.postproc import (
    load_seg_oriented, per_case_transform,
    template_grid, voxelize_mask_to_grid, make_grid_affine,
)


CART_CLASSES: dict[int, str] = {
    2: "femur_cart",
    4: "tibia_cart",
}
TIBIA_BONE_LABEL_DESS_4CLASS = 3   # same value as PD nnU-Net but different scheme

DEFAULT_DESS_GLOB = (
    r"E:/KneeMR/Datasets/OAI/Legacy/OAI_MOAKS/48M_PRED_MINI/"
    r"predicted_3d_good_moaks_score_RIGHT/*.nii.gz"
)
DEFAULT_TIBIA_VTK = r"E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk"
DEFAULT_FEMUR_VTK = r"E:/KneeMR/Datasets/Bone-Atlas/femur_template_full_with_subch.vtk"
DEFAULT_OUT_DIR   = r"E:/KneeMR/Datasets/Bone-Atlas/soft_tissue"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dess_glob",       default=DEFAULT_DESS_GLOB)
    p.add_argument("--tibia_template",  default=DEFAULT_TIBIA_VTK)
    p.add_argument("--femur_template",  default=DEFAULT_FEMUR_VTK)
    p.add_argument("--out_dir",         default=DEFAULT_OUT_DIR)
    p.add_argument("--n_run",           type=int, default=None)
    p.add_argument("--dry_run",         action="store_true")
    p.add_argument("--grid_spacing",    type=float, default=1.0)
    p.add_argument("--margin_mm",       type=float, default=30.0)
    args = p.parse_args()

    files = sorted(glob(args.dess_glob))
    if args.n_run is not None:
        files = files[: args.n_run]
    print(f"[input] {len(files)} cases from {args.dess_glob}")

    target = pv.read(args.tibia_template)
    print(f"[anchor] tibia = {Path(args.tibia_template).name}  "
          f"({target.n_points} verts)")

    grid_origin, grid_spacing, grid_dims = template_grid(
        args.femur_template, args.tibia_template,
        spacing_mm=args.grid_spacing, margin_mm=args.margin_mm,
    )
    print(f"[grid] origin={grid_origin.round(1).tolist()}  "
          f"spacing={grid_spacing}mm  dims={grid_dims}")

    accum = {lbl: np.zeros(grid_dims, dtype=np.float64) for lbl in CART_CLASSES}
    n_used = 0
    fail = []
    t0 = time.time()

    pbar = tqdm(files, desc="cases")
    for path in pbar:
        try:
            seg, spacing, side = load_seg_oriented(path)
            scale, translation = per_case_transform(
                seg, spacing, target,
                tibia_bone_label=TIBIA_BONE_LABEL_DESS_4CLASS,
            )
            for lbl in CART_CLASSES:
                accum[lbl] += voxelize_mask_to_grid(
                    (seg == lbl), spacing, scale, translation,
                    grid_origin, grid_spacing, grid_dims,
                )
            n_used += 1
            pbar.set_postfix(n=n_used, t=f"{(time.time()-t0)/n_used:.1f}s/case")
        except Exception as e:  # noqa: BLE001
            fail.append((Path(path).name, repr(e))); continue

    print(f"\n[done] used={n_used} / failed={len(fail)}")
    for n, r in fail[:10]:
        print(f"  fail: {n}  {r}")
    if n_used == 0:
        return

    probs = {lbl: (accum[lbl] / n_used).astype(np.float32) for lbl in CART_CLASSES}
    print("\n[probabilities]  (max  /  vox where p>=0.5)")
    for lbl, name in CART_CLASSES.items():
        v = probs[lbl]
        print(f"  lbl{lbl} {name:14s}  max={v.max():.2f}  "
              f"p>=0.5: {int((v >= 0.5).sum()):8d} vox  "
              f"p>=0.1: {int((v >= 0.1).sum()):8d} vox")
    if args.dry_run:
        print("\n[dry_run] not saving."); return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    aff = make_grid_affine(grid_origin, grid_spacing)
    for lbl, name in CART_CLASSES.items():
        nib.save(nib.Nifti1Image(probs[lbl], aff), str(out_dir / f"{name}.nii.gz"))

    meta_path = out_dir / "_grid_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "grid_origin_mm":     grid_origin.tolist(),
        "grid_spacing_mm":    grid_spacing,
        "grid_dims":          list(grid_dims),
        "tibia_template":     str(Path(args.tibia_template).resolve()),
        "femur_template":     str(Path(args.femur_template).resolve()),
        "tibia_target_ml_mm": 75.0,
        "tibia_target_ap_mm": 55.0,
    })
    meta.setdefault("builds", {})["cart"] = {
        "n_cases":           n_used,
        "failed":            [{"file": n, "reason": r} for n, r in fail],
        "classes":           CART_CLASSES,
        "dess_glob":         args.dess_glob,
        "label_scheme":      "DESS 4-class (1=femur_b, 2=femur_c, 3=tibia_b, 4=tibia_c)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
