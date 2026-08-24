"""
Build per-class soft-tissue probability volumes (NIfTI) in the existing
tibia-template coordinate frame, from manually-labeled 11-class DESS segs.

Cohort:  E:/KneeMR/Datasets/DESS18seg/3D_DESS_labels/  (110 cases)
Anchor:  E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk

Classes (PD nnU-Net Dataset204_Knee3D scheme):
    1=patella, 4=patella_cart, 7=medial_meniscus, 8=lateral_meniscus,
    9=acl, 10=pcl, 11=patellar_tendon

(Femur / tibia / their cart come from the bone+cart templates and from
build_cart_atlas.py — NOT this script.)
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
    TIBIA_BONE_LABEL_DEFAULT,
)


SOFT_TISSUE_CLASSES: dict[int, str] = {
    1:  "patella",
    4:  "patella_cart",
    7:  "medial_meniscus",
    8:  "lateral_meniscus",
    9:  "acl",
    10: "pcl",
    11: "patellar_tendon",
}

DEFAULT_DESS_GLOB = r"E:/KneeMR/Datasets/DESS18seg/3D_DESS_labels/*.nii.gz"
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
          f"({target.n_points} verts, bounds={tuple(round(x,1) for x in target.bounds)})")

    grid_origin, grid_spacing, grid_dims = template_grid(
        args.femur_template, args.tibia_template,
        spacing_mm=args.grid_spacing, margin_mm=args.margin_mm,
    )
    print(f"[grid] origin={grid_origin.round(1).tolist()}  "
          f"spacing={grid_spacing}mm  dims={grid_dims}  "
          f"voxels={int(np.prod(grid_dims)):,}")

    accum = {lbl: np.zeros(grid_dims, dtype=np.float64) for lbl in SOFT_TISSUE_CLASSES}
    n_used = 0
    fail = []
    t0 = time.time()

    pbar = tqdm(files, desc="cases")
    for path in pbar:
        try:
            seg, spacing, side = load_seg_oriented(path)
            scale, translation = per_case_transform(seg, spacing, target,
                                                     tibia_bone_label=TIBIA_BONE_LABEL_DEFAULT)
            for lbl in SOFT_TISSUE_CLASSES:
                accum[lbl] += voxelize_mask_to_grid(
                    (seg == lbl), spacing, scale, translation,
                    grid_origin, grid_spacing, grid_dims,
                )
            n_used += 1
            pbar.set_postfix(side=side, n=n_used,
                              t=f"{(time.time()-t0)/n_used:.1f}s/case")
        except Exception as e:  # noqa: BLE001
            fail.append((Path(path).name, repr(e))); continue

    print(f"\n[done] used={n_used} / failed={len(fail)}")
    for n, r in fail[:10]:
        print(f"  fail: {n}  {r}")
    if n_used == 0:
        return

    probs = {lbl: (accum[lbl] / n_used).astype(np.float32) for lbl in SOFT_TISSUE_CLASSES}
    print("\n[probabilities]  (max  /  vox where p>=0.5)")
    for lbl, name in SOFT_TISSUE_CLASSES.items():
        v = probs[lbl]
        print(f"  lbl{lbl:2d} {name:18s}  max={v.max():.2f}  "
              f"p>=0.5: {int((v >= 0.5).sum()):8d} vox  "
              f"p>=0.1: {int((v >= 0.1).sum()):8d} vox")
    if args.dry_run:
        print("\n[dry_run] not saving."); return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    aff = make_grid_affine(grid_origin, grid_spacing)
    for lbl, name in SOFT_TISSUE_CLASSES.items():
        nib.save(nib.Nifti1Image(probs[lbl], aff), str(out_dir / f"{name}.nii.gz"))

    # Merge with any existing meta (so the cart-atlas build can co-exist)
    meta_path = out_dir / "_grid_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "grid_origin_mm":      grid_origin.tolist(),
        "grid_spacing_mm":     grid_spacing,
        "grid_dims":           list(grid_dims),
        "tibia_template":      str(Path(args.tibia_template).resolve()),
        "femur_template":      str(Path(args.femur_template).resolve()),
        "tibia_target_ml_mm":  75.0,
        "tibia_target_ap_mm":  55.0,
    })
    meta.setdefault("builds", {})["soft_tissue"] = {
        "n_cases":           n_used,
        "failed":            [{"file": n, "reason": r} for n, r in fail],
        "classes":           SOFT_TISSUE_CLASSES,
        "dess_glob":         args.dess_glob,
        "label_scheme":      "PD nnU-Net 11-class (Dataset204_Knee3D)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
