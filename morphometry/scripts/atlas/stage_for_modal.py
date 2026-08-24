"""
Stage PD test set images for Modal upload.

Strips the trailing `_0000` (nnU-Net channel-id) suffix from the source
imagesTs filenames, since the Modal app at scripts/modal_batch_inference.py
in knee-mr-seg internally re-adds `_0000` before calling nnU-Net.

Source: E:/KneeMR/eval/PD_seg_test/imagesTs/{stem}_0000.nii.gz
Dest:   E:/KneeMR/eval/PD_seg_test/staged/{stem}.nii.gz   (flat)
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

DEFAULT_SRC = Path("E:/KneeMR/eval/PD_seg_test/imagesTs")
DEFAULT_DST = Path("E:/KneeMR/eval/PD_seg_test/staged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run",  type=int, default=None)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.nii.gz"))
    if args.n_run is not None:
        files = files[: args.n_run]

    n_done = n_skip = 0
    for src in files:
        stem = src.name.replace(".nii.gz", "")
        if stem.endswith("_0000"):
            stem = stem[:-5]
        dst = args.dst / f"{stem}.nii.gz"
        if dst.exists():
            n_skip += 1
            continue
        if args.dry_run:
            print(f"  [dry] {src.name} -> {dst.name}")
        else:
            shutil.copy2(src, dst)
        n_done += 1

    print(f"[stage] copied {n_done}, skipped {n_skip}")
    print(f"[stage] dst = {args.dst}")


if __name__ == "__main__":
    main()
