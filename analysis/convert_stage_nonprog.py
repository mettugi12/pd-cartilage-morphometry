"""v8 non-progressor arm — DICOM -> NIfTI conversion + Modal staging.

Converts pulled series (nonprog_pull/IW_{00m,48m}/{pid}_{side}/*.dcm) to NIfTI
staged flat for the Modal batch segmenter:
    E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull/staged_{00m,48m}/{pid}_{side}.nii.gz

Safety: verifies each converted volume is SAG by NIfTI affine (through-plane =
medial-lateral) and has a plausible IW slice count (25-50); failures are reported
and excluded from staging.

CLI: --dry_run, --n_run N, --tp {00m,48m,both}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np
import SimpleITK as sitk
import nibabel as nib

ROOT = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull")


def detect_plane(nii_path) -> str | None:
    nii = nib.load(str(nii_path))
    spacings = np.linalg.norm(nii.affine[:3, :3], axis=0)
    codes = nib.orientations.aff2axcodes(nii.affine)
    code = codes[int(np.argmax(spacings))]
    if code in ("R", "L"):
        return "SAG"
    if code in ("S", "I"):
        return "AX"
    if code in ("A", "P"):
        return "COR"
    return None


def dicom_to_nifti(dicom_folder: Path, nifti_path: Path) -> bool:
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_folder))
        if not series_ids:
            return False
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_folder), series_ids[0])
        if not files:
            return False
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        img = reader.Execute()
        nifti_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(nifti_path))
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--tp", choices=["00m", "48m", "both"], default="both")
    args = ap.parse_args()

    tps = ["00m", "48m"] if args.tp == "both" else [args.tp]
    for tp in tps:
        src_root = ROOT / f"IW_{tp}"
        dst_root = ROOT / f"staged_{tp}"
        folders = sorted(d for d in src_root.iterdir() if d.is_dir())
        if args.n_run is not None:
            folders = folders[: args.n_run]
        print(f"[{tp}] {len(folders)} series -> {dst_root}")

        ok = skip = fail = notsag = 0
        t0 = time()
        for i, d in enumerate(folders, 1):
            out = dst_root / f"{d.name}.nii.gz"
            if out.exists() and out.stat().st_size > 0:
                skip += 1
                continue
            if args.dry_run:
                print(f"  DRY {d.name}")
                continue
            if not dicom_to_nifti(d, out):
                fail += 1
                print(f"  [{tp}] FAIL convert {d.name}")
                continue
            plane = detect_plane(out)
            n_slices = nib.load(str(out)).shape[np.argmax(
                np.linalg.norm(nib.load(str(out)).affine[:3, :3], axis=0))]
            if plane != "SAG" or not (25 <= n_slices <= 50):
                notsag += 1
                print(f"  [{tp}] QC-FAIL {d.name}: plane={plane} slices={n_slices} — unstaged")
                out.rename(out.with_name(d.name + ".QCFAIL"))  # keep out of *.nii.gz upload glob
                continue
            ok += 1
            if i % 25 == 0:
                print(f"  [{tp}] {i}/{len(folders)} ok={ok} skip={skip} fail={fail} "
                      f"({i/max(1e-6,time()-t0):.1f}/s)")
        print(f"[{tp}] DONE ok={ok} skip={skip} fail={fail} qcfail={notsag}")


if __name__ == "__main__":
    main()
