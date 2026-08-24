"""
DICOM to NIfTI conversion utilities.

Provides functions for:
- Extracting laterality information from DICOM headers
- Collecting DICOM series metadata from folders
- Converting DICOM series to NIfTI format
- Reorienting NIfTI volumes to RAS orientation (canonical version)

The ``reorient_to_RAS`` function defined here is the single canonical
implementation.  All other modules should import it from this module:

    from knee_mr_seg.io_utils import reorient_to_RAS
"""

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff,
)
import pandas as pd
import pydicom
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def reorient_to_RAS(data, affine):
    """
    Reorient a 3-D volume and its affine to RAS+ orientation.

    This is the canonical implementation -- all other modules in the package
    should import from here rather than defining their own copy.

    Parameters
    ----------
    data : np.ndarray
        3-D (or higher) image array.
    affine : np.ndarray
        4x4 affine matrix mapping voxel indices to scanner (world) coordinates.

    Returns
    -------
    data_ras : np.ndarray
        Data reoriented to RAS.  Always a copy.
    affine_ras : np.ndarray
        Affine matrix corresponding to the reoriented data.
    """
    current_ornt = aff2axcodes(affine)
    target_ornt = ('R', 'A', 'S')

    if current_ornt == target_ornt:
        return data.copy(), affine.copy()

    ornt_trans = ornt_transform(
        axcodes2ornt(current_ornt), axcodes2ornt(target_ornt)
    )
    data_ras = apply_orientation(data, ornt_trans)
    transform = inv_ornt_aff(ornt_trans, data.shape)
    affine_ras = affine @ transform

    return data_ras, affine_ras


# ---------------------------------------------------------------------------
# Laterality patterns (shared across all DICOM tag lookups)
# ---------------------------------------------------------------------------

_LEFT_PATTERNS = [
    'LEFT', 'L KNEE', 'L_KNEE', 'L-KNEE',
    'LEFT KNEE', 'LEFT_KNEE', 'LEFT-KNEE',
    'LT', 'LT KNEE', 'LT_KNEE', 'LT-KNEE',
]

_RIGHT_PATTERNS = [
    'RIGHT', 'R KNEE', 'R_KNEE', 'R-KNEE',
    'RIGHT KNEE', 'RIGHT_KNEE', 'RIGHT-KNEE',
    'RT', 'RT KNEE', 'RT_KNEE', 'RT-KNEE',
]


def _match_laterality(text, left_patterns=_LEFT_PATTERNS, right_patterns=_RIGHT_PATTERNS):
    """Return 'Lt', 'Rt', or None based on pattern matching in *text*."""
    upper = text.upper()
    for pat in left_patterns:
        if pat in upper:
            return 'Lt'
    for pat in right_patterns:
        if pat in upper:
            return 'Rt'
    return None


# ---------------------------------------------------------------------------
# DICOM laterality extraction
# ---------------------------------------------------------------------------

def extract_laterality_from_dicom_detailed(dicom_path):
    """
    Extract laterality information from a DICOM file using multiple header tags.

    The function inspects tags in priority order:
    1. Laterality (0020,0060)
    2. RequestedProcedureDescription (0032,1060)
    3. SeriesDescription
    4. ProtocolName
    5. StudyDescription

    Parameters
    ----------
    dicom_path : str or Path
        Path to a single DICOM file.

    Returns
    -------
    dict
        Keys: ``laterality`` ('Lt', 'Rt', or 'Unknown'),
        ``source`` (tag name that determined laterality),
        ``details`` (raw tag values read).
    """
    result = {
        'laterality': 'Unknown',
        'source': 'None',
        'details': {},
    }

    try:
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    except Exception as e:
        result['details']['error'] = str(e)
        return result

    # 1. Laterality tag
    laterality = getattr(ds, 'Laterality', None)
    if laterality:
        result['details']['Laterality_Tag'] = laterality
        upper = laterality.upper()
        if upper in ('L', 'LEFT', 'LT'):
            result['laterality'] = 'Lt'
            result['source'] = 'Laterality_Tag'
            return result
        elif upper in ('R', 'RIGHT', 'RT'):
            result['laterality'] = 'Rt'
            result['source'] = 'Laterality_Tag'
            return result

    # 2-5. Text-based tags in priority order
    _tag_priority = [
        ('RequestedProcedureDescription', None),
        ('SeriesDescription', ['KNEE', 'TSE', 'FSE', 'PD', 'T2', 'T1']),
        ('ProtocolName', None),
        ('StudyDescription', None),
    ]

    for tag_name, required_keywords in _tag_priority:
        value = getattr(ds, tag_name, None)
        if not value:
            continue
        result['details'][tag_name] = value

        # Some tags require a knee-related keyword to be present
        if required_keywords:
            if not any(kw in value.upper() for kw in required_keywords):
                continue

        match = _match_laterality(value)
        if match:
            result['laterality'] = match
            result['source'] = tag_name
            return result

    return result


# ---------------------------------------------------------------------------
# DICOM folder scanning
# ---------------------------------------------------------------------------

def collect_dicom_info_from_folder(folder_path):
    """
    Collect metadata from all DICOM files in *folder_path*.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing ``.dcm`` files.

    Returns
    -------
    pd.DataFrame or None
        One row per DICOM file with columns for common tags,
        plus ``SliceIndexInSeries`` and ``TotalSlicesInSeries``.
        Returns ``None`` if no valid DICOM files are found.
    """
    rows = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.dcm'):
            continue
        path = os.path.join(folder_path, fname)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            rows.append({
                'filepath': path,
                'SeriesInstanceUID': getattr(ds, 'SeriesInstanceUID', None),
                'SeriesNumber': getattr(ds, 'SeriesNumber', None),
                'InstanceNumber': getattr(ds, 'InstanceNumber', None),
                'SliceLocation': getattr(ds, 'SliceLocation', None),
                'StudyDescription': getattr(ds, 'StudyDescription', None),
                'SeriesDescription': getattr(ds, 'SeriesDescription', None),
                'ProtocolName': getattr(ds, 'ProtocolName', None),
                'StudyDate': getattr(ds, 'StudyDate', None),
            })
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(['SeriesInstanceUID', 'InstanceNumber'])
    df['SliceIndexInSeries'] = df.groupby('SeriesInstanceUID').cumcount() + 1
    df['TotalSlicesInSeries'] = df.groupby('SeriesInstanceUID')['filepath'].transform('count')
    return df


# ---------------------------------------------------------------------------
# Series-level NIfTI conversion
# ---------------------------------------------------------------------------

def convert_series_to_nifti(series_group, output_path, laterality='Unknown', dry_run=False):
    """
    Convert a single DICOM series to a NIfTI file.

    Parameters
    ----------
    series_group : pd.DataFrame
        Rows belonging to one SeriesInstanceUID (must contain 'filepath'
        and 'InstanceNumber' columns).
    output_path : str or Path
        Destination ``.nii.gz`` path.
    laterality : str
        Laterality label for logging ('Lt', 'Rt', or 'Unknown').
    dry_run : bool
        If True, skip actual conversion and only print what would happen.

    Returns
    -------
    bool
        True on success (or dry-run), False on failure.
    """
    try:
        sorted_files = series_group.sort_values('InstanceNumber')['filepath'].tolist()

        if dry_run:
            print(f"  [DRY RUN] Would convert {len(sorted_files)} files -> {Path(output_path).name}")
            return True

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_path))
        return True

    except Exception as e:
        print(f"  [ERROR] Conversion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Patient / batch conversion
# ---------------------------------------------------------------------------

def process_patient_folder(patient_path, output_dir, target_keywords=None,
                           folder_structure="auto", dry_run=False):
    """
    Walk a patient folder, find DICOM series, and convert each to NIfTI.

    Parameters
    ----------
    patient_path : str or Path
        Root folder for one patient (may contain sub-folders).
    output_dir : str or Path
        Where to write the resulting ``.nii.gz`` files.
    target_keywords : list[str] or None
        If given, only convert series whose description contains **all**
        of these keywords (case-insensitive).
    folder_structure : str
        Folder-structure hint (currently unused, reserved for future use).
    dry_run : bool
        If True, only print what would be done.

    Returns
    -------
    tuple[int, int]
        (number of successful conversions, number of errors).
    """
    patient_name = Path(patient_path).name
    converted_count = 0
    error_count = 0

    print(f"[INFO] Processing patient: {patient_name}")

    for root, _dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue

        print(f"  Found DICOM folder: {Path(root).name}")

        df = collect_dicom_info_from_folder(root)
        if df is None:
            print(f"  [WARN] No readable DICOM files")
            continue

        for series_uid, series_group in df.groupby('SeriesInstanceUID'):
            if series_uid is None:
                continue

            first_row = series_group.iloc[0]
            series_desc = str(first_row.get('SeriesDescription', 'Unknown'))
            study_date = str(first_row.get('StudyDate', 'Unknown'))

            laterality_result = extract_laterality_from_dicom_detailed(first_row['filepath'])
            laterality = laterality_result['laterality']

            print(f"    Series: {series_desc} (Laterality: {laterality})")

            # Keyword filtering
            if target_keywords:
                series_upper = series_desc.upper()
                if not all(kw.upper() in series_upper for kw in target_keywords):
                    print(f"      Skipped (keyword mismatch)")
                    continue

            # Build safe output filename
            safe_desc = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in series_desc
            ).strip()
            safe_date = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in study_date
            ).strip()

            if laterality != 'Unknown':
                output_filename = f"{patient_name}_{laterality}_{safe_desc}_{safe_date}.nii.gz"
            else:
                output_filename = f"{patient_name}_{safe_desc}_{safe_date}.nii.gz"

            output_path = Path(output_dir) / output_filename

            if convert_series_to_nifti(series_group, output_path, laterality, dry_run):
                action = "Would convert" if dry_run else "Converted"
                print(f"      {action}: {output_filename}")
                converted_count += 1
            else:
                error_count += 1

    return converted_count, error_count


def integrated_dicom_to_nifti(input_dir, output_dir, target_keywords=None,
                              folder_structure="auto", dry_run=False):
    """
    Batch-convert DICOM folders (one per patient) to NIfTI.

    Parameters
    ----------
    input_dir : str or Path
        Top-level directory whose immediate sub-directories are patient folders.
    output_dir : str or Path
        Destination directory for ``.nii.gz`` files.
    target_keywords : list[str] or None
        If given, only series matching **all** keywords are converted.
        Defaults to ``['PD']``.
    folder_structure : str
        Folder-structure hint (reserved).
    dry_run : bool
        If True, no files are written.
    """
    if target_keywords is None:
        target_keywords = ['PD']

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if dry_run:
        print("=== DRY RUN ===")
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Keywords: {target_keywords}")
    print("-" * 50)

    total_converted = 0
    total_errors = 0

    for item in sorted(input_path.iterdir()):
        if not item.is_dir():
            continue
        conv, errs = process_patient_folder(
            str(item), str(output_path), target_keywords, folder_structure, dry_run
        )
        total_converted += conv
        total_errors += errs

    print("-" * 50)
    label = "Would convert" if dry_run else "Converted"
    print(f"{label}: {total_converted} series")
    print(f"Errors:   {total_errors}")


def test_laterality_extraction(dicom_path):
    """
    Print laterality extraction results for a single DICOM file (debug helper).

    Parameters
    ----------
    dicom_path : str or Path
        Path to the DICOM file to test.
    """
    print("=== Laterality extraction test ===")
    if os.path.exists(dicom_path):
        result = extract_laterality_from_dicom_detailed(dicom_path)
        print(f"File:        {dicom_path}")
        print(f"Laterality:  {result['laterality']}")
        print(f"Source:      {result['source']}")
        print(f"Details:     {result['details']}")
    else:
        print(f"File not found: {dicom_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for DICOM-to-NIfTI conversion."""
    parser = argparse.ArgumentParser(
        description='Convert DICOM series to NIfTI, with automatic laterality detection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default conversion (series matching 'PD' keyword)
  python -m knee_mr_seg.io_utils -i /data/patients -o /data/nifti

  # Custom keyword filter
  python -m knee_mr_seg.io_utils -i /data/patients -o /data/nifti -k PD SAG

  # Dry run (preview only)
  python -m knee_mr_seg.io_utils -i /data/patients -o /data/nifti --dry-run

  # Test laterality extraction on a single file
  python -m knee_mr_seg.io_utils --test /data/example.dcm
        """,
    )
    parser.add_argument('--input', '-i',
                        help='Input directory (patient sub-folders with DICOM files)')
    parser.add_argument('--output', '-o',
                        help='Output directory for .nii.gz files')
    parser.add_argument('--keywords', '-k', nargs='*', default=['PD', 'SAG'],
                        help='Series description keywords to filter on (default: PD SAG)')
    parser.add_argument('--folder-structure', '-f',
                        choices=['auto', 'patient_visit', 'flat'], default='auto',
                        help='Folder structure hint (default: auto)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Preview without converting')
    parser.add_argument('--test', '-t',
                        help='Test laterality extraction on a single DICOM file')

    args = parser.parse_args()

    if args.test:
        test_laterality_extraction(args.test)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required (unless using --test)")

    if not os.path.isdir(args.input):
        parser.error(f"Input path is not a directory: {args.input}")

    integrated_dicom_to_nifti(
        input_dir=args.input,
        output_dir=args.output,
        target_keywords=args.keywords,
        folder_structure=args.folder_structure,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
