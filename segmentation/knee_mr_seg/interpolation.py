"""
Mask interpolation for sparse sagittal slices.

Interpolates segmentation masks between slices to create smoother 3-D surfaces.
This is superior to post-hoc mesh smoothing because it creates actual intermediate
data rather than just moving mesh vertices.

For PD images with ~40 sagittal slices (2--3 mm spacing), this can upsample to
~160 slices (matching DESS resolution of ~0.7 mm spacing).

Three methods are provided, in increasing order of quality:

- **nearest** -- fast, slight stepping at label boundaries
- **linear** -- linear interpolation per label + threshold at 0.5
- **morphological** -- signed-distance-field interpolation (gold standard)
"""

import numpy as np
import nibabel as nib
from scipy import ndimage

from knee_mr_seg.io_utils import reorient_to_RAS


# ---------------------------------------------------------------------------
# Low-level interpolation kernels
# ---------------------------------------------------------------------------

def interpolate_mask_nearest(mask, target_slices, axis=0):
    """
    Interpolate a label mask using nearest-neighbour resampling.

    Parameters
    ----------
    mask : np.ndarray
        3-D integer label array.
    target_slices : int
        Desired number of slices along *axis*.
    axis : int
        Axis to interpolate along (0, 1, or 2).

    Returns
    -------
    np.ndarray
        Resampled label array.
    """
    current_slices = mask.shape[axis]
    if current_slices == target_slices:
        return mask

    zoom_factors = [1.0, 1.0, 1.0]
    zoom_factors[axis] = target_slices / current_slices

    return ndimage.zoom(mask, zoom_factors, order=0)


def interpolate_mask_linear_then_threshold(mask, target_slices, axis=0, labels=None):
    """
    Interpolate each label with linear interpolation, then threshold at 0.5.

    For each label a binary mask is created, linearly interpolated, and
    re-binarised.  This produces smooth transitions between slices.

    Parameters
    ----------
    mask : np.ndarray
        3-D integer label array.
    target_slices : int
        Desired number of slices along *axis*.
    axis : int
        Axis to interpolate along.
    labels : array-like or None
        Label values to process.  Auto-detected (excluding 0) when *None*.

    Returns
    -------
    np.ndarray
        Resampled label array with same dtype as *mask*.
    """
    current_slices = mask.shape[axis]
    if current_slices == target_slices:
        return mask.copy()

    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]

    zoom_factors = [1.0, 1.0, 1.0]
    zoom_factors[axis] = target_slices / current_slices

    output_shape = list(mask.shape)
    output_shape[axis] = target_slices
    result = np.zeros(output_shape, dtype=mask.dtype)

    print(f"  Interpolating {len(labels)} labels along axis {axis}")
    print(f"  Zoom factor: {zoom_factors[axis]:.2f}x ({current_slices} -> {target_slices} slices)")

    for label in labels:
        binary = (mask == label).astype(np.float32)
        interpolated = ndimage.zoom(binary, zoom_factors, order=1)
        binary_result = interpolated > 0.5
        result[binary_result & (result == 0)] = label

    return result


def interpolate_mask_morphological(mask, target_slices, axis=0, labels=None):
    """
    Morphological (signed-distance-field) label interpolation.

    For each pair of adjacent slices the signed distance transforms are
    linearly blended and thresholded.  This is the gold-standard method,
    producing the smoothest boundaries.

    Parameters
    ----------
    mask : np.ndarray
        3-D integer label array.
    target_slices : int
        Desired number of slices along *axis*.
    axis : int
        Axis to interpolate along.
    labels : array-like or None
        Label values to process.  Auto-detected (excluding 0) when *None*.

    Returns
    -------
    np.ndarray
        Resampled label array with same dtype as *mask*.
    """
    current_slices = mask.shape[axis]
    if current_slices == target_slices:
        return mask.copy()

    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]

    scale_factor = target_slices / current_slices

    output_shape = list(mask.shape)
    output_shape[axis] = target_slices
    result = np.zeros(output_shape, dtype=mask.dtype)

    print(f"  Morphological interpolation along axis {axis}")
    print(f"  Scale factor: {scale_factor:.2f}x ({current_slices} -> {target_slices} slices)")

    for label in labels:
        binary = (mask == label).astype(np.float32)

        for i in range(current_slices - 1):
            start_idx = int(round(i * scale_factor))
            end_idx = int(round((i + 1) * scale_factor))

            if axis == 0:
                slice1 = binary[i, :, :]
                slice2 = binary[i + 1, :, :]
            elif axis == 1:
                slice1 = binary[:, i, :]
                slice2 = binary[:, i + 1, :]
            else:
                slice1 = binary[:, :, i]
                slice2 = binary[:, :, i + 1]

            if np.any(slice1):
                dist1 = ndimage.distance_transform_edt(slice1) - ndimage.distance_transform_edt(1 - slice1)
            else:
                dist1 = -ndimage.distance_transform_edt(np.ones_like(slice1))

            if np.any(slice2):
                dist2 = ndimage.distance_transform_edt(slice2) - ndimage.distance_transform_edt(1 - slice2)
            else:
                dist2 = -ndimage.distance_transform_edt(np.ones_like(slice2))

            num_intermediate = end_idx - start_idx
            for j in range(num_intermediate):
                t = j / num_intermediate
                dist_interp = (1 - t) * dist1 + t * dist2
                interp_slice = (dist_interp > 0).astype(mask.dtype) * label

                out_idx = start_idx + j
                if axis == 0:
                    result[out_idx, :, :] = np.maximum(result[out_idx, :, :], interp_slice)
                elif axis == 1:
                    result[:, out_idx, :] = np.maximum(result[:, out_idx, :], interp_slice)
                else:
                    result[:, :, out_idx] = np.maximum(result[:, :, out_idx], interp_slice)

        # Handle last slice
        last_out_idx = int(round((current_slices - 1) * scale_factor))
        if last_out_idx < target_slices:
            if axis == 0:
                result[last_out_idx:, :, :] = np.maximum(
                    result[last_out_idx:, :, :],
                    np.broadcast_to(binary[-1:, :, :] * label, result[last_out_idx:, :, :].shape),
                )
            elif axis == 1:
                result[:, last_out_idx:, :] = np.maximum(
                    result[:, last_out_idx:, :],
                    np.broadcast_to(binary[:, -1:, :] * label, result[:, last_out_idx:, :].shape),
                )
            else:
                result[:, :, last_out_idx:] = np.maximum(
                    result[:, :, last_out_idx:],
                    np.broadcast_to(binary[:, :, -1:] * label, result[:, :, last_out_idx:].shape),
                )

    return result


# ---------------------------------------------------------------------------
# High-level NIfTI-to-NIfTI interpolation
# ---------------------------------------------------------------------------

def interpolate_segmentation(input_path, output_path, target_spacing=None,
                             target_slices=None, method='morphological',
                             axis=None, labels=None):
    """
    Interpolate a segmentation NIfTI file to create smoother surfaces.

    Parameters
    ----------
    input_path : str or Path
        Path to input NIfTI segmentation file.
    output_path : str or Path
        Path to save the interpolated segmentation.
    target_spacing : float, optional
        Target voxel spacing in mm along the interpolation axis
        (e.g. 0.7 to match DESS resolution).
    target_slices : int, optional
        Target number of slices along the interpolation axis.
        Exactly one of *target_spacing* or *target_slices* must be given.
    method : {'nearest', 'linear', 'morphological'}
        Interpolation strategy.
    axis : int or None
        Axis to interpolate along (0=R, 1=A, 2=S in RAS).
        Auto-detects the coarsest axis when *None*.
    labels : list[int] or None
        Label values to interpolate.  Auto-detected when *None*.

    Returns
    -------
    interpolated_data : np.ndarray
        Interpolated segmentation array.
    new_affine : np.ndarray
        Updated 4x4 affine matrix with new spacing.
    """
    print(f"[INFO] Loading: {input_path}")
    nii = nib.load(str(input_path))
    data_orig = nii.get_fdata().astype(np.int16)
    affine_orig = nii.affine

    print("[INFO] Reorienting to RAS...")
    data, affine = reorient_to_RAS(data_orig, affine_orig)
    print(f"  Shape: {data.shape}")

    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    print(f"  Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm")

    if axis is None:
        axis = int(np.argmax(spacing))
        print(f"  Auto-detected interpolation axis: {axis} ({'RAS'[axis]})")

    current_spacing = spacing[axis]
    current_slices = data.shape[axis]

    if target_spacing is not None:
        target_slices = int(round(current_slices * current_spacing / target_spacing))
        print(f"  Target spacing: {target_spacing:.3f} mm -> {target_slices} slices")
    elif target_slices is not None:
        target_spacing = current_slices * current_spacing / target_slices
        print(f"  Target slices: {target_slices} -> {target_spacing:.3f} mm spacing")
    else:
        raise ValueError("Either target_spacing or target_slices must be provided")

    if labels is None:
        labels = np.unique(data)
        labels = labels[labels != 0]
        print(f"  Found labels: {labels.tolist()}")

    print(f"\n[INFO] Interpolating with method='{method}'...")
    if method == 'nearest':
        result = interpolate_mask_nearest(data, target_slices, axis)
    elif method == 'linear':
        result = interpolate_mask_linear_then_threshold(data, target_slices, axis, labels)
    elif method == 'morphological':
        result = interpolate_mask_morphological(data, target_slices, axis, labels)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[OK] Interpolation complete.  Output shape: {result.shape}")

    new_affine = affine.copy()
    scale_factor = current_slices / target_slices
    new_affine[:3, axis] *= scale_factor

    new_spacing = np.sqrt(np.sum(new_affine[:3, :3] ** 2, axis=0))
    print(f"  New spacing: [{new_spacing[0]:.3f}, {new_spacing[1]:.3f}, {new_spacing[2]:.3f}] mm")

    if output_path:
        print(f"[INFO] Saving to: {output_path}")
        out_nii = nib.Nifti1Image(result.astype(np.int16), new_affine)
        nib.save(out_nii, str(output_path))
        print("[OK] Saved successfully")

    return result, new_affine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for mask interpolation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interpolate segmentation masks between slices for smoother 3-D surfaces",
    )
    parser.add_argument("input", help="Input NIfTI segmentation file")
    parser.add_argument("output", help="Output NIfTI file path")
    parser.add_argument("--target-spacing", type=float, default=None,
                        help="Target spacing in mm (e.g. 0.7 for DESS-like resolution)")
    parser.add_argument("--target-slices", type=int, default=None,
                        help="Target number of slices")
    parser.add_argument("--method", choices=['nearest', 'linear', 'morphological'],
                        default='morphological',
                        help="Interpolation method (default: morphological)")
    parser.add_argument("--axis", type=int, default=None,
                        help="Axis to interpolate (0=R, 1=A, 2=S). Auto-detects if omitted.")
    parser.add_argument("--labels", type=int, nargs='+', default=None,
                        help="Label values to interpolate (auto-detects if omitted)")

    args = parser.parse_args()

    if args.target_spacing is None and args.target_slices is None:
        parser.error("Either --target-spacing or --target-slices must be provided")

    interpolate_segmentation(
        args.input,
        args.output,
        target_spacing=args.target_spacing,
        target_slices=args.target_slices,
        method=args.method,
        axis=args.axis,
        labels=args.labels,
    )


if __name__ == "__main__":
    main()
