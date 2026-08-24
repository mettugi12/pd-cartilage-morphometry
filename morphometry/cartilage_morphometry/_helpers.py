"""Vendored helpers from the legacy MOAKS_seg pipeline.

Originally lived in `Knee_MR/_archive/legacy_code/MOAKS_seg/standardization/new_standard.py`.
Vendored here so the package is self-contained.
"""
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
from skimage.measure import find_contours


def create_pv_mesh(vertices, face_array):
    """Build a pyvista PolyData from marching-cubes output."""
    if len(face_array) == 0:
        return None
    faces_pv = np.hstack(
        [np.full((face_array.shape[0], 1), 3), face_array]
    ).astype(np.int64)
    return pv.PolyData(vertices, faces_pv.flatten())


def _force_close_contour(contour, shape):
    ny, nz = shape
    is_open = not np.allclose(contour[0], contour[-1])
    touches_edge = (
        (contour[:, 0] <= 1).any() or (contour[:, 0] >= ny - 2).any()
        or (contour[:, 1] <= 1).any() or (contour[:, 1] >= nz - 2).any()
    )
    if is_open and touches_edge:
        bridge = np.linspace(contour[0], contour[-1], 10)
        return np.vstack([contour, bridge])
    return contour


def _align_contour_start(c1, c2):
    """Rotate c2 so its starting point is closest to c1[0]."""
    dists = np.linalg.norm(c2 - c1[0], axis=1)
    return np.roll(c2, -int(np.argmin(dists)), axis=0)


def interpolate_segmentation_filled(seg, spacing, bone_label, cartilage_label,
                                    slices_between=5):
    """Per-slice contour interpolation densifier for sparse PD-style segs.

    Walks adjacent slices along axis 2 (ML), aligns their largest contour for
    `bone_label` / `cartilage_label`, linearly interpolates between them, and
    rasterises the interpolated contours back. Used as a fallback when a
    DESS-modality input has slice spacing > 1.5 mm.
    """
    ny, nz, nx = seg.shape
    new_nx = nx + (nx - 1) * slices_between
    new_spacing = (spacing[0], spacing[1], spacing[2] / (slices_between + 1))
    seg_interp = np.zeros((ny, nz, new_nx), dtype=np.uint8)

    for label in (bone_label, cartilage_label):
        for x in range(nx - 1):
            slc1 = (seg[:, :, x] == label).astype(np.uint8)
            slc2 = (seg[:, :, x + 1] == label).astype(np.uint8)
            if not slc1.any() or not slc2.any():
                continue
            contours1 = find_contours(slc1, level=0.5)
            contours2 = find_contours(slc2, level=0.5)
            if not contours1 or not contours2:
                continue
            c1 = _force_close_contour(max(contours1, key=len), slc1.shape)
            c2 = _force_close_contour(max(contours2, key=len), slc2.shape)
            c2 = _align_contour_start(c1, c2)
            n_pts = min(len(c1), len(c2))
            c1 = c1[np.linspace(0, len(c1) - 1, n_pts).astype(int)]
            c2 = c2[np.linspace(0, len(c2) - 1, n_pts).astype(int)]
            for i in range(slices_between + 2):
                alpha = i / (slices_between + 1)
                interp = (1 - alpha) * c1 + alpha * c2
                rr, cc = polygon(interp[:, 0], interp[:, 1], shape=(ny, nz))
                x_new = x * (slices_between + 1) + i
                mask = np.zeros((ny, nz), dtype=bool)
                mask[rr, cc] = True
                mask = binary_fill_holes(mask)
                seg_interp[:, :, x_new][mask] = label

    return seg_interp, new_spacing
