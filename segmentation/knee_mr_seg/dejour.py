"""
Dejour trochlear dysplasia classification from knee MRI.

Implements automated measurement of:
- Sulcus angle (axial)
- Lateral trochlear inclination (LTI) angle (axial)
- Sagittal supratrochlear bump height

These are combined into a Dejour classification (Type 0--3).

Required inputs are an MR image and a corresponding multi-label
segmentation mask containing femur bone and cartilage labels.
"""

import numpy as np
import cv2
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import zoom, rotate
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def mr_to_uint8_robust(mr_array, percentile_clip=99.5, lower_percentile=0.5):
    """
    Normalise an MR intensity array to uint8 using robust percentile clipping.

    Parameters
    ----------
    mr_array : np.ndarray
        Raw MR intensity values.
    percentile_clip : float
        Upper percentile for windowing.
    lower_percentile : float
        Lower percentile for windowing.

    Returns
    -------
    np.ndarray
        uint8 image in [0, 255].
    """
    non_zero = mr_array[mr_array > 0]
    if len(non_zero) > 0:
        lower_bound = np.percentile(non_zero, lower_percentile)
        upper_bound = np.percentile(non_zero, percentile_clip)
    else:
        lower_bound = np.percentile(mr_array, lower_percentile)
        upper_bound = np.percentile(mr_array, percentile_clip)

    clipped = np.clip(mr_array, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    return (normalized * 255).astype(np.uint8)


def mr_to_uint8_adaptive(mr_array):
    """
    Adaptive uint8 normalisation that selects percentiles based on variance.

    Parameters
    ----------
    mr_array : np.ndarray
        Raw MR intensity values.

    Returns
    -------
    np.ndarray
        uint8 image in [0, 255].
    """
    non_zero = mr_array[mr_array > 0]
    mean_val = np.mean(non_zero)
    std_val = np.std(non_zero)
    cv = std_val / mean_val

    if cv > 1.5:
        return mr_to_uint8_robust(mr_array, 98, 2)
    elif cv > 1.0:
        return mr_to_uint8_robust(mr_array, 99, 1)
    else:
        return mr_to_uint8_robust(mr_array, 99.5, 0.5)


# ---------------------------------------------------------------------------
# Image annotation helpers
# ---------------------------------------------------------------------------

def add_ml_labels_to_rgb_image(ax_img, side):
    """
    Overlay 'M' (medial) and 'L' (lateral) text labels on an RGB axial image.

    Parameters
    ----------
    ax_img : np.ndarray
        Axial RGB image (H, W, 3).
    side : str
        Knee laterality: 'Rt' or 'Lt'.

    Returns
    -------
    np.ndarray
        Copy of the image with text labels added.
    """
    labeled_img = ax_img.copy()
    if labeled_img.dtype != np.uint8:
        lo, hi = labeled_img.min(), labeled_img.max()
        if hi > lo:
            labeled_img = ((labeled_img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            labeled_img = np.zeros_like(labeled_img, dtype=np.uint8)

    height, width = labeled_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    text_y = height // 2
    left_x = int(width * 0.05)
    right_x = int(width * 0.95)

    if side.upper() in ('RT', 'RIGHT', 'R'):
        medial_pos, lateral_pos = (left_x, text_y), (right_x, text_y)
    else:
        medial_pos, lateral_pos = (right_x, text_y), (left_x, text_y)

    cv2.putText(labeled_img, 'M', medial_pos, font, font_scale, text_color, thickness)
    cv2.putText(labeled_img, 'L', lateral_pos, font, font_scale, text_color, thickness)
    return labeled_img


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_mri_to_isotropic(mri_array, original_spacing, target_spacing=None,
                              is_mask=False):
    """
    Resample a 3-D array to isotropic voxel spacing.

    Parameters
    ----------
    mri_array : np.ndarray
        3-D volume.
    original_spacing : tuple[float, float, float]
        Voxel size in mm for each axis.
    target_spacing : float or None
        Desired isotropic spacing.  Defaults to the finest original spacing.
    is_mask : bool
        If True, use nearest-neighbour interpolation to preserve labels.

    Returns
    -------
    resampled : np.ndarray
    new_shape : tuple
    actual_spacing : float
    """
    if target_spacing is None:
        target_spacing = min(original_spacing)

    zoom_factors = [original_spacing[i] / target_spacing for i in range(3)]

    if is_mask:
        resampled = zoom(mri_array, zoom_factors, order=0, prefilter=False)
    else:
        resampled = zoom(mri_array, zoom_factors, order=1, prefilter=True)

    if is_mask:
        original_labels = np.unique(mri_array)
        resampled_labels = np.unique(resampled)
        if not np.array_equal(original_labels, resampled_labels):
            print("[WARN] Label values changed during resampling")

    return resampled, resampled.shape, target_spacing


# ---------------------------------------------------------------------------
# Morphological helpers
# ---------------------------------------------------------------------------

def fill_and_close_mask(mask, kernel_size=5):
    """
    Fill holes and morphologically close a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input binary (or 0/1 integer) mask.
    kernel_size : int
        Diameter of the elliptical structuring element.

    Returns
    -------
    np.ndarray (bool)
        Cleaned binary mask.
    """
    filled = ndimage.binary_fill_holes(mask > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def remove_small_islands_multilabel_scipy(mask, min_size=50):
    """
    Remove small connected components from a multi-label mask.

    Parameters
    ----------
    mask : np.ndarray
        Integer label array (0 = background).
    min_size : int
        Minimum component area in voxels.

    Returns
    -------
    np.ndarray
        Cleaned label array.
    """
    cleaned = np.zeros_like(mask)
    for label_val in np.unique(mask):
        if label_val == 0:
            continue
        binary = (mask == label_val)
        labeled, _ = ndimage.label(binary)
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_size
        keep[0] = False
        cleaned[keep[labeled]] = label_val
    return cleaned


# ---------------------------------------------------------------------------
# Contour / geometry utilities
# ---------------------------------------------------------------------------

def DstBtw2Points(p1, p2):
    """
    Euclidean distance between two 2-D points.

    Parameters
    ----------
    p1, p2 : array-like
        (x, y) coordinates.

    Returns
    -------
    float
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cnt_hull(y_mask):
    """
    Find the largest contour and compute its convex hull.

    Parameters
    ----------
    y_mask : np.ndarray
        Binary mask (uint8 or bool).

    Returns
    -------
    cnt : np.ndarray
        Largest contour in OpenCV format.
    hull_img : np.ndarray
        BGR visualisation image of the mask.
    """
    y_mask = y_mask.astype(np.uint8)
    if len(y_mask.shape) == 3:
        y_mask = cv2.cvtColor(y_mask, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=len)

    hull_img = cv2.cvtColor(255 * y_mask, cv2.COLOR_GRAY2BGR)
    return cnt, hull_img


def clean_cnt(cnt):
    """
    Clean a contour by fixing self-intersections via Shapely buffer(0).

    Parameters
    ----------
    cnt : np.ndarray
        OpenCV contour (N, 1, 2).

    Returns
    -------
    np.ndarray
        Cleaned contour in OpenCV format.
    """
    cnt_points = cnt.squeeze()
    if cnt_points.ndim == 1:
        cnt_points = np.expand_dims(cnt_points, axis=0)

    poly = Polygon(cnt_points)
    clean_poly = poly.buffer(0)

    if clean_poly.geom_type == 'MultiPolygon':
        clean_poly = max(clean_poly.geoms, key=lambda a: a.area)

    cleaned_coords = np.array(clean_poly.exterior.coords, dtype=np.int32)
    return cleaned_coords.reshape((-1, 1, 2))


def savgol_smooth_contour(contour, window=15, poly=9):
    """
    Smooth an OpenCV contour with a Savitzky-Golay filter.

    Parameters
    ----------
    contour : np.ndarray
        OpenCV contour (N, 1, 2).
    window : int
        Savgol window length (must be odd).
    poly : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed contour in OpenCV format.
    """
    points = contour.squeeze()
    smooth_x = savgol_filter(points[:, 0], window, poly)
    smooth_y = savgol_filter(points[:, 1], window, poly)
    smoothed = np.column_stack((smooth_x, smooth_y)).astype(np.int32)
    return np.expand_dims(smoothed, axis=1)


# ---------------------------------------------------------------------------
# Anterior contour and M-shape detection
# ---------------------------------------------------------------------------

def extract_anterior_contour_horizontal(binary_mask, plot_results=True):
    """
    Extract the anterior (top-edge) contour of a horizontally oriented bone mask.

    For each x-coordinate, the minimum y-value on the contour is taken.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary 2-D mask.
    plot_results : bool
        If True, show a matplotlib figure.

    Returns
    -------
    np.ndarray
        (N, 2) array of (x, y) points along the anterior contour.
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No contours found")

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour[:, 0, :]

    x_to_y = defaultdict(list)
    for x, y in contour_points:
        x_to_y[x].append(y)

    anterior_points = np.array([[x, min(ys)] for x, ys in sorted(x_to_y.items())])

    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(binary_mask, cmap='viridis')
        ax1.plot(contour_points[:, 0], contour_points[:, 1], 'b-', lw=1, alpha=0.7, label='Full contour')
        ax1.plot(anterior_points[:, 0], anterior_points[:, 1], 'r-', lw=3, label='Anterior contour')
        ax1.set_title('Anterior Contour Extraction')
        ax1.legend()
        ax1.axis('equal')
        ax2.plot(anterior_points[:, 0], anterior_points[:, 1], 'r-', lw=2, marker='o', ms=2)
        ax2.set_title('Anterior Contour (Y vs X)')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        plt.tight_layout()
        plt.show()

    return anterior_points


def find_m_shape_points_simple(binary_mask, plot_results=True):
    """
    Detect the M-shaped trochlear groove on a femoral axial contour.

    Identifies the medial valley, central peak (sulcus), and lateral valley.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary 2-D mask of the femur cross-section.
    plot_results : bool
        If True, show a matplotlib figure.

    Returns
    -------
    dict
        Keys: ``medial_valley``, ``central_peak``, ``lateral_valley`` (each (x, y)),
        ``n_peaks``, ``n_valleys``, ``anterior_contour``.
    """
    anterior_points = extract_anterior_contour_horizontal(binary_mask, plot_results=False)
    x_coords = anterior_points[:, 0]
    y_coords = anterior_points[:, 1]

    if len(y_coords) > 5:
        window_size = min(100, len(y_coords) // 3 * 2 + 1)
        y_smooth = savgol_filter(y_coords, window_size, 2)
    else:
        y_smooth = y_coords

    peak_indices, _ = find_peaks(
        y_smooth, height=np.percentile(y_smooth, 0), distance=len(y_smooth) // 8
    )
    valley_indices, _ = find_peaks(
        -y_smooth, height=-np.percentile(y_smooth, 100), distance=len(y_smooth) // 8
    )

    n_peaks = len(peak_indices)
    n_valleys = len(valley_indices)

    if n_peaks > 0:
        central_peak_idx = peak_indices[np.argmax(y_smooth[peak_indices])]
        central_peak = anterior_points[central_peak_idx]
    else:
        central_peak_idx = np.argmax(y_smooth)
        central_peak = anterior_points[central_peak_idx]

    if n_valleys >= 2:
        valley_heights = y_smooth[valley_indices]
        medial_valley = anterior_points[valley_indices[0]]
        lateral_valley = anterior_points[valley_indices[-1]]
    elif n_valleys == 1:
        medial_end, lateral_end = anterior_points[0], anterior_points[-1]
        if anterior_points[valley_indices[0]][0] < (medial_end[0] + lateral_end[0]) / 2:
            medial_valley = anterior_points[valley_indices[0]]
            lateral_valley = lateral_end
        else:
            medial_valley = medial_end
            lateral_valley = anterior_points[valley_indices[0]]
    else:
        medial_valley = anterior_points[0]
        lateral_valley = anterior_points[-1]

    if plot_results:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask, cmap='gray')
        plt.plot(anterior_points[:, 0], anterior_points[:, 1], 'r-', lw=1, label='Anterior contour')
        plt.plot(medial_valley[0], medial_valley[1], 'go', ms=6, label='Medial valley')
        plt.plot(central_peak[0], central_peak[1], 'ro', ms=6, label='Central peak')
        plt.plot(lateral_valley[0], lateral_valley[1], 'bo', ms=6, label='Lateral valley')
        plt.title('M-Shape Detection on Anterior Contour Only')
        plt.legend()
        plt.axis('equal')
        plt.show()

    return {
        'medial_valley': medial_valley,
        'central_peak': central_peak,
        'lateral_valley': lateral_valley,
        'n_peaks': n_peaks,
        'n_valleys': n_valleys,
        'anterior_contour': anterior_points,
    }


# ---------------------------------------------------------------------------
# Convexity-defect / notch detection  (lateral view)
# ---------------------------------------------------------------------------

def find_notch(cnt, hull_img):
    """
    Find the two deepest convexity defects on a contour.

    Parameters
    ----------
    cnt : np.ndarray
        OpenCV contour.
    hull_img : np.ndarray
        BGR image for visualisation.

    Returns
    -------
    farpoints : dict
        {(x, y): depth} for the two deepest defect points.
    defect_img : np.ndarray
        Annotated BGR image.
    """
    cnt = clean_cnt(cnt)
    hull_index = cv2.convexHull(cnt, clockwise=True, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_index)

    farpoints = {}
    defect_img = hull_img.copy()
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        far = tuple(cnt[f][0])
        farpoints[far] = d

    farpoints = dict(sorted(farpoints.items(), key=lambda item: item[1], reverse=True)[:2])

    for far in farpoints:
        cv2.circle(defect_img, far, 5, (255, 0, 0), -1)

    return farpoints, defect_img


def find_notch_with_visualization(cnt, hull_img, start_fraction=0.4, right=True):
    """
    Find deepest convexity defects, restricted to the anterior portion.

    Parameters
    ----------
    cnt : np.ndarray
        OpenCV contour.
    hull_img : np.ndarray
        BGR image for visualisation.
    start_fraction : float
        Fraction of contour length defining the anterior boundary.
    right : bool
        If True, the anterior portion is the first *start_fraction* of points.

    Returns
    -------
    farpoints : dict
        {(x, y): depth} for up to two deepest anterior defects.
    defect_img : np.ndarray
        Annotated BGR image.
    """
    cnt = clean_cnt(cnt)
    total_points = len(cnt)
    anterior_start_idx = int(total_points * start_fraction)

    if right:
        anterior_cnt = cnt[:anterior_start_idx]
    else:
        anterior_cnt = cnt[anterior_start_idx:]

    hull_index = cv2.convexHull(cnt, clockwise=True, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_index)

    farpoints = {}
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far = tuple(cnt[f][0])
            if right and f <= anterior_start_idx:
                farpoints[far] = d
            elif not right and f >= anterior_start_idx:
                farpoints[far] = d

    farpoints = dict(sorted(farpoints.items(), key=lambda item: item[1], reverse=True)[:2])

    defect_img = hull_img.copy()
    if len(anterior_cnt) > 0:
        cv2.drawContours(defect_img, [anterior_cnt], -1, (128, 255, 255), 2)
    for far in farpoints:
        cv2.circle(defect_img, far, 5, (255, 0, 0), -1)

    return farpoints, defect_img


# ---------------------------------------------------------------------------
# Geometry helpers for anterior line / posterior point
# ---------------------------------------------------------------------------

def vector_distance(x, v):
    """
    Perpendicular distance from point *x* to the line through the origin
    along unit vector *v*.

    Parameters
    ----------
    x : np.ndarray
        2-D point vector.
    v : np.ndarray
        Unit direction vector.

    Returns
    -------
    float
    """
    return np.linalg.norm(x - (v @ x) * v)


def twopoint_distance(x1, x2):
    """
    Euclidean distance between two 2-D points.

    Parameters
    ----------
    x1, x2 : array-like
        (x, y) coordinates.

    Returns
    -------
    float
    """
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def find_posterior(cnt, linepoint, line_v):
    """
    Find the posterior-most contour point relative to the anterior line.

    Parameters
    ----------
    cnt : np.ndarray
        OpenCV contour.
    linepoint : np.ndarray
        A point on the anterior reference line.
    line_v : np.ndarray
        Direction vector of the anterior line.

    Returns
    -------
    post_point : np.ndarray
        Posterior-most point coordinates.
    foot : tuple[int, int]
        Foot of perpendicular from *post_point* onto the anterior line.
    apwidth : float
        Anterior-posterior distance in pixels.
    """
    cnt = cnt[len(cnt) // 2:]
    line_unitv = line_v / np.linalg.norm(line_v)
    post_point = max(cnt - linepoint, key=lambda x: vector_distance(x[0], v=line_unitv))[0] + linepoint
    foot = linepoint + ((post_point - linepoint) @ line_unitv) * line_unitv
    apwidth = twopoint_distance(foot, post_point)
    foot = (int(foot[0]), int(foot[1]))
    return post_point, foot, apwidth


def antline_connect_right(lat_xr, cnt, farpoints, spacing):
    """
    Compute the anterior reference line from the notch and contour.

    Parameters
    ----------
    lat_xr : np.ndarray
        Lateral projection mask (used for shape only).
    cnt : np.ndarray
        OpenCV contour.
    farpoints : list[tuple]
        Sorted list of notch points (anterior first).
    spacing : float
        Pixel spacing in mm (used to scale the 30 mm look-up distance).

    Returns
    -------
    ant_v : np.ndarray
        Anterior line direction vector.
    f_notch : tuple
        Anterior notch point.
    """
    f_notch = farpoints[0]
    back_notch = farpoints[-1]
    ant_cnt = cnt[:len(cnt) // 2]
    post_cnt = cnt[len(cnt) // 2:]
    ant_cnt = np.array([i for i in ant_cnt if i[0][1] <= f_notch[1]])
    post_cnt = np.array([i for i in post_cnt if i[0][1] <= back_notch[1]])
    [vx, vy, x0, y0] = cv2.fitLine(np.concatenate((ant_cnt, post_cnt)), cv2.DIST_L2, 0, 0.01, 0.01)
    v = np.array([vx, vy])
    v = v / np.linalg.norm(v)
    length = 30 / spacing  # 30 mm above the notch
    new_point_y = max(f_notch[1] - length * abs(v[1]), 20)
    new_point = min(ant_cnt, key=lambda x: abs(x[0][1] - new_point_y))
    new_point = (new_point[0][0], new_point[0][1])
    ant_v = np.array(f_notch) - np.array(new_point)
    return ant_v, f_notch


def get_points(y_mask, spacing, right=True):
    """
    Compute anterior line, notch point, and an annotated image.

    Parameters
    ----------
    y_mask : np.ndarray
        Binary lateral-projection mask.
    spacing : float
        Pixel spacing in mm.
    right : bool
        Whether this is a right knee.

    Returns
    -------
    ant_v : np.ndarray
        Anterior line direction vector.
    f_notch : tuple
        Anterior notch point.
    img : np.ndarray
        BGR annotated image.
    """
    cnt, hull_img = cnt_hull(y_mask)
    farpoints, defect_img = find_notch_with_visualization(cnt, hull_img, right=right)
    ant_v, f_notch = antline_connect_right(y_mask, cnt, list(farpoints.keys()), spacing)
    post_point, foot, apwidth = find_posterior(cnt, f_notch, ant_v)

    img = y_mask.copy() * 255
    img = np.stack([img, img, img], axis=-1)
    cv2.circle(img, f_notch, 4, (0, 255, 0), -1)
    cv2.line(img, f_notch - ant_v * 500, f_notch + ant_v * 500, (0, 0, 255), 1)
    return ant_v, f_notch, img


# ---------------------------------------------------------------------------
# Point / line side test and contour filtering
# ---------------------------------------------------------------------------

def is_point_left_of_line(point, line_point1, line_point2):
    """
    Test whether *point* lies to the left of the directed line from
    *line_point1* to *line_point2* (via 2-D cross product).

    Parameters
    ----------
    point : array-like
        (x, y) test point.
    line_point1, line_point2 : array-like
        Points defining the directed line.

    Returns
    -------
    bool
    """
    line_vec = np.array(line_point2) - np.array(line_point1)
    point_vec = np.array(point) - np.array(line_point1)
    return (line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) > 0


def filter_contour_left_of_line(contour, line_point1, line_point2):
    """
    Keep only contour points that are to the left of a directed line.

    Parameters
    ----------
    contour : np.ndarray
        OpenCV contour (N, 1, 2) or (N, 2).
    line_point1, line_point2 : array-like
        Points defining the directed line.

    Returns
    -------
    np.ndarray
        Filtered contour in the same format as the input.
    """
    is_cv_format = len(contour.shape) == 3 and contour.shape[1] == 1
    points = contour.reshape(-1, 2) if is_cv_format else np.array(contour)

    filtered = np.array([p for p in points if is_point_left_of_line(p, line_point1, line_point2)])

    if len(filtered) == 0:
        return np.array([]).reshape(0, 2)

    if is_cv_format:
        return filtered.reshape(-1, 1, 2)
    return filtered


# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------

def read_prepare_pbcl(mr_path, mask_path, femur_bone_code, femur_cart_code):
    """
    Load MRI + segmentation, resample to isotropic, and align femoral condyles.

    Parameters
    ----------
    mr_path : str or Path
        Path to the MR NIfTI file.
    mask_path : str or Path
        Path to the segmentation NIfTI file.
    femur_bone_code : int
        Label value for femur bone in the segmentation.
    femur_cart_code : int
        Label value for femur cartilage.

    Returns
    -------
    mr_aligned : np.ndarray
        Aligned MR volume (isotropic).
    seg_aligned : np.ndarray
        Aligned segmentation volume (isotropic).
    spacing : tuple
        Original voxel spacing.
    condyle_upper_border : int
        Upper border row index of the femoral condyle region.
    condyle_lower_border : int
        Lower border row index.
    """
    nii = nib.load(str(mask_path))
    seg = nii.get_fdata().astype(int)
    spacing = nii.header.get_zooms()

    nii_mr = nib.load(str(mr_path))
    mr = nii_mr.get_fdata().astype(np.float32)

    mr_resampled, _, _ = resample_mri_to_isotropic(mr, spacing, is_mask=False)
    seg_resampled, _, _ = resample_mri_to_isotropic(seg, spacing, is_mask=True)

    mask_femur = np.where(
        np.isin(seg_resampled, [femur_bone_code, femur_cart_code]), seg_resampled, 0
    )
    mask_femur_bone = np.where(mask_femur == femur_bone_code, femur_bone_code, 0)

    femur_bone_image = sitk.GetImageFromArray(mask_femur_bone)
    femur_bone_lateral = sitk.MaximumProjection(femur_bone_image, projectionDimension=0)
    sag_project = sitk.GetArrayFromImage(femur_bone_lateral).squeeze(-1).T

    cnt, hull_img = cnt_hull(sag_project)
    farpoints, _ = find_notch(cnt, hull_img)
    condyle_upper_border = sorted(farpoints, key=lambda x: x[0])[1][1]
    condyle_lower_border = np.argwhere(sag_project)[:, 0].max()

    condyle_mid_slice = mask_femur[:, (condyle_lower_border + condyle_upper_border) // 2, :]

    posterior = cv2.flip((condyle_mid_slice == femur_bone_code).astype(np.uint8), 0)
    result = find_m_shape_points_simple(posterior, plot_results=False)

    # NOTE: 384 is the expected resampled grid size -- kept as-is to preserve algorithm
    medial_condyle_post = np.array([result['medial_valley'][0], 384 - result['medial_valley'][1]])
    lateral_condyle_post = np.array([result['lateral_valley'][0], 384 - result['lateral_valley'][1]])

    theta = np.degrees(np.arctan2(
        lateral_condyle_post[1] - medial_condyle_post[1],
        lateral_condyle_post[0] - medial_condyle_post[0],
    ))

    seg_aligned = rotate(seg_resampled, theta, axes=(0, 2), reshape=False, order=0, prefilter=False)
    mr_aligned = rotate(mr_resampled, theta, axes=(0, 2), reshape=False, order=1, prefilter=False)

    return mr_aligned, seg_aligned, spacing, condyle_upper_border, condyle_lower_border


def get_angles_axial(mr_aligned, seg_aligned, upper_border, lower_border,
                     side, bone_code, cart_code, show_img=False):
    """
    Measure sulcus angle and lateral trochlear inclination on an axial slice.

    Parameters
    ----------
    mr_aligned : np.ndarray
        Aligned MR volume.
    seg_aligned : np.ndarray
        Aligned segmentation volume.
    upper_border, lower_border : int
        Row range for searching the best axial slice.
    side : str
        'Rt' or 'Lt'.
    bone_code, cart_code : int
        Label values for femur bone and cartilage.
    show_img : bool
        If True, return an annotated image as the first element.

    Returns
    -------
    If show_img:
        (mr_slice_bgr, concave, sulcus_angle, lti_angle, gc)
    Else:
        (concave, sulcus_angle, lti_angle, gc)
    """
    mask_femur = np.where(np.isin(seg_aligned, [bone_code, cart_code]), seg_aligned, 0)

    found = False
    for slice_idx in range(upper_border, lower_border):
        slc = mask_femur[:, slice_idx, :].copy()
        results = find_m_shape_points_simple((slc == bone_code).astype(np.uint8), plot_results=False)
        if results is None:
            continue
        slc[200:, :] = 0
        slc = remove_small_islands_multilabel_scipy(slc, min_size=10)
        x_coords = np.argwhere(slc == cart_code)[:, 1]
        if len(x_coords) == 0:
            continue
        x_min, x_max = x_coords.min(), x_coords.max()
        if x_min <= results['medial_valley'][0] + 5 and x_max >= results['lateral_valley'][0] - 5:
            found = True
            break

    if not found:
        raise ValueError("No suitable slice found with medial and lateral cartilage points")

    cart_mask = fill_and_close_mask((slc == cart_code).astype(np.uint8), kernel_size=5)
    results = find_m_shape_points_simple(cart_mask, plot_results=False)
    mtc = results['medial_valley']
    gc = results['central_peak']
    ltc = results['lateral_valley']

    angle1 = np.degrees(np.arctan2(mtc[1] - gc[1], mtc[0] - gc[0]))
    angle2 = np.degrees(np.arctan2(ltc[1] - gc[1], ltc[0] - gc[0]))

    if angle1 >= 0 or angle2 >= 0:
        concave = False
        sulcus_angle = 'unmeasurable'
        lti_angle = 'unmeasurable'
    else:
        concave = True

    sulcus_angle = angle2 - angle1

    if side == 'Rt':
        lti_angle = -angle2
    elif side == 'Lt':
        lti_angle = 180 + angle1

    if show_img:
        mr_slice = mr_to_uint8_adaptive(mr_aligned[:, slice_idx, :])
        mr_slice = cv2.cvtColor(mr_slice, cv2.COLOR_GRAY2BGR)
        mr_slice = add_ml_labels_to_rgb_image(mr_slice, side)
        cv2.circle(mr_slice, (mtc[0], mtc[1]), 2, (255, 0, 0), -1)
        cv2.circle(mr_slice, (gc[0], gc[1]), 2, (255, 0, 0), -1)
        cv2.line(mr_slice, (gc[0] - 500, gc[1]), (gc[0] + 500, gc[1]), (255, 255, 0))
        cv2.circle(mr_slice, (ltc[0], ltc[1]), 2, (255, 0, 0), -1)
        return mr_slice, concave, sulcus_angle, lti_angle, gc
    else:
        return concave, sulcus_angle, lti_angle, gc


def sagittal_bump(mr_aligned, seg_aligned, side, spacing, gc,
                  bone_code, cart_code, show_img=False):
    """
    Measure the supratrochlear bump height on a sagittal slice.

    Parameters
    ----------
    mr_aligned : np.ndarray
        Aligned MR volume.
    seg_aligned : np.ndarray
        Aligned segmentation volume.
    side : str
        'Rt' or 'Lt'.
    spacing : tuple
        Original voxel spacing (used for mm conversion).
    gc : array-like
        (x, y) groove centre from axial measurement.
    bone_code, cart_code : int
        Label values for femur bone and cartilage.
    show_img : bool
        If True, return (annotated image, bump_width_mm).

    Returns
    -------
    If show_img:
        (img, bump_width_mm)
    Else:
        bump_width_mm : float
    """
    mask_femur = np.where(np.isin(seg_aligned, [bone_code, cart_code]), seg_aligned, 0)
    sag_slice = mask_femur[:, :, gc[0]].copy().T
    ant_v, f_notch, img = get_points(sag_slice, spacing[0])

    fcart_cnt, hull_img = cnt_hull(sag_slice == cart_code)
    fcart_cnt_ant = filter_contour_left_of_line(fcart_cnt, f_notch, f_notch + ant_v)
    if len(fcart_cnt_ant) == 0:
        raise ValueError("No anterior contour points found left of the anterior line")

    bump_point, foot, bump_width = find_posterior(fcart_cnt_ant, f_notch, ant_v)
    bump_width_mm = bump_width * spacing[0]

    if show_img:
        mr_sag_slice = mr_to_uint8_adaptive(mr_aligned[:, :, gc[0]].copy().T)
        img = cv2.cvtColor(mr_sag_slice, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [fcart_cnt], -1, (0, 255, 0), 1)
        cv2.circle(img, bump_point.astype(int), 2, (255, 0, 0), -1)
        cv2.circle(img, foot, 2, (255, 0, 0), -1)
        cv2.line(img, f_notch - ant_v * 200, f_notch + ant_v * 200, (255, 255, 0), 1)
        return img, bump_width_mm
    else:
        return bump_width_mm


# ---------------------------------------------------------------------------
# Top-level classification
# ---------------------------------------------------------------------------

def dejour_classification(mr_path, mask_path, side, femur_bone_code,
                          femur_cart_code, show_img=False):
    """
    Run the full Dejour trochlear dysplasia classification pipeline.

    Parameters
    ----------
    mr_path : str or Path
        Path to the MR NIfTI file.
    mask_path : str or Path
        Path to the segmentation NIfTI file.
    side : str
        'Rt' or 'Lt'.
    femur_bone_code : int
        Segmentation label for femur bone.
    femur_cart_code : int
        Segmentation label for femur cartilage.
    show_img : bool
        If True, return annotated axial and sagittal images.

    Returns
    -------
    If show_img:
        (ax_img, sag_img, results_dict)
    Else:
        results_dict  (or None on error)
    """
    mr_aligned, seg_aligned, spacing, upper, lower = read_prepare_pbcl(
        mr_path, mask_path, femur_bone_code, femur_cart_code
    )

    try:
        ax_img, concave, sulcus_angle, lti_angle, gc = get_angles_axial(
            mr_aligned, seg_aligned, upper, lower, side,
            femur_bone_code, femur_cart_code, show_img=True,
        )
    except Exception:
        print("[ERROR] Axial angle calculation failed.")
        return None

    try:
        sag_img, bump_width_mm = sagittal_bump(
            mr_aligned, seg_aligned, side, spacing, gc,
            femur_bone_code, femur_cart_code, show_img=True,
        )
    except Exception:
        print("[ERROR] Sagittal bump calculation failed.")
        return None

    if concave:
        if sulcus_angle < 157 and lti_angle >= 14:
            classification = 'Dejour Type 0: No trochlear dysplasia'
        elif (sulcus_angle >= 157 or lti_angle < 14) and bump_width_mm < 5:
            classification = 'Dejour Type 1: Low grade trochlear dysplasia'
        elif not concave and bump_width_mm < 5:
            classification = 'Dejour Type 2: Moderate grade trochlear dysplasia'
        elif (not concave or sulcus_angle >= 157 or lti_angle < 14) and bump_width_mm >= 5:
            classification = 'Dejour Type 3: High grade trochlear dysplasia'
        else:
            classification = 'Unclassified'
    else:
        classification = 'error'

    results = {
        "concave": concave,
        "sulcus angle": sulcus_angle,
        "LTI angle": lti_angle,
        "Sagittal bump(mm)": bump_width_mm,
        "classification": classification,
    }

    if show_img:
        return ax_img, sag_img, results
    else:
        return results
