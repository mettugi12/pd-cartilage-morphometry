"""knee_mr_cartilage.pipeline — patient → template cartilage thickness.

Orchestration only. Each load-bearing decision (cleanup, anchor, subch, IDW)
is selected by `PipelineConfig` and resolved via the strategy registry in
`knee_mr_cartilage.strategies`. Default config reproduces the v1 canonical
behavior exactly.

Documented in Obsidian: knee-MR/Applications/Cartilage-Analysis.md

Per-bone, per-patient steps:
  1. Load seg + LEFT-flip (axis 2 ML reversal).
  2. PD modality: keep_largest_bone_components on the seg label array
     (mandatory; spurious bone islands are always wrong).
  3. PD modality: RECON 4× depth super-resolution to densify slabs.
     DESS modality: per-slice contour interpolation if spacing[2] > 1.5 mm.
  4. Apply each cfg.cart_cleanup strategy to the (bone_mask, cart_mask) pair.
  5. Build patient bone surface mesh (marching cubes in physical mm).
  6. Per-vertex thickness via PD-vs-DESS canonical EDT.
  7. Resolve cfg.anchor → aligned patient verts in template frame.
  8. Resolve cfg.subch_method → per-vert subch_prob, is_subchondral.
  9. IDW remap thickness to template using cfg.idw_k / cfg.idw_mutual_nn.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
    label as cc_label,
)
from skimage import measure

from ._helpers import create_pv_mesh, interpolate_segmentation_filled
from .config import PipelineConfig
from .strategies import get_anchor, get_cleanup, get_subch, get_thickness
from .strategies.anchor import (
    TARGET_SCALE_MM,
    _articular_source_mask as _articular_source_mask,
    _femur_scale_factors as _femur_scale_factors_impl,
    _tibia_scale_factors as _tibia_scale_factors_impl,
    _apply_aniso as _apply_aniso_impl,
    _rigid_icp as _rigid_icp_impl,
)
from .remap import remap_thickness_to_template
from .strategies.subch import tpl_nn as _tpl_nn
from .thickness import compute_full_bone_thickness


# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
TEMPLATE_PATHS = {
    "femur": Path(r"E:/KneeMR/Datasets/Bone-Atlas/femur_template_full_with_subch.vtk"),
    "tibia": Path(r"E:/KneeMR/Datasets/Bone-Atlas/tibia_template_full_with_subch.vtk"),
}

# DESS labels (matches KLG=0 templates: tibia bone=3 cart=4, femur bone=1 cart=2)
DESS_LABELS = {
    "femur": {"bone": 1, "cart": 2},
    "tibia": {"bone": 3, "cart": 4},
}
# PD nnU-Net labels (0=BG, 1=Patella, 2=Femur, 3=Tibia, 4=PC, 5=FC, 6=TC, ...)
PD_LABELS = {
    "femur": {"bone": 2, "cart": 5},
    "tibia": {"bone": 3, "cart": 6},
}

RECON_WEIGHTS = Path(r"E:/KneeMR/Models/RECON_Net/weights/11class_RECON_best.pt")

_PD_BONE_LABELS = (1, 2, 3)


# Module-level RECON output cache. Keyed by str(seg_path). When the same seg
# is processed multiple times (e.g. femur AND tibia calls back-to-back), we
# only run RECON once and slice different channels for each bone.
_RECON_CACHE: dict[str, tuple] = {}

# Optional disk cache directory — if set via `set_recon_disk_cache_dir()`,
# RECON outputs are also serialised to compressed npz so subsequent runs
# (in a fresh Python process) can skip RECON entirely.
_RECON_DISK_CACHE_DIR: Path | None = None


def clear_recon_cache():
    """Drop the cached RECON outputs (call between cases if memory matters)."""
    _RECON_CACHE.clear()


def set_recon_disk_cache_dir(path: Path | str | None):
    """Enable / disable on-disk caching of RECON 4× densified outputs.

    When set, `densify_pd_with_recon` saves dense_chw + spacing to
    `<dir>/<seg_filename>.recon.npz` after running RECON, and loads from there
    on future calls (across Python processes). Disable with `None`.
    """
    global _RECON_DISK_CACHE_DIR
    if path is None:
        _RECON_DISK_CACHE_DIR = None
    else:
        _RECON_DISK_CACHE_DIR = Path(path)
        _RECON_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _disk_cache_path(cache_key: str) -> Path | None:
    if _RECON_DISK_CACHE_DIR is None or cache_key is None:
        return None
    stem = Path(cache_key).name.replace(".nii.gz", "").replace(".nii", "")
    return _RECON_DISK_CACHE_DIR / f"{stem}.recon.npz"


# ---------------------------------------------------------------------------
# Step 1 — load + bone-island cleanup (mandatory for PD) + densify
# ---------------------------------------------------------------------------
def _empirical_flip_needed(seg: np.ndarray, spacing: tuple, bone_label: int = 3) -> bool:
    """Empirical ML-flip check (knee-mr-atlas convention).

    Builds the largest-CC tibia mesh from the seg, registers it to the
    right-knee tibia template, computes the mean nearest-neighbour residual.
    Flips on axis 2 and repeats. Returns True if the flipped version fits
    the template better than as-is — meaning the input is a LEFT-knee-like
    voxel layout that needs flipping to match the right-knee convention.

    Falls back to False (no flip) on any error — safer than mis-flipping.

    This is robust to DICOM tag mislabels, slice-ordering variance during
    NIfTI assembly, and unusual axcodes. The DICOM `Laterality` tag tells
    you which anatomical knee was scanned, but it doesn't tell you whether
    SimpleITK/dcm2niix assembled the slices in a "left-knee voxel layout"
    or "right-knee voxel layout" — that depends on slice direction and
    sorting strategy, which varies by scanner.
    """
    try:
        from scipy.ndimage import label as cc_label
        from scipy.spatial import cKDTree
        from skimage import measure
        # Largest-CC tibia (label 3 in PD scheme, 3 in DESS scheme too)
        mask = (seg == bone_label)
        if mask.sum() < 1000:
            return False
        cc, n = cc_label(mask)
        if n > 1:
            sizes = np.bincount(cc.ravel())[1:]
            mask = cc == (int(np.argmax(sizes)) + 1)

        target = pv.read(str(TEMPLATE_PATHS["tibia"]))
        target_pts = np.asarray(target.points)
        target_tree = cKDTree(target_pts)
        target_ml = 75.0; target_ap = 55.0

        def _resid(m):
            if m.sum() < 500:
                return float("inf")
            verts, faces, *_ = measure.marching_cubes(
                m.astype(float), level=0.5, spacing=spacing, step_size=2,
            )
            # aniso scale (bone bounds): match ML, AP, leave SI=1
            ml_r = max(np.ptp(verts[:, 2]), 1e-6)
            ap_r = max(np.ptp(verts[:, 0]), 1e-6)
            scale = np.array([target_ap / ap_r, 1.0, target_ml / ml_r])
            scaled = verts * scale
            # translation-only: centroid match
            t = target_pts.mean(0) - scaled.mean(0)
            moved = scaled + t
            d, _ = target_tree.query(moved)
            return float(d.mean())

        r_R = _resid(mask)
        r_L = _resid(mask[:, :, ::-1])
        return r_L < r_R
    except Exception:
        return False


def load_seg_oriented(seg_path: Path, side: str | None = None,
                      detect_side: bool = True):
    """Load NIfTI seg; apply ML flip on axis 2 if needed for right-knee convention.

    The flip decision is **empirical** by default — tries both orientations
    against the right-knee tibia template and picks the one with lower
    registration residual. This is robust to DICOM-tag mislabels, axcode
    variance, and slice-ordering quirks during NIfTI assembly.

    Args:
        seg_path: NIfTI path
        side:     Anatomical side from DICOM. Kept for logging only when
                  `detect_side=True`. If `detect_side=False`, the legacy
                  "side==LEFT → flip" rule is used.
        detect_side: If True (default), the flip decision is empirical.
                  If False, falls back to the legacy DICOM-tag-based logic.

    Returns: (seg array in right-knee-convention voxel space, spacing).
    """
    nii = nib.load(str(seg_path))
    seg = nii.get_fdata().astype(np.int16)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    if detect_side:
        flip = _empirical_flip_needed(seg, spacing, bone_label=3)
        if flip:
            seg = seg[:, :, ::-1].copy()
            print(f"  [orient] empirical check → FLIP applied "
                  f"(DICOM said side={side!r}; voxel layout was left-knee-like)")
        else:
            print(f"  [orient] empirical check → no flip "
                  f"(DICOM said side={side!r}; voxel layout already right-knee-like)")
    else:
        # Legacy: trust the side argument literally
        if side == "LEFT":
            seg = seg[:, :, ::-1].copy()
    return seg, spacing


def keep_largest_bone_components(seg: np.ndarray) -> tuple[np.ndarray, dict]:
    """For each PD bone label (Patella=1, Femur=2, Tibia=3): keep only the
    single largest connected component; zero spurious islands. Mandatory for
    PD modality (PD seg of the knee has exactly one CC per bone label).
    """
    seg_out = seg.copy()
    removed = {}
    for lbl in _PD_BONE_LABELS:
        mask = seg == lbl
        if mask.sum() == 0:
            removed[lbl] = 0
            continue
        labeled, n = cc_label(mask)
        if n <= 1:
            removed[lbl] = 0
            continue
        sizes = np.bincount(labeled.ravel())[1:]
        biggest = int(np.argmax(sizes)) + 1
        spurious = mask & (labeled != biggest)
        seg_out[spurious] = 0
        removed[lbl] = int(spurious.sum())
    return seg_out, removed


def densify_pd_with_recon(seg, spacing, bone_name, cache_key: str | None = None):
    """RECON 4× depth super-resolution. Returns (bone_mask, cart_mask, new_spacing).

    If `cache_key` is provided AND already in `_RECON_CACHE`, reuse the cached
    11-channel densified volume and just slice this bone's channels. Saves
    ~50% of total runtime when processing multiple bones from the same seg.
    """
    if cache_key is not None and cache_key in _RECON_CACHE:
        dense_chw, new_spacing = _RECON_CACHE[cache_key]
    else:
        # On-disk cache check: skip RECON entirely if a prior run cached it.
        disk_path = _disk_cache_path(cache_key) if cache_key is not None else None
        if disk_path is not None and disk_path.exists():
            print(f"  [recon] loading cached dense_chw from {disk_path.name}")
            data = np.load(disk_path)
            dense_chw = data["dense_chw"]
            new_spacing = tuple(float(v) for v in data["new_spacing"])
            if cache_key is not None:
                _RECON_CACHE[cache_key] = (dense_chw, new_spacing)
        else:
            from knee_mr_seg.recon import supersample_volume

            seg_dhw = seg.transpose(2, 0, 1)
            dense_chw = supersample_volume(
                mask_path=None, mask_array=seg_dhw,
                checkpoint_path=str(RECON_WEIGHTS),
                mode="11class", device="cuda",
            )
            new_spacing = (spacing[0], spacing[1], spacing[2] / 4.0)
            if cache_key is not None:
                _RECON_CACHE[cache_key] = (dense_chw, new_spacing)
            if disk_path is not None:
                # Threshold-quantise to bool channels for tiny disk footprint
                # (the calling code only uses `> 0.5` thresholded masks anyway).
                # ~150 MB → ~5-15 MB compressed for typical PD knee.
                dense_bool = (dense_chw > 0.5).astype(np.uint8)
                np.savez_compressed(
                    disk_path,
                    dense_chw=dense_bool,    # uint8 0/1 instead of float
                    new_spacing=np.asarray(new_spacing, dtype=np.float64),
                )
                print(f"  [recon] cached to {disk_path.name}")

    ch = {"femur_bone": 1, "tibia_bone": 2, "femur_cart": 4, "tibia_cart": 5}
    bone_dhw = dense_chw[ch[f"{bone_name}_bone"]] > 0.5
    cart_dhw = dense_chw[ch[f"{bone_name}_cart"]] > 0.5
    bone_mask = bone_dhw.transpose(1, 2, 0)
    cart_mask = cart_dhw.transpose(1, 2, 0)
    return bone_mask, cart_mask, new_spacing


def get_patient_masks(seg_path: Path, side: str, modality: str, bone_name: str,
                      config: PipelineConfig | None = None):
    """Load seg → bone-island cleanup (PD only) → densify → cart cleanup pipeline.

    Returns (bone_mask, cart_mask, spacing, seg_native, spacing_native, cleanup_meta).
    """
    import time as _t
    if config is None:
        config = PipelineConfig()
    _t0 = _t.time()
    seg_native, spacing_native = load_seg_oriented(seg_path, side)
    print(f"    [timing] load_seg_oriented (incl. empirical-flip check): {_t.time()-_t0:.1f}s")
    cleanup_meta = {"bone_island_voxels_removed": {}, "cart_cleanup_steps": []}

    if modality == "pd":
        _t0 = _t.time()
        seg_native, removed = keep_largest_bone_components(seg_native)
        cleanup_meta["bone_island_voxels_removed"] = {int(k): int(v) for k, v in removed.items()}
        if any(v > 0 for v in removed.values()):
            print(f"  [bone cleanup] " + ", ".join(
                f"lbl{k}={v}vox" for k, v in removed.items() if v > 0
            ))
        print(f"    [timing] keep_largest_bone_components: {_t.time()-_t0:.1f}s")
        _t0 = _t.time()
        bone_mask, cart_mask, spacing = densify_pd_with_recon(
            seg_native, spacing_native, bone_name,
            cache_key=str(seg_path),   # cache RECON across bone calls for the same seg
        )
        print(f"    [timing] densify_pd_with_recon (RECON, cached if 2nd+ call): {_t.time()-_t0:.1f}s")
    else:  # dess
        labels = DESS_LABELS[bone_name]
        if spacing_native[2] > 1.5:
            seg_d, spacing = interpolate_segmentation_filled(
                seg_native, spacing_native, labels["bone"], labels["cart"], slices_between=4,
            )
        else:
            seg_d, spacing = seg_native, spacing_native
        bone_mask = (seg_d == labels["bone"])
        cart_mask = (seg_d == labels["cart"])

    # Apply each configured cart-cleanup strategy in order
    for name in config.cart_cleanup:
        import time as _t
        _t0 = _t.time()
        fn = get_cleanup(name)
        bone_mask, cart_mask, meta = fn(bone_mask, cart_mask, spacing, config)
        meta["strategy"] = name
        cleanup_meta["cart_cleanup_steps"].append(meta)
        rm = meta.get("cart_voxels_removed", meta.get("cart_voxels_removed_near_bone_filter", 0))
        if rm:
            print(f"  [cart cleanup {name}] removed {rm} voxels ({_t.time()-_t0:.1f}s)")
        else:
            print(f"  [cart cleanup {name}] no voxels removed ({_t.time()-_t0:.1f}s)")

    return bone_mask, cart_mask, spacing, seg_native, spacing_native, cleanup_meta


# ---------------------------------------------------------------------------
# Step 2 — meshes
# ---------------------------------------------------------------------------
def build_meshes(bone_mask, cart_mask, spacing,
                 bone_smooth_iters: int = 0,
                 bone_taubin_pass_band: float = 0.1):
    """Build bone + cart surface meshes in physical mm.

    Bone mesh: binary_closing + gaussian σ=1.0 → marching_cubes. Mesh-level
    Taubin smoothing is OFF by default (v0.5.1+) because:
      - EDT thickness (the default) uses no mesh normals → smoothing is wasted compute
      - PyVista `add_mesh(smooth_shading=True)` interpolates normals at render
        time → the same smooth visual appearance without geometric distortion

    Set `bone_smooth_iters > 0` to enable Taubin smoothing — useful when the
    thickness method is `raycast` / `raycast_to_far` (which DO use mesh normals)
    or when you're exporting the bone mesh for downstream surface-based
    analysis.
    """
    bone_closed = binary_closing(bone_mask, structure=np.ones((3, 3, 3)), iterations=2)
    # Stronger mask smoothing (σ=1.0) reduces the marching-cubes voxel staircase
    # significantly before mesh-level Taubin smoothing finishes the job.
    bone_smooth_vol = gaussian_filter(bone_closed.astype(float), sigma=1.0)
    bv, bf, _, _ = measure.marching_cubes(
        bone_smooth_vol, level=0.5, spacing=spacing, step_size=1,
    )
    bone_mesh = create_pv_mesh(bv, bf)
    # Taubin smoothing — volume-preserving, removes stair-steps without shrinking
    if bone_smooth_iters > 0:
        try:
            bone_mesh = bone_mesh.smooth_taubin(
                n_iter=bone_smooth_iters,
                pass_band=bone_taubin_pass_band,
                feature_smoothing=False,
                boundary_smoothing=False,
            )
        except AttributeError:
            # Older pyvista without smooth_taubin → fall back to Laplacian
            bone_mesh = bone_mesh.smooth(
                n_iter=bone_smooth_iters, relaxation_factor=0.05,
                feature_smoothing=False, boundary_smoothing=False,
            )
    bone_mesh.compute_normals(
        point_normals=True, cell_normals=False, inplace=True,
        auto_orient_normals=True,
    )
    cart_mesh = None
    if cart_mask.sum() > 0:
        cv, cf, _, _ = measure.marching_cubes(
            cart_mask.astype(float), level=0.5, spacing=spacing, step_size=1,
        )
        cart_mesh = create_pv_mesh(cv, cf)
    return bone_mesh, cart_mesh


# ---------------------------------------------------------------------------
# Backward-compat shims (preserve top-level API surface)
# ---------------------------------------------------------------------------
def patient_to_template_align(bone_mesh_phys, cart_mask, spacing,
                              template_mesh, bone_name, modality,
                              articular_radius_mm=5.0):
    """Legacy entry point — preserves old return shape (aligned_points,
    scale_factors, translation, R_icp, t_icp). Wraps the registry's
    `aniso_rigid` strategy.
    """
    cfg = PipelineConfig(anchor="aniso_rigid", articular_radius_mm=articular_radius_mm)
    aligned, meta = get_anchor("aniso_rigid")(
        bone_mesh_phys, cart_mask, spacing, template_mesh, bone_name, modality, cfg
    )
    # Reconstruct old return shape best-effort (no R/t exposed by registry,
    # so callers depending on exact R/t should switch to PipelineConfig API).
    scale_factors = meta.get("scale_factors")
    translation = np.asarray(meta.get("translation_mm", [0.0, 0.0, 0.0]))
    # R_icp / t_icp not available in modular shape; return identity / zero.
    R_icp = np.eye(3)
    t_icp = np.zeros(3)
    return aligned, scale_factors, translation, R_icp, t_icp


def pull_subchondral_from_template(aligned_points, template_mesh, thresh=0.5):
    """Legacy entry point — pure tpl_nn behavior, threshold-based."""
    cfg = PipelineConfig(subch_method="tpl_nn", subch_threshold=thresh)
    # Build a dummy "bone_mesh"-like for the strategy signature
    return _tpl_nn(aligned_points, template_mesh, None, None, None, None, cfg)


# rigid_icp / aniso scale helpers re-exported for legacy callers
def rigid_icp(source_points, target_points, max_iter=30, tol=1e-4):
    return _rigid_icp_impl(source_points, target_points, max_iter, tol)


def apply_anisotropic_scale_to_points(points, scale_factors):
    return _apply_aniso_impl(points, scale_factors)


def compute_femur_scale_factors_from_cart_mask(cart_mask, spacing, target_ml, target_ap):
    return _femur_scale_factors_impl(cart_mask, spacing, target_ml, target_ap)


def compute_tibia_scale_factors_from_bone_mesh(bone_mesh, target_ml, target_ap):
    return _tibia_scale_factors_impl(bone_mesh, target_ml, target_ap)


# Legacy cleanup function shims (still importable from `pipeline` for old code)
def drop_small_cart_components(cart_mask, min_size_vox=100):
    """Legacy signature — returns (cleaned_cart_mask, n_voxels_removed)."""
    from .strategies.cleanup import drop_small_components
    # Build minimal cfg + dummy bone_mask of same shape
    cfg = PipelineConfig()
    dummy_bone = np.zeros_like(cart_mask, dtype=bool)
    _, cleaned, meta = drop_small_components(dummy_bone, cart_mask, (1.0, 1.0, 1.0), cfg)
    return cleaned, meta["cart_voxels_removed"]


def filter_cart_near_bone_per_slice(cart_mask, bone_mask, spacing, r_mm=6.0, slice_axis=2):
    """Legacy signature — returns (cleaned_cart_mask, n_voxels_removed)."""
    from .strategies.cleanup import filter_near_bone_per_slice
    cfg = PipelineConfig()
    _, cleaned, meta = filter_near_bone_per_slice(bone_mask, cart_mask, spacing, cfg)
    return cleaned, meta["cart_voxels_removed_near_bone_filter"]


# ---------------------------------------------------------------------------
# Top-level per-patient pipeline
# ---------------------------------------------------------------------------
def process_one_patient(seg_path: Path, side: str, bone_name: str, modality: str,
                        config: PipelineConfig | None = None,
                        max_thickness_mm: float = 6.0,
                        out_dir: Path | None = None,
                        case_id: str | None = None,
                        save_unregistered_mesh: bool = False):
    """End-to-end pipeline. Returns a result dict.

    config: PipelineConfig | None — if None, defaults to PipelineConfig()
            (current v1 canonical behavior).
    save_unregistered_mesh: if True, additionally saves a `_patient_raw.vtk`
            in patient physical-mm frame (pre-registration) for visualization.
    """
    if config is None:
        config = PipelineConfig()

    t_total = time.time()
    template_mesh = pv.read(str(TEMPLATE_PATHS[bone_name]))
    if "subch_prob" not in template_mesh.point_data:
        raise RuntimeError(f"Template {TEMPLATE_PATHS[bone_name]} missing 'subch_prob' field")

    t = time.time()
    print(f"  loading + densify ({modality})...")
    bone_mask, cart_mask, spacing, seg_native, spacing_native, cleanup_meta = get_patient_masks(
        seg_path, side, modality, bone_name, config=config,
    )
    t_load = time.time() - t
    print(f"    bone={int(bone_mask.sum())}, cart={int(cart_mask.sum())}, "
          f"spacing={tuple(round(s, 3) for s in spacing)}")

    # Cart pre/post bbox for QA logging
    cart_idx = np.argwhere(cart_mask.astype(bool))
    if len(cart_idx):
        cart_mm = cart_idx * np.asarray(spacing, dtype=np.float64)
        cart_ap_extent_mm = float(np.ptp(cart_mm[:, 0]))
        cart_ml_extent_mm = float(np.ptp(cart_mm[:, 2]))
        from scipy.ndimage import label as _cc
        _, n_cart_cc = _cc(cart_mask)
    else:
        cart_ap_extent_mm = cart_ml_extent_mm = 0.0
        n_cart_cc = 0

    t = time.time()
    print("  building meshes...")
    bone_mesh, cart_mesh = build_meshes(bone_mask, cart_mask, spacing)
    t_mesh = time.time() - t
    print(f"    bone_mesh: {bone_mesh.n_points} verts | "
          f"cart_mesh: {cart_mesh.n_points if cart_mesh else 0} verts")

    # For template-frame raycast, thickness is computed AFTER the anchor (it
    # needs the patient→template transform). For all other thickness methods,
    # compute on the patient bone mesh BEFORE the anchor.
    is_template_raycast = (config.thickness_method == "template_raycast")
    thickness_fn = get_thickness(config.thickness_method)

    if not is_template_raycast:
        t = time.time()
        print(f"  computing per-vertex thickness (method={config.thickness_method})...")
        thickness = thickness_fn(bone_mesh, bone_mask, cart_mask, spacing, config)
        t_thick = time.time() - t
        bone_mesh["thickness_mm"] = thickness
        pos = thickness > 0
        if pos.any():
            print(f"    n_with_thickness={int(pos.sum())}/{len(thickness)}; "
                  f"mean(>0)={thickness[pos].mean():.2f} mm; max={thickness.max():.2f} mm")
    else:
        thickness = np.zeros(bone_mesh.n_points, dtype=np.float32)
        bone_mesh["thickness_mm"] = thickness  # placeholder; viz uses template_thickness
        t_thick = 0.0

    # Save unregistered patient bone (in physical mm) BEFORE registration, if asked
    raw_mesh = bone_mesh.copy() if save_unregistered_mesh else None

    t = time.time()
    print(f"  anchor: {config.anchor}...")
    anchor_fn = get_anchor(config.anchor)
    aligned_pts, transform_meta = anchor_fn(
        bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, config,
    )
    t_reg = time.time() - t
    print(f"    rot={transform_meta.get('rot_deg', float('nan')):.2f}°, "
          f"trans={transform_meta.get('trans_mm', float('nan')):.2f} mm; "
          f"meta={ {k: v for k, v in transform_meta.items() if k not in ('translation_mm','sv_final')} }")

    print(f"  subch: {config.subch_method}...")
    subch_fn = get_subch(config.subch_method)
    subch_prob_v, is_subch = subch_fn(
        aligned_pts, template_mesh, cart_mask, spacing, bone_mesh, thickness, config,
    )
    bone_mesh["subch_prob"] = subch_prob_v
    bone_mesh["is_subchondral"] = is_subch
    print(f"    is_subchondral: {int(is_subch.sum())}/{len(is_subch)} verts")

    if is_template_raycast:
        t = time.time()
        print(f"  computing template-frame raycast thickness "
              f"(skip IDW; native template values)...")
        template_thickness = thickness_fn(
            bone_mesh, bone_mask, cart_mask, spacing, config,
            template_mesh=template_mesh, aligned_pts=aligned_pts,
        )
        # Quality metrics: compute ASSD between aligned patient subch verts and
        # template subch zone (registration QA — same calculation as IDW path)
        from scipy.spatial import cKDTree
        template_pts_arr = np.asarray(template_mesh.points)
        template_subch_mask_arr = np.asarray(
            template_mesh.point_data["subch_prob"]
        ) >= config.subch_threshold
        is_subch_p = subch_prob_v >= config.subch_threshold
        patient_subch_pts = aligned_pts[is_subch_p] if is_subch_p.any() else np.zeros((0, 3))
        template_subch_pts = template_pts_arr[template_subch_mask_arr]
        if len(patient_subch_pts) and len(template_subch_pts):
            patient_tree = cKDTree(patient_subch_pts)
            tmpl_subch_tree = cKDTree(template_subch_pts)
            d_t2p, _ = patient_tree.query(template_subch_pts, k=1)
            d_p2t, _ = tmpl_subch_tree.query(patient_subch_pts, k=1)
            assd = float((d_t2p.sum() + d_p2t.sum())
                         / max(len(d_t2p) + len(d_p2t), 1))
            mean_t2p = float(d_t2p.mean())
        else:
            assd = float("nan")
            mean_t2p = float("nan")
        quality = {
            "n_template_total": int(template_mesh.n_points),
            "n_template_subch": int(template_subch_mask_arr.sum()),
            "n_patient_subch": int(is_subch_p.sum()),
            "n_mutual": 0, "frac_mutual": 0.0,
            "mean_mapping_dist_mm_all_template": float("nan"),
            "mean_mapping_dist_mm_subch_only": mean_t2p,
            "median_mapping_dist_mm_subch_only": float("nan"),
            "max_mapping_dist_mm_subch_only": float("nan"),
            "assd_subch_mm": assd,
        }
        t_remap = time.time() - t
    else:
        t = time.time()
        print(f"  IDW remap (k={config.idw_k}, mutual_nn={config.idw_mutual_nn})...")
        template_thickness, quality = remap_thickness_to_template(
            template_mesh, aligned_pts, thickness, subch_prob_v,
            k=config.idw_k, mutual_nn=config.idw_mutual_nn,
            subch_threshold=config.subch_threshold,
        )
        t_remap = time.time() - t
    template_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= config.subch_threshold
    template_thickness_masked = template_thickness.copy()
    template_thickness_masked[~template_subch] = np.nan
    n_valid = int(np.isfinite(template_thickness_masked).sum())
    print(f"    template thickness: n_valid={n_valid}, "
          f"mean={np.nanmean(template_thickness_masked):.2f} mm, "
          f"max={np.nanmax(template_thickness_masked):.2f} mm")
    print(f"    mapping: mutual_NN={quality['frac_mutual']:.1%}, "
          f"subch_mean_dist={quality['mean_mapping_dist_mm_subch_only']:.2f} mm, "
          f"ASSD={quality['assd_subch_mm']:.2f} mm")
    print(f"  TIMING: load={t_load:.1f}s, mesh={t_mesh:.1f}s, thickness={t_thick:.1f}s, "
          f"register={t_reg:.1f}s, remap={t_remap:.1f}s, TOTAL={time.time()-t_total:.1f}s")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cid = case_id or Path(seg_path).stem.replace(".nii", "")
        vtk_path = out_dir / f"{cid}_{bone_name}_{side}_patient.vtk"
        npy_path = out_dir / f"{cid}_{bone_name}_{side}_template_thickness.npy"
        aligned_npy = out_dir / f"{cid}_{bone_name}_{side}_aligned_pts.npy"
        bone_mesh.save(str(vtk_path))
        np.save(npy_path, template_thickness_masked)
        np.save(aligned_npy, np.asarray(aligned_pts, dtype=np.float32))
        if raw_mesh is not None:
            raw_path = out_dir / f"{cid}_{bone_name}_{side}_patient_raw.vtk"
            raw_mesh.save(str(raw_path))
            print(f"    saved {raw_path}")
        print(f"    saved {vtk_path}")
        print(f"    saved {npy_path}")
        print(f"    saved {aligned_npy}")

    return {
        "bone_mesh": bone_mesh,
        "bone_mesh_raw": raw_mesh,
        "aligned_pts": aligned_pts,
        "template_thickness": template_thickness_masked,
        "quality": quality,
        "transform_meta": transform_meta,
        "cleanup_meta": cleanup_meta,
        "config": config.to_dict(),
        "cart_qa": {
            "cart_ap_extent_mm": cart_ap_extent_mm,
            "cart_ml_extent_mm": cart_ml_extent_mm,
            "n_cart_components": int(n_cart_cc),
        },
    }


def process_one_patient_levels(seg_path: Path, side: str, bone_name: str,
                               modality: str,
                               thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
                               config: PipelineConfig | None = None,
                               compute_levels: bool = True):
    """Multi-threshold variant of `process_one_patient` for the web viewer.

    Runs the expensive work ONCE (load + densify, mesh, per-vertex thickness,
    anchor, patient subch_prob), then loops `thresholds`, calling the cheap IDW
    remap + template-mask tail for each. Each level is an INDEPENDENT remap, so
    thickness values recompute with the threshold (the patient IDW source set
    `patient_subch_prob >= T` changes), not just the displayed extent.

    Returns:
        {
          "bone_mesh": pv.PolyData,            # patient bone mesh (thickness_mm)
          "template_mesh": pv.PolyData,        # fixed template (has subch_prob)
          "aligned_pts": np.ndarray,
          "thickness": np.ndarray,             # patient per-vert thickness
          "subch_prob_v": np.ndarray,          # patient per-vert subch prob
          "template_thickness_by_t": {T: np.ndarray(n_template,) masked NaN out-of-zone},
          "quality_by_t": {T: dict},
          "transform_meta": dict,
          "cleanup_meta": dict,
        }

    Only the IDW remap path (thickness_method in {edt, raycast, raycast_2d}) is
    supported; `template_raycast` (no IDW) is not used by the web viewer.
    """
    if config is None:
        config = PipelineConfig()
    if config.thickness_method == "template_raycast":
        raise NotImplementedError(
            "process_one_patient_levels does not support thickness_method="
            "'template_raycast' (no IDW remap). Use the web default 'raycast_2d'."
        )

    template_mesh = pv.read(str(TEMPLATE_PATHS[bone_name]))
    if "subch_prob" not in template_mesh.point_data:
        raise RuntimeError(f"Template {TEMPLATE_PATHS[bone_name]} missing 'subch_prob' field")

    bone_mask, cart_mask, spacing, _seg_native, _spacing_native, cleanup_meta = get_patient_masks(
        seg_path, side, modality, bone_name, config=config,
    )
    bone_mesh, _cart_mesh = build_meshes(bone_mask, cart_mask, spacing)

    # Coordinate-frame info so the web viewer can map a patient bone-mesh vertex
    # (built in the *flipped* physical-mm frame) back to the ORIGINAL NIfTI world
    # mm that the slice viewer (niivue) reports. bone_mesh.points = orig_voxel *
    # spacing_native in the flipped frame; undo the axis-2 (ML) flip then apply
    # the affine. (Densification only refines z sampling; physical mm is
    # preserved, so spacing_native is the correct divisor.)
    _nii = nib.load(str(seg_path))
    _affine = np.asarray(_nii.affine, dtype=np.float64)
    _frame = {
        "affine": _affine.tolist(),
        "spacing": [float(s) for s in _spacing_native],
        "n2": int(_nii.shape[2]),
        "flip2": bool(_empirical_flip_needed(
            _nii.get_fdata().astype(np.int16), _spacing_native, bone_label=3)),
    }

    thickness_fn = get_thickness(config.thickness_method)
    thickness = thickness_fn(bone_mesh, bone_mask, cart_mask, spacing, config)
    bone_mesh["thickness_mm"] = thickness

    anchor_fn = get_anchor(config.anchor)
    aligned_pts, transform_meta = anchor_fn(
        bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, config,
    )

    subch_fn = get_subch(config.subch_method)
    subch_prob_v, _is_subch = subch_fn(
        aligned_pts, template_mesh, cart_mask, spacing, bone_mesh, thickness, config,
    )
    bone_mesh["subch_prob"] = subch_prob_v

    template_subch_prob = np.asarray(template_mesh.point_data["subch_prob"])
    template_thickness_by_t: dict[float, np.ndarray] = {}
    quality_by_t: dict[float, dict] = {}
    # The web viewer recomputes every level via the RADIUS remap in
    # build_web_artifacts (cross-slice de-striped), so the study k-NN remap here
    # is pure waste for that path (~25 s/bone). Skip it with compute_levels=False;
    # only aligned_pts + subch_prob_v + thickness are needed downstream.
    if compute_levels:
        for t in thresholds:
            template_thickness, quality = remap_thickness_to_template(
                template_mesh, aligned_pts, thickness, subch_prob_v,
                k=config.idw_k, mutual_nn=config.idw_mutual_nn,
                subch_threshold=t,
            )
            masked = template_thickness.copy()
            masked[~(template_subch_prob >= t)] = np.nan
            template_thickness_by_t[t] = masked
            quality_by_t[t] = quality
            n_valid = int(np.isfinite(masked).sum())
            print(f"    [levels] T={t:.2f}: n_subch={n_valid}, "
                  f"mean={np.nanmean(masked) if n_valid else float('nan'):.2f} mm")

    return {
        "bone_mesh": bone_mesh,
        "template_mesh": template_mesh,
        "aligned_pts": aligned_pts,
        "thickness": thickness,
        "subch_prob_v": subch_prob_v,
        "template_thickness_by_t": template_thickness_by_t,
        "quality_by_t": quality_by_t,
        "transform_meta": transform_meta,
        "cleanup_meta": cleanup_meta,
        "frame": _frame,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_path", required=True, type=Path)
    p.add_argument("--side", choices=["RIGHT", "LEFT"], required=True)
    p.add_argument("--bone", choices=["femur", "tibia", "both"], default="both")
    p.add_argument("--modality", choices=["pd", "dess"], required=True)
    p.add_argument("--anchor", default="aniso_rigid")
    p.add_argument("--cart_cleanup", default="",
                   help="Comma-separated cleanup strategy names (e.g. drop_small_components,filter_near_bone_per_slice)")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--case_id", default=None)
    p.add_argument("--max_thickness_mm", type=float, default=6.0)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = None if args.dry_run else args.out_dir
    cart_cleanup = tuple(s for s in args.cart_cleanup.split(",") if s)
    config = PipelineConfig(anchor=args.anchor, cart_cleanup=cart_cleanup)
    bones = ["femur", "tibia"] if args.bone == "both" else [args.bone]
    for b in bones:
        print(f"\n=== {b} ({args.side}, {args.modality}) anchor={args.anchor} ===")
        process_one_patient(
            seg_path=args.seg_path,
            side=args.side,
            bone_name=b,
            modality=args.modality,
            config=config,
            max_thickness_mm=args.max_thickness_mm,
            out_dir=out_dir,
            case_id=args.case_id,
        )


if __name__ == "__main__":
    main()
