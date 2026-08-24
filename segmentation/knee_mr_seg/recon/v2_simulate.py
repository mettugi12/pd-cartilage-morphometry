"""
RECONv2 forward model: simulate clinical PD acquisitions from a dense DESS mask.

Motivation
----------
Current RECON-Net takes only **sparse PD SAG** (40 slices, 3 mm thick) and
upsamples depth 4x to a DESS-equivalent isotropic 0.7 mm volume. This is a
single-axis depth-SR problem that cannot reconstruct what the SAG plane never
saw — features in the medial-lateral direction that are sub-3 mm. Adding a
**PD COR** view (slice axis = anterior-posterior, also 2.0–3.6 mm thick) gives
a perpendicular high-resolution sample of the cartilage volume, leaving only
the AP-axis at thick spacing in COR and ML-axis at thick spacing in SAG;
both views share S-I in-plane at high resolution. Fusing them should
reconstruct the dense isotropic DESS volume more reliably.

To test feasibility before building real model, we simulate paired
(SAG_thick, COR_thick) inputs from existing dense DESS GT masks
(`DESS_gt_43`, n=43; plus `DESS_not_revised`, n=66).

Spacing reality (n=12 NAS sample, knee-mri-lesion-bbox dataset 2018):
- SAG: in-plane 0.20–0.36 mm, slice 2.0–3.6 mm (median 2.5), slice axis = R-L
- COR: in-plane 0.18–0.31 mm, slice 2.0–3.6 mm (median 2.5), slice axis = A-P
- DESS GT: 0.365 × 0.365 × 0.7 mm, slice axis = R-L

Slab averaging vs subsampling
-----------------------------
A clinical 2.5 mm PD slice is NOT one DESS voxel sampled at every 4th index.
It is a **slab integral** of all tissue spanning that 2.5 mm thickness, i.e.
~3-4 DESS voxels averaged together. For a binary segmentation:

  hard simulation  : slab_mask = ANY(DESS_voxels in slab) (logical OR)
  soft simulation  : slab_mask = MEAN(DESS_voxels in slab) (partial volume)

`soft` produces non-binary probabilities matching the real partial-volume
effect at slab edges; `hard` is a coarser approximation. Default = `soft`.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

ASSUMED_DESS_VOXEL_MM = (0.365, 0.365, 0.700)
DEFAULT_PD_SLICE_THICKNESS_MM = 2.5  # median of NAS bbox sample
DEFAULT_PD_INPLANE_MM = 0.35  # leave inplane near native; resampling not strictly needed


def _slab_average_axis(
    mask: np.ndarray,
    axis: int,
    in_voxel_mm: float,
    out_voxel_mm: float,
    mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    """Resample `mask` along `axis` from in_voxel_mm to out_voxel_mm spacing.

    Each output voxel covers `out_voxel_mm` of physical extent and aggregates
    all input voxels within that extent. With out > in, this is slab averaging
    (downsampling); with out < in, this is up-sampling (replicates input).

    The output length is round(N_in * in/out). Output voxel k spans physical
    range [k*out, (k+1)*out] in mm; the input voxels overlapping this range
    contribute weighted by their overlap fraction.

    `mask` may be 3-D (D, H, W) or 4-D (C, D, H, W) etc. Operates on `axis`.
    """
    if abs(in_voxel_mm - out_voxel_mm) < 1e-6:
        return mask.copy()

    n_in = mask.shape[axis]
    total_mm = n_in * in_voxel_mm
    n_out = max(1, int(round(total_mm / out_voxel_mm)))

    # Output voxel k physical range:  [k*out_voxel_mm, (k+1)*out_voxel_mm]
    # Input voxel j physical range:   [j*in_voxel_mm,  (j+1)*in_voxel_mm]
    # Overlap = max(0, min(end_o,end_i) - max(start_o,start_i))
    out_starts = np.arange(n_out) * out_voxel_mm
    out_ends = out_starts + out_voxel_mm

    # Build a sparse-ish weight matrix W (n_out, n_in) — clamped to nonzero rows.
    # For typical out~3.5x in, each output row has ~4 nonzero entries; we just
    # iterate it with a small Python loop and rely on vectorized accumulation.
    out_shape = list(mask.shape)
    out_shape[axis] = n_out
    out = np.zeros(out_shape, dtype=np.float32)
    # Move axis to front for easy slicing
    mask_m = np.moveaxis(mask, axis, 0)
    out_m = np.moveaxis(out, axis, 0)

    for k in range(n_out):
        s_o, e_o = out_starts[k], out_ends[k]
        j_lo = int(np.floor(s_o / in_voxel_mm))
        j_hi = int(np.ceil(e_o / in_voxel_mm))
        j_lo = max(0, j_lo)
        j_hi = min(n_in, j_hi)
        if j_hi <= j_lo:
            continue
        w_total = 0.0
        acc = np.zeros_like(mask_m[0], dtype=np.float32)
        for j in range(j_lo, j_hi):
            s_i, e_i = j * in_voxel_mm, (j + 1) * in_voxel_mm
            ov = max(0.0, min(e_o, e_i) - max(s_o, s_i))
            if ov <= 0:
                continue
            acc += ov * mask_m[j].astype(np.float32)
            w_total += ov
        if w_total > 0:
            slab = acc / w_total
            if mode == "hard":
                slab = (slab >= 0.5).astype(np.float32)
            out_m[k] = slab

    out = np.moveaxis(out_m, 0, axis)
    return out


def simulate_pd_view(
    dess_mask: np.ndarray,
    dess_voxel_mm: Tuple[float, float, float] = ASSUMED_DESS_VOXEL_MM,
    plane: Literal["SAG", "COR"] = "SAG",
    slice_thickness_mm: float = DEFAULT_PD_SLICE_THICKNESS_MM,
    inplane_mm: float | None = DEFAULT_PD_INPLANE_MM,
    mode: Literal["soft", "hard"] = "soft",
    dess_axes: Tuple[str, str, str] = ("AP", "SI", "ML"),
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Simulate a thick-slab PD acquisition from a dense DESS one-hot mask.

    Args:
        dess_mask: (C, D0, D1, D2) one-hot or label-prob, or (D0, D1, D2)
                   integer label map.
        dess_voxel_mm: voxel spacing of the dense DESS mask (in DESS axis order).
        plane: 'SAG' or 'COR' — defines which anatomical axis becomes the
               thick-slice axis. SAG slabs along ML, COR slabs along AP.
        slice_thickness_mm: PD slab thickness.
        inplane_mm: PD in-plane voxel size; if None, leaves DESS in-plane
                    spacing untouched. The default is ~native DESS, so leaving
                    in-plane untouched is normally fine.
        mode: 'soft' partial-volume average (probabilities), or 'hard' OR.
        dess_axes: anatomical interpretation of the 3 spatial axes of dess_mask.
                   Default ('AP', 'SI', 'ML') matches the DESS_gt_43 NIfTIs:
                   axis 0 = anterior-posterior (in-plane row),
                   axis 1 = superior-inferior (in-plane col),
                   axis 2 = medial-lateral (slice axis, 0.7 mm).

    Returns:
        (sim_mask, sim_voxel_mm) — same channel layout (added if needed),
        downsampled along the slab axis, optionally inplane-downsampled.
    """
    arr = dess_mask
    if arr.ndim == 3:
        # integer label map → one-hot (drop BG)
        labels = arr.astype(int)
        n_lab = int(labels.max()) + 1
        oh = np.zeros((n_lab,) + labels.shape, dtype=np.float32)
        for c in range(n_lab):
            oh[c] = (labels == c).astype(np.float32)
        arr = oh
        added_channel = True
    else:
        added_channel = False

    if arr.ndim != 4:
        raise ValueError(f"Expected (C, D0, D1, D2) or (D0, D1, D2), got {arr.shape}")

    target_axis_anat = "ML" if plane == "SAG" else "AP"
    if target_axis_anat not in dess_axes:
        raise ValueError(
            f"DESS axes {dess_axes} do not contain {target_axis_anat}; "
            "either remap dess_axes or transpose mask first."
        )
    slab_axis_in_data = dess_axes.index(target_axis_anat) + 1  # +1 for channel dim
    slab_in_mm = dess_voxel_mm[slab_axis_in_data - 1]

    out = _slab_average_axis(
        arr,
        axis=slab_axis_in_data,
        in_voxel_mm=slab_in_mm,
        out_voxel_mm=slice_thickness_mm,
        mode=mode,
    )
    out_voxel = list(dess_voxel_mm)
    out_voxel[slab_axis_in_data - 1] = slice_thickness_mm

    # Optional in-plane downsampling
    if inplane_mm is not None:
        for ax_anat, ax_idx in zip(dess_axes, range(3)):
            if ax_anat == target_axis_anat:
                continue
            in_mm = out_voxel[ax_idx]
            if abs(in_mm - inplane_mm) > 1e-6:
                out = _slab_average_axis(
                    out,
                    axis=ax_idx + 1,
                    in_voxel_mm=in_mm,
                    out_voxel_mm=inplane_mm,
                    mode=mode,
                )
                out_voxel[ax_idx] = inplane_mm

    if added_channel and out.shape[0] > 1:
        # restore label-map by argmax (only meaningful in 'hard' mode)
        if mode == "hard":
            out = np.argmax(out, axis=0).astype(np.uint8)
    return out, tuple(out_voxel)


def perturb_rigid(
    view_chw: np.ndarray,
    voxel_mm: Tuple[float, float, float],
    trans_mm: float,
    rot_deg: float,
    seed: int = 0,
) -> np.ndarray:
    """Apply a known rigid perturbation in physical (mm) space about the volume center.

    Used by the feasibility test to simulate misregistration between SAG and COR
    views before image-based registration is wired in. Rotation axis and
    translation direction are random unit vectors; magnitudes are the args.
    Sampling at order=1 (linear) preserves partial-volume probabilities.

    Args:
        view_chw: (C, D0, D1, D2) one-hot or partial-volume mask on a regular grid.
        voxel_mm: (mm, mm, mm) per spatial axis of `view_chw`.
        trans_mm: scalar magnitude of translation in mm.
        rot_deg: scalar magnitude of rotation in degrees.
        seed: rng seed (reproducible per case+level).

    Returns:
        same-shape array, perturbed.
    """
    from scipy.ndimage import affine_transform  # local import keeps cold-start cheap

    rng = np.random.default_rng(seed)
    if rot_deg > 0:
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        theta = np.deg2rad(rot_deg)
        K = np.array(
            [[0.0, -axis[2], axis[1]],
             [axis[2], 0.0, -axis[0]],
             [-axis[1], axis[0], 0.0]]
        )
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    else:
        R = np.eye(3)
    if trans_mm > 0:
        t_axis = rng.normal(size=3)
        t_axis = t_axis / np.linalg.norm(t_axis)
        t_mm = t_axis * trans_mm
    else:
        t_mm = np.zeros(3)

    # scipy.ndimage.affine_transform: out[k] = in[M @ k + offset], with k in voxels.
    # Want a rigid in mm about the volume center:
    #   p_in = R (p_out - c_mm) + c_mm + t_mm
    # In voxels (D = diag(voxel_mm)):
    #   M_vox = D^-1 R D
    #   offset = c_vox - M_vox c_vox + D^-1 t_mm
    D = np.diag(voxel_mm)
    Dinv = np.diag([1.0 / v for v in voxel_mm])
    M_vox = Dinv @ R @ D
    shape = view_chw.shape[1:]
    c_vox = np.array([(s - 1) / 2.0 for s in shape])
    offset = c_vox - M_vox @ c_vox + Dinv @ t_mm

    out = np.zeros_like(view_chw)
    for c in range(view_chw.shape[0]):
        out[c] = affine_transform(
            view_chw[c], M_vox, offset=offset,
            order=1, mode="constant", cval=0.0,
        )
    return out


def simulate_pd_pair(
    dess_mask: np.ndarray,
    dess_voxel_mm: Tuple[float, float, float] = ASSUMED_DESS_VOXEL_MM,
    sag_slice_mm: float = DEFAULT_PD_SLICE_THICKNESS_MM,
    cor_slice_mm: float = DEFAULT_PD_SLICE_THICKNESS_MM,
    mode: Literal["soft", "hard"] = "soft",
    dess_axes: Tuple[str, str, str] = ("AP", "SI", "ML"),
) -> dict:
    """Simulate paired (SAG_thick, COR_thick) PD inputs from one DESS mask."""
    sag, sag_v = simulate_pd_view(
        dess_mask, dess_voxel_mm, plane="SAG",
        slice_thickness_mm=sag_slice_mm, inplane_mm=None,
        mode=mode, dess_axes=dess_axes,
    )
    cor, cor_v = simulate_pd_view(
        dess_mask, dess_voxel_mm, plane="COR",
        slice_thickness_mm=cor_slice_mm, inplane_mm=None,
        mode=mode, dess_axes=dess_axes,
    )
    return {
        "sag_mask": sag, "sag_voxel_mm": sag_v,
        "cor_mask": cor, "cor_voxel_mm": cor_v,
        "dess_voxel_mm": dess_voxel_mm,
    }
