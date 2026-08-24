"""World-coordinate trilinear resampling of nnU-Net softmax probability volumes
into a shared DESS voxel grid.

Inputs:
  * Per-class softmax volumes from Dataset211 (PD SAG) in PD_SAG voxel space.
  * Per-class softmax volumes from Dataset212 (PD COR) in PD_COR voxel space.
  * The reference DESS image's affine + shape.

Output:
  * (8, H, W, D) float32 tensor where channels 0..3 = SAG class probs in DESS
    grid (Femur, FC, Tibia, TC), channels 4..7 = COR class probs in DESS grid.
    Background channel is dropped (recovered as 1 - sum); this keeps the
    network's input compact and avoids redundant 9th/10th channels.

The mapping is exact world-coordinate trilinear interpolation:

    For each DESS voxel index (i, j, k):
        world_xyz = DESS_affine @ (i, j, k, 1)
        sag_ijk   = inv(SAG_affine) @ (world_xyz, 1)        # fractional
        sample SAG prob at sag_ijk via trilinear interp
    same for COR.

We implement this with a single torch.nn.functional.grid_sample call per plane
(per-channel batched) for speed on GPU.

DESS label scheme used downstream (matches knee_mr_cartilage.DESS_LABELS):
    0 = background, 1 = Femur bone, 2 = Femur cart (FC),
    3 = Tibia bone, 4 = Tibia cart (TC).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Fusion target label scheme — identical to knee_mr_cartilage.DESS_LABELS so
# the wrapper can pass the saved NIfTI straight into process_one_patient
# with modality='dess'.
DESS_FUSION_LABELS = {
    "background": 0,
    "Femur": 1,
    "Femur cartilage": 2,  # FC
    "Tibia": 3,
    "Tibia cartilage": 4,  # TC
}
N_FUSION_CLASSES = 5  # BG, Fem, FC, Tib, TC
N_FUSION_FG = 4       # Fem, FC, Tib, TC (BG dropped from input channels)


def _to_torch(arr: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if device is not None:
        t = t.to(device)
    return t


def _build_dess_grid_in_src_voxel_space(
    dess_shape: tuple[int, int, int],
    dess_affine: np.ndarray,
    src_shape: tuple[int, int, int],
    src_affine: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return a (1, D_dess, H_dess, W_dess, 3) grid in normalised src-voxel
    coords suitable for ``F.grid_sample`` on the source volume.

    grid_sample expects coordinates in [-1, 1] where -1 = index 0 and
    +1 = index (N - 1). The grid's last dim ordering is (x, y, z) where
    x indexes the LAST tensor dim (width), z the first spatial dim.

    Our convention: a tensor of shape (1, C, D, H, W) sampled by a grid of
    shape (1, D_out, H_out, W_out, 3) where grid[..., 0] -> W axis (= src
    voxel axis 2), grid[..., 1] -> H axis (= src voxel axis 1),
    grid[..., 2] -> D axis (= src voxel axis 0). I.e. (x_norm, y_norm, z_norm)
    must correspond to src voxel axes (2, 1, 0) respectively.
    """
    Hd, Wd, Dd = dess_shape  # we keep the same axis ordering as the underlying
                             # numpy arrays. We'll arrange the grid_sample input
                             # tensor to match below.
    # Build a (Hd, Wd, Dd, 4) tensor of homogeneous DESS voxel indices.
    ii, jj, kk = torch.meshgrid(
        torch.arange(Hd, device=device, dtype=torch.float32),
        torch.arange(Wd, device=device, dtype=torch.float32),
        torch.arange(Dd, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(ii)
    dess_ijk_h = torch.stack([ii, jj, kk, ones], dim=-1)  # (Hd, Wd, Dd, 4)

    A_dess = torch.from_numpy(np.ascontiguousarray(dess_affine)).to(device=device, dtype=torch.float32)
    A_src = torch.from_numpy(np.ascontiguousarray(src_affine)).to(device=device, dtype=torch.float32)
    M = torch.linalg.inv(A_src) @ A_dess  # 4x4

    # Apply to every DESS voxel
    flat = dess_ijk_h.reshape(-1, 4)
    src_ijk = (flat @ M.T)[:, :3].reshape(Hd, Wd, Dd, 3)  # (Hd, Wd, Dd, 3) src voxel coords (i, j, k)

    # Normalise to [-1, 1] using the SRC volume's shape per axis.
    Hs, Ws, Ds = src_shape
    denom = torch.tensor([max(Hs - 1, 1), max(Ws - 1, 1), max(Ds - 1, 1)],
                         device=device, dtype=torch.float32)
    norm = (src_ijk / denom) * 2.0 - 1.0  # (Hd, Wd, Dd, 3) with last dim = (i_n, j_n, k_n)

    # grid_sample's last-dim ordering is (x=W, y=H, z=D) which corresponds to
    # src voxel axes (2, 1, 0). Reorder.
    grid = torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], dim=-1)  # (Hd, Wd, Dd, 3)

    # grid_sample needs the grid in (N, D_out, H_out, W_out, 3) — but our
    # src tensor will be (N, C, D, H, W). We must arrange so that the OUTPUT
    # tensor (which we'll reinterpret as (C, Hd, Wd, Dd)) has axes consistent
    # with the DESS array. The safest convention: build src as
    # (1, C, src_axis0, src_axis1, src_axis2) -> i.e. (1, C, Hs, Ws, Ds),
    # then ``grid_sample`` treats the THREE spatial dims as (D, H, W) =
    # (Hs, Ws, Ds). Our grid's last-dim ordering must be (x=W_src=Ds,
    # y=H_src=Ws, z=D_src=Hs) = (k, j, i). That is exactly what we built above.
    #
    # The output will be shape (1, C, Hd, Wd, Dd) iff the grid is
    # (1, Hd, Wd, Dd, 3) — yes. So we add a leading batch dim.
    return grid.unsqueeze(0)


def _resample_prob_volume(
    probs_src: np.ndarray,
    src_affine: np.ndarray,
    dess_shape: tuple[int, int, int],
    dess_affine: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Resample a (C, Hs, Ws, Ds) probability volume into DESS grid.

    Returns a (C, Hd, Wd, Dd) float32 tensor on ``device`` with values in [0, 1].
    Out-of-bounds DESS voxels get 0 (padding_mode='zeros').
    """
    C = probs_src.shape[0]
    src_shape = tuple(probs_src.shape[1:])
    src_tensor = _to_torch(probs_src.astype(np.float32), device=device).unsqueeze(0)  # (1, C, Hs, Ws, Ds)
    grid = _build_dess_grid_in_src_voxel_space(
        dess_shape, dess_affine, src_shape, src_affine, device=device,
    )  # (1, Hd, Wd, Dd, 3)
    # F.grid_sample with mode='bilinear' is trilinear for 5D tensors.
    out = F.grid_sample(
        src_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (1, C, Hd, Wd, Dd)
    return out[0]


def resample_probs_to_dess_grid(
    sag_probs: np.ndarray,
    sag_affine: np.ndarray,
    cor_probs: np.ndarray,
    cor_affine: np.ndarray,
    dess_shape: tuple[int, int, int],
    dess_affine: np.ndarray,
    device: torch.device | str = "cpu",
    drop_background: bool = True,
) -> np.ndarray:
    """Resample SAG + COR softmax probability volumes to the DESS voxel grid.

    Parameters
    ----------
    sag_probs : (C, Hs, Ws, Ds) float
        Per-class softmax probabilities from Dataset211 in PD SAG voxel space.
        C is 5 (BG, Fem, FC, Tib, TC) for the v7.0 4-class scheme.
    sag_affine : (4, 4) float
        PD SAG voxel-to-world affine (typically from the PD_SAG.nii.gz header).
    cor_probs, cor_affine : analogous from Dataset212.
    dess_shape : tuple
        Target DESS grid shape (H, W, D).
    dess_affine : (4, 4) float
        DESS voxel-to-world affine.
    device : torch device — use 'cuda' for batch caching, 'cpu' is fine for one-off.
    drop_background : if True (default), the BG channel is excluded from the
        returned array. Output shape is (8, H, W, D); first 4 are SAG class
        probs (Fem, FC, Tib, TC), last 4 are COR class probs.

    Returns
    -------
    np.ndarray of shape (8, H, W, D) float32 (or (10, H, W, D) if not dropping BG).
    """
    dev = torch.device(device)
    sag_re = _resample_prob_volume(sag_probs, sag_affine, dess_shape, dess_affine, dev)  # (C, H, W, D)
    cor_re = _resample_prob_volume(cor_probs, cor_affine, dess_shape, dess_affine, dev)
    if drop_background:
        # Channel 0 is BG by nnU-Net convention.
        sag_re = sag_re[1:]
        cor_re = cor_re[1:]
    fused = torch.cat([sag_re, cor_re], dim=0).clamp_(0.0, 1.0).cpu().numpy().astype(np.float32)
    return fused


def save_dess_grid_mask_nifti(
    mask: np.ndarray,
    dess_affine: np.ndarray,
    out_path: Path | str,
    dess_header: nib.Nifti1Header | None = None,
) -> Path:
    """Save a 4-class DESS-grid argmax mask as a NIfTI with the DESS affine.

    Labels match DESS_FUSION_LABELS (= knee_mr_cartilage.DESS_LABELS): 1 Fem,
    2 FC, 3 Tib, 4 TC. The downstream cartilage pipeline expects these exact
    integer labels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dess_header is None:
        header = nib.Nifti1Header()
    else:
        header = dess_header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), dess_affine, header), str(out_path))
    return out_path
