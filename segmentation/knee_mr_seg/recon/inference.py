"""
RECON Net inference wrapper: apply 4x depth super-resolution to a PD volume.

Input:  (40, 384, 384) PD segmentation mask — 3.0mm slice spacing
Output: (160, 384, 384) dense mask — 0.75mm equivalent spacing

Usage:
    from knee_mr_seg.recon import supersample_volume, visualize_result
    dense = supersample_volume(
        '/path/to/pd_mask.npy',
        '/path/to/best.pt',
        mode='4class'
    )
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

from .model import LabelReconUNet, sliding_window_inference

CLASS_NAMES_4 = ['Femur', 'Tibia', 'FC', 'TC']
CLASS_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']


def load_recon_model(
    checkpoint_path: str,
    device: str = 'cuda',
    seed: int = 42,
) -> 'LabelReconUNet':
    """Load and return a RECON model ready for inference.

    Call once per session and pass the returned model to supersample_volume
    to avoid repeated disk I/O and GPU transfers across a batch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_ch = ckpt.get('in_channels', 11)
    factor = ckpt.get('factor', 4)
    model = LabelReconUNet(in_channels=n_ch, factor=factor)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    print(f"[RECON] Model loaded: epoch {ckpt.get('epoch','?')}, "
          f"val_dice={ckpt.get('val_dice', 0):.4f}  device={device}")
    return model


def supersample_volume(
    mask_path: str | None,
    checkpoint_path: str,
    mode: str = '4class',
    device: str = 'cuda',
    save_path: str = None,
    mask_array: np.ndarray | None = None,
    seed: int = 42,
    model: 'LabelReconUNet | None' = None,
) -> np.ndarray:
    """
    Load a PD mask, resize to model size, run 4x depth SR, resize back.

    Args:
        mask_path: Path to .npy/.npz with shape (C, 40, H, W) or (40, H, W).
                   Pass None when supplying mask_array directly.
        checkpoint_path: Path to best.pt. Ignored when `model` is provided.
        mode: '4class' or '11class'.
        device: 'cuda' or 'cpu'.
        save_path: If given, save result as .npy.
        mask_array: Optional pre-loaded array (C, D, H, W) or (D, H, W).
                    When provided, mask_path is ignored — no file I/O.
        seed: Fixed random seed for deterministic CUDA inference (default 42).
        model: Pre-loaded LabelReconUNet (from load_recon_model). When provided,
               skips checkpoint loading — use this in batch loops to avoid
               repeated disk I/O and GPU transfers.

    Returns:
        dense: (C, D*factor, H, W) binary mask.
    """
    # ── Determinism (Upgrade C) ──
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Load mask ──
    if mask_array is not None:
        mask = mask_array.astype(np.float32)
    elif mask_path.endswith('.npz'):
        mask = np.load(mask_path)['data'].astype(np.float32)
    else:
        mask = np.load(mask_path).astype(np.float32)

    # ── Detect format and convert to one-hot ──
    # 11-class order: 0=BG,1=Femur,2=Tibia,3=Patella,4=FC,5=TC,6=PC,7=MedMen,8=LatMen,9=ACL,10=PCL
    # FOUR_CLASS_INDICES = [1, 2, 4, 5] → Femur, Tibia, FC, TC
    FOUR_CLASS_INDICES = [1, 2, 4, 5]

    if mask.ndim == 3:
        # Integer label map (40, H, W) — convert to 11-class one-hot first
        D, H, W = mask.shape
        labels = mask.astype(int)
        n_labels = int(labels.max()) + 1
        print(f"Integer label map detected: {n_labels} unique labels (incl. background)")
        mask = np.zeros((n_labels, D, H, W), dtype=np.float32)
        for c in range(n_labels):
            mask[c] = (labels == c).astype(np.float32)
        # Drop background channel (index 0)
        mask = mask[1:]  # (n_labels-1, D, H, W)

    C, D_sparse, H_orig, W_orig = mask.shape
    print(f"Input: ({C}, {D_sparse}, {H_orig}, {W_orig})")

    # ── Select channels for 4-class mode ──
    if mode == '4class':
        if C == 10 or C == 11:
            # 11-class system (with or without BG already stripped)
            # If BG is still there (C=11), indices are as-is; if stripped (C=10), shift by -1
            if C == 11:
                idx = FOUR_CLASS_INDICES
            else:
                idx = [i - 1 for i in FOUR_CLASS_INDICES]  # [0, 1, 3, 4]
            mask = mask[idx]
            print(f"Selected 4 channels from 11-class: Femur, Tibia, FC, TC → shape {mask.shape}")
        elif C == 4:
            print(f"Input already 4-channel, using as-is (assuming Femur, Tibia, FC, TC)")
        else:
            print(f"WARNING: unexpected {C} channels, using first 4")
            mask = mask[:4]

    C, D_sparse, H_orig, W_orig = mask.shape
    print(f"After channel selection: ({C}, {D_sparse}, {H_orig}, {W_orig})")

    # ── Resize each slice: (H_orig, W_orig) → (192, 192) ──
    MODEL_HW = 192
    if H_orig != MODEL_HW or W_orig != MODEL_HW:
        scale_h, scale_w = MODEL_HW / H_orig, MODEL_HW / W_orig
        # zoom: (C, D, H, W) → only scale H and W
        mask_resized = zoom(mask, (1, 1, scale_h, scale_w), order=0)
    else:
        mask_resized = mask

    print(f"Resized to model input: {mask_resized.shape}")

    # ── Load model (skip if caller passed a pre-loaded one) ──
    if model is None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        n_ch = ckpt.get('in_channels', C)
        factor = ckpt.get('factor', 4)
        model = LabelReconUNet(in_channels=n_ch, factor=factor)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval().to(device)
        print(f"Model loaded: epoch {ckpt.get('epoch','?')}, "
              f"val_dice={ckpt.get('val_dice', 0):.4f}")

    # ── Sliding window inference ──
    dense_small = sliding_window_inference(
        model, mask_resized,
        chunk_size=10, overlap=3,
        device=device, batch_size=2,
    )  # (C, D_sparse*4, 192, 192) float probabilities

    dense_small = (dense_small > 0.5).astype(np.float32)
    print(f"Model output: {dense_small.shape}")

    # ── Restore original slices at every factor-th position ──
    # The network fills all positions via learned interpolation.
    # Original sparse slices must be preserved exactly so that downstream
    # comparison between PD_dense and DESS is not contaminated by network
    # reconstruction error at known positions.
    factor_used = dense_small.shape[1] // D_sparse
    for i in range(D_sparse):
        dense_small[:, i * factor_used, :, :] = mask_resized[:, i, :, :]
    print(f"Restored {D_sparse} original slices at every {factor_used}-th position.")

    # ── Resize back: (192, 192) → (H_orig, W_orig) ──
    D_dense = dense_small.shape[1]
    if H_orig != MODEL_HW or W_orig != MODEL_HW:
        scale_h_back = H_orig / MODEL_HW
        scale_w_back = W_orig / MODEL_HW
        dense = zoom(dense_small, (1, 1, scale_h_back, scale_w_back), order=0)
    else:
        dense = dense_small

    print(f"Final output: {dense.shape}")  # (C, 160, 384, 384)

    if save_path:
        np.save(save_path, dense)
        print(f"Saved to {save_path}")

    return dense


def visualize_result(dense, n_slices=10, class_names=None, save_path=None):
    """
    Show evenly-spaced axial slices from the super-resolved volume.
    Green border on slices that correspond to original input positions (every 4th).
    """
    if class_names is None:
        class_names = CLASS_NAMES_4
    C, D, H, W = dense.shape
    indices = np.linspace(0, D - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices * 2.2, 2.5))
    if n_slices == 1:
        axes = [axes]

    for col, d_idx in enumerate(indices):
        ax = axes[col]
        canvas = np.ones((H, W, 3), dtype=np.float32) * 0.1
        for c in range(C):
            rgb = np.array(mcolors.to_rgb(CLASS_COLORS[c % len(CLASS_COLORS)]))
            m = dense[c, d_idx] > 0.5
            for ch in range(3):
                canvas[:, :, ch] = np.where(m, rgb[ch], canvas[:, :, ch])

        ax.imshow(np.clip(canvas, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])

        is_known = (d_idx % 4 == 0)
        if is_known:
            for spine in ax.spines.values():
                spine.set_edgecolor('lime')
                spine.set_linewidth(3)

        ax.set_title(f'{d_idx}{"*" if is_known else ""}', fontsize=8)

    # Legend
    patches = [plt.Line2D([0], [0], color=CLASS_COLORS[i % len(CLASS_COLORS)],
               linewidth=6, label=class_names[i]) for i in range(C)]
    fig.legend(handles=patches, loc='lower center', ncol=C, fontsize=8)
    plt.suptitle('4x Depth Super-Resolution  (* = original input slice)', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_consecutive(dense, start=60, n=16, class_names=None, save_path=None):
    """Show consecutive slices to see smooth interpolation between known slices."""
    if class_names is None:
        class_names = CLASS_NAMES_4
    C, D, H, W = dense.shape
    indices = range(start, min(start + n, D))

    fig, axes = plt.subplots(1, len(indices), figsize=(len(indices) * 1.8, 2.2))
    for col, d_idx in enumerate(indices):
        ax = axes[col]
        canvas = np.ones((H, W, 3), dtype=np.float32) * 0.1
        for c in range(C):
            rgb = np.array(mcolors.to_rgb(CLASS_COLORS[c % len(CLASS_COLORS)]))
            m = dense[c, d_idx] > 0.5
            for ch in range(3):
                canvas[:, :, ch] = np.where(m, rgb[ch], canvas[:, :, ch])
        ax.imshow(np.clip(canvas, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        is_known = (d_idx % 4 == 0)
        if is_known:
            for spine in ax.spines.values():
                spine.set_edgecolor('lime')
                spine.set_linewidth(3)
        ax.set_title(f'{d_idx}', fontsize=7, color='lime' if is_known else 'white')
    plt.suptitle('Consecutive slices (green = original input)', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
