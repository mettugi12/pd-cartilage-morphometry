"""
RECON Net: Patch-based Depth Super-Resolution for Knee MR Label Completion.

True 4x depth SR: input N sparse slices → output N*4 dense slices.
Sliding-window inference with overlap averaging for full volume reconstruction.

Designed for Colab L4 (24GB VRAM) with mixed precision + gradient accumulation.

=============================================================================
USAGE
=============================================================================

1. Training (11-class DESS):
    python -m knee_mr_seg.recon.model --data_dir /path/to/data --metadata_path /path/to/metadata.json \
        --output_dir /path/to/output --mode 11class

2. Training (4-class OAI-ZIB compatible):
    python -m knee_mr_seg.recon.model --data_dir /path/to/data --metadata_path /path/to/metadata.json \
        --output_dir /path/to/output --mode 4class

3. From Python / Colab:
    from knee_mr_seg.recon.model import train_recon
    train_recon(
        data_dir='/path/to/data',
        metadata_path='/path/to/metadata.json',
        output_dir='/path/to/output',
        mode='11class',
        chunk_size=10,       # sparse slices per patch
        overlap=3,           # overlap in sparse slices
        chunks_per_volume=8, # random chunks per volume per epoch
    )

4. Inference (sliding window):
    from knee_mr_seg.recon.model import sliding_window_inference, LabelReconUNet
    import torch

    model = LabelReconUNet(in_channels=11, factor=4)
    ckpt = torch.load('best.pt')
    model.load_state_dict(ckpt['model_state_dict'])

    sparse_vol = ...  # (C, 40, H, W) sparse input
    dense_vol = sliding_window_inference(
        model, sparse_vol, chunk_size=10, overlap=3, device='cuda'
    )  # → (C, 160, H, W)

5. Quick sanity test:
    from knee_mr_seg.recon.model import quick_test
    quick_test()

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import json
import csv
import argparse
import time

try:
    from tqdm import tqdm
except ImportError:
    print("[INFO] tqdm not found. Install with: pip install tqdm")
    def tqdm(iterable, *args, **kwargs):
        return iterable


# =============================================================================
# DATASET
# =============================================================================

# Channel mapping for 4-class mode
# 11-class order: 0=bg, 1=Femur, 2=Tibia, 3=Patella, 4=FC, 5=TC, 6=PC,
#                 7=MedMen, 8=LatMen, 9=ACL, 10=PCL
FOUR_CLASS_INDICES = [1, 2, 4, 5]  # Femur, Tibia, FC, TC


class ReconDataset(Dataset):
    """
    Patch-based dataset for depth super-resolution.

    Each sample is a random depth chunk:
      - Input:  (C+1, chunk_size, H, W)  — N sparse slices + positional encoding channel
      - Target: (C, chunk_size * factor, H, W) — corresponding dense slices

    The +1 channel encodes normalized depth position (0→1) so the model knows
    where in the volume each chunk comes from.
    """

    def __init__(self, data_dir: str, case_ids: List[str],
                 metadata: Dict, mode: str = '11class',
                 undersample_factor: int = 4,
                 chunk_size: int = 10, chunks_per_volume: int = 8,
                 augment: bool = False):
        self.data_dir = data_dir
        self.case_ids = case_ids
        self.metadata = metadata
        self.mode = mode
        self.factor = undersample_factor
        self.chunk_size = chunk_size
        self.chunks_per_volume = chunks_per_volume
        self.augment = augment

    def __len__(self):
        return len(self.case_ids) * self.chunks_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.chunks_per_volume
        case_id = self.case_ids[vol_idx]
        meta = self.metadata.get(case_id, {})

        # Load dense mask (C, D_dense, H, W)
        npz_path = os.path.join(self.data_dir, f"{case_id}.npz")
        npy_path = os.path.join(self.data_dir, f"{case_id}.npy")
        if os.path.exists(npz_path):
            mask = np.load(npz_path)['data'].astype(np.float32)
        else:
            mask = np.load(npy_path).astype(np.float32)

        # Handle partial data (OAI-ZIB 4ch -> 11ch)
        is_partial = meta.get('is_partial', False)
        if is_partial and mask.shape[0] == 4:
            full_mask = np.zeros((11,) + mask.shape[1:], dtype=np.float32)
            full_mask[1] = mask[0]
            full_mask[4] = mask[1]
            full_mask[2] = mask[2]
            full_mask[5] = mask[3]
            mask = full_mask

        # Crop H,W to (192, 192) — divisible by 8 (3 levels of stride-2)
        # 192/2/2/2 = 24, clean at all encoder levels
        # Keep all 160 depth slices (divisible by 4)
        if mask.shape[2] == 200 and mask.shape[3] == 200:
            mask = mask[:, :, 4:196, 4:196]  # (C, D, 192, 192)

        # Select channels
        if self.mode == '4class':
            mask = mask[FOUR_CLASS_INDICES]

        C, D_dense, H, W = mask.shape
        # D_dense should be 160, sparse = 40
        D_sparse = D_dense // self.factor

        # Create sparse volume: every factor-th slice
        sparse = mask[:, ::self.factor]  # (C, D_sparse, H, W)

        # Random chunk start (in sparse indices)
        max_start = D_sparse - self.chunk_size
        if max_start <= 0:
            start_s = 0
            cs = D_sparse  # use all slices if chunk_size > D_sparse
        else:
            start_s = np.random.randint(0, max_start + 1)
            cs = self.chunk_size

        end_s = start_s + cs

        # Extract sparse chunk and corresponding dense chunk
        sparse_chunk = sparse[:, start_s:end_s]           # (C, cs, H, W)
        dense_start = start_s * self.factor
        dense_end = end_s * self.factor
        dense_chunk = mask[:, dense_start:dense_end]       # (C, cs*factor, H, W)

        # Positional encoding channel: normalized depth position
        # Tells the model where this chunk is in the full volume
        pos = np.linspace(start_s / D_sparse, (end_s - 1) / D_sparse,
                          cs, dtype=np.float32)
        pos_ch = np.broadcast_to(pos[None, :, None, None],
                                 (1, cs, H, W)).copy()  # (1, cs, H, W)

        # Input: (C+1, cs, H, W)
        inp = np.concatenate([sparse_chunk, pos_ch], axis=0)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:  # L/R flip
                inp = inp[:, :, :, ::-1].copy()
                dense_chunk = dense_chunk[:, :, :, ::-1].copy()
            if np.random.rand() > 0.5:  # A/P flip
                inp = inp[:, :, ::-1, :].copy()
                dense_chunk = dense_chunk[:, :, ::-1, :].copy()

        return {
            'input': torch.from_numpy(inp).float(),
            'target': torch.from_numpy(dense_chunk).float(),
            'case_id': case_id,
            'start_sparse': start_s,
        }


# =============================================================================
# MODEL: LabelReconUNet (Asymmetric Depth Super-Resolution)
# =============================================================================

class ResBlock3D(nn.Module):
    """3D Residual Block: Conv-GN-ReLU-Conv-GN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        g = min(32, channels // 4) if channels >= 4 else 1
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(g, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class DepthUpsample(nn.Module):
    """Upsample depth by a given factor using learned convolution."""

    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.factor = factor
        # PixelShuffle-style: expand channels then reshape
        self.conv = nn.Conv3d(channels, channels * factor, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(32, channels // 4) if channels >= 4 else 1, channels)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = self.conv(x)  # (B, C*factor, D, H, W)
        # Reshape: (B, C, factor, D, H, W) → (B, C, D*factor, H, W)
        x = x.view(B, C, self.factor, D, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, C, D, factor, H, W)
        x = x.view(B, C, D * self.factor, H, W)
        return self.norm(x)


class LabelReconUNet(nn.Module):
    """
    Asymmetric 3D U-Net for depth super-resolution.

    Input:  (B, C_in+1, D_sparse, H, W) — sparse slices + positional channel
    Output: (B, C_in, D_sparse * factor, H, W) — dense label probabilities

    The encoder operates at sparse depth resolution, downsampling only H and W.
    The decoder upsamples H,W back AND upsamples depth by `factor` using learned
    depth-shuffle layers.
    """

    def __init__(self, in_channels: int, factor: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.factor = factor
        cin = in_channels + 1  # +1 for positional channel

        # --- Encoder (downsample H,W only, keep depth intact) ---
        self.enc0 = nn.Sequential(
            nn.Conv3d(cin, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            ResBlock3D(32), ResBlock3D(32),
        )
        # Stride (1,2,2): keep depth, halve H,W
        self.down1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3),
                               stride=(1, 2, 2), padding=1)
        self.enc1 = nn.Sequential(nn.GroupNorm(16, 64), nn.ReLU(inplace=True),
                                  ResBlock3D(64), ResBlock3D(64))

        self.down2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3),
                               stride=(1, 2, 2), padding=1)
        self.enc2 = nn.Sequential(nn.GroupNorm(32, 128), nn.ReLU(inplace=True),
                                  ResBlock3D(128), ResBlock3D(128))

        self.down3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3),
                               stride=(1, 2, 2), padding=1)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(32, 256), nn.ReLU(inplace=True),
            ResBlock3D(256), ResBlock3D(256),
        )

        # --- Decoder (upsample H,W; depth upsample at the end) ---
        # Upsample H,W by 2 (keep depth)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(256, 128, 1))
        self.dec3 = nn.Sequential(ResBlock3D(256), ResBlock3D(256),
                                  nn.Conv3d(256, 128, 1))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, 1))
        self.dec2 = nn.Sequential(ResBlock3D(128), ResBlock3D(128),
                                  nn.Conv3d(128, 64, 1))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, 1))
        self.dec1 = nn.Sequential(ResBlock3D(64), ResBlock3D(64),
                                  nn.Conv3d(64, 32, 1))

        # --- Depth super-resolution head ---
        # At this point: (B, 32, D_sparse, H, W)
        # Upsample depth by factor using learned depth shuffle
        self.depth_up = DepthUpsample(32, factor)
        self.depth_refine = nn.Sequential(
            ResBlock3D(32), ResBlock3D(32),
        )

        # Final projection
        self.final = nn.Conv3d(32, in_channels, 1)

    def forward(self, x):
        # x: (B, C_in+1, D_sparse, H, W)

        # Encoder (depth preserved, H/W halved each level)
        e0 = self.enc0(x)                          # (B, 32, D, H, W)
        e1 = self.enc1(self.down1(e0))              # (B, 64, D, H/2, W/2)
        e2 = self.enc2(self.down2(e1))              # (B, 128, D, H/4, W/4)
        b = self.bottleneck(self.down3(e2))         # (B, 256, D, H/8, W/8)

        # Decoder (H/W restored via skip connections)
        # Crop upsampled feature to match encoder skip size (handles odd dimensions)
        d3 = self.up3(b)
        d3 = d3[:, :, :e2.shape[2], :e2.shape[3], :e2.shape[4]]
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = d2[:, :, :e1.shape[2], :e1.shape[3], :e1.shape[4]]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = d1[:, :, :e0.shape[2], :e0.shape[3], :e0.shape[4]]
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        # Depth super-resolution: (B, 32, D, H, W) → (B, 32, D*factor, H, W)
        d1 = self.depth_up(d1)
        d1 = self.depth_refine(d1)

        return self.final(d1)                           # (B, C_in, D*factor, H, W) logits


def load_4class_into_11class(
    model_11: LabelReconUNet,
    ckpt_path_4class: str,
    device: str = 'cpu',
) -> LabelReconUNet:
    """
    Transfer weights from a trained 4-class model into an 11-class model.

    4-class channels [0,1,2,3] = Femur, Tibia, FC, TC
    11-class channels [0..10] = BG, Femur, Tibia, Patella, FC, TC, PC, MedMen, LatMen, ACL, PCL
    Mapping (4class idx → 11class idx, BG-stripped): 0→0, 1→1, 2→3, 3→4
    """
    ckpt = torch.load(ckpt_path_4class, map_location=device, weights_only=False)
    sd4 = ckpt['model_state_dict']
    sd11 = model_11.state_dict()

    # 4-class channel positions in 11-class system (BG already excluded in model channels)
    # 11-class model has in_channels=11, so channels 0..10 map to BG..PCL
    # But the model never sees BG — in_channels=11 means 11 tissue classes without BG
    # Actually: in_channels=11 for 11class mode, channels are [BG,Femur,...,PCL]
    # FOUR_CLASS_INDICES = [1, 2, 4, 5] in the 11-class system
    # So 4class ch0 (Femur) → 11class ch1, ch1 (Tibia) → ch2, ch2 (FC) → ch4, ch3 (TC) → ch5
    MAP_4_TO_11 = [1, 2, 4, 5]

    transferred, skipped = 0, 0
    for name, param11 in sd11.items():
        if name not in sd4:
            skipped += 1
            continue

        param4 = sd4[name]

        if param4.shape == param11.shape:
            # Identical shape — direct copy
            sd11[name] = param4.clone()
            transferred += 1

        elif name == 'enc0.0.weight':
            # First conv: (32, C_in+1, 3, 3, 3)
            # 4class: (32, 5, 3,3,3), 11class: (32, 12, 3,3,3)
            # Copy class channels and positional channel
            for i4, i11 in enumerate(MAP_4_TO_11):
                sd11[name][:, i11] = param4[:, i4]
            sd11[name][:, -1] = param4[:, -1]  # positional channel (last)
            transferred += 1

        elif name == 'enc0.0.bias':
            # Bias is per-output-channel (32,) — same shape, direct copy
            sd11[name] = param4.clone()
            transferred += 1

        elif name == 'final.weight':
            # Last conv: (C_out, 32, 1, 1, 1)
            # 4class: (4, 32, 1,1,1), 11class: (11, 32, 1,1,1)
            for i4, i11 in enumerate(MAP_4_TO_11):
                sd11[name][i11] = param4[i4]
            transferred += 1

        elif name == 'final.bias':
            # (C_out,) — 4class: (4,), 11class: (11,)
            for i4, i11 in enumerate(MAP_4_TO_11):
                sd11[name][i11] = param4[i4]
            transferred += 1

        else:
            skipped += 1

    model_11.load_state_dict(sd11)
    print(f"[RECON] Transferred {transferred} params from 4-class checkpoint, "
          f"skipped {skipped}")
    return model_11


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

@torch.no_grad()
def sliding_window_inference(
    model: LabelReconUNet,
    sparse_vol: np.ndarray,
    chunk_size: int = 10,
    overlap: int = 3,
    device: str = 'cuda',
    batch_size: int = 1,
) -> np.ndarray:
    """
    Reconstruct a full dense volume from sparse input using sliding window.

    Args:
        model: Trained LabelReconUNet.
        sparse_vol: (C, D_sparse, H, W) sparse label volume.
        chunk_size: Number of sparse slices per window.
        overlap: Overlap in sparse slices between adjacent windows.
        device: 'cuda' or 'cpu'.
        batch_size: How many chunks to process at once.

    Returns:
        dense_vol: (C, D_sparse * factor, H, W) reconstructed dense volume.
    """
    model.eval()
    model = model.to(device)
    factor = model.factor
    C, D_sparse, H, W = sparse_vol.shape
    D_dense = D_sparse * factor

    # Output accumulation buffers
    output_sum = np.zeros((C, D_dense, H, W), dtype=np.float64)
    output_count = np.zeros((1, D_dense, 1, 1), dtype=np.float64)

    # Generate chunk start positions
    stride = chunk_size - overlap
    if stride < 1:
        stride = 1
    starts = list(range(0, D_sparse - chunk_size + 1, stride))
    # Ensure we cover the last slices
    if len(starts) == 0 or starts[-1] + chunk_size < D_sparse:
        starts.append(max(0, D_sparse - chunk_size))

    # Process in batches
    all_inputs = []
    all_dense_starts = []
    for s in starts:
        chunk = sparse_vol[:, s:s + chunk_size]  # (C, cs, H, W)
        # Positional encoding
        pos = np.linspace(s / D_sparse, (s + chunk_size - 1) / D_sparse,
                          chunk_size, dtype=np.float32)
        pos_ch = np.broadcast_to(pos[None, :, None, None],
                                 (1, chunk_size, H, W)).copy()
        inp = np.concatenate([chunk, pos_ch], axis=0)  # (C+1, cs, H, W)
        all_inputs.append(inp)
        all_dense_starts.append(s * factor)

    # Run model in batches
    for i in range(0, len(all_inputs), batch_size):
        batch_inp = np.stack(all_inputs[i:i + batch_size])  # (B, C+1, cs, H, W)
        batch_t = torch.from_numpy(batch_inp).float().to(device)

        with autocast('cuda' if 'cuda' in device else 'cpu'):
            pred = model(batch_t)  # (B, C, cs*factor, H, W)

        pred_np = torch.sigmoid(pred).cpu().numpy()

        for j in range(pred_np.shape[0]):
            ds = all_dense_starts[i + j]
            de = ds + chunk_size * factor
            output_sum[:, ds:de] += pred_np[j]
            output_count[:, ds:de] += 1.0

    # Average overlapping predictions
    output_count = np.maximum(output_count, 1.0)
    dense_vol = (output_sum / output_count).astype(np.float32)

    return dense_vol


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-class soft Dice loss, averaged over classes. Expects logits."""
    pred = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Voxel-wise BCE with logits (autocast-safe)."""
    return F.binary_cross_entropy_with_logits(logits, target)


def sparse_slice_consistency_loss(
    logits: torch.Tensor, sparse_input: torch.Tensor, factor: int
) -> torch.Tensor:
    """
    The known sparse slices appear at indices 0, factor, 2*factor, ... in the
    dense output. The model should reproduce them exactly.

    logits: (B, C, D_dense, H, W)
    sparse_input: (B, C, D_sparse, H, W) — the original sparse class channels
    """
    # Extract predictions at known-slice positions
    pred_at_known = torch.sigmoid(logits[:, :, ::factor])  # (B, C, D_sparse, H, W)
    return F.mse_loss(pred_at_known, sparse_input)


class ReconLoss(nn.Module):
    """Combined loss for depth super-resolution."""

    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0,
                 consistency_w: float = 2.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.consistency_w = consistency_w

    def forward(self, pred, target, sparse_input, factor):
        l_dice = dice_loss(pred, target)
        l_bce = bce_loss(pred, target)
        l_cons = sparse_slice_consistency_loss(pred, sparse_input, factor)

        total = self.dice_w * l_dice + self.bce_w * l_bce + self.consistency_w * l_cons
        return {
            'total': total,
            'dice': l_dice.item(),
            'bce': l_bce.item(),
            'consistency': l_cons.item(),
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_dice_per_class(logits: torch.Tensor, target: torch.Tensor,
                           threshold: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    """Compute Dice score per class from logits. Returns (C,) array."""
    pred_bin = (torch.sigmoid(logits) > threshold).float()
    dims = (0, 2, 3, 4)
    intersection = (pred_bin * target).sum(dim=dims)
    union = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.cpu().numpy()


# =============================================================================
# TRAINER
# =============================================================================

class ReconTrainer:
    """Trainer with mixed precision, gradient accumulation, and early stopping."""

    def __init__(self, model: LabelReconUNet, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 accumulation_steps: int = 4, num_epochs: int = 300,
                 dice_w: float = 1.0, bce_w: float = 1.0,
                 consistency_w: float = 2.0):
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.factor = model.factor

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, eta_min=1e-6)
        self.scaler = GradScaler('cuda')
        self.loss_fn = ReconLoss(dice_w, bce_w, consistency_w)

        self.best_dice = 0.0
        self.patience_counter = 0

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_losses = {'total': 0, 'dice': 0, 'bce': 0, 'consistency': 0}
        self.optimizer.zero_grad()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
        for batch_idx, batch in pbar:
            inp = batch['input'].to(self.device)       # (B, C+1, cs, H, W)
            target = batch['target'].to(self.device)   # (B, C, cs*factor, H, W)
            C_in = self.model.in_channels
            sparse_input = inp[:, :C_in]                # (B, C, cs, H, W)

            with autocast('cuda'):
                pred = self.model(inp)
                losses = self.loss_fn(pred, target, sparse_input, self.factor)
                loss = losses['total'] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k in total_losses:
                total_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        n = len(dataloader)
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.eval()
        total_losses = {'total': 0, 'dice': 0, 'bce': 0, 'consistency': 0}
        all_dice = []

        for batch in tqdm(dataloader, desc="Val"):
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            C_in = self.model.in_channels
            sparse_input = inp[:, :C_in]

            with autocast('cuda'):
                pred = self.model(inp)
                losses = self.loss_fn(pred, target, sparse_input, self.factor)

            for k in total_losses:
                total_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

            dice_scores = compute_dice_per_class(pred, target)
            all_dice.append(dice_scores)

        n = len(dataloader)
        avg_losses = {k: v / n for k, v in total_losses.items()}
        avg_dice = np.mean(all_dice, axis=0)
        return avg_losses, avg_dice

    def save_checkpoint(self, path: str, epoch: int, val_dice: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_dice': self.best_dice,
            'val_dice': val_dice,
            'factor': self.factor,
            'in_channels': self.model.in_channels,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_dice = ckpt.get('best_dice', 0.0)
        return ckpt['epoch']


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

CLASS_NAMES_11 = ['BG', 'Femur', 'Tibia', 'Patella', 'FC', 'TC', 'PC',
                  'MedMen', 'LatMen', 'ACL', 'PCL']
CLASS_NAMES_4 = ['Femur', 'Tibia', 'FC', 'TC']


def train_recon(
    data_dir: str,
    metadata_path: str,
    output_dir: str,
    mode: str = '11class',
    undersample_factor: int = 4,
    chunk_size: int = 10,
    overlap: int = 3,
    chunks_per_volume: int = 8,
    batch_size: int = 2,
    num_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    accumulation_steps: int = 4,
    patience: int = 40,
    dice_w: float = 1.0,
    bce_w: float = 1.0,
    consistency_w: float = 2.0,
    resume_from: str = None,
    pretrained_4class: str = None,
    val_split: float = 0.2,
    num_workers: int = 2,
):
    """
    Train the patch-based depth SR RECON net.

    Args:
        data_dir: Directory containing .npy/.npz mask files (dense, 160 slices).
        metadata_path: Path to metadata JSON.
        output_dir: Checkpoint and log directory.
        mode: '11class' or '4class'.
        undersample_factor: Depth SR factor (default 4: 40→160 slices).
        chunk_size: Sparse slices per training patch (default 10).
        overlap: Sparse-slice overlap for inference (default 3).
        chunks_per_volume: Random chunks sampled per volume per epoch (default 8).
        batch_size: Per-GPU batch size (default 2).
        num_epochs: Max epochs (default 300).
        lr, weight_decay: AdamW params.
        accumulation_steps: Gradient accumulation (default 4, effective batch=8).
        patience: Early stopping patience.
        dice_w, bce_w, consistency_w: Loss weights.
        resume_from: Checkpoint path to resume training.
        pretrained_4class: Path to 4-class checkpoint for 11-class transfer learning.
        val_split: Validation fraction (default 0.2).
        num_workers: DataLoader workers.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[RECON] Device: {device}")
    print(f"[RECON] Mode: {mode}, Factor: {undersample_factor}x")
    print(f"[RECON] Chunk: {chunk_size} sparse slices → {chunk_size * undersample_factor} dense slices")
    print(f"[RECON] Overlap: {overlap} sparse slices, Chunks/vol: {chunks_per_volume}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_cases = sorted(metadata.keys())
    print(f"[RECON] Total cases: {len(all_cases)}")

    # For 11-class mode, exclude partial (4-class only) cases
    if mode == '11class':
        n_before = len(all_cases)
        all_cases = [c for c in all_cases
                     if not metadata[c].get('is_partial', False)]
        print(f"[RECON] Filtered to {len(all_cases)} full 11-class cases "
              f"(excluded {n_before - len(all_cases)} partial)")

    # Separate original and synthetic cases
    original_cases = sorted([c for c in all_cases
                             if metadata[c].get('timepoint') != 'synthetic'])
    synthetic_cases = sorted([c for c in all_cases
                              if metadata[c].get('timepoint') == 'synthetic'])

    # Train/val split on ORIGINAL cases only (deterministic, identical to non-synthetic run)
    np.random.seed(42)
    indices = np.random.permutation(len(original_cases))
    n_val = max(1, int(len(original_cases) * val_split))
    val_ids = [original_cases[i] for i in indices[:n_val]]
    train_ids = [original_cases[i] for i in indices[n_val:]]

    # Add synthetic cases to training only
    train_ids += synthetic_cases
    if synthetic_cases:
        print(f"[RECON] Original cases: {len(original_cases)}, Synthetic cases: {len(synthetic_cases)}")
    print(f"[RECON] Train: {len(train_ids)} ({len(train_ids) - len(synthetic_cases)} original + {len(synthetic_cases)} synthetic), Val: {len(val_ids)} (original only)")

    # Datasets
    train_ds = ReconDataset(data_dir, train_ids, metadata, mode=mode,
                            undersample_factor=undersample_factor,
                            chunk_size=chunk_size,
                            chunks_per_volume=chunks_per_volume,
                            augment=True)
    val_ds = ReconDataset(data_dir, val_ids, metadata, mode=mode,
                          undersample_factor=undersample_factor,
                          chunk_size=chunk_size,
                          chunks_per_volume=chunks_per_volume,
                          augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model
    n_classes = 4 if mode == '4class' else 11
    model = LabelReconUNet(in_channels=n_classes, factor=undersample_factor)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[RECON] Model parameters: {n_params:.2f}M")

    # Transfer learning: load 4-class weights into 11-class model
    if mode == '11class' and pretrained_4class and os.path.exists(pretrained_4class):
        print(f"[RECON] Loading 4-class pretrained weights from: {pretrained_4class}")
        model = load_4class_into_11class(model, pretrained_4class, device=str(device))

    # Trainer
    trainer = ReconTrainer(
        model, device, lr=lr, weight_decay=weight_decay,
        accumulation_steps=accumulation_steps, num_epochs=num_epochs,
        dice_w=dice_w, bce_w=bce_w, consistency_w=consistency_w
    )

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch = trainer.load_checkpoint(resume_from) + 1
        print(f"[RECON] Resumed from epoch {start_epoch}")

    # Logging
    class_names = CLASS_NAMES_4 if mode == '4class' else CLASS_NAMES_11
    log_path = os.path.join(output_dir, 'training_log.csv')
    log_fields = ['epoch', 'train_loss', 'val_loss', 'mean_dice'] + \
                 [f'dice_{n}' for n in class_names]
    if not os.path.exists(log_path) or start_epoch == 0:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(log_fields)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}  (lr={trainer.optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'='*60}")

        train_losses = trainer.train_epoch(train_loader)
        val_losses, val_dice = trainer.evaluate(val_loader)
        trainer.scheduler.step()

        mean_dice = val_dice.mean()
        elapsed = time.time() - t0

        print(f"\nTrain loss: {train_losses['total']:.4f} "
              f"(dice={train_losses['dice']:.4f}, bce={train_losses['bce']:.4f}, "
              f"cons={train_losses['consistency']:.4f})")
        print(f"Val   loss: {val_losses['total']:.4f}  |  Mean Dice: {mean_dice:.4f}")
        for i, name in enumerate(class_names):
            if i < len(val_dice):
                print(f"  {name:>10s}: {val_dice[i]:.4f}")
        print(f"Elapsed: {elapsed:.1f}s")

        row = [epoch + 1, train_losses['total'], val_losses['total'], mean_dice]
        row += [val_dice[i] if i < len(val_dice) else 0.0 for i in range(len(class_names))]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        trainer.save_checkpoint(os.path.join(output_dir, 'last.pt'), epoch, mean_dice)

        if mean_dice > trainer.best_dice:
            trainer.best_dice = mean_dice
            trainer.patience_counter = 0
            trainer.save_checkpoint(os.path.join(output_dir, 'best.pt'), epoch, mean_dice)
            print(f"  >> New best Dice: {mean_dice:.4f}")
        else:
            trainer.patience_counter += 1
            print(f"  >> No improvement ({trainer.patience_counter}/{patience})")

        if trainer.patience_counter >= patience:
            print(f"\n[RECON] Early stopping at epoch {epoch+1}")
            break

    print(f"\n[RECON] Training complete. Best Dice: {trainer.best_dice:.4f}")
    print(f"[RECON] Checkpoints saved to: {output_dir}")
    return trainer


# =============================================================================
# QUICK TEST
# =============================================================================

def quick_test():
    """
    Quick sanity test with synthetic data.
    Verifies: shapes, forward pass, overfitting, sliding window inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TEST] Device: {device}")

    n_classes = 4
    factor = 4
    D_sparse, H, W = 10, 64, 64  # small for speed
    D_dense = D_sparse * factor   # 40

    # --- 1. Shape test ---
    print("\n[TEST 1] Shape verification...")
    model = LabelReconUNet(in_channels=n_classes, factor=factor).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {n_params:.2f}M")

    dummy = torch.randn(2, n_classes + 1, D_sparse, H, W, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, n_classes, D_dense, H, W), \
        f"Shape mismatch: expected {(2, n_classes, D_dense, H, W)}, got {out.shape}"
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}  ✓")

    # --- 2. Overfit test ---
    print("\n[TEST 2] Overfitting on synthetic spheres...")

    # Create synthetic volumes: dense (C, 40, 64, 64)
    volumes = []
    for _ in range(4):
        vol = np.zeros((n_classes, D_dense, H, W), dtype=np.float32)
        for c in range(n_classes):
            cz, cy, cx = D_dense // 2 + c * 3, H // 2, W // 2
            zz, yy, xx = np.ogrid[:D_dense, :H, :W]
            r = 8 + c * 3
            vol[c] = (((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2) < r**2).astype(np.float32)
        volumes.append(vol)

    class SyntheticDS(Dataset):
        def __len__(self):
            return len(volumes)
        def __getitem__(self, idx):
            dense = volumes[idx]                          # (C, D_dense, H, W)
            sparse = dense[:, ::factor]                   # (C, D_sparse, H, W)
            pos = np.linspace(0, 1, D_sparse, dtype=np.float32)
            pos_ch = np.broadcast_to(pos[None, :, None, None],
                                     (1, D_sparse, H, W)).copy()
            inp = np.concatenate([sparse, pos_ch], axis=0)
            return {
                'input': torch.from_numpy(inp).float(),
                'target': torch.from_numpy(dense).float(),
            }

    loader = DataLoader(SyntheticDS(), batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = ReconLoss()

    model.train()
    for step in range(80):
        for batch in loader:
            inp = batch['input'].to(device)
            target = batch['target'].to(device)
            sparse_input = inp[:, :n_classes]

            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                pred = model(inp)
                losses = loss_fn(pred, target, sparse_input, factor)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

        if (step + 1) % 20 == 0:
            dice = compute_dice_per_class(pred.detach(), target)
            print(f"  Step {step+1}: loss={losses['total'].item():.4f}, "
                  f"mean_dice={dice.mean():.4f}")

    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['input'].to(device))
            dice = compute_dice_per_class(pred, batch['target'].to(device))
            print(f"  Final Dice: {dice}, mean={dice.mean():.4f}")
            break

    # --- 3. Sliding window test ---
    print("\n[TEST 3] Sliding window inference...")
    # Full sparse volume: (C, 40, 64, 64) → expect (C, 160, 64, 64)
    D_sparse_full = 40
    full_sparse = np.random.rand(n_classes, D_sparse_full, H, W).astype(np.float32)
    dense_out = sliding_window_inference(
        model, full_sparse, chunk_size=10, overlap=3,
        device=str(device), batch_size=2
    )
    assert dense_out.shape == (n_classes, D_sparse_full * factor, H, W), \
        f"Stitcher shape mismatch: {dense_out.shape}"
    print(f"  Sparse input:  {full_sparse.shape}")
    print(f"  Dense output:  {dense_out.shape}  ✓")
    print(f"  Value range:   [{dense_out.min():.3f}, {dense_out.max():.3f}]")

    if dice.mean() > 0.6:
        print("\n[TEST] PASS")
    else:
        print("\n[TEST] WARNING — Dice lower than expected, may need more steps")

    return model


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RECON Net: Patch-based Depth SR for Knee Label Completion')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./recon_output')
    parser.add_argument('--mode', type=str, default='11class',
                        choices=['11class', '4class'])
    parser.add_argument('--undersample_factor', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Sparse slices per training patch')
    parser.add_argument('--overlap', type=int, default=3,
                        help='Sparse-slice overlap for inference')
    parser.add_argument('--chunks_per_volume', type=int, default=8,
                        help='Random chunks per volume per epoch')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--pretrained_4class', type=str, default=None,
                        help='Path to 4-class checkpoint for 11-class transfer learning')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick sanity test instead of training')

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    else:
        train_recon(**{k: v for k, v in vars(args).items()
                       if k not in ('quick_test',)})
