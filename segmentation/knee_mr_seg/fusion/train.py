"""Train loop for the v2 PD dual-plane fusion net (Framing A, slab-deblur SVR).

Inputs: per-case cached ``.npz`` files written by ``build_fusion_training_inputs.py``,
each containing
    inputs  : (8, H, W, D) float16  — slab-blurred DESS_mask. Channels 0..3 =
                                       SAG slab-blur (FB, FC, TB, TC) along
                                       R-L axis (DESS axis 2 in PIR); 4..7 =
                                       COR slab-blur along A-P axis (DESS
                                       axis 1). Values in [0, 1] = per-class
                                       fractional occupancy under the rect
                                       kernel.
    target  : (H, W, D) uint8       — original DESS_mask 4-class argmax
                                       (0 BG, 1 Fem, 2 FC, 3 Tib, 4 TC).
    affine  : (4, 4) float32        — DESS affine (kept for inference NIfTI save).
    case_id : scalar string.

At test time the same network consumes D211 + D212 softmax predictions
resampled to the DESS grid (different distribution, see
v2_pipeline.run_dualplane_inference). The train/test gap under this framing
is bounded by D211/D212's deviation from a perfect slab-blur (~0.13 Dice on
cartilage) and is measured empirically on the fold-0 val set.

Training strategy
-----------------
* Random 128x128x128 patches per iteration (oversample foreground: 80% of
  patches drawn so that the patch centre falls on a foreground voxel).
* Loss = mean of CE(logits, target) + (1 - mean foreground-Dice).
* Adam, lr=1e-3, cosine decay over total iters.
* Per-epoch (= 200 iters) val pass: full-volume sliding-window inference on
  the fold's val cases, per-class Dice vs the cached target.
* Fold split = splits_final.json from Dataset211 prep (provenance-stratified
  5-fold; mirrors the single-plane fold definitions so the val cases for
  fold 0 of D211/D212/Fusion are identical).

Run from CLI:
    python -m knee_mr_seg.fusion.train --cache_dir E:/.../fusion_inputs \\
        --splits_json E:/.../Dataset211_PD_SAG_DESSEq/splits_final.json \\
        --fold 0 --out_dir E:/.../fusion_runs/fold0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .net import build_fusion_net
from .resample import N_FUSION_CLASSES, DESS_FUSION_LABELS


PATCH = 128
N_CLASSES = N_FUSION_CLASSES   # 5 (BG + 4)
FG_CLASSES = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Cache-format adapter
# ---------------------------------------------------------------------------
def expand_inputs_to_8ch(inputs: np.ndarray) -> np.ndarray:
    """Normalise a cached inputs array to the (8, H, W, D) float32 format
    FusionKNet expects.

    Two cache formats are supported transparently:

    - **Option A (legacy)**: ``inputs`` is already (8, H, W, D) float16 --
      4 SAG foreground softmax channels followed by 4 COR foreground
      softmax channels. Just cast to float32.

    - **Option C (current default)**: ``inputs`` is (4, H, W, D) uint8 with
      ch0=SAG argmax (0..4), ch1=SAG winner-confidence (0..255), ch2=COR
      argmax, ch3=COR confidence. Expand to soft one-hot:

          out[c-1]   = (sag_argmax == c) * (sag_conf / 255)   for c in 1..4
          out[c-1+4] = (cor_argmax == c) * (cor_conf / 255)   for c in 1..4

    Per CLAUDE.md memory the slab-blurred Option A cache (Framing A) and
    the argmax+conf Option C cache (Framing B production) end up in
    different ``fusion_inputs[_a|_b]`` dirs but share the same train.py
    code path because of this adapter.
    """
    if inputs.ndim != 4:
        raise ValueError(f"expected 4D inputs (C, H, W, D), got {inputs.shape}")
    c0 = inputs.shape[0]
    if c0 == 8:
        return inputs.astype(np.float32, copy=False)
    if c0 == 4:
        # Option C: (sag_argmax, sag_conf, cor_argmax, cor_conf) uint8
        sag_arg = inputs[0]                           # (H, W, D) uint8
        sag_conf = inputs[1].astype(np.float32) / 255.0
        cor_arg = inputs[2]
        cor_conf = inputs[3].astype(np.float32) / 255.0
        out = np.zeros((8,) + inputs.shape[1:], dtype=np.float32)
        for ci in range(4):
            cls = ci + 1  # foreground class label 1..4
            out[ci]     = (sag_arg == cls).astype(np.float32) * sag_conf
            out[ci + 4] = (cor_arg == cls).astype(np.float32) * cor_conf
        return out
    raise ValueError(
        f"unexpected cached input channel count {c0}; supported: 8 (Option A "
        f"softmax full), 4 (Option C argmax+conf). For Option B (argmax only, "
        f"2 channels) add the appropriate branch here."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FusionPatchDataset(Dataset):
    """Random foreground-biased 128^3 patches from cached fusion inputs."""

    def __init__(self, cache_paths: list[Path], patch_size: int = PATCH,
                 fg_ratio: float = 0.8, iters_per_epoch: int = 200):
        self.cache_paths = [Path(p) for p in cache_paths]
        self.patch = patch_size
        self.fg_ratio = fg_ratio
        self.iters_per_epoch = iters_per_epoch
        # Quick sanity load of first file
        if self.cache_paths:
            with np.load(self.cache_paths[0]) as f:
                _ = f["inputs"].shape

    def __len__(self):
        return self.iters_per_epoch

    def _sample_patch(self, inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, H, W, D = inputs.shape
        ph = pw = pd = self.patch
        # Pad if any spatial dim < patch
        pad = [(0, 0), (0, max(0, ph - H)), (0, max(0, pw - W)), (0, max(0, pd - D))]
        if any(p[1] > 0 for p in pad):
            inputs = np.pad(inputs, pad)
            target = np.pad(target, [(p[1], p[2]) if i > 0 else (0, 0) for i, p in enumerate(pad)])
            _, H, W, D = inputs.shape
        if random.random() < self.fg_ratio:
            fg = np.argwhere(np.isin(target, FG_CLASSES))
            if len(fg) > 0:
                cy, cx, cz = fg[random.randrange(len(fg))]
            else:
                cy = random.randrange(H); cx = random.randrange(W); cz = random.randrange(D)
        else:
            cy = random.randrange(H); cx = random.randrange(W); cz = random.randrange(D)
        # Clamp so patch is in-bounds
        y0 = max(0, min(cy - ph // 2, H - ph))
        x0 = max(0, min(cx - pw // 2, W - pw))
        z0 = max(0, min(cz - pd // 2, D - pd))
        x_patch = inputs[:, y0:y0 + ph, x0:x0 + pw, z0:z0 + pd]
        t_patch = target[y0:y0 + ph, x0:x0 + pw, z0:z0 + pd]
        return x_patch, t_patch

    def __getitem__(self, idx: int):
        # Pick a random case each iter (uniform; small dataset, ok).
        # Sample the patch from the cached compact format first (uint8 if
        # Option C, float16 if Option A) so worker RAM stays small; only
        # the patch expands to 8-channel float32 for FusionKNet.
        path = random.choice(self.cache_paths)
        with np.load(path) as f:
            inputs = np.asarray(f["inputs"])    # (4, ...) uint8 OR (8, ...) float16
            target = np.asarray(f["target"])    # (H, W, D) uint8
        x_patch, t_patch = self._sample_patch(inputs, target)
        x = expand_inputs_to_8ch(x_patch)       # (8, ph, pw, pd) float32
        y = t_patch.astype(np.int64, copy=False)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


# ---------------------------------------------------------------------------
# Losses + metrics
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean foreground Dice loss (1 - Dice) across classes 1..C-1.

    logits: (B, C, ...) ; target: (B, ...).
    """
    probs = torch.softmax(logits, dim=1)
    tgt_oh = F.one_hot(target.clamp(0, probs.shape[1] - 1), num_classes=probs.shape[1])
    tgt_oh = tgt_oh.permute(0, -1, *range(1, target.ndim)).float()  # (B, C, ...)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * tgt_oh).sum(dim=dims)
    denom = probs.sum(dim=dims) + tgt_oh.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)         # (B, C)
    fg = dice[:, 1:]                                  # drop BG
    return 1.0 - fg.mean()


def per_class_dice(pred_argmax: np.ndarray, target: np.ndarray, n_classes: int) -> dict[int, float]:
    out = {}
    for c in range(n_classes):
        a = pred_argmax == c
        b = target == c
        denom = a.sum() + b.sum()
        out[c] = float((2 * (a & b).sum() + 1e-6) / (denom + 1e-6))
    return out


# ---------------------------------------------------------------------------
# Sliding-window inference (full volume)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_infer(model: torch.nn.Module, inputs: np.ndarray,
                         patch_size: int = PATCH, overlap: float = 0.25,
                         device: torch.device | str = "cuda") -> np.ndarray:
    """Tiled full-volume inference. Returns argmax (H, W, D) uint8.

    Keeps the accumulator on GPU -- previously this used numpy arrays of
    shape (5, 384, 384, 160) float32 (~470 MB) and did per-patch numpy
    adds, which dominated wall time (10+ s/case on A100). With everything
    on GPU the val drops to ~2 s/case.
    """
    dev = torch.device(device)
    model.eval()
    C_in, H, W, D = inputs.shape
    P = patch_size
    pad_h = max(0, P - H); pad_w = max(0, P - W); pad_d = max(0, P - D)
    if pad_h or pad_w or pad_d:
        inputs = np.pad(inputs, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)))
    _, Hp, Wp, Dp = inputs.shape
    step = max(int(P * (1 - overlap)), 1)

    # Move full volume + accumulators to GPU once.
    x_gpu = torch.from_numpy(np.ascontiguousarray(inputs)).float().to(dev)  # (C, Hp, Wp, Dp)
    accum = torch.zeros((N_CLASSES, Hp, Wp, Dp), dtype=torch.float32, device=dev)
    weight = torch.full((Hp, Wp, Dp), 1e-6, dtype=torch.float32, device=dev)

    ys = list(range(0, max(1, Hp - P + 1), step))
    if ys[-1] != Hp - P: ys.append(Hp - P)
    xs = list(range(0, max(1, Wp - P + 1), step))
    if xs[-1] != Wp - P: xs.append(Wp - P)
    zs = list(range(0, max(1, Dp - P + 1), step))
    if zs[-1] != Dp - P: zs.append(Dp - P)

    for y0 in ys:
        for x0 in xs:
            for z0 in zs:
                patch = x_gpu[:, y0:y0 + P, x0:x0 + P, z0:z0 + P].unsqueeze(0)  # (1, C, P, P, P)
                logits = model(patch)                              # (1, N, P, P, P)
                probs = torch.softmax(logits, dim=1)[0]            # (N, P, P, P)
                accum[:, y0:y0 + P, x0:x0 + P, z0:z0 + P] += probs
                weight[y0:y0 + P, x0:x0 + P, z0:z0 + P] += 1.0     # uniform weight ok at overlap=0.25
    pred = torch.argmax(accum / weight.unsqueeze(0), dim=0).to(torch.uint8).cpu().numpy()
    return pred[:H, :W, :D]


# ---------------------------------------------------------------------------
# Train / eval driver
# ---------------------------------------------------------------------------
def load_splits(splits_json: Path, fold: int) -> tuple[list[str], list[str]]:
    splits = json.loads(Path(splits_json).read_text())
    return splits[fold]["train"], splits[fold]["val"]


def select_cache_paths(cache_dir: Path, case_ids: list[str]) -> list[Path]:
    paths = []
    missing = []
    for cid in case_ids:
        p = cache_dir / f"{cid}.npz"
        if p.exists():
            paths.append(p)
        else:
            missing.append(cid)
    if missing:
        print(f"  [warn] {len(missing)} cases missing in cache_dir; first 5: {missing[:5]}")
    return paths


def train(args):
    print(f"[fusion train] start  cache_dir={args.cache_dir}  out_dir={args.out_dir}",
          flush=True)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = load_splits(Path(args.splits_json), args.fold)
    print(f"[fusion train] fold {args.fold}: train={len(train_ids)}  val={len(val_ids)}",
          flush=True)
    train_paths = select_cache_paths(cache_dir, train_ids)
    val_paths = select_cache_paths(cache_dir, val_ids)
    if args.n_run is not None:
        train_paths = train_paths[: args.n_run]
        val_paths = val_paths[: max(2, args.n_run // 5)]
        print(f"  --n_run cap: train={len(train_paths)} val={len(val_paths)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fusion train] device={device}  num_workers={args.num_workers}  "
          f"batch={args.batch_size}  patch={PATCH}  val_every={args.val_every}", flush=True)
    model = build_fusion_net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    ds = FusionPatchDataset(train_paths, patch_size=PATCH, fg_ratio=0.8,
                            iters_per_epoch=args.iters_per_epoch)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        drop_last=True,
                        persistent_workers=(args.num_workers > 0))
    print(f"[fusion train] DataLoader ready ({len(loader)} batches/epoch)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_iters = args.epochs * args.iters_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters, eta_min=args.lr * 0.01)

    if args.dry_run:
        # Quick smoke test: 5 iters + 1 val case
        print("[fusion train] DRY RUN - 5 iters + 1 val case")
        model.train()
        it = iter(loader)
        for i in range(5):
            x, y = next(it)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  iter {i}: loss={loss.item():.4f}  shape_in={tuple(x.shape)}")
        if val_paths:
            with np.load(val_paths[0]) as f:
                raw = np.asarray(f["inputs"]); target = f["target"].astype(np.int64)
            inputs = expand_inputs_to_8ch(raw)
            pred = sliding_window_infer(model, inputs, patch_size=PATCH, device=device)
            d = per_class_dice(pred, target, N_CLASSES)
            print(f"  val[0] {val_paths[0].stem} Dice per class: {d}")
        return

    log_csv = out_dir / "train_log.csv"
    with log_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,iter,loss,lr\n")

    best_dice = -1.0
    history = []
    last_val: dict | None = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item()
            if (it + 1) % 20 == 0:
                with log_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{epoch},{it},{loss.item():.5f},{opt.param_groups[0]['lr']:.6f}\n")
        avg = running / max(1, len(loader))
        t_train = time.time() - t0

        # Validation: only every --val_every epochs (and always on the last)
        do_val = ((epoch + 1) % max(1, args.val_every) == 0) or (epoch == args.epochs - 1)
        if do_val:
            t_v = time.time()
            # Optionally validate on a subset (deterministic per epoch via
            # epoch-seeded RNG so cycles are comparable across runs).
            if args.val_subset is not None and args.val_subset < len(val_paths):
                rng = random.Random(0xC0DE + epoch)
                this_val_paths = rng.sample(val_paths, args.val_subset)
            else:
                this_val_paths = val_paths
            val_dice = _validate(model, this_val_paths, device)
            mean_fg = float(np.mean([val_dice[c] for c in FG_CLASSES]))
            t_val = time.time() - t_v
            last_val = {"epoch": epoch, "val_dice": val_dice, "mean_fg": mean_fg}
            print(f"epoch {epoch:3d}/{args.epochs} loss={avg:.4f}  "
                  f"val Dice Fem={val_dice[1]:.3f} FC={val_dice[2]:.3f} "
                  f"Tib={val_dice[3]:.3f} TC={val_dice[4]:.3f}  meanFG={mean_fg:.3f}  "
                  f"({t_train:.0f}s train + {t_val:.0f}s val)", flush=True)
        else:
            mean_fg = last_val["mean_fg"] if last_val else float("nan")
            val_dice = last_val["val_dice"] if last_val else {c: float("nan") for c in range(N_CLASSES)}
            print(f"epoch {epoch:3d}/{args.epochs} loss={avg:.4f}  (no val this epoch)  "
                  f"({t_train:.0f}s train)", flush=True)

        history.append({"epoch": epoch, "loss": avg, "val_dice": val_dice, "mean_fg": mean_fg,
                        "elapsed_s": time.time() - t0, "did_val": do_val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        # Save last + best (only update best when we actually validated)
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice},
                   out_dir / "checkpoint_last.pth")
        if do_val and mean_fg > best_dice:
            best_dice = mean_fg
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice,
                        "mean_fg": mean_fg},
                       out_dir / "checkpoint_best.pth")
            print(f"  [save] new best (meanFG={mean_fg:.3f}) -> checkpoint_best.pth", flush=True)
    print(f"[fusion train] done. best meanFG Dice={best_dice:.3f}", flush=True)


@torch.no_grad()
def _validate(model, val_paths, device):
    """Return mean per-class Dice across val cases.

    Each val case loads a full (8, H, W, D) volume + does sliding-window
    inference. With patch=128, overlap=0.5 on a 384x384x160 volume that's
    ~32 patches/case x ~50ms/patch on A100 = ~1.5 s/case forward + python
    overhead ~1 s = ~3 s/case. 114 cases = ~5-6 minutes.

    Cast the full input to float32 only at sliding-window-infer time --
    the function ingests np.ndarray and feeds patch-sized tensors to GPU.
    """
    model.eval()
    per_case = []
    n_val = len(val_paths)
    t_start = time.time()
    print(f"  [val] {n_val} cases...", flush=True)
    for i, p in enumerate(val_paths, 1):
        t_case = time.time()
        with np.load(p) as f:
            raw = np.asarray(f["inputs"])
            target = np.asarray(f["target"]).astype(np.int64)
        inputs = expand_inputs_to_8ch(raw)
        pred = sliding_window_infer(model, inputs, patch_size=PATCH, device=device)
        d = per_class_dice(pred, target, N_CLASSES)
        per_case.append(d)
        del raw, inputs, target, pred
        # Tick every 10 cases + first case + last case so the user sees
        # progress and can estimate ETA. First case alone tells us per-case
        # time so we don't sit silently for 10 min wondering if it hung.
        if i == 1 or i == n_val or (i % 10 == 0):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_val - i) / rate if rate > 0 else 0
            mean_fg = float(np.mean([d[c] for c in FG_CLASSES]))
            print(f"  [val] {i:3d}/{n_val}  meanFG={mean_fg:.3f}  "
                  f"({time.time()-t_case:.1f}s/case, {elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining)", flush=True)
    mean = {c: float(np.mean([d[c] for d in per_case])) for c in range(N_CLASSES)}
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, required=True,
                    help="Dir of per-case .npz files from build_fusion_training_inputs.py")
    ap.add_argument("--splits_json", type=Path, required=True,
                    help="Dataset211/Dataset212 splits_final.json (identical across planes).")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iters_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers. Each fork copies parent RAM "
                         "(big on Colab); start at 2, raise only if loader "
                         "is the bottleneck. Use 0 to debug OOM.")
    ap.add_argument("--val_every", type=int, default=5,
                    help="Run full-volume val every N epochs (and on the "
                         "last). Reduces per-epoch overhead.")
    ap.add_argument("--val_subset", type=int, default=None,
                    help="If set, validate on a random subset of N val cases "
                         "per cycle (still deterministic per-cycle via "
                         "epoch-seeded RNG). 20 is a fast smoke estimate; "
                         "None = all val cases.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_run", type=int, default=None,
                    help="Cap train cases (smoke testing).")
    ap.add_argument("--dry_run", action="store_true",
                    help="5 iters + 1 val case, then exit.")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
