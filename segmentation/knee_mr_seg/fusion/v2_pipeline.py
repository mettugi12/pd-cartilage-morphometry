"""End-to-end v2 wrapper: PD SAG + PD COR images -> DESS-grid 4-class mask
-> drop straight into ``knee_mr_cartilage.process_one_patient(modality='dess')``.

Composition::

    PD_SAG.nii.gz ─┐
                    ├─> Dataset211 fold-0  ─┐
                    │                       │
                    └─> SAG softmax (Hs,Ws,Ds)
                                            ├─> world-coord trilinear resample to DESS grid
    PD_COR.nii.gz ─┐                       │   (8 channels: 4 SAG + 4 COR fg classes)
                    ├─> Dataset212 fold-0  ─┤
                    │                       │
                    └─> COR softmax (Hc,Wc,Dc)
                                            ▼
                             Fusion 3D U-Net (8 -> 5 classes)
                                            ▼
                             argmax DESS-grid mask
                                  + DESS affine
                                            ▼
                             save as NIfTI (DESS_LABELS scheme)
                                            ▼
        knee_mr_cartilage.process_one_patient(seg_path=mask.nii.gz,
                                              modality='dess', ...)

The downstream cartilage projection is modality-agnostic: it doesn't know or
care that the DESS mask came from a fusion net rather than from a hand-drawn
DESS GT. Same template registration, EDT thickness, IDW remap.

Per HANDOFF Phase 4 — sister session ``cartilage-validation`` will import
``run_dualplane_inference()`` here as its ``pd_dualplane_fusion_provider``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .inference import make_predictor, predict_softmax
from .net import build_fusion_net
from .resample import (
    resample_probs_to_dess_grid,
    save_dess_grid_mask_nifti,
    DESS_FUSION_LABELS,
)
from .train import sliding_window_infer, PATCH, expand_inputs_to_8ch


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_fusion_model(weights_path: Path, device: torch.device):
    model = build_fusion_net().to(device)
    ck = torch.load(str(weights_path), map_location=device)
    state = ck.get("model", ck)
    model.load_state_dict(state)
    model.eval()
    return model


# Module-level singletons so repeated calls (e.g. batch validation) don't
# rebuild predictors / reload weights per patient. Clear with ``reset_cache``.
_PREDICTOR_CACHE: dict[tuple, object] = {}
_FUSION_MODEL_CACHE: dict[tuple, object] = {}


def reset_cache():
    _PREDICTOR_CACHE.clear()
    _FUSION_MODEL_CACHE.clear()


def _get_predictors(nnunet_results_dir, fold, device):
    key = (str(nnunet_results_dir) if nnunet_results_dir else None, fold, str(device))
    if key not in _PREDICTOR_CACHE:
        sag = make_predictor("Dataset211_PD_SAG_DESSEq", fold=fold,
                             results_dir=nnunet_results_dir, device=str(device))
        cor = make_predictor("Dataset212_PD_COR_DESSEq", fold=fold,
                             results_dir=nnunet_results_dir, device=str(device))
        _PREDICTOR_CACHE[key] = (sag, cor)
    return _PREDICTOR_CACHE[key]


def _get_fusion_model(weights_path, device):
    key = (str(weights_path), str(device))
    if key not in _FUSION_MODEL_CACHE:
        _FUSION_MODEL_CACHE[key] = _load_fusion_model(Path(weights_path), device)
    return _FUSION_MODEL_CACHE[key]


def run_dualplane_inference(
    pd_sag_path: Path | str,
    pd_cor_path: Path | str,
    dess_path: Path | str,
    fusion_weights: Path | str,
    nnunet_results_dir: Path | str | None = None,
    fold: int = 0,
    device: str | torch.device | None = None,
    out_mask_path: Path | str | None = None,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Run D211 + D212 + fusion → DESS-grid 4-class NIfTI.

    Parameters
    ----------
    pd_sag_path, pd_cor_path : NIfTI of PD SAG / PD COR (canonical layer
        format — already in the post-reorient voxel layout that D211/D212
        were trained on).
    dess_path : NIfTI of the matched DESS image — provides the target voxel
        grid (shape + affine). For external eval where there IS a DESS image,
        this is canonical/cases/<id>/DESS.nii.gz. For deployment cases with
        no DESS available, see HANDOFF gotcha: build a synthetic DESS grid by
        re-affining PD COR to 0.7 mm iso in the same world coords.
    fusion_weights : .pth from train.py output (key 'model' in checkpoint).
    nnunet_results_dir : dir containing Dataset211/212 fold weights, or None
        to read from $NNUNET_RESULTS_DIR / $nnUNet_results env.
    out_mask_path : where to save the DESS-grid mask NIfTI. If None, writes
        a temp file (caller responsible for cleanup).

    Returns
    -------
    (mask_path, mask_array, dess_affine)
        mask_path : Path to the saved NIfTI.
        mask_array : (H, W, D) uint8 — same shape as the DESS reference.
        dess_affine : (4, 4) float — same as the DESS reference.
    """
    device = _resolve_device(device)

    dess_nii = nib.load(str(dess_path))
    dess_shape = tuple(int(s) for s in dess_nii.shape[:3])
    dess_affine = np.asarray(dess_nii.affine, dtype=np.float64)

    sag_pred, cor_pred = _get_predictors(nnunet_results_dir, fold, device)

    sag_probs, sag_affine, _ = predict_softmax(sag_pred, pd_sag_path)
    cor_probs, cor_affine, _ = predict_softmax(cor_pred, pd_cor_path)

    # Resample BOTH planes' 5-channel softmax to DESS grid in one call.
    # Keep BG so we can argmax over all 5 classes.
    fused_5 = resample_probs_to_dess_grid(
        sag_probs, sag_affine, cor_probs, cor_affine,
        dess_shape, dess_affine, device=device, drop_background=False,
    )  # (10, H, W, D) float32

    # Match the training distribution (Option C cache: argmax + winner-conf).
    # If we fed raw softmax instead, the fusion net would see a distribution
    # it never saw during training and the val Dice wouldn't transfer.
    sag_5 = fused_5[0:5]
    cor_5 = fused_5[5:10]
    sag_argmax = np.argmax(sag_5, axis=0).astype(np.uint8)
    sag_conf = np.clip(np.round(np.max(sag_5, axis=0) * 255), 0, 255).astype(np.uint8)
    cor_argmax = np.argmax(cor_5, axis=0).astype(np.uint8)
    cor_conf = np.clip(np.round(np.max(cor_5, axis=0) * 255), 0, 255).astype(np.uint8)
    compact = np.stack([sag_argmax, sag_conf, cor_argmax, cor_conf], axis=0)  # (4, H, W, D)
    fused = expand_inputs_to_8ch(compact)  # (8, H, W, D) float32 -- soft one-hot

    model = _get_fusion_model(fusion_weights, device)
    pred_mask = sliding_window_infer(model, fused, patch_size=PATCH, device=device)  # (H, W, D) uint8

    if out_mask_path is None:
        out_mask_path = Path(tempfile.mkdtemp(prefix="v2_dess_mask_")) / "v2_dess_mask.nii.gz"
    save_dess_grid_mask_nifti(pred_mask, dess_affine, out_mask_path,
                              dess_header=dess_nii.header)
    return Path(out_mask_path), pred_mask, dess_affine


def v2_process_one_patient(
    pd_sag_path: Path | str,
    pd_cor_path: Path | str,
    dess_path: Path | str,
    side: str,
    bone_name: str,
    fusion_weights: Path | str,
    nnunet_results_dir: Path | str | None = None,
    fold: int = 0,
    device: str | torch.device | None = None,
    config=None,
    out_dir: Path | str | None = None,
    case_id: str | None = None,
    save_unregistered_mesh: bool = False,
    keep_intermediate_mask: bool = False,
):
    """End-to-end PD dual-plane -> cartilage thickness on template.

    The DESS-grid mask produced by ``run_dualplane_inference`` is dropped into
    ``knee_mr_cartilage.process_one_patient(modality='dess', ...)`` — the
    cartilage projection runs unchanged.

    Returns whatever ``process_one_patient`` returns (a result dict).
    """
    # Lazy import — knee_mr_cartilage isn't a hard dependency of knee_mr_seg.
    from knee_mr_cartilage import PipelineConfig, process_one_patient

    if config is None:
        config = PipelineConfig()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_out = out_dir / f"{case_id or 'patient'}_v2_dess_mask.nii.gz"
    else:
        mask_out = None  # temp

    mask_path, _, _ = run_dualplane_inference(
        pd_sag_path=pd_sag_path,
        pd_cor_path=pd_cor_path,
        dess_path=dess_path,
        fusion_weights=fusion_weights,
        nnunet_results_dir=nnunet_results_dir,
        fold=fold,
        device=device,
        out_mask_path=mask_out,
    )

    result = process_one_patient(
        seg_path=mask_path,
        side=side,
        bone_name=bone_name,
        modality="dess",
        config=config,
        out_dir=out_dir,
        case_id=case_id,
        save_unregistered_mesh=save_unregistered_mesh,
    )

    if not keep_intermediate_mask and out_dir is None:
        try:
            mask_path.unlink()
            mask_path.parent.rmdir()
        except OSError:
            pass

    if isinstance(result, dict):
        result.setdefault("v2_fusion", {})
        result["v2_fusion"]["dess_mask_path"] = str(mask_path)
        result["v2_fusion"]["fusion_weights"] = str(fusion_weights)
        result["v2_fusion"]["fold"] = fold
    return result
