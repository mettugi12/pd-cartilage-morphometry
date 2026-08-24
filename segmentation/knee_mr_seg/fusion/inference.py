"""Single-plane nnU-Net inference helpers for Dataset211 (PD SAG) and
Dataset212 (PD COR), returning per-class softmax probability volumes in the
input's native voxel space.

Why a wrapper? ``nnUNetPredictor.predict_from_files`` writes a NIfTI argmax to
disk; for the fusion net we need the soft probabilities, not the argmax. The
predictor's ``predict_logits_from_preprocessed_data`` API returns logits but
expects pre-preprocessed input. We use ``predict_from_files`` with
``save_probabilities=True`` which yields a sidecar ``.npz`` containing the
per-class softmax already at the input's original voxel resolution.

Per-fold weight layout (after train_dualplane_pd.ipynb run):
    <NNUNET_RESULTS>/
        Dataset211_PD_SAG_DESSEq/
            nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres/
                fold_0/checkpoint_best.pth
                dataset.json, plans.json
        Dataset212_PD_COR_DESSEq/
            ...

Set the ``NNUNET_RESULTS_DIR`` env var or pass ``results_dir=`` explicitly.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_TRAINER = "nnUNetTrainer_500epochs"
DEFAULT_PLANS = "nnUNetResEncUNetMPlans"
DEFAULT_CONFIG = "3d_fullres"


def _resolve_model_dir(
    dataset_name: str,
    results_dir: Path | None,
    trainer: str = DEFAULT_TRAINER,
    plans: str = DEFAULT_PLANS,
    config: str = DEFAULT_CONFIG,
) -> Path:
    """Find the trainer dir under one of the conventional layouts:

        <results_dir>/<dataset_name>/<trainer>__<plans>__<config>/
        <results_dir>/<dataset_name>_results/<trainer>__<plans>__<config>/
        <results_dir>/<trainer>__<plans>__<config>/   (results_dir already at dataset level)
    """
    if results_dir is None:
        env = os.environ.get("NNUNET_RESULTS_DIR") or os.environ.get("nnUNet_results")
        if env is None:
            raise RuntimeError(
                "No results_dir given and neither NNUNET_RESULTS_DIR nor "
                "nnUNet_results env var is set. Point at the directory that "
                f"contains {dataset_name}[_results]/{trainer}__{plans}__{config}/fold_N/."
            )
        results_dir = Path(env)
    root = Path(results_dir)
    trainer_subpath = f"{trainer}__{plans}__{config}"
    candidates = [
        root / dataset_name / trainer_subpath,
        root / f"{dataset_name}_results" / trainer_subpath,
        root / trainer_subpath,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Model dir not found under {root}. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def make_predictor(
    dataset_name: str,
    fold: int = 0,
    results_dir: Path | None = None,
    trainer: str = DEFAULT_TRAINER,
    plans: str = DEFAULT_PLANS,
    config: str = DEFAULT_CONFIG,
    use_mirroring: bool = False,
    device: str | None = None,
):
    """Build an nnUNetPredictor for a (dataset, fold) pair.

    ``use_mirroring=False`` per CLAUDE.md feedback — doubles runtime for
    marginal gain.
    """
    # Lazy import — nnunetv2 is heavy; only required at call time.
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = _resolve_model_dir(dataset_name, results_dir, trainer, plans, config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(fold,), checkpoint_name="checkpoint_best.pth"
    )
    return predictor


def predict_softmax(
    predictor,
    image_path: Path | str,
    case_id: str | None = None,
    work_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Run nnU-Net inference and return per-class softmax probabilities in
    the input image's native voxel grid.

    Returns
    -------
    probs : (C, H, W, D) float32
        Per-class softmax probability volume at the input image's original
        spatial resolution.
    affine : (4, 4) float
        The input image's voxel-to-world affine, for downstream resampling.
    shape : tuple
        Original (H, W, D) of the input.

    nnU-Net's saved ``.npz`` from ``save_probabilities=True`` stores the
    softmax as a (C, ...) array under key 'probabilities' (older versions) or
    similar; we read it back, transpose if needed, and undo nnU-Net's
    internal axis transpose so the output shares the input NIfTI's voxel
    indexing.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if case_id is None:
        case_id = image_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")

    # Load the input to grab affine + shape (for resample step downstream)
    nii = nib.load(str(image_path))
    affine = nii.affine.copy()
    shape = tuple(nii.shape[:3])

    keep_tempdir = work_dir is not None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_pred_{case_id}_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net's predict_from_files wants a 4D-channel-stack list-of-lists:
    #   [[ch0_path], [ch0_path_case2], ...]
    # Single-channel here.
    out_argmax_path = work_dir / f"{case_id}.nii.gz"
    predictor.predict_from_files(
        [[str(image_path)]],
        [str(out_argmax_path)],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    # nnU-Net writes <out_dir>/<case_id>.npz (probabilities) + <case_id>.pkl (props)
    npz_path = work_dir / f"{case_id}.npz"
    pkl_path = work_dir / f"{case_id}.pkl"
    if not npz_path.exists():
        # Some nnU-Net versions use _seg.npz or similar
        candidates = list(work_dir.glob(f"{case_id}*.npz"))
        if not candidates:
            raise RuntimeError(
                f"nnU-Net produced no probability sidecar for {case_id} in {work_dir}. "
                f"Files: {[p.name for p in work_dir.iterdir()]}"
            )
        npz_path = candidates[0]
    with np.load(npz_path) as f:
        keys = list(f.keys())
        if "probabilities" in keys:
            probs = f["probabilities"]
        else:
            # Fallback: take the only array
            probs = f[keys[0]]

    # nnU-Net stores probs in (C, *spatial) where the spatial layout may
    # NOT match the input voxel order -- specifically for 3d_fullres, the
    # internal pipeline transposes the slab axis to position 0 (verified
    # against plans.json patch_size = [slab, *in_plane]). When the input
    # NIfTI has two equal spatial sizes (e.g. PD COR (384, 384, 37)), the
    # naive "match by size" heuristic that used to live here CANNOT pick
    # the right permutation -- I↔R got silently swapped, which collapsed
    # the COR-on-DESS-grid Dice to ~0.17 in the 2026-05-26 fusion run.
    #
    # The robust fix: nnU-Net also saves the argmax NIfTI with the input's
    # EXACT affine + shape. Pick the unique probs-axis permutation whose
    # argmax agrees with that NIfTI -- this disambiguates 384=384 cases.
    if probs.shape[1:] != shape:
        if sorted(probs.shape[1:]) != sorted(shape):
            raise RuntimeError(
                f"nnU-Net probs spatial shape {probs.shape[1:]} not a "
                f"permutation of input shape {shape}. Cannot reconcile."
            )
        if not out_argmax_path.exists():
            raise RuntimeError(
                f"probs spatial shape {probs.shape[1:]} != input {shape} "
                f"and no argmax NIfTI at {out_argmax_path} to disambiguate."
            )
        ref_argmax = np.asarray(nib.load(str(out_argmax_path)).dataobj).astype(np.uint8)
        if ref_argmax.shape != shape:
            raise RuntimeError(
                f"argmax NIfTI shape {ref_argmax.shape} != input shape "
                f"{shape}; nnU-Net invariant violated."
            )
        # Try all 6 permutations of the 3 spatial axes; pick the one
        # whose argmax agrees most with the saved argmax NIfTI.
        from itertools import permutations
        best_perm = None
        best_agree = -1.0
        for perm in permutations(range(3)):
            probs_p = np.transpose(probs, (0,) + tuple(p + 1 for p in perm))
            if probs_p.shape[1:] != shape:
                continue
            am = np.argmax(probs_p, axis=0).astype(np.uint8)
            agree = float((am == ref_argmax).mean())
            if agree > best_agree:
                best_agree = agree; best_perm = perm
        if best_perm is None or best_agree < 0.99:
            raise RuntimeError(
                f"Could not align probs axes to argmax NIfTI: best perm "
                f"{best_perm} achieved {best_agree:.3f} agreement (<0.99)."
            )
        probs = np.transpose(probs, (0,) + tuple(p + 1 for p in best_perm))
    probs = probs.astype(np.float32, copy=False)

    if not keep_tempdir:
        # Best-effort cleanup; ignore errors so a stale lock doesn't break the
        # outer pipeline.
        for p in (out_argmax_path, npz_path, pkl_path):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            work_dir.rmdir()
        except OSError:
            pass

    return probs, affine, shape


def predict_dataset211_softmax(image_path, fold=0, predictor=None, **kwargs):
    """Convenience: D211 (PD SAG) → (C, H, W, D) softmax + affine + shape."""
    if predictor is None:
        predictor = make_predictor("Dataset211_PD_SAG_DESSEq", fold=fold, **kwargs)
    return predict_softmax(predictor, image_path)


def predict_dataset212_softmax(image_path, fold=0, predictor=None, **kwargs):
    """Convenience: D212 (PD COR) → (C, H, W, D) softmax + affine + shape."""
    if predictor is None:
        predictor = make_predictor("Dataset212_PD_COR_DESSEq", fold=fold, **kwargs)
    return predict_softmax(predictor, image_path)
