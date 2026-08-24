"""Cohort loaders + segmentation providers.

Two cohorts, both backed by data on E:/.

- Cross-sectional: 61 PD↔DESS pairs.
    manifest:   E:/Others/KneeMR/Datasets/PD-DESS-Pairs/PD_DESS_pairs/metadata_61_pairs.json
    pair folder: E:/Others/KneeMR/Datasets/PD-DESS-Pairs/Chae_pairs/nifti/<folder>/
                {PD_labels_native, DESS_labels}.nii.gz
    Manifest keys are <pid>_3D_DESS_<SIDE>[_flipped2R]; the folder name drops the
    "_3D_DESS_" infix (e.g. manifest key `9013941_3D_DESS_RIGHT` → folder
    `9013941_RIGHT`).

- Longitudinal v3.3:
    cohort CSV: E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/cohort_v33_combined.csv
    PD segs:    E:/KneeMR/Datasets/segmentation_canonical/legacy/PDvDESS/
                v32_seg_{00m,48m}/<pid>_<SIDE>_seg.nii.gz
    (Manuscript dedup is to 166 progressors via seed=42; this scaffold loads
    all rows and lets the driver subset/dedup.)

Seg providers are callables (case_id, modality, side) -> Path[.nii.gz]. The
default `dataset204_provider` just resolves to the on-disk paths above.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


# --- canonical heavy-data paths (E:/ — verified 2026-05-23) ----------------
PAIRS_MANIFEST = Path(
    r"E:/Others/KneeMR/Datasets/PD-DESS-Pairs/PD_DESS_pairs/metadata_61_pairs.json"
)
PAIRS_NIFTI_ROOT = Path(
    r"E:/Others/KneeMR/Datasets/PD-DESS-Pairs/Chae_pairs/nifti"
)

V33_COHORT_CSV = Path(
    r"E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/cohort_v33_combined.csv"
)
IW_SEG_00M_DIR = Path(
    r"E:/KneeMR/Datasets/segmentation_canonical/legacy/PDvDESS/v32_seg_00m"
)
IW_SEG_48M_DIR = Path(
    r"E:/KneeMR/Datasets/segmentation_canonical/legacy/PDvDESS/v32_seg_48m"
)

# Dataset211 (PD SAG → DESS-equivalent 5-class) inference cache.
# Layout: <root>/{61pairs,long_00m,long_48m}/<stem>_seg.nii.gz
# Produced by KneeMR/Repos/knee-mr-seg/scripts/modal_batch_inference.py --model d211_sag
D211_INFERENCE_ROOT = Path(
    r"E:/KneeMR/Datasets/segmentation_canonical/derived/Dataset211_inference"
)

# Default RECON disk-cache root — set by api.validate() if the caller doesn't
# override. Persists densified PD volumes across runs so swapping anchor /
# subch / thickness configs doesn't pay RECON again.
DEFAULT_RECON_CACHE_DIR = Path(r"E:/KneeMR/Studies/cartilage-validation/recon_cache")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
SegProvider = Callable[[str, str, str], Path]
"""(case_id, modality, side) -> Path to an 11-class label NIfTI."""


@dataclass
class PairCase:
    case_id: str            # folder name, e.g. "9013941_RIGHT" or "9014883_LEFT_flipped2R"
    manifest_key: str       # original manifest key, e.g. "9013941_3D_DESS_RIGHT"
    side_anatomical: str    # "LEFT" or "RIGHT" — preserved from manifest
    kl_grade: int
    split: str
    pd_seg_path: Path
    dess_seg_path: Path
    folder: Path

    @property
    def side_for_pipeline(self) -> str:
        """`_flipped2R` folders are pre-flipped → pass RIGHT so the library's
        empirical ML-flip doesn't double-flip on top of pre-flipped voxels."""
        return "RIGHT" if str(self.folder).endswith("_flipped2R") else self.side_anatomical


@dataclass
class LongCase:
    pid: str
    side: str
    KL_00: int
    BMI_00: float
    JSW_00: float
    JSW_48: float
    dJSW_48: float
    worst_dJSW: float
    cohort: str             # "progressor" | "nonprogressor"
    seg_00m_path: Path
    seg_48m_path: Path


# ---------------------------------------------------------------------------
# Cross-sectional cohort loader
# ---------------------------------------------------------------------------
def _manifest_to_folder(manifest_key: str) -> str:
    """`9013941_3D_DESS_RIGHT` → `9013941_RIGHT`; preserves trailing `_flipped2R`."""
    return manifest_key.replace("_3D_DESS_", "_")


def load_pair_cases(manifest_path: Path | None = None,
                    nifti_root: Path | None = None) -> list[PairCase]:
    """Load the 61 PD↔DESS pairs; drop entries whose folder is missing."""
    manifest_path = Path(manifest_path or PAIRS_MANIFEST)
    nifti_root = Path(nifti_root or PAIRS_NIFTI_ROOT)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    out: list[PairCase] = []
    for key, meta in raw.items():
        case_id = _manifest_to_folder(key)
        folder = nifti_root / case_id
        pd_seg = folder / "PD_labels_native.nii.gz"
        dess_seg = folder / "DESS_labels.nii.gz"
        if not (pd_seg.exists() and dess_seg.exists()):
            continue
        out.append(PairCase(
            case_id=case_id,
            manifest_key=key,
            side_anatomical=meta["side"],
            kl_grade=int(meta["kl_grade"]),
            split=meta.get("split", "unknown"),
            pd_seg_path=pd_seg,
            dess_seg_path=dess_seg,
            folder=folder,
        ))
    return out


# ---------------------------------------------------------------------------
# Longitudinal cohort loader
# ---------------------------------------------------------------------------
def load_v33_cohort(csv_path: Path | None = None,
                    progressor_only: bool = True,
                    seg_00m_dir: Path | None = None,
                    seg_48m_dir: Path | None = None,
                    require_segs_present: bool = True) -> list[LongCase]:
    """Load v3.3 cohort and attach 00m / 48m PD seg paths.

    Args:
        progressor_only: keep only `cohort == "progressor"` rows.
        require_segs_present: drop rows whose 00m or 48m seg file is missing.
            With both files required, this leaves only the cases the
            validator can actually run.
    """
    csv_path = Path(csv_path or V33_COHORT_CSV)
    seg_00m_dir = Path(seg_00m_dir or IW_SEG_00M_DIR)
    seg_48m_dir = Path(seg_48m_dir or IW_SEG_48M_DIR)

    df = pd.read_csv(csv_path)
    if progressor_only:
        df = df[df["cohort"] == "progressor"].copy()

    out: list[LongCase] = []
    for _, row in df.iterrows():
        stem = f"{row['pid']}_{row['side']}_seg.nii.gz"
        p00 = seg_00m_dir / stem
        p48 = seg_48m_dir / stem
        if require_segs_present and not (p00.exists() and p48.exists()):
            continue
        out.append(LongCase(
            pid=str(row["pid"]),
            side=str(row["side"]),
            KL_00=int(row.get("KL_00", -1)),
            BMI_00=float(row.get("BMI_00", float("nan"))),
            JSW_00=float(row.get("JSW_00", float("nan"))),
            JSW_48=float(row.get("JSW_48", float("nan"))),
            dJSW_48=float(row.get("dJSW_48", float("nan"))),
            worst_dJSW=float(row.get("worst_dJSW", float("nan"))),
            cohort=str(row["cohort"]),
            seg_00m_path=p00,
            seg_48m_path=p48,
        ))
    return out


# ---------------------------------------------------------------------------
# Seg providers
# ---------------------------------------------------------------------------
def dataset204_provider(case_id: str, modality: str, side: str) -> Path:
    """Default provider for v1: PD segs come from on-disk Dataset204 inferences
    (Chae_pairs/nifti/<id>/PD_labels_native.nii.gz for cohort 1,
    v32_seg_{00m,48m}/<pid>_<side>_seg.nii.gz for cohort 2). DESS GT is the
    manual 11-class label file shipped with the pair.

    Implementation note: this provider is not actually called by the v1
    drivers — they read paths directly off the cohort objects (PairCase /
    LongCase) which already encode the modality/timepoint. The provider
    exists for v2 (dual-plane fusion), where the seg has to be generated
    per call instead of fetched off disk.
    """
    raise NotImplementedError(
        "dataset204_provider is unused for v1 — paths come off PairCase / "
        "LongCase. Use it once you wire a non-default seg backend."
    )


def dataset211_provider(case_id: str, modality: str, side: str) -> Path:
    """Dataset211 PD-SAG-to-DESS-equivalent 5-class inference cache.

    Resolves to a pre-computed nnU-Net D211 SAG inference output. Outputs are
    5-class (BG, FB, FC, TB, TC) on the PD-SAG voxel grid — matches the
    library's default `DESS_LABELS` (femur bone=1 cart=2, tibia bone=3 cart=4),
    so the cartilage pipeline must process this as ``modality='dess',
    dess_is_11class=False``. The validator's `pd_is_dess_equivalent=True`
    flag routes the PD arm through that code path.

    Modality keys this provider understands:
      - "pd"       → cross-sectional 61pairs cohort
      - "pd_00m"   → longitudinal 00m timepoint
      - "pd_48m"   → longitudinal 48m timepoint
    """
    if modality == "pd":
        return D211_INFERENCE_ROOT / "61pairs" / f"{case_id}_seg.nii.gz"
    if modality == "pd_00m":
        return D211_INFERENCE_ROOT / "long_00m" / f"{case_id}_seg.nii.gz"
    if modality == "pd_48m":
        return D211_INFERENCE_ROOT / "long_48m" / f"{case_id}_seg.nii.gz"
    raise ValueError(
        f"dataset211_provider: unknown modality {modality!r}; "
        "expected 'pd' | 'pd_00m' | 'pd_48m'"
    )


def dualplane_fusion_provider(case_id: str, modality: str, side: str) -> Path:
    """v2 stub — PDvDESS v7 dual-plane fusion seg. Folds 1-4 + fusion net are
    still training (2026-05-23)."""
    raise NotImplementedError("dual-plane fusion net is still training — v2 placeholder.")
