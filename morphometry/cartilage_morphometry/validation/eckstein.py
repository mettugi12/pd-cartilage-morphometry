"""OAI Eckstein QCart loader — manual cartilage thickness GT for validation.

Source: `E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII/`
    kMRI_QCart_Eckstein00.txt  → V00 (baseline)
    kmri_qcart_eckstein06.txt  → V06 (= 48m followup)

Per the v3.3 manuscript pipeline (`Studies/PD-vs-DESS/v3.3/analysis/
analysis_eckstein_style.py`):

- Aggregate by mean across `readprj` rows for each `(ID, SIDE)`. The v3.3
  paper used project 22 (FNIH) but in practice the loader takes the mean
  across whatever projects are present, which equals the project-22 value
  for FNIH-only IDs.
- SIDE in the QCart file is encoded `1=RIGHT, 2=LEFT`. We expose both the
  encoded int (`SIDE`) and the string (`side`) for ease of merging.

Region → QCart column mapping (Eckstein convention, matches v3.3 paper):
    cMF  ← V{vv}BMFMTH   central weight-bearing medial femur (75% ROI)
    cLF  ← V{vv}BLFMTH   central weight-bearing lateral femur
    MT   ← V{vv}WMTMTH   whole medial tibia plate
    LT   ← V{vv}WLTMTH   whole lateral tibia plate
    MFTC = cMF + MT      total medial femoro-tibial cartilage
    LFTC = cLF + LT      total lateral femoro-tibial cartilage

Returned columns from `load_qcart_deltas()`:
    ID, SIDE, side, KL_grade,
    eck_cMF_00, eck_cLF_00, eck_MT_00, eck_LT_00, eck_MFTC_00, eck_LFTC_00,
    eck_cMF_48, eck_cLF_48, eck_MT_48, eck_LT_48, eck_MFTC_48, eck_LFTC_48,
    eck_cMF_d,  eck_cLF_d,  eck_MT_d,  eck_LT_d,  eck_MFTC_d,  eck_LFTC_d
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

OAI_ASCII_ROOT = Path(r"E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII")

# {region_name: QCart stem suffix}
ECK_COL = {
    "cMF": "BMFMTH",   # central medial femur, mean thickness
    "cLF": "BLFMTH",   # central lateral femur, mean thickness
    "MT":  "WMTMTH",   # whole medial tibia, mean thickness
    "LT":  "WLTMTH",   # whole lateral tibia, mean thickness
}

SIDE_INT_TO_STR = {1: "RIGHT", 2: "LEFT"}
SIDE_STR_TO_INT = {"RIGHT": 1, "LEFT": 2}


def _parse_prefix_code(val):
    """OAI SIDE strings look like "1: Right knee" — return the int prefix."""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if ":" in s:
        try:
            return int(s.split(":")[0].strip())
        except ValueError:
            return pd.NA
    try:
        return int(float(s))
    except ValueError:
        return pd.NA


def _resolve_qcart_file(visit_suffix: str, root: Path) -> Path:
    """Resolve `kMRI_QCart_Eckstein{vv}.txt` (case-insensitive)."""
    for cand in (root / f"kMRI_QCart_Eckstein{visit_suffix}.txt",
                 root / f"kmri_qcart_eckstein{visit_suffix}.txt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"QCart Eckstein file not found for visit {visit_suffix} under {root}"
    )


def _load_one_visit(visit_suffix: str, visit_tag: str,
                    root: Path = OAI_ASCII_ROOT) -> pd.DataFrame:
    """Returns ID, SIDE (int 1/2), eck_<region>_<tag> for each region."""
    fn = _resolve_qcart_file(visit_suffix, root)
    df = pd.read_csv(fn, sep="|", dtype={"ID": str}, low_memory=False)
    df["SIDE_i"] = df["SIDE"].map(_parse_prefix_code)
    prefix = f"V{visit_suffix}"
    cols_needed = ["ID", "SIDE_i"] + [f"{prefix}{stem}" for stem in ECK_COL.values()]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {fn.name}: {missing}")
    sub = df[cols_needed].rename(columns={"SIDE_i": "SIDE"}).copy()
    for region, stem in ECK_COL.items():
        sub[f"eck_{region}_{visit_tag}"] = pd.to_numeric(
            sub[f"{prefix}{stem}"], errors="coerce",
        )
    sub = sub.dropna(subset=["SIDE"])
    sub["SIDE"] = sub["SIDE"].astype(int)
    keep = ["ID", "SIDE"] + [f"eck_{r}_{visit_tag}" for r in ECK_COL]
    return sub[keep].groupby(["ID", "SIDE"], as_index=False).mean(numeric_only=True)


def load_qcart_deltas(root: Path | None = None) -> pd.DataFrame:
    """Load V00 + V06 (= 48m followup); merge; compute Δ + MFTC/LFTC totals.

    Returns one row per (ID, SIDE) with the 00m, 48m, and Δ values plus a
    string `side` column ("RIGHT" / "LEFT") for direct merging with cohort
    rows.
    """
    root = Path(root or OAI_ASCII_ROOT)
    df00 = _load_one_visit("00", "00", root=root)
    df48 = _load_one_visit("06", "48", root=root)
    out = df00.merge(df48, on=["ID", "SIDE"], how="inner")
    for region in ECK_COL:
        out[f"eck_{region}_d"] = out[f"eck_{region}_48"] - out[f"eck_{region}_00"]
    out["eck_MFTC_00"] = out["eck_cMF_00"] + out["eck_MT_00"]
    out["eck_LFTC_00"] = out["eck_cLF_00"] + out["eck_LT_00"]
    out["eck_MFTC_48"] = out["eck_cMF_48"] + out["eck_MT_48"]
    out["eck_LFTC_48"] = out["eck_cLF_48"] + out["eck_LT_48"]
    out["eck_MFTC_d"] = out["eck_cMF_d"] + out["eck_MT_d"]
    out["eck_LFTC_d"] = out["eck_cLF_d"] + out["eck_LT_d"]
    out["side"] = out["SIDE"].map(SIDE_INT_TO_STR)
    return out


def lookup_qcart_row(qcart: pd.DataFrame, pid: str, side: str) -> dict | None:
    """Return the QCart row for one (pid, side) as a plain dict, or None if
    not present."""
    side_int = SIDE_STR_TO_INT.get(side.upper())
    if side_int is None:
        return None
    m = (qcart["ID"] == str(pid)) & (qcart["SIDE"] == side_int)
    rows = qcart[m]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    # Convert pandas NaN → None for JSON-safety
    return {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in ("SIDE",)}
