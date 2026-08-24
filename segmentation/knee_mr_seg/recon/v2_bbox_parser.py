"""
RECONv2 bbox-dataset orientation parser.

The SNUH lesion-bbox NAS dataset
(`/volume1/homes/donghyuk/knee_mri_lesion_bounding_box/images/<pid>/{A,B,C[,D,E]}/`)
uses A/B/C/D/E subfolders per patient, but the letter does NOT fix orientation.
Survey of all 726 patients (2026-05-05) confirmed the same letter can be SAG,
COR, or AX in different patients. Always parse the filename for the orientation
token; never trust the subfolder letter.

Coverage on the full 726-patient survey:
- 714/726 have detectable SAG, 709/726 detectable COR, 702/726 BOTH.
- 32 files unknown/ambig (mostly scout/AP/lateral X-rays, plus 5
  Philips `ssPD mDixonCOR Water` Dixon scans where the COR token jams into
  CamelCase — handled here by an additional substring fallback).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

COR_TOKENS = {"COR", "CORONAL"}
AX_TOKENS = {"AX", "AXL", "AXIAL", "TRA", "TRANS", "TRANSVERSE"}
SAG_TOKENS = {"SAG", "SAGITTAL"}

# Substring fallback for jammed CamelCase patterns the tokenizer misses
# (e.g. `mDixonCOR`, `ssPDsagWater`). Only used when no token matched any class.
_FALLBACK = [
    ("COR", re.compile(r"(?i)cor(?=[A-Z]|_|\s|\.|$)")),
    ("SAG", re.compile(r"(?i)sag(?=[A-Z]|_|\s|\.|$)")),
    ("AX", re.compile(r"(?i)\bax(?=[A-Z]|_|\s|\.|$)")),
]


def detect_orientation(filename: str) -> Optional[str]:
    """Return 'SAG', 'COR', 'AX', or None (unknown/ambiguous).

    Tokenizes filename on non-alphanumeric, then checks for orientation tokens.
    If exactly one orientation class matches, returns it. If zero, falls back
    to a CamelCase boundary search (catches `mDixonCOR`, etc.). If multiple
    match unambiguously, returns None.
    """
    toks = {t.upper() for t in re.split(r"[^A-Za-z0-9]+", filename) if t}
    has_cor = bool(toks & COR_TOKENS)
    has_ax = bool(toks & AX_TOKENS)
    has_sag = bool(toks & SAG_TOKENS)
    n = sum([has_cor, has_ax, has_sag])
    if n == 1:
        return "COR" if has_cor else ("AX" if has_ax else "SAG")
    if n == 0:
        # CamelCase fallback
        hits = []
        for tag, pat in _FALLBACK:
            if pat.search(filename):
                hits.append(tag)
        if len(hits) == 1:
            return hits[0]
    return None


def index_patient(patient_dir: Path) -> dict[str, Path]:
    """Index one patient folder. Returns {'SAG': path, 'COR': path, 'AX': path}.

    For each orientation, picks the first NIfTI matching that orientation
    across all subfolders (A/B/C/D/E).
    """
    out: dict[str, Path] = {}
    for sub in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        for nii in sorted(sub.glob("*.nii.gz")):
            o = detect_orientation(nii.name)
            if o and o not in out:
                out[o] = nii
    return out


def index_dataset(images_root: Path) -> dict[str, dict[str, Path]]:
    """Index all patients under <images_root>/<pid>/."""
    return {
        pdir.name: index_patient(pdir)
        for pdir in sorted(p for p in images_root.iterdir() if p.is_dir())
    }
