"""Δ-thickness agreement vs Eckstein QCart.

Per-region Pearson r between pipeline Δ (V06 − V00, projected to cMF / cLF /
MT / LT subregions) and QCart Δ (BMFMTH / BLFMTH / WMTMTH / WLTMTH).

Manuscript reference (n=166 progressors):
    cMF r vs QCart Δ ≈ 0.67
    MT  r vs QCart Δ ≈ 0.48
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def region_delta_correlation(
    pipeline_delta: pd.DataFrame,    # columns: pid, side, cMF, cLF, MT, LT
    qcart_delta: pd.DataFrame,       # columns: ID, SIDE, BMFMTH, BLFMTH, WMTMTH, WLTMTH
) -> dict[str, float]:
    """Inner-join on (pid/ID, side/SIDE); return per-region Pearson r."""
    raise NotImplementedError(
        "Port from Studies/PD-vs-DESS/v3.3/analysis/analysis_eckstein_style.py."
    )
