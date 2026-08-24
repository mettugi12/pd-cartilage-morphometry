"""Standardized response mean.

SRM = mean(Δ) / SD(Δ). Manuscript reference (n=166 progressors):
MFTC SRM = −1.59. Requires a cohort — returns NaN for n<2.
"""
from __future__ import annotations

import numpy as np


def srm(delta: np.ndarray) -> dict:
    d = np.asarray(delta, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        return {"srm": float("nan"), "mean_delta_mm": float(d.mean()) if n else float("nan"), "n": n}
    sd = float(d.std(ddof=1))
    mu = float(d.mean())
    return {"srm": (mu / sd) if sd > 0 else float("nan"), "mean_delta_mm": mu, "n": n}
