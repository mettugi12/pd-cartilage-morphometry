"""Bland-Altman bias + 95% limits of agreement on paired (a − b)."""
from __future__ import annotations

import numpy as np


def bias_loa(a: np.ndarray, b: np.ndarray) -> dict:
    """Returns {bias_mm, sd_mm, loa_lower_mm, loa_upper_mm, n}."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 2:
        return {"bias_mm": float("nan"), "sd_mm": float("nan"),
                "loa_lower_mm": float("nan"), "loa_upper_mm": float("nan"),
                "n": n}
    d = a[m] - b[m]
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    return {
        "bias_mm": mu, "sd_mm": sd,
        "loa_lower_mm": mu - 1.96 * sd,
        "loa_upper_mm": mu + 1.96 * sd,
        "n": n,
    }
