"""ValidationReport — typed container + JSON (de)serialiser.

A report is a snapshot of one (PipelineConfig, seg_provider) run over the
cohorts. Two tracks live inside: `cross_sectional` (per-bone metrics over the
61 PD↔DESS pairs) and `longitudinal` (per-region Δ metrics over the v3.3
progressor cohort).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationReport:
    config: dict[str, Any] = field(default_factory=dict)        # PipelineConfig.to_dict()
    seg_provider: str = "dataset204"
    cross_sectional: dict[str, Any] = field(default_factory=dict)
    longitudinal: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0
    cache_hits: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=_default), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> "ValidationReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _default(o):
    # numpy / pathlib fall-through
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Unserialisable: {type(o)}")
