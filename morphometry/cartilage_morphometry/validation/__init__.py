"""cartilage_validation — single-call validation harness for cartilage_morphometry.

Public surface:
    validate(config, ...) -> ValidationReport
    ValidationReport (dataclass, JSON-serialisable)
    dataset204_provider, dataset211_provider, dualplane_fusion_provider (seg providers)
"""
from .api import validate
from .reports import ValidationReport
from .cohorts import dataset204_provider, dataset211_provider, dualplane_fusion_provider

__all__ = [
    "validate",
    "ValidationReport",
    "dataset204_provider",
    "dataset211_provider",
    "dualplane_fusion_provider",
]

__version__ = "0.1.0"
