"""
cartilage_morphometry — canonical knee-MR cartilage morphometry pipeline.

Consumes segmentation masks from knee_mr_seg (nnU-Net direct DESS, or PD via
RECON) and computes morphometry: per-vertex thickness, atlas-anchored
standardization, anatomical subregions, and DESS validation.

Submodules:
- pipeline / thickness / strategies — per-patient thickness computation
- subregions — anatomical 3D regions (Eckstein AP-distance) + 2D projections
- atlas — tibia-anchored atlas builders (soft-tissue / cartilage / patella)
- validation — DESS validation harness (locked PD-DESS cohorts, gate tests)

Renamed from knee-mr-cartilage to cartilage-morphometry in the 2026-05-28
refactor. The old `knee_mr_cartilage` package name is kept as a deprecation
shim for one release; update your imports to `cartilage_morphometry`.

Documented in Obsidian: knee-MR/Applications/Cartilage-Analysis.md
"""
from .config import PipelineConfig
from .pipeline import (
    process_one_patient,
    pull_subchondral_from_template,
    patient_to_template_align,
    rigid_icp,
    TEMPLATE_PATHS,
    DESS_LABELS,
    PD_LABELS,
)
from .thickness import compute_full_bone_thickness
from .remap import remap_thickness_to_template
from .strategies.anchor import TARGET_SCALE_MM
from .strategies import (
    register_anchor,
    register_cleanup,
    register_subch,
    register_smoothing,
    register_thickness,
    get_anchor,
    get_cleanup,
    get_subch,
    get_smoothing,
    get_thickness,
    list_strategies,
)
from .projection_2d import (
    template_thickness_2d_femur,
    template_thickness_2d_tibia,
    patient_thickness_2d_femur,
)
from .subregions import (
    FemurSubregions,
    fit_circle_2d,
    compute_femur_template_subregions,
    annotate_femur_template,
    femur_region_means_3d,
    TibiaSubregions,
    compute_tibia_template_subregions,
    annotate_tibia_template,
    tibia_region_means_3d,
    # Eckstein AP-distance femur subregions (anatomically correct cMF/cLF)
    FemurEcksteinSubregions,
    find_femur_eckstein_landmarks,
    compute_femur_eckstein_apsplit_subregions,
    femur_eckstein_region_means_3d,
    annotate_femur_eckstein_template,
    # Region label constants
    LBL_NON_SUBCH,
    LBL_MED,
    LBL_LAT,
    LBL_AP_NA,
    LBL_AP_ANTERIOR,
    LBL_AP_CENTRAL,
    LBL_AP_POSTERIOR,
    LBL_NOTCH_PRE,
    LBL_CMF_REGION,
    LBL_PMF_REGION,
    LBL_CLF_REGION,
    LBL_PLF_REGION,
)

__version__ = "0.6.0"

__all__ = [
    "PipelineConfig",
    "process_one_patient",
    "compute_full_bone_thickness",
    "remap_thickness_to_template",
    "pull_subchondral_from_template",
    "patient_to_template_align",
    "rigid_icp",
    "register_anchor",
    "register_cleanup",
    "register_subch",
    "register_smoothing",
    "register_thickness",
    "get_anchor",
    "get_cleanup",
    "get_subch",
    "get_smoothing",
    "get_thickness",
    "list_strategies",
    "template_thickness_2d_femur",
    "template_thickness_2d_tibia",
    "patient_thickness_2d_femur",
    "FemurSubregions",
    "fit_circle_2d",
    "compute_femur_template_subregions",
    "annotate_femur_template",
    "femur_region_means_3d",
    "TibiaSubregions",
    "compute_tibia_template_subregions",
    "annotate_tibia_template",
    "tibia_region_means_3d",
    "FemurEcksteinSubregions",
    "find_femur_eckstein_landmarks",
    "compute_femur_eckstein_apsplit_subregions",
    "femur_eckstein_region_means_3d",
    "annotate_femur_eckstein_template",
    "LBL_NON_SUBCH",
    "LBL_MED",
    "LBL_LAT",
    "LBL_AP_NA",
    "LBL_AP_ANTERIOR",
    "LBL_AP_CENTRAL",
    "LBL_AP_POSTERIOR",
    "LBL_NOTCH_PRE",
    "LBL_CMF_REGION",
    "LBL_PMF_REGION",
    "LBL_CLF_REGION",
    "LBL_PLF_REGION",
    "TEMPLATE_PATHS",
    "TARGET_SCALE_MM",
    "DESS_LABELS",
    "PD_LABELS",
]
