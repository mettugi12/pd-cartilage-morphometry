"""PipelineConfig — swappable strategies for the cartilage thickness pipeline.

Each load-bearing decision (segmentation cleanup, registration anchor, subchondral
classifier, IDW remap, Δ-map smoothing) is selected by a string name that
resolves through the strategy registry in `knee_mr_cartilage.strategies`.

Default `PipelineConfig()` reproduces the current v1 behavior exactly so that
existing call sites that don't pass a config continue to work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    """Selects strategies for the per-patient cartilage pipeline.

    cart_cleanup: ordered tuple of cleanup-strategy names applied to the
        post-densify cart mask before meshing. Each strategy is looked up in
        the registry and called in order. Empty tuple = no cleanup (current
        default; PDvDESS 2026-05-21 recommends ("drop_small_components",
        "filter_near_bone_per_slice") for any modality whose seg comes from a
        network prone to far-from-bone hallucinations).

    anchor: registration-anchor strategy name. One of
        - "aniso_rigid"              (current default — aniso scale + translation + articular-subset rigid ICP)
        - "rigid_only"               (translation + rigid ICP on articular subset; NO scale)
        - "rigid_only_iso_scale"     (single isotropic scale from bone diagonal + translation + rigid ICP)
        - "bounded_affine"           (articular-subset bounded-affine ICP, sv clamped to [sv_min, sv_max])
        - "rigid_then_bounded_affine" (rigid first, then bounded-affine refinement; cascade)

    subch_method: subchondral classifier on aligned patient verts. One of
        - "tpl_nn"     (canonical: KDTree NN to template, subch_prob ≥ subch_threshold)
        - "cart_hit"   (subch ≡ thickness > 0; loses denudation analysis — diagnostic only)
        - "cart_prox"  (subch ≡ patient vert ≤ cart_prox_mm of any cart voxel)

    idw_k / idw_mutual_nn: forwarded to remap.remap_thickness_to_template.

    delta_smooth_sigma_mm: if set, callers doing paired/longitudinal Δ-maps
        can apply border-preserving Gaussian smoothing at this sigma on the
        template subch zone. Not applied inside process_one_patient (it sees
        only one patient); callers difference two results then call
        strategies.smoothing.border_preserving_gaussian.

    bounded_affine_sv_range: (sv_min, sv_max) for bounded-affine anchors.
    rigid_icp_max_iter / rigid_icp_tol: ICP knobs.
    articular_radius_mm: source mask for articular-subset alignment.
    """

    # Cart cleanup default since v0.5.2: drop CCs < 100 vox only. Protects the
    # femur aniso scale (cart-bbox-based) from being corrupted by far-from-bone
    # segmentation artefacts. PDvDESS 2026-05-21 root-cause analysis on OAI
    # 9226021 showed 28 spurious voxels could inflate sf_y by 1.87× and push
    # rigid ICP into a 42° local minimum — `drop_small_components` (min 100 vox)
    # catches those outliers cleanly.
    #
    # `filter_near_bone_per_slice` was dropped from the default in v0.5.2
    # because the per-slice 2D EDT on a 512×512×128 densified array is very
    # slow (~6 minutes per call). Available via opt-in for batch QA runs.
    # Set to () to disable cleanup entirely.
    cart_cleanup: Tuple[str, ...] = ("drop_small_components",)
    anchor: str = "aniso_rigid"
    subch_method: str = "tpl_nn"
    subch_threshold: float = 0.5
    cart_prox_mm: float = 5.0
    idw_k: int = 3
    idw_mutual_nn: bool = False
    delta_smooth_sigma_mm: float | None = None
    bounded_affine_sv_range: Tuple[float, float] = (0.7, 1.3)
    rigid_icp_max_iter: int = 30
    rigid_icp_tol: float = 1e-4
    articular_radius_mm: float = 5.0
    # Thickness method options:
    #   "edt"               DEFAULT (classical PD-vs-DESS canonical algorithm,
    #                       library default since v0.1.0). For each cart voxel,
    #                       distance-transform-edt to nearest bone voxel; per
    #                       bone vert, max over (2r+1)³ neighborhood = local
    #                       slab thickness. Mesh-normal-free → robust to
    #                       marching-cubes voxel-grid artefacts. Mild
    #                       under-estimate on thick / oblique cart.
    #   "template_raycast"  Raycast from TEMPLATE subch verts (smooth normals)
    #                       against patient cart mesh transformed into template
    #                       frame. Stable rendering but underestimates whenever
    #                       patient cart slab is tilted relative to template
    #                       normal, plus the affine cart-transform compresses
    #                       the slab along the aniso-scaling axes.
    #   "raycast"           Patient-frame first-slab raycast (per-vertex on
    #                       patient bone mesh, then IDW remap to template).
    #                       Voxel-grid mesh artefacts make this noisy.
    #   "raycast_to_far"    Legacy v6 variant (bone-to-far-cart). Over-counts
    #                       by the bone-cart gap. Provided for backward compat.
    #   "raycast_2d"        Per-sagittal-slice 2D raycast from the bone contour
    #                       through the cart mask. This is the repo default and
    #                       what the web app actually runs (its callers pass
    #                       PipelineConfig(thickness_method="raycast_2d") — the
    #                       web app's config.py still ships "edt" as a dead
    #                       default, but no caller uses it). edt is retired here.
    thickness_method: str = "raycast_2d"
    max_thickness_mm: float = 6.0
    cart_smooth_iters: int = 15
    edt_search_radius_voxels: int = 3
    min_art_dist_mm: float = 0.5

    # Regional projection method (longitudinal validation path only — the
    # web-app single-timepoint pipeline does not use this):
    #   "template"      DEFAULT. Map per-vertex thickness onto the fixed
    #                   cross-patient atlas bone template, read regional means off
    #                   the template. What the deployed web app / validate() uses.
    #   "baseline_grid" OPT-IN. Restore the v3.3 PD-DESS method: parameterise each
    #                   knee by a 40x40 ML(D)xAP(W) grid built from its OWN baseline
    #                   (00m) bone geometry, project both timepoints onto that
    #                   intrinsic grid. Recovers stronger longitudinal QCart
    #                   agreement (esp. medial tibia) by avoiding cross-patient
    #                   template-registration variance. See cartilage_morphometry
    #                   .baseline_grid and Studies/PD-vs-DESS v8b.
    region_projection: str = "template"

    # Femur unwrap for region_projection="baseline_grid" (FC only):
    #   "per_slice"        v3.3 default — arc around the per-ML-slice bone centroid.
    #   "best_fit_circle"  web-app style — single straight ML axis at the best-fit
    #                      circle centre; smoother arc. Changes femur regional means
    #                      (cMF/cLF/MFTC); tibia unaffected. See cartilage_morphometry
    #                      .baseline_grid and Studies/PD-vs-DESS v8b.
    femur_unwrap: str = "per_slice"

    def to_dict(self) -> dict:
        return {
            "cart_cleanup": list(self.cart_cleanup),
            "anchor": self.anchor,
            "subch_method": self.subch_method,
            "subch_threshold": self.subch_threshold,
            "cart_prox_mm": self.cart_prox_mm,
            "idw_k": self.idw_k,
            "idw_mutual_nn": self.idw_mutual_nn,
            "delta_smooth_sigma_mm": self.delta_smooth_sigma_mm,
            "bounded_affine_sv_range": list(self.bounded_affine_sv_range),
            "rigid_icp_max_iter": self.rigid_icp_max_iter,
            "rigid_icp_tol": self.rigid_icp_tol,
            "articular_radius_mm": self.articular_radius_mm,
            "thickness_method": self.thickness_method,
            "max_thickness_mm": self.max_thickness_mm,
            "cart_smooth_iters": self.cart_smooth_iters,
            "edt_search_radius_voxels": self.edt_search_radius_voxels,
            "min_art_dist_mm": self.min_art_dist_mm,
            "region_projection": self.region_projection,
            "femur_unwrap": self.femur_unwrap,
        }
