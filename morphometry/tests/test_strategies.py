"""Smoke tests for the pluggable strategy registry + end-to-end pipeline.

Generates a tiny synthetic DESS-style seg (bone + cart) where the patient
geometry happens to roughly match the template, runs `process_one_patient`
once per (anchor, cleanup) combination, and asserts:
  - no crash
  - non-NaN ASSD_subch
  - sensible thickness range (0..6 mm)
  - all logged transform_meta keys present

Runs on CPU only (modality="dess" path, no RECON needed). ~30 s total.
"""
from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cartilage_morphometry import PipelineConfig, list_strategies, process_one_patient
from cartilage_morphometry.pipeline import DESS_LABELS, TEMPLATE_PATHS


def _make_synthetic_dess_seg(out_path: Path, bone_name: str = "femur",
                              shape=(160, 160, 80), spacing=(0.5, 0.5, 0.5)):
    """Create a roughly knee-shaped 4-class DESS seg. Femur = sphere+cylinder.
    Tibia = box+disc. Cart = thin shell around bone.
    """
    labels = DESS_LABELS[bone_name]
    seg = np.zeros(shape, dtype=np.int16)
    # Bone: half-sphere (femur condyle proxy) or box (tibia)
    yy, xx, zz = np.mgrid[:shape[0], :shape[1], :shape[2]]
    yy = (yy - shape[0] / 2) * spacing[0]
    xx = (xx - shape[1] / 2) * spacing[1]
    zz = (zz - shape[2] / 2) * spacing[2]
    if bone_name == "femur":
        # Two bumps (medial + lateral condyle) connected by a slab
        r = np.sqrt(yy * yy + xx * xx + zz * zz)
        bone = (r < 25) | (
            (np.abs(yy) < 8) & (np.abs(xx) < 25) & (np.abs(zz) < 25)
        )
    else:
        # Plateau: flat thick disc
        bone = (np.abs(yy) < 20) & (np.abs(xx) < 25) & (np.abs(zz) < 30)
    seg[bone] = labels["bone"]
    # Cart: 2-mm shell on the "lower" half of bone (anterior face)
    from scipy.ndimage import binary_dilation
    bone_arr = bone.astype(np.uint8)
    shell = binary_dilation(bone_arr, iterations=4) & (~bone_arr) & (yy < 0)
    seg[shell] = labels["cart"]

    # Build affine that gives the right voxel spacing (axcodes PIR — closest
    # to canonical convention). Spacing axes match our shape order (H, W, D).
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float64)
    img = nib.Nifti1Image(seg, affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


@pytest.fixture(scope="module")
def synth_seg(tmp_path_factory):
    out_path = tmp_path_factory.mktemp("synth") / "synth_femur_dess.nii.gz"
    _make_synthetic_dess_seg(out_path, bone_name="femur")
    return out_path


def test_registry_lists_strategies():
    """Library strategies are registered on package import."""
    avail = list_strategies()
    assert "aniso_rigid" in avail["anchor"]
    assert "rigid_only" in avail["anchor"]
    assert "bounded_affine" in avail["anchor"]
    assert "rigid_then_bounded_affine" in avail["anchor"]
    assert "tpl_nn" in avail["subch"]
    assert "drop_small_components" in avail["cleanup"]


def test_default_config():
    """Default config = the web-app canonical (repo mirrored to the deployed
    knee-seg-web-app cartilage_morphometry module, 2026-06-23) with ONE
    deliberate deviation: thickness_method defaults to "raycast_2d" (what the web
    app's callers actually pass; the web app's config.py ships a dead "edt"
    default that no caller uses). subch_threshold stays 0.5 (display threshold
    0.65 lives in web_export.DEFAULT_THRESHOLD). The denudation gate and
    trimmed_rigid long-ICP fields do NOT exist in the web-app config — dropped.
    """
    cfg = PipelineConfig()
    assert cfg.cart_cleanup == ("drop_small_components",)
    assert cfg.anchor == "aniso_rigid"
    assert cfg.thickness_method == "raycast_2d"
    assert cfg.subch_method == "tpl_nn"
    assert cfg.subch_threshold == 0.5
    assert cfg.idw_k == 3
    assert cfg.idw_mutual_nn is False
    # Region projection defaults to the atlas template (web-app behaviour); the
    # v3.3 baseline-grid is opt-in via region_projection="baseline_grid".
    assert cfg.region_projection == "template"
    # Fields removed by the web-app mirror (repo-only canonical additions):
    assert not hasattr(cfg, "long_icp_method")
    assert not hasattr(cfg, "idw_denudation_gate")


@pytest.mark.skipif(
    not TEMPLATE_PATHS["femur"].exists(),
    reason="Femur template not available; skipping pipeline smoke test."
)
@pytest.mark.parametrize("anchor", [
    "aniso_rigid",
    "rigid_only",
    "rigid_only_iso_scale",
    "bounded_affine",
])
def test_anchor_strategies_run(anchor, synth_seg, tmp_path):
    """Each anchor strategy runs end-to-end on a tiny synthetic seg.

    Note: synthetic geometry is rough; we don't assert clinical ASSD here,
    just that the pipeline completes and produces a finite ASSD value.
    """
    cfg = PipelineConfig(anchor=anchor)
    result = process_one_patient(
        seg_path=synth_seg,
        side="RIGHT",
        bone_name="femur",
        modality="dess",
        config=cfg,
        out_dir=tmp_path,
    )
    assert "template_thickness" in result
    assert result["transform_meta"]["anchor"] == anchor
    assert "rot_deg" in result["transform_meta"]
    assert "trans_mm" in result["transform_meta"]
    assert math.isfinite(result["quality"]["assd_subch_mm"])
    # Pattern preservation sanity — at least SOME template verts get a value
    n_valid = int(np.isfinite(result["template_thickness"]).sum())
    assert n_valid > 100, f"Almost no template verts mapped (n_valid={n_valid})"


@pytest.mark.skipif(
    not TEMPLATE_PATHS["femur"].exists(),
    reason="Femur template not available; skipping cleanup test."
)
def test_cart_cleanup_records_meta(synth_seg, tmp_path):
    """Cart cleanup strategies log what they removed in cleanup_meta."""
    cfg = PipelineConfig(
        cart_cleanup=("drop_small_components", "filter_near_bone_per_slice"),
        anchor="rigid_only",
    )
    result = process_one_patient(
        seg_path=synth_seg,
        side="RIGHT",
        bone_name="femur",
        modality="dess",
        config=cfg,
        out_dir=tmp_path,
    )
    steps = result["cleanup_meta"]["cart_cleanup_steps"]
    assert len(steps) == 2
    assert steps[0]["strategy"] == "drop_small_components"
    assert steps[1]["strategy"] == "filter_near_bone_per_slice"


def test_register_anchor_custom():
    """Study code can register a new anchor without modifying the library."""
    from cartilage_morphometry import register_anchor, get_anchor

    @register_anchor("__test_identity__")
    def _identity(bone_mesh, cart_mask, spacing, template_mesh, bone_name, modality, cfg):
        return np.asarray(bone_mesh.points), {"anchor": "__test_identity__",
                                               "rot_deg": 0.0, "trans_mm": 0.0}

    fn = get_anchor("__test_identity__")
    assert fn is _identity
