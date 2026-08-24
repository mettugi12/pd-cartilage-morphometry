"""Top-level validate() entry point.

One call → one ValidationReport. Drives the cross-sectional and longitudinal
tracks with the same PipelineConfig + seg provider so submodule comparisons
(anchor / thickness / subch / cleanup / fusion seg) are a one-liner.

Default registration mode is **shared-mesh** (the v6 canonical), so the
PD↔DESS cross-sectional and 00m↔48m longitudinal numbers track the
manuscript's `raycast_shared_mesh` configuration.
"""
from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyvista as pv

from cartilage_morphometry import (
    PipelineConfig,
    TEMPLATE_PATHS,
    compute_femur_eckstein_apsplit_subregions,
    compute_femur_template_subregions,
    compute_tibia_template_subregions,
    femur_eckstein_region_means_3d,
    tibia_region_means_3d,
)
from cartilage_morphometry import pipeline as _pipeline

from .cohorts import (
    DEFAULT_RECON_CACHE_DIR,
    SegProvider,
    dataset204_provider,
    dataset211_provider,
    dualplane_fusion_provider,
    load_pair_cases,
    load_v33_cohort,
)
from .eckstein import load_qcart_deltas, lookup_qcart_row
from .metrics.bland_altman import bias_loa
from .metrics.correlation import per_vertex_mae, per_vertex_pearson, r_2d_smooth
from .metrics.srm import srm
from .patient_regions import femur_patient_regions, tibia_patient_regions
from .reports import ValidationReport
from .shared_mesh import (
    process_long_independent,
    process_long_shared_mesh,
    process_pair_independent,
    process_pair_shared_mesh,
)

_PROVIDERS: dict[str, SegProvider] = {
    "dataset204": dataset204_provider,
    "dataset211": dataset211_provider,
    "pd_dualplane_fusion": dualplane_fusion_provider,
}
# Providers whose PD arm yields a DESS-equivalent 5-class mask on the PD voxel
# grid (vs the legacy 12-class SNUH labelling). Drives `pd_is_dess_equivalent`
# in the shared-mesh / independent dispatchers.
_DESS_EQUIVALENT_PROVIDERS = frozenset({"dataset211", "pd_dualplane_fusion"})

BONES = ("femur", "tibia")
# Region names produced by the library's region_means helpers (matches QCart).
FEMUR_REGIONS = ("cMF", "pMF", "cLF", "pLF")
TIBIA_REGIONS = ("MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT")

# Eckstein QCart columns we attach per case as GT (cMF/cLF/MT/LT + composites)
QCART_REGIONS = ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC")
# Map our pipeline region key → QCart region key for cohort-level r vs QCart
PIPELINE_TO_QCART = {"cMF": "cMF", "cLF": "cLF", "MT": "MT", "LT": "LT"}


# ---------------------------------------------------------------------------
# Template-aware cache
# ---------------------------------------------------------------------------
_TEMPLATE_CACHE: dict[str, dict] = {}


def _template_pack(bone_name: str) -> dict:
    """Load template mesh + precompute the two subregion variants we need.

    Femur uses `fraction=0.75` to match Eckstein's manuscript convention
    (BMFMTH = central weight-bearing 75 % ROI between trochlear notch and
    posterior condyle tip). Library default is 0.60 which is too anterior
    and under-samples the progressor-loss region.
    """
    if bone_name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[bone_name]
    mesh = pv.read(str(TEMPLATE_PATHS[bone_name]))
    if bone_name == "femur":
        pack = {
            "mesh": mesh,
            "eckstein_sub": compute_femur_eckstein_apsplit_subregions(mesh, fraction=0.75),
            "projection_sub": compute_femur_template_subregions(mesh),
        }
    else:
        pack = {
            "mesh": mesh,
            "eckstein_sub": compute_tibia_template_subregions(mesh),
            "projection_sub": None,
        }
    _TEMPLATE_CACHE[bone_name] = pack
    return pack


def _region_means(bone_name: str, thickness: np.ndarray) -> dict:
    """Region means with zero-imputation, matching the v3.3 manuscript +
    Eckstein QCart conventions (denuded subch verts count as 0 mm thickness,
    not NaN). With zero_impute=False, severe-OA cases with denuded patches
    have shrunken-magnitude regional Δ → SRM + r-vs-QCart both drop because
    QCart is area-weighted with-zeros."""
    pack = _template_pack(bone_name)
    if bone_name == "femur":
        return femur_eckstein_region_means_3d(thickness, pack["eckstein_sub"], zero_impute=True)
    return tibia_region_means_3d(thickness, pack["eckstein_sub"], zero_impute=True)


def _r_2d_smooth_pair(bone_name: str, a: np.ndarray, b: np.ndarray, sigma: float = 1.5):
    pack = _template_pack(bone_name)
    return r_2d_smooth(pack["mesh"], a, b, bone_name, sigma=sigma,
                       femur_subregions=pack["projection_sub"])


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def validate(
    config: PipelineConfig,
    seg_provider: str | SegProvider = "dataset204",
    cross_sectional: bool = True,
    longitudinal: bool = True,
    progressor_only: bool = True,
    n_run: int | None = None,
    recon_cache_dir: Path | None = "default",
    shared_mesh: bool = True,
    r_2d_sigma: float = 1.5,
    save_thickness_dir: Path | None = None,
) -> ValidationReport:
    """Run the cross-sectional + longitudinal validation tracks.

    Args:
        config: PipelineConfig under test.
        seg_provider: "dataset204" (default, 12-class SNUH PD seg from on-disk
            Chae_pairs + v32_seg_{00m,48m}) | "dataset211" (5-class DESS-equivalent
            PD seg from cached Modal inference at
            E:/KneeMR/Datasets/segmentation_canonical/derived/Dataset211_inference/)
            | "pd_dualplane_fusion" (v2 stub).
        cross_sectional / longitudinal: enable each track.
        progressor_only: longitudinal cohort filter.
        n_run: cap N cases per cohort. Use small (1-10) for smoke tests.
        recon_cache_dir: "default" → DEFAULT_RECON_CACHE_DIR; None disables.
        shared_mesh: True (default) → v6 canonical (one bone mesh as the
            shared sampling surface for both arms). False → independent
            registration of each arm.
        r_2d_sigma: σ (grid cells, not mm) for the border-preserving Gaussian
            in r_2d_smooth. Manuscript value is 1.5.
    """
    provider_name = seg_provider if isinstance(seg_provider, str) else getattr(seg_provider, "__name__", "callable")
    pd_is_dess_equivalent = provider_name in _DESS_EQUIVALENT_PROVIDERS

    if recon_cache_dir == "default":
        recon_cache_dir = DEFAULT_RECON_CACHE_DIR
    if recon_cache_dir is not None:
        Path(recon_cache_dir).mkdir(parents=True, exist_ok=True)
        _pipeline.set_recon_disk_cache_dir(Path(recon_cache_dir))
        _patch_disk_cache_path_unique()
        print(f"[recon] disk cache: {recon_cache_dir}")

    print(f"[mode] shared_mesh={shared_mesh}  thickness={config.thickness_method}  "
          f"anchor={config.anchor}  r_2d_sigma={r_2d_sigma}")

    t0 = time.time()
    cs_metrics: dict = {}
    long_metrics: dict = {}
    notes: list[str] = []

    save_cs_dir = save_long_dir = None
    if save_thickness_dir is not None:
        save_cs_dir = Path(save_thickness_dir) / "cross_sectional"
        save_long_dir = Path(save_thickness_dir) / "longitudinal"
        save_cs_dir.mkdir(parents=True, exist_ok=True)
        save_long_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save] thickness arrays → {save_thickness_dir}")

    if cross_sectional:
        cs_metrics = _run_cross_sectional(config, n_run=n_run,
                                          shared_mesh=shared_mesh,
                                          r_2d_sigma=r_2d_sigma,
                                          save_dir=save_cs_dir,
                                          provider_name=provider_name,
                                          pd_is_dess_equivalent=pd_is_dess_equivalent)

    if longitudinal:
        try:
            qcart = load_qcart_deltas()
            print(f"[qcart] loaded {len(qcart)} (ID, SIDE) rows from OAI Eckstein")
        except Exception as e:
            qcart = None
            notes.append(f"qcart load failed: {e}")
            print(f"[qcart] FAILED to load ({e}); longitudinal report will not have QCart GT")
        long_metrics = _run_longitudinal(
            config, progressor_only=progressor_only, n_run=n_run,
            shared_mesh=shared_mesh, qcart=qcart,
            save_dir=save_long_dir,
            provider_name=provider_name,
            pd_is_dess_equivalent=pd_is_dess_equivalent,
        )

    return ValidationReport(
        config=config.to_dict(),
        seg_provider=provider_name,
        cross_sectional=cs_metrics,
        longitudinal=long_metrics,
        runtime_s=time.time() - t0,
        cache_hits={"recon_cache_dir": str(recon_cache_dir) if recon_cache_dir else None},
        notes=notes + [f"shared_mesh={shared_mesh}", f"r_2d_sigma={r_2d_sigma}"],
    )


def _patch_disk_cache_path_unique():
    """Make the library's RECON disk-cache key path-unique AND flip-aware.

    Two reasons to override the library default:
    1. Library keys on filename only → collides across pairs / 00m vs 48m
       that share the same stem (e.g. PD_labels_native.nii.gz in every
       pair folder). Fix: md5(full_path).
    2. When `cartilage_validation.shared_mesh.ACTIVE_FLIP_OVERRIDE` is set
       (v1.2 consistent-flip mode for longitudinal), the actual ML-flip
       applied may differ from the per-timepoint empirical decision used
       the first time RECON ran on this seg. Old cache file holds densified
       output in the empirical orientation; new run wants override
       orientation. Fix: tag the cache key with `_flipT` / `_flipF`.
       Tag is added only when override is active — backward-compat with
       pre-1.2 caches that ran without any override.
    """
    import hashlib

    from . import shared_mesh as _sm

    def _disk_cache_path(cache_key):
        if _pipeline._RECON_DISK_CACHE_DIR is None or cache_key is None:
            return None
        stem = Path(cache_key).name.replace(".nii.gz", "").replace(".nii", "")
        h = hashlib.md5(str(cache_key).encode("utf-8")).hexdigest()[:12]
        flip_tag = ""
        if _sm.ACTIVE_FLIP_OVERRIDE is not None:
            flip_tag = f"_flip{'T' if _sm.ACTIVE_FLIP_OVERRIDE else 'F'}"
        return _pipeline._RECON_DISK_CACHE_DIR / f"{stem}{flip_tag}__{h}.recon.npz"

    _pipeline._disk_cache_path = _disk_cache_path


# ---------------------------------------------------------------------------
# Cross-sectional — 61 PD↔DESS pairs
# ---------------------------------------------------------------------------
def _run_cross_sectional(config: PipelineConfig, n_run: int | None,
                         shared_mesh: bool, r_2d_sigma: float,
                         save_dir: Path | None = None,
                         provider_name: str = "dataset204",
                         pd_is_dess_equivalent: bool = False) -> dict:
    cases = load_pair_cases()
    if n_run is not None:
        cases = cases[:n_run]
    if not cases:
        return {"per_case": [], "per_bone": {}, "n_total": 0}

    # If a non-default seg provider was selected, swap each case's PD seg path
    # for the provider's resolved path. Cases whose D211 inference file is
    # missing are skipped with a warning (no point dispatching a guaranteed
    # FileNotFoundError).
    if provider_name == "dataset211":
        provider = _PROVIDERS["dataset211"]
        kept: list = []
        for c in cases:
            p = provider(c.case_id, "pd", c.side_anatomical)
            if not p.exists():
                print(f"[cs skip] {c.case_id}: D211 inference missing → {p}")
                continue
            c.pd_seg_path = p
            kept.append(c)
        print(f"[cs] dataset211: {len(kept)}/{len(cases)} pairs have D211 inference present")
        cases = kept

    fn = process_pair_shared_mesh if shared_mesh else process_pair_independent
    extra_kw = {"pd_is_dess_equivalent": True} if pd_is_dess_equivalent else {}
    per_case: list[dict] = []
    per_bone_rows: dict[str, list[dict]] = {b: [] for b in BONES}

    for ci, case in enumerate(cases, 1):
        # Free the in-memory RECON cache between cases — the library keeps
        # densified volumes (~150 MB each) in module-level _RECON_CACHE so a
        # full run accumulates GB of RAM. Disk cache stays, so re-reads on
        # second visit are still fast.
        _pipeline.clear_recon_cache()
        gc.collect()
        print(f"\n=== [cs {ci}/{len(cases)}] {case.case_id} KL={case.kl_grade} "
              f"side={case.side_for_pipeline} ===")
        case_row: dict = {"case_id": case.case_id, "kl_grade": case.kl_grade,
                          "side_anatomical": case.side_anatomical,
                          "split": case.split, "bones": {}}
        for bone in BONES:
            print(f"  --- bone={bone} ---")
            try:
                res = fn(case.pd_seg_path, case.dess_seg_path,
                         side=case.side_for_pipeline, bone_name=bone, config=config,
                         dess_is_11class=True, **extra_kw)
                pd_t, de_t = res["pd_template_thickness"], res["dess_template_thickness"]
                if save_dir is not None:
                    np.save(save_dir / f"{case.case_id}_{bone}_pd.npy", pd_t)
                    np.save(save_dir / f"{case.case_id}_{bone}_dess.npy", de_t)
                r = per_vertex_pearson(pd_t, de_t)
                mae = per_vertex_mae(pd_t, de_t)
                ba = bias_loa(pd_t, de_t)
                r2d = _r_2d_smooth_pair(bone, pd_t, de_t, sigma=r_2d_sigma)
                bone_row = {
                    "r_pearson": r["r"], "n_valid": r["n"],
                    "r_2d_smooth": r2d["r"], "n_valid_2d": r2d["n_valid"],
                    "mae_mm": mae["mae_mm"],
                    "bias_mm": ba["bias_mm"], "sd_mm": ba["sd_mm"],
                    "loa_lower_mm": ba["loa_lower_mm"], "loa_upper_mm": ba["loa_upper_mm"],
                    "pd_quality": res.get("pd_quality"),
                    "dess_quality": res.get("dess_quality"),
                    "pair_quality": res.get("pair_quality"),
                }
                print(f"    r_vert={bone_row['r_pearson']:+.3f}  "
                      f"r_2d_smooth={bone_row['r_2d_smooth']:+.3f}  "
                      f"bias={bone_row['bias_mm']:+.3f}mm  MAE={bone_row['mae_mm']:.3f}mm  n={bone_row['n_valid']}")
            except Exception as e:
                traceback.print_exc()
                bone_row = {"error": f"{type(e).__name__}: {e}"}
            case_row["bones"][bone] = bone_row
            per_bone_rows[bone].append({"case_id": case.case_id, **bone_row})
        per_case.append(case_row)

    def _agg(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), float) and np.isfinite(r[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                len(vals))

    per_bone_summary: dict = {}
    for bone, rows in per_bone_rows.items():
        m_rv, s_rv, n_rv = _agg(rows, "r_pearson")
        m_r2, s_r2, n_r2 = _agg(rows, "r_2d_smooth")
        m_b, s_b, _ = _agg(rows, "bias_mm")
        m_m, _, _ = _agg(rows, "mae_mm")
        per_bone_summary[bone] = {
            "r_pearson_mean": m_rv, "r_pearson_std": s_rv,
            "r_2d_smooth_mean": m_r2, "r_2d_smooth_std": s_r2,
            "bias_mm_mean": m_b, "bias_mm_std": s_b,
            "mae_mm_mean": m_m, "n_cases": n_rv,
        }
    return {"per_case": per_case, "per_bone": per_bone_summary, "n_total": len(cases)}


# ---------------------------------------------------------------------------
# Longitudinal — v3.3 progressor 00m↔48m + Eckstein QCart GT
# ---------------------------------------------------------------------------
def _qcart_block(qcart: pd.DataFrame | None, pid: str, side: str) -> dict | None:
    """Return a compact QCart GT block for this case, JSON-friendly."""
    if qcart is None:
        return None
    row = lookup_qcart_row(qcart, pid, side)
    if row is None:
        return {"present": False}
    out: dict = {"present": True}
    for r in QCART_REGIONS:
        out[f"{r}_00mm"] = row.get(f"eck_{r}_00")
        out[f"{r}_48mm"] = row.get(f"eck_{r}_48")
        out[f"{r}_d"] = row.get(f"eck_{r}_d")
    return out


def _run_longitudinal(config: PipelineConfig, progressor_only: bool, n_run: int | None,
                      shared_mesh: bool, qcart: pd.DataFrame | None,
                      save_dir: Path | None = None,
                      provider_name: str = "dataset204",
                      pd_is_dess_equivalent: bool = False) -> dict:
    cases = load_v33_cohort(progressor_only=progressor_only)
    if n_run is not None:
        cases = cases[:n_run]
    if not cases:
        return {"per_case": [], "per_region": {}, "n_total": 0}

    # Swap seg paths if a non-default provider is selected. For dataset211,
    # also drop cases that overlap with the D211 training pool (data leak):
    # we only inferred + cached the 107 non-train progressor PIDs. Cases
    # without both timepoints cached are skipped with a warning.
    if provider_name == "dataset211":
        provider = _PROVIDERS["dataset211"]
        kept = []
        for c in cases:
            p00 = provider(f"{c.pid}_{c.side}", "pd_00m", c.side)
            p48 = provider(f"{c.pid}_{c.side}", "pd_48m", c.side)
            if not (p00.exists() and p48.exists()):
                continue
            c.seg_00m_path = p00
            c.seg_48m_path = p48
            kept.append(c)
        print(f"[long] dataset211: {len(kept)}/{len(cases)} cases have both 00m+48m D211 "
              f"inference present (others either in D211 train pool or missing)")
        cases = kept

    fn = process_long_shared_mesh if shared_mesh else process_long_independent
    extra_kw = {"pd_is_dess_equivalent": True} if pd_is_dess_equivalent else {}
    per_case: list[dict] = []
    region_deltas: dict[str, list[float]] = {}
    pipeline_vs_qcart_pairs: dict[str, list[tuple[float, float]]] = {
        f"{b}.{r}": [] for b, regs in (("femur", FEMUR_REGIONS), ("tibia", TIBIA_REGIONS)) for r in regs
        if r in PIPELINE_TO_QCART
    }
    # MFTC / LFTC composites — femur.cMF + tibia.MT  / femur.cLF + tibia.LT
    # paired against QCart MFTC_d / LFTC_d. Manuscript: MFTC SRM = -1.59,
    # cMF r vs QCart = 0.67, MT r vs QCart = 0.48 (over n=166 progressors).
    composite_deltas: dict[str, list[float]] = {"MFTC": [], "LFTC": []}
    composite_vs_qcart: dict[str, list[tuple[float, float]]] = {"MFTC": [], "LFTC": []}
    # Patient-frame parallel accumulators (diagnostic: is template anchoring
    # the source of bad r vs QCart?)
    pf_region_deltas: dict[str, list[float]] = {}
    pf_vs_qcart_pairs: dict[str, list[tuple[float, float]]] = {}
    pf_composite_deltas: dict[str, list[float]] = {"MFTC": [], "LFTC": []}
    pf_composite_vs_qcart: dict[str, list[tuple[float, float]]] = {"MFTC": [], "LFTC": []}

    for ci, case in enumerate(cases, 1):
        # Free in-memory RECON cache between cases (see cross_sectional note).
        # Longitudinal needs this more — two PD timepoints per case = ~300 MB
        # per case staying resident otherwise. OOM hit ~case 10 without it.
        _pipeline.clear_recon_cache()
        gc.collect()
        print(f"\n=== [long {ci}/{len(cases)}] {case.pid}_{case.side} "
              f"KL00={case.KL_00} JSW00={case.JSW_00:.2f}→48={case.JSW_48:.2f} "
              f"dJSW48={case.dJSW_48:+.2f} ===")
        case_row: dict = {
            "pid": case.pid, "side": case.side, "KL_00": case.KL_00,
            "BMI_00": case.BMI_00, "JSW_00": case.JSW_00, "JSW_48": case.JSW_48,
            "dJSW_48": case.dJSW_48, "worst_dJSW": case.worst_dJSW,
            "cohort": case.cohort,
            "eckstein_qcart": _qcart_block(qcart, case.pid, case.side),
            "bones": {},
        }
        for bone in BONES:
            print(f"  --- bone={bone} ---")
            try:
                res = fn(case.seg_00m_path, case.seg_48m_path,
                         side=case.side, bone_name=bone, config=config, **extra_kw)
                t00, t48 = res["tp00_template_thickness"], res["tp48_template_thickness"]
                if save_dir is not None:
                    case_tag = f"{case.pid}_{case.side}"
                    np.save(save_dir / f"{case_tag}_{bone}_00m.npy", t00)
                    np.save(save_dir / f"{case_tag}_{bone}_48m.npy", t48)
                delta = t48 - t00
                d = delta[np.isfinite(delta)]
                mean_delta = float(d.mean()) if d.size else float("nan")
                regions_00 = _region_means(bone, t00)
                regions_48 = _region_means(bone, t48)
                region_dlt = {k: regions_48[k] - regions_00[k] for k in regions_00}

                # Patient-frame region means (no template, 00m bone is the
                # shared sampling surface). Requires shared_mesh wrapper
                # (process_long_shared_mesh) to have returned bone_mesh_00m +
                # th00_patient + th48_at_00_patient.
                pf_regions_00 = pf_regions_48 = pf_regions_dlt = None
                if shared_mesh and "bone_mesh_00m" in res:
                    pf_fn = femur_patient_regions if bone == "femur" else tibia_patient_regions
                    pf_regions_00 = pf_fn(res["bone_mesh_00m"], res["th00_patient"])
                    pf_regions_48 = pf_fn(res["bone_mesh_00m"], res["th48_at_00_patient"])
                    pf_regions_dlt = {k: (pf_regions_48[k] - pf_regions_00[k])
                                      for k in pf_regions_00
                                      if np.isfinite(pf_regions_00[k]) and np.isfinite(pf_regions_48[k])}

                bone_row = {
                    "mean_delta_mm": mean_delta, "n_valid": int(d.size),
                    "regions_00m": regions_00,
                    "regions_48m": regions_48,
                    "regions_delta": region_dlt,
                    "regions_00m_patient_frame": pf_regions_00,
                    "regions_48m_patient_frame": pf_regions_48,
                    "regions_delta_patient_frame": pf_regions_dlt,
                    "pair_quality": res.get("pair_quality"),
                }
                for k, v in region_dlt.items():
                    region_deltas.setdefault(f"{bone}.{k}", []).append(v)
                if pf_regions_dlt is not None:
                    for k, v in pf_regions_dlt.items():
                        pf_region_deltas.setdefault(f"{bone}.{k}", []).append(v)
                # Per-case pipeline-vs-QCart pairing (for cohort r computation)
                if qcart is not None and case_row["eckstein_qcart"] and case_row["eckstein_qcart"].get("present"):
                    for pipeline_k, qcart_k in PIPELINE_TO_QCART.items():
                        if pipeline_k in region_dlt:
                            q = case_row["eckstein_qcart"].get(f"{qcart_k}_d")
                            if q is not None and np.isfinite(q) and np.isfinite(region_dlt[pipeline_k]):
                                pipeline_vs_qcart_pairs.setdefault(f"{bone}.{pipeline_k}", []).append(
                                    (float(region_dlt[pipeline_k]), float(q))
                                )
                                if pf_regions_dlt is not None and pipeline_k in pf_regions_dlt:
                                    pf_vs_qcart_pairs.setdefault(f"{bone}.{pipeline_k}", []).append(
                                        (float(pf_regions_dlt[pipeline_k]), float(q))
                                    )
                rdt = ", ".join(f"{k}={v:+.3f}" for k, v in region_dlt.items() if np.isfinite(v))
                print(f"    mean_delta={mean_delta:+.4f}mm  n={d.size}")
                print(f"    pipeline Δ: {rdt}")
                if case_row["eckstein_qcart"] and case_row["eckstein_qcart"].get("present"):
                    qd = ", ".join(
                        f"{r}={case_row['eckstein_qcart'].get(f'{r}_d'):+.3f}"
                        for r in QCART_REGIONS
                        if case_row["eckstein_qcart"].get(f"{r}_d") is not None
                    )
                    print(f"    qcart    Δ: {qd}")
                else:
                    print(f"    qcart    Δ: (no QCart row for {case.pid}_{case.side})")
            except Exception as e:
                traceback.print_exc()
                bone_row = {"error": f"{type(e).__name__}: {e}"}
            case_row["bones"][bone] = bone_row

        # Composite MFTC / LFTC for this case (only if both bones succeeded)
        fem = case_row["bones"].get("femur", {})
        tib = case_row["bones"].get("tibia", {})
        case_composites: dict = {}
        for comp_name, fem_k, tib_k, qcart_k in (
            ("MFTC", "cMF", "MT", "MFTC_d"),
            ("LFTC", "cLF", "LT", "LFTC_d"),
        ):
            fem_d = (fem.get("regions_delta") or {}).get(fem_k)
            tib_d = (tib.get("regions_delta") or {}).get(tib_k)
            if fem_d is not None and tib_d is not None and np.isfinite(fem_d) and np.isfinite(tib_d):
                comp_d = float(fem_d + tib_d)
                case_composites[f"{comp_name}_d"] = comp_d
                composite_deltas[comp_name].append(comp_d)
                if case_row["eckstein_qcart"] and case_row["eckstein_qcart"].get("present"):
                    qd = case_row["eckstein_qcart"].get(qcart_k)
                    if qd is not None and np.isfinite(qd):
                        composite_vs_qcart[comp_name].append((comp_d, float(qd)))
            # Patient-frame composites
            fem_d_pf = (fem.get("regions_delta_patient_frame") or {}).get(fem_k)
            tib_d_pf = (tib.get("regions_delta_patient_frame") or {}).get(tib_k)
            if (fem_d_pf is not None and tib_d_pf is not None
                    and np.isfinite(fem_d_pf) and np.isfinite(tib_d_pf)):
                comp_d_pf = float(fem_d_pf + tib_d_pf)
                case_composites[f"{comp_name}_d_patient_frame"] = comp_d_pf
                pf_composite_deltas[comp_name].append(comp_d_pf)
                if case_row["eckstein_qcart"] and case_row["eckstein_qcart"].get("present"):
                    qd = case_row["eckstein_qcart"].get(qcart_k)
                    if qd is not None and np.isfinite(qd):
                        pf_composite_vs_qcart[comp_name].append((comp_d_pf, float(qd)))
        case_row["composites"] = case_composites
        if case_composites:
            cstr = "  ".join(f"{k}={v:+.3f}" for k, v in case_composites.items())
            print(f"  composites: {cstr}")
        per_case.append(case_row)

    # Per-region SRM + r vs QCart (cohort-level)
    per_region_summary: dict = {}
    for key, vals in region_deltas.items():
        v = np.asarray(vals, dtype=np.float64); v = v[np.isfinite(v)]
        summary = srm(v)
        pairs = pipeline_vs_qcart_pairs.get(key, [])
        if len(pairs) >= 3:
            xs = np.asarray([p[0] for p in pairs]); ys = np.asarray([p[1] for p in pairs])
            if xs.std() > 0 and ys.std() > 0:
                summary["r_vs_qcart"] = float(np.corrcoef(xs, ys)[0, 1])
            else:
                summary["r_vs_qcart"] = float("nan")
            summary["n_qcart"] = len(pairs)
        else:
            summary["r_vs_qcart"] = float("nan"); summary["n_qcart"] = len(pairs)
        per_region_summary[key] = summary

    # MFTC / LFTC composite summaries (manuscript headline)
    per_composite: dict = {}
    for name in ("MFTC", "LFTC"):
        v = np.asarray(composite_deltas[name], dtype=np.float64)
        v = v[np.isfinite(v)]
        summary = srm(v)
        pairs = composite_vs_qcart[name]
        if len(pairs) >= 3:
            xs = np.asarray([p[0] for p in pairs]); ys = np.asarray([p[1] for p in pairs])
            summary["r_vs_qcart"] = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float("nan")
            summary["n_qcart"] = len(pairs)
        else:
            summary["r_vs_qcart"] = float("nan"); summary["n_qcart"] = len(pairs)
        per_composite[name] = summary

    # Patient-frame parallel summaries (diagnostic: bypass template anchoring)
    per_region_patient_frame: dict = {}
    for key, vals in pf_region_deltas.items():
        v = np.asarray(vals, dtype=np.float64); v = v[np.isfinite(v)]
        summary = srm(v)
        pairs = pf_vs_qcart_pairs.get(key, [])
        if len(pairs) >= 3:
            xs = np.asarray([p[0] for p in pairs]); ys = np.asarray([p[1] for p in pairs])
            summary["r_vs_qcart"] = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float("nan")
            summary["n_qcart"] = len(pairs)
        else:
            summary["r_vs_qcart"] = float("nan"); summary["n_qcart"] = len(pairs)
        per_region_patient_frame[key] = summary

    per_composite_patient_frame: dict = {}
    for name in ("MFTC", "LFTC"):
        v = np.asarray(pf_composite_deltas[name], dtype=np.float64)
        v = v[np.isfinite(v)]
        summary = srm(v)
        pairs = pf_composite_vs_qcart[name]
        if len(pairs) >= 3:
            xs = np.asarray([p[0] for p in pairs]); ys = np.asarray([p[1] for p in pairs])
            summary["r_vs_qcart"] = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float("nan")
            summary["n_qcart"] = len(pairs)
        else:
            summary["r_vs_qcart"] = float("nan"); summary["n_qcart"] = len(pairs)
        per_composite_patient_frame[name] = summary

    return {"per_case": per_case,
            "per_region": per_region_summary,
            "per_composite": per_composite,
            "per_region_patient_frame": per_region_patient_frame,
            "per_composite_patient_frame": per_composite_patient_frame,
            "n_total": len(cases)}
