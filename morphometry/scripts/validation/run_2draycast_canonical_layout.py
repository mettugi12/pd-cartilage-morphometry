"""Run 2D-per-slice raycast on cross-sectional + longitudinal cohorts,
render the V1.1-CANONICAL 2x3 grids (matches figures/v1.1_canonical layout).

For each case + bone:
  1. Compute per-vertex thickness via 2D-per-slice raycast
  2. Register patient bone to template (library `patient_to_template_align`)
  3. Compute subch_prob (tpl_nn)
  4. IDW remap to template (compartment-aware on tibia, plain on femur)
  5. Save the template thickness array as .npy in the layout expected by
     `make_pair_figure` / `make_long_figure`:
       <save_dir>/pair_thickness/<case_id>_<bone>_<pd|dess>.npy
       <save_dir>/long_thickness/<pid>_<side>_<bone>_<00m|48m>.npy
  6. Render the 2x3 grid (femur+tibia rows, modality/timepoint cols + Δ)

Usage:
  python -m scripts.run_2draycast_canonical_layout            # default: 5 worst long + 5 pair
  python -m scripts.run_2draycast_canonical_layout --cases_long 9477205_RIGHT 9235666_LEFT
  python -m scripts.run_2draycast_canonical_layout --cases_pair 9013941_RIGHT 9039627_LEFT
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from cartilage_morphometry import (
    PipelineConfig,
    TEMPLATE_PATHS,
    get_subch,
    patient_to_template_align,
    remap_thickness_to_template,
)
from cartilage_morphometry import pipeline as _pipeline

from cartilage_morphometry.validation.api import (
    DEFAULT_RECON_CACHE_DIR, _patch_disk_cache_path_unique,
)
from cartilage_morphometry.validation.cohorts import load_v33_cohort, load_pair_cases
from cartilage_morphometry.validation.shared_mesh import (
    _force_ml_flip, _ml_flip_decision, _dess_labels_11class,
    compartment_aware_remap, _articular_icp, _idw_sample, _bone_assd,
)
from cartilage_morphometry.validation.viz import (
    make_long_figure_proportional, make_pair_figure_proportional,
)
from cartilage_morphometry.validation.eckstein import load_qcart_deltas, lookup_qcart_row
from cartilage_morphometry.validation.api import _template_pack, _region_means
from scripts.viz_3d_thickness_2dslice import compute_2dslice_thickness_per_vert


# Worst-pMT tibia long cases (from v1.1 canonical investigation)
WORST_PMT_TIBIA = [
    "9477205_RIGHT", "9235666_LEFT", "9567504_RIGHT",
    "9705992_LEFT", "9832566_RIGHT",
]


def _inherit_compartment(aligned_pts, template_mesh, subch_threshold=0.5):
    tpl_pts = np.asarray(template_mesh.points)
    tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= subch_threshold
    sp_idx = np.where(tpl_subch)[0]
    tree = cKDTree(tpl_pts[sp_idx])
    d, idx = tree.query(np.asarray(aligned_pts), k=1)
    return tpl_comp[sp_idx][idx].astype(np.int8), d.astype(np.float32)


def _compute_template_thickness(seg_path: Path, side: str, bone: str,
                                  modality: str, template_mesh: pv.PolyData,
                                  args, flip_override: bool | None = None,
                                  dess_is_11class: bool = False) -> np.ndarray:
    """End-to-end: seg → 2D-per-slice raycast → registration → IDW remap.

    Returns the masked template thickness array (n_template,), NaN outside the
    template subch zone.
    """
    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)
    flip_cm = _force_ml_flip(flip_override) if flip_override is not None else contextlib.nullcontext()
    label_cm = _dess_labels_11class() if (modality == "dess" and dess_is_11class) else contextlib.nullcontext()
    with flip_cm, label_cm:
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, modality, bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing,
                                            bone_smooth_iters=5)

    # 2D-per-slice raycast (works for both PD and DESS modalities — same
    # bone-cart relationship per slice; DESS has thinner slices so even better)
    th_2d = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm,
    )

    aligned_pts, *_ = patient_to_template_align(
        bone_mesh, cart3d, spacing, template_mesh, bone, modality)
    subch_fn = get_subch(cfg.subch_method)
    subch_prob, _ = subch_fn(
        aligned_pts, template_mesh, cart3d, spacing, bone_mesh, th_2d, cfg)

    has_compartment = "compartment" in template_mesh.point_data
    if has_compartment:
        inh_comp, _ = _inherit_compartment(aligned_pts, template_mesh)
        template_thickness, _ = compartment_aware_remap(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, inh_comp,
            k=cfg.idw_k, subch_threshold=cfg.subch_threshold,
        )
    else:
        template_thickness, _ = remap_thickness_to_template(
            template_mesh, aligned_pts, th_2d.astype(np.float32),
            subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold,
        )
        tpl_subch_mask = (np.asarray(template_mesh.point_data["subch_prob"])
                           >= cfg.subch_threshold)
        template_thickness = template_thickness.copy()
        template_thickness[~tpl_subch_mask] = np.nan
    # Always mask non-subch verts to NaN (compartment_aware_remap already does
    # this, but be defensive in case caller mutates)
    template_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= cfg.subch_threshold
    template_thickness = template_thickness.copy()
    template_thickness[~template_subch] = np.nan
    return template_thickness.astype(np.float32)


def _thickness_2d_for_seg(seg_path, side, bone, args, flip_override):
    """seg → 2D-per-slice raycast per-vert thickness on the patient bone mesh.
    Returns (bone_mesh, bone3d, cart3d, spacing, thickness)."""
    cfg = PipelineConfig(thickness_method="raycast")
    with _force_ml_flip(flip_override):
        bone3d, cart3d, spacing, _, _, _ = _pipeline.get_patient_masks(
            seg_path, side, "pd", bone, config=cfg)
    bone_mesh, _ = _pipeline.build_meshes(bone3d, cart3d, spacing, bone_smooth_iters=5)
    th = compute_2dslice_thickness_per_vert(
        bone_mesh, bone3d, cart3d, spacing,
        bone_sigma=args.bone_sigma, cart_sigma=args.cart_sigma,
        near_cart_mm=args.near_cart_mm, max_mm=args.max_mm,
        max_query_mm=args.max_query_mm)
    return bone_mesh, bone3d, cart3d, spacing, th


def _compute_long_shared_mesh(seg00, seg48, side, bone, template_mesh, args,
                                flip) -> tuple[np.ndarray, np.ndarray]:
    """CANONICAL longitudinal recipe (shared-mesh, 2D-per-slice raycast):
      1. 2D raycast thickness on 00m bone (th00) and 48m bone (th48)
      2. 48m → 00m bone trimmed-rigid articular ICP (partial-overlap robust)
      3. resample th48 onto 00m bone verts  → th48_at_00
      4. register 00m bone → template ONCE  → aligned_00
      5. remap BOTH th00 and th48_at_00 with the SAME aligned_00 + subch
         → tpl00, tpl48  (zero inter-timepoint registration drift)

    Returns (tpl00, tpl48), both masked to the template subch zone.
    """
    cfg = PipelineConfig(thickness_method="raycast", subch_method="tpl_nn", idw_k=3)
    m00, b00, c00, sp00, th00 = _thickness_2d_for_seg(seg00, side, bone, args, flip)
    m48, b48, c48, sp48, th48 = _thickness_2d_for_seg(seg48, side, bone, args, flip)

    # 48m → 00m bone ICP (trimmed_rigid → robust to ROI-size mismatch)
    p00 = np.asarray(m00.points); p48 = np.asarray(m48.points)
    p48_aligned, A, t, used_art, *_ = _articular_icp(
        p48, c48, sp48, p00, c00, sp00,
        articular_radius_mm=args.articular_radius_mm,
        method=args.icp_method, trim_fraction=args.trim_fraction)
    rot = float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
    print(f"    48→00 {args.icp_method} ICP: rot={rot:.1f}° |t|={np.linalg.norm(t):.1f}mm "
          f"ASSD→{_bone_assd(p48_aligned, p00):.2f}mm")
    th48_at_00 = _idw_sample(p00, p48_aligned, th48, k=3).astype(np.float32)

    # 00m → template ONCE; both timepoints share this mapping + subch
    aligned_00, *_ = patient_to_template_align(m00, c00, sp00, template_mesh, bone, "pd")
    subch_fn = get_subch(cfg.subch_method)
    subch_prob, _ = subch_fn(aligned_00, template_mesh, c00, sp00, m00, th00, cfg)

    has_comp = "compartment" in template_mesh.point_data
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= cfg.subch_threshold
    if has_comp:
        inh, _ = _inherit_compartment(aligned_00, template_mesh)
        tpl00, _ = compartment_aware_remap(template_mesh, aligned_00, th00.astype(np.float32),
                                            subch_prob, inh, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
        tpl48, _ = compartment_aware_remap(template_mesh, aligned_00, th48_at_00.astype(np.float32),
                                            subch_prob, inh, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
    else:
        tpl00, _ = remap_thickness_to_template(template_mesh, aligned_00, th00.astype(np.float32),
                                                subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
        tpl48, _ = remap_thickness_to_template(template_mesh, aligned_00, th48_at_00.astype(np.float32),
                                                subch_prob, k=cfg.idw_k, subch_threshold=cfg.subch_threshold)
    tpl00 = tpl00.copy(); tpl00[~tpl_subch] = np.nan
    tpl48 = tpl48.copy(); tpl48[~tpl_subch] = np.nan
    return tpl00.astype(np.float32), tpl48.astype(np.float32)


def _pipeline_regions_from_npy(case_tag: str, save_dir: Path) -> dict:
    """Compute Eckstein regional Δ (48m - 00m) for cMF/cLF/pMF/pLF/MT/LT
    from the saved 00m/48m thickness arrays. Returns
    `{"femur": {"cMF": Δmm, "cLF": Δmm, "pMF": Δmm, "pLF": Δmm},
       "tibia": {"MT": Δmm, "LT": Δmm}}`.
    """
    out: dict[str, dict[str, float]] = {}
    for bone in ("femur", "tibia"):
        a = np.load(save_dir / f"{case_tag}_{bone}_00m.npy")
        b = np.load(save_dir / f"{case_tag}_{bone}_48m.npy")
        r00 = _region_means(bone, a)
        r48 = _region_means(bone, b)
        out[bone] = {k: float(r48[k] - r00[k]) for k in r00.keys() if k in r48}
    return out


def run_long(case_tags: list[str], args, save_root: Path, fig_dir: Path):
    cohort = {f"{c.pid}_{c.side}": c for c in load_v33_cohort(progressor_only=True)}
    save_dir = save_root / "long_thickness"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load OAI Eckstein QCart manuscript deltas once (used by every case)
    try:
        qcart_df = load_qcart_deltas()
        print(f"  loaded OAI QCart deltas: {len(qcart_df)} rows")
    except Exception as e:
        print(f"  [warn] could not load QCart deltas ({e}); bar row will be skipped")
        qcart_df = None

    for case_tag in case_tags:
        case = cohort.get(case_tag)
        if case is None:
            print(f"[skip] {case_tag}: not in cohort"); continue
        # Single flip decision from 00m, reused on 48m (v1.1 fix)
        flip = _ml_flip_decision(case.seg_00m_path)
        for bone in ("femur", "tibia"):
            template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
            npy00 = save_dir / f"{case_tag}_{bone}_00m.npy"
            npy48 = save_dir / f"{case_tag}_{bone}_48m.npy"
            if npy00.exists() and npy48.exists() and not args.force:
                print(f"  [cache] {npy00.name}, {npy48.name}"); continue
            print(f"  [long] {case_tag} {bone} (shared-mesh 48m→00m→template)")
            # CANONICAL shared-mesh recipe — single 00m→template map applied to
            # both timepoints, 48m→00m trimmed-rigid bone ICP first.
            tpl00, tpl48 = _compute_long_shared_mesh(
                case.seg_00m_path, case.seg_48m_path, case.side, bone,
                template_mesh, args, flip)
            np.save(npy00, tpl00)
            np.save(npy48, tpl48)

        # QCart bar row input — `lookup_qcart_row` returns columns with the
        # `eck_` prefix (e.g. `eck_cMF_d`). The bar function expects `cMF_d`
        # (matches `_qcart_block` in api.py). Strip the prefix.
        qcart_row = None
        if qcart_df is not None:
            try:
                raw = lookup_qcart_row(qcart_df, case.pid, case.side)
                if raw is not None:
                    qcart_row = {"present": True}
                    for r in ("cMF", "cLF", "MT", "LT", "MFTC", "LFTC"):
                        qcart_row[f"{r}_00mm"] = raw.get(f"eck_{r}_00")
                        qcart_row[f"{r}_48mm"] = raw.get(f"eck_{r}_48")
                        qcart_row[f"{r}_d"] = raw.get(f"eck_{r}_d")
            except Exception as e:
                print(f"  [warn] qcart lookup failed: {e}")

        out_png = fig_dir / f"long_{case_tag}.png"
        try:
            # Don't pre-compute pipeline_regions here — `make_long_figure_
            # proportional` builds them from the saved thickness arrays using
            # arc-based cMF/cLF (not AP-distance), matching the per-panel
            # annotation in the figure.
            make_long_figure_proportional(
                case_tag, save_dir, out_png,
                vmax_thick=4.0, vabs_delta=1.5,
                smooth_sigma=args.smooth_sigma,
                qcart_row=qcart_row, pipeline_regions=None,
            )
            print(f"  [ok] {out_png}")
        except Exception as e:
            print(f"  [ERR render] {e}")


def run_pair(case_ids: list[str], args, save_root: Path, fig_dir: Path):
    pair_cohort = {pc.case_id: pc for pc in load_pair_cases()}
    save_dir = save_root / "pair_thickness"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        pc = pair_cohort.get(case_id)
        if pc is None:
            print(f"[skip] pair {case_id}: not in cohort"); continue
        side = pc.side_anatomical
        for bone in ("femur", "tibia"):
            template_mesh = pv.read(str(TEMPLATE_PATHS[bone]))
            for modality, seg, is_11 in (
                ("pd",   pc.pd_seg_path,   False),
                ("dess", pc.dess_seg_path, True),
            ):
                npy = save_dir / f"{case_id}_{bone}_{modality}.npy"
                if npy.exists() and not args.force:
                    print(f"  [cache] {npy.name}"); continue
                print(f"  [pair] {case_id} {bone} {modality}")
                th = _compute_template_thickness(
                    seg, side, bone, modality, template_mesh, args,
                    flip_override=None, dess_is_11class=is_11)
                np.save(npy, th)
        out_png = fig_dir / f"pair_{case_id}.png"
        try:
            make_pair_figure_proportional(
                case_id, save_dir, out_png,
                vmax_thick=4.0, vabs_delta=1.5,
                smooth_sigma=args.smooth_sigma,
            )
            print(f"  [ok] {out_png}")
        except Exception as e:
            print(f"  [ERR render] {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_long", nargs="+", default=WORST_PMT_TIBIA,
                   help="Longitudinal case tags (e.g. 9477205_RIGHT).")
    p.add_argument("--cases_pair", nargs="+", default=None,
                   help="Pair case_ids (e.g. 9013941_RIGHT). If omitted, picks "
                        "the first 5 pairs available.")
    p.add_argument("--bone_sigma", type=float, default=1.8)
    p.add_argument("--cart_sigma", type=float, default=0.8)
    p.add_argument("--near_cart_mm", type=float, default=4.0)
    p.add_argument("--max_mm", type=float, default=6.0)
    p.add_argument("--max_query_mm", type=float, default=1.5)
    # Longitudinal shared-mesh 48m→00m bone ICP (canonical)
    p.add_argument("--articular_radius_mm", type=float, default=5.0)
    p.add_argument("--icp_method", default="trimmed_rigid",
                   choices=["rigid", "trimmed_rigid"],
                   help="48m→00m bone ICP. trimmed_rigid is partial-overlap robust.")
    p.add_argument("--trim_fraction", type=float, default=0.8)
    p.add_argument("--smooth_sigma", type=float, default=1.5,
                   help="2D-figure Gaussian sigma in grid cells (v1.1 default).")
    p.add_argument("--force", action="store_true",
                   help="Re-compute thickness even if .npy exists.")
    p.add_argument("--out_root", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/figures/v1.2_raycast2d_canonical"))
    p.add_argument("--save_root", type=Path,
                   default=Path(r"E:/KneeMR/Studies/cartilage-validation/thickness_raycast2d"))
    p.add_argument("--skip_long", action="store_true")
    p.add_argument("--skip_pair", action="store_true")
    args = p.parse_args()

    DEFAULT_RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline.set_recon_disk_cache_dir(DEFAULT_RECON_CACHE_DIR)
    _patch_disk_cache_path_unique()

    if not args.skip_long:
        print("\n========== LONGITUDINAL ==========")
        run_long(args.cases_long, args, args.save_root, args.out_root / "long")

    if not args.skip_pair:
        print("\n========== CROSS-SECTIONAL PAIRS ==========")
        pair_ids = args.cases_pair
        if pair_ids is None:
            all_pairs = load_pair_cases()
            pair_ids = [p.case_id for p in all_pairs[:5]]
            print(f"  (no --cases_pair given; using first 5: {pair_ids})")
        run_pair(pair_ids, args, args.save_root, args.out_root / "pairs")


if __name__ == "__main__":
    main()
