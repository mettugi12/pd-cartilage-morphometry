"""2D thickness-map figures — SNUH-test-canonical style (grid_size=150).

Two layouts:

- `make_pair_figure(case_id, ...)` — 2×3 grid for a PD↔DESS pair:
    rows: femur, tibia
    cols: PD,  DESS,  Δ (PD − DESS)
    Per-row title: r_2d (PD↔DESS Pearson r over the dense grid).

- `make_long_figure(case_id, ...)` — 2×3 thickness/Δ grid for a 00m↔48m
  longitudinal pair, plus an optional bottom row showing pipeline Δ vs
  OAI Eckstein QCart Δ (cMF/cLF/MT/LT + MFTC/LFTC composites).

Why grid_size=150 (was 40): at grid=40 each cell is ~1.9 mm wide on the
template, the figures look pixelated, and a heavy Gaussian (σ=1.5 grid cells
≈ 2.8 mm) was needed to hide the binning. At grid=150 each cell is ~0.5 mm
and the same σ=1.5 cells (~0.75 mm) becomes a barely-visible touch-up — the
raw projection already reads cleanly, so the redundant smoothed-row goes
away. Matches the cartilage-v1-snuh-test/v1.0/deliverable convention.

Both expect the per-knee template_thickness arrays to already be on disk under
the dir written by `validate(save_thickness_dir=...)`:
    <save_dir>/cross_sectional/<case_id>_<bone>_{pd,dess}.npy
    <save_dir>/longitudinal/<pid>_<side>_<bone>_{00m,48m}.npy
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

from cartilage_morphometry import (
    TEMPLATE_PATHS,
    compute_femur_template_subregions,
    template_thickness_2d_femur,
    template_thickness_2d_tibia,
)

# Module-level cache: template meshes + (femur) projection subregions.
_TEMPLATE: dict[str, dict] = {}


def _template_pack(bone: str) -> dict:
    if bone in _TEMPLATE:
        return _TEMPLATE[bone]
    mesh = pv.read(str(TEMPLATE_PATHS[bone]))
    pack = {"mesh": mesh}
    if bone == "femur":
        pack["subregions"] = compute_femur_template_subregions(mesh)
    _TEMPLATE[bone] = pack
    return pack


def _project(thickness_1d: np.ndarray, bone: str, grid_size: int = 150) -> np.ndarray:
    pack = _template_pack(bone)
    if bone == "femur":
        g, _, _ = template_thickness_2d_femur(
            pack["mesh"], thickness_1d, grid_size=grid_size,
            subregions=pack["subregions"],
        )
        return g
    g, _, _ = template_thickness_2d_tibia(pack["mesh"], thickness_1d, grid_size=grid_size)
    return g


def _smooth_border_preserving(g: np.ndarray, sigma: float,
                               support_threshold: float = 0.05) -> np.ndarray:
    """Gaussian smooth on a 2D grid; fill sparse interior bins, preserve border.

    Matches the cartilage-v1-snuh-test/v1.0/deliverable convention:
    sm_weights < `support_threshold` (default 5 %) marks "outside the bone
    footprint" and stays NaN; everything else gets the weighted Gaussian
    average. This is what kills the freckles at grid=150 (which has lots of
    interior bins with zero template-verts that the old "restore original
    NaN mask" recipe punched back into white dots).

    Why a single weight threshold is enough to preserve the border:
    just outside the subch zone, summed Gaussian weights drop sharply because
    the kernel has fewer valid neighbors → those cells fall below 5 % and
    stay NaN. Inside the bone, even sparse bins have neighbors within σ, so
    weights stay >5 % and the smooth fills them in.
    """
    nan_mask = np.isnan(g)
    g0 = g.copy(); g0[nan_mask] = 0.0
    weights = (~nan_mask).astype(np.float32)
    num = gaussian_filter(g0, sigma=sigma)
    den = gaussian_filter(weights, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.maximum(den, 1e-6)
    out[den < support_threshold] = np.nan
    return out


def _r2d(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    x = a[m]; y = b[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _decorate(ax, n_cols: int) -> None:
    mid = n_cols // 2
    ax.axvline(mid - 0.5, color="white", lw=1.2, alpha=0.7)
    for x, label, ha in ((0.02, "medial", "left"), (0.98, "lateral", "right")):
        ax.text(x, 0.97, label, transform=ax.transAxes, color="white",
                fontsize=8, va="top", ha=ha,
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
    for y, label, va in ((0.02, "anterior", "bottom"), (0.97, "posterior", "top")):
        ax.text(0.5, y, label, transform=ax.transAxes, color="white",
                fontsize=8, va=va, ha="center",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
    ax.set_xticks([]); ax.set_yticks([])


def _render_grid(case_id: str,
                 col_keys: tuple[str, str, str],   # ("pd","dess","delta") or ("00m","48m","delta")
                 col_labels: tuple[str, str, str],
                 arrays: dict[tuple[str, str], np.ndarray],  # (bone, col_key) -> 1D template_thickness
                 out_png: Path,
                 title_suffix: str,
                 vmax_thick: float = 4.0, vabs_delta: float = 1.5,
                 grid_size: int = 150,
                 smooth_sigma: float = 1.5,
                 qcart_row: dict | None = None,
                 pipeline_regions: dict[str, dict] | None = None) -> Path:
    """Render the 2×3 thickness/Δ figure (femur, tibia × col0, col1, Δ) at
    `grid_size` cells per side (default 150 → ~0.5 mm/cell, SNUH-test
    canonical). A light border-preserving Gaussian smooth (σ=`smooth_sigma`
    cells, default 1.5 ≈ 0.75 mm) takes off the last bit of binning. If
    `qcart_row` + `pipeline_regions` are given, append a 3rd wide row with
    pipeline Δ vs QCart Δ bars (longitudinal only).
    """
    if col_keys[2] != "delta":
        raise ValueError("col_keys[2] must be 'delta'")

    grids: dict[tuple[str, str], np.ndarray] = {}
    for bone in ("femur", "tibia"):
        a = _project(arrays[(bone, col_keys[0])], bone, grid_size=grid_size)
        b = _project(arrays[(bone, col_keys[1])], bone, grid_size=grid_size)
        if smooth_sigma and smooth_sigma > 0:
            a = _smooth_border_preserving(a, smooth_sigma)
            b = _smooth_border_preserving(b, smooth_sigma)
        # Δ = follow-up − baseline (longitudinal) OR PD − DESS (cross-sectional)
        d = (b - a) if col_keys[1] == "48m" else (a - b)
        grids[(bone, col_keys[0])] = a
        grids[(bone, col_keys[1])] = b
        grids[(bone, "delta")] = d

    show_qcart_row = qcart_row is not None and qcart_row.get("present") and pipeline_regions is not None
    if show_qcart_row:
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.55])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
        ax_bars = fig.add_subplot(gs[2, :])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        ax_bars = None
    rows = [("femur", "Femur"), ("tibia", "Tibia")]
    cols = [(col_keys[0], col_labels[0]), (col_keys[1], col_labels[1]), ("delta", col_labels[2])]

    for r, (bone, row_label) in enumerate(rows):
        r_val = _r2d(grids[(bone, col_keys[0])], grids[(bone, col_keys[1])])
        for c, (key, col_label) in enumerate(cols):
            ax = axes[r, c]
            g = grids[(bone, key)]
            if key == "delta":
                im = ax.imshow(g, origin="upper", cmap="RdBu",
                               vmin=-vabs_delta, vmax=vabs_delta, interpolation="nearest")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("Delta thickness (mm)", fontsize=9)
            else:
                im = ax.imshow(g, origin="upper", cmap="jet_r",
                               vmin=0.0, vmax=vmax_thick, interpolation="nearest")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("thickness (mm)", fontsize=9)
            title = f"{row_label} -- {col_label}"
            if c == 0:
                title += f"     [r_2d={r_val:.3f}]"
            ax.set_title(title, fontsize=11)
            _decorate(ax, g.shape[1])
            with np.errstate(invalid="ignore"):
                if key == "delta":
                    txt = f"mean Delta = {np.nanmean(g):+.3f} mm"
                else:
                    txt = f"mean = {np.nanmean(g):.3f} mm"
            ax.text(0.5, -0.06, txt, transform=ax.transAxes,
                    fontsize=9, va="top", ha="center")

    if ax_bars is not None:
        _draw_qcart_bar_row(ax_bars, qcart_row, pipeline_regions)

    fig.suptitle(f"{case_id}  {title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ===========================================================================
# 3D viz — patient bone mesh + template thickness, deliverable style
# ===========================================================================
# Ported from cartilage-v1-snuh-test/v1.0/analysis/render_case_variant.py.
# Off-screen PyVista screenshot → mirror horizontally → matplotlib compose
# with M/L/A/P labels + colorbar + title. Articular surface up; medial-left.

THICKNESS_VMIN_3D, THICKNESS_VMAX_3D = 0.0, 3.0
DELTA_VABS_3D = 1.0


def _set_oblique_top_down(pl, mesh, bone_name: str) -> None:
    """Camera ABOVE (superior) looking DOWN (inferior). Articular plateau
    of the tibia, trochlear groove + condyle tops of the femur — both viewed
    from the head's perspective. Medial-left after horizontal mirror.
    """
    pts = np.asarray(mesh.points)
    center = pts.mean(axis=0)
    bounds = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bounds))
    d = 1.4 * diag if bone_name == "femur" else 1.2 * diag
    # axis 1 = SI (inferior+), so superior = LOW SI. Camera at LOW SI looking
    # toward center = top-down view of the articular surface.
    cam = (center[0] - 0.4 * d, center[1] - 0.9 * d, center[2] - 0.2 * d)
    view_up = (-1.0, 0.0, 0.0)  # -AP up → anterior at top of image
    pl.camera_position = [cam, tuple(center), view_up]
    pl.camera.zoom(1.1)


def _screenshot_to_ax(ax, mesh, bone_name: str, scalar: str,
                       cmap: str, vmin: float, vmax: float,
                       title: str, cbar_label: str,
                       mask_nan: bool = True) -> None:
    """Render `mesh` colored by `scalar` to an off-screen PyVista screenshot,
    horizontally mirror (to put medial-left), and draw onto `ax` with
    M/L/A/P labels + colorbar + title."""
    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.set_background("white")
    mesh_to_render = mesh.copy()
    if scalar in mesh.point_data:
        if mask_nan:
            arr = np.asarray(mesh_to_render[scalar], dtype=np.float32).copy()
            sub = np.asarray(mesh_to_render.point_data.get("subch_prob", arr)) >= 0.5
            # Only mask for *template* meshes (have subch_prob); patient
            # meshes don't have subch_prob, so just leave thickness as-is.
            if "subch_prob" in mesh_to_render.point_data:
                arr[~sub] = np.nan
            mesh_to_render[scalar] = arr
        pl.add_mesh(
            mesh_to_render, scalars=scalar, cmap=cmap,
            clim=(vmin, vmax), nan_color="lightgrey",
            show_scalar_bar=False, smooth_shading=True,
        )
    else:
        pl.add_mesh(mesh_to_render, color="lightgrey",
                    show_scalar_bar=False, smooth_shading=True)
    _set_oblique_top_down(pl, mesh_to_render, bone_name)
    img = pl.screenshot(return_img=True, transparent_background=False)
    pl.close()

    img_mirrored = np.fliplr(img)
    ax.imshow(img_mirrored)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for tx, ty, label in ((-0.04, 0.5, "M"), (1.04, 0.5, "L"),
                          (0.5, 1.04, "A"), (0.5, -0.04, "P")):
        ax.text(tx, ty, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontweight="bold")
    # Synthetic colorbar via ScalarMappable so each subplot gets its own
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)


def make_pair_3d_figure(case_id: str, vtk_dir: Path, bone_name: str,
                        out_png: Path,
                        vmax_thick: float = THICKNESS_VMAX_3D,
                        vabs_delta: float = DELTA_VABS_3D) -> Path:
    """2×2 PD↔DESS 3D viz for one (case, bone). Top: patient PD/DESS bone
    meshes in patient mm. Bottom: template mesh with PD-remapped and
    DESS-remapped thickness (same view of canonical bone)."""
    vtk_dir = Path(vtk_dir)
    pd_pt = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_pd_patient.vtk"))
    de_pt = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_dess_patient.vtk"))
    tpl = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_template.vtk"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    _screenshot_to_ax(axes[0, 0], pd_pt,  bone_name, "thickness_mm",
                       "jet", 0.0, vmax_thick,
                       f"PD patient bone  [{case_id} {bone_name}]",
                       "thickness (mm)", mask_nan=False)
    _screenshot_to_ax(axes[0, 1], de_pt,  bone_name, "thickness_mm",
                       "jet", 0.0, vmax_thick,
                       f"DESS patient bone  [{case_id} {bone_name}]",
                       "thickness (mm)", mask_nan=False)
    _screenshot_to_ax(axes[1, 0], tpl, bone_name, "pd_thickness_mm",
                       "jet", 0.0, vmax_thick,
                       "Template (PD remapped)", "thickness (mm)")
    _screenshot_to_ax(axes[1, 1], tpl, bone_name, "dess_thickness_mm",
                       "jet", 0.0, vmax_thick,
                       "Template (DESS remapped)", "thickness (mm)")
    fig.suptitle(f"{case_id}  {bone_name}  — PD vs DESS 3D thickness", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def make_long_3d_figure(case_id: str, vtk_dir: Path, bone_name: str,
                        out_png: Path,
                        vmax_thick: float = THICKNESS_VMAX_3D,
                        vabs_delta: float = DELTA_VABS_3D) -> Path:
    """2×3 longitudinal 3D viz: row1 = 00m / 48m patient bones; row2 = template
    (00m remapped, 48m remapped, Δ). Use to spot implausible-thickening cases."""
    vtk_dir = Path(vtk_dir)
    pt00 = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_00m_patient.vtk"))
    pt48 = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_48m_patient.vtk"))
    tpl = pv.read(str(vtk_dir / f"{case_id}_{bone_name}_template.vtk"))

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    _screenshot_to_ax(axes[0, 0], pt00, bone_name, "thickness_mm",
                       "jet", 0.0, vmax_thick,
                       f"00m patient bone  [{case_id} {bone_name}]",
                       "thickness (mm)", mask_nan=False)
    _screenshot_to_ax(axes[0, 1], pt48, bone_name, "thickness_mm",
                       "jet", 0.0, vmax_thick,
                       f"48m patient bone  [{case_id} {bone_name}]",
                       "thickness (mm)", mask_nan=False)
    axes[0, 2].axis("off")
    _screenshot_to_ax(axes[1, 0], tpl, bone_name, "tp00_thickness_mm",
                       "jet", 0.0, vmax_thick,
                       "Template (00m remapped)", "thickness (mm)")
    _screenshot_to_ax(axes[1, 1], tpl, bone_name, "tp48_thickness_mm",
                       "jet", 0.0, vmax_thick,
                       "Template (48m remapped)", "thickness (mm)")
    _screenshot_to_ax(axes[1, 2], tpl, bone_name, "delta_mm",
                       "RdBu_r", -vabs_delta, vabs_delta,
                       "Template Delta (48m - 00m)", "Delta thickness (mm)")
    fig.suptitle(f"{case_id}  {bone_name}  — longitudinal 3D thickness", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _draw_qcart_bar_row(ax, qcart_row: dict, pipeline_regions: dict[str, dict]) -> None:
    """Side-by-side bar comparison: pipeline Δ vs QCart Δ for the 4 matched
    regions {cMF, cLF, MT, LT} (plus composites MFTC, LFTC).

    `pipeline_regions` is `{"femur": {<region>: Δmm}, "tibia": {<region>: Δmm}}`
    (the per-case dicts straight off the JSON report).
    """
    region_map = [
        ("cMF", "femur", "cMF"),    # display, pipeline_bone, pipeline_key
        ("cLF", "femur", "cLF"),
        ("MT",  "tibia", "MT"),
        ("LT",  "tibia", "LT"),
        ("MFTC", "_composite", "MFTC"),
        ("LFTC", "_composite", "LFTC"),
    ]
    labels: list[str] = []; pipe: list[float] = []; qcart: list[float] = []
    for disp, bone, pkey in region_map:
        # Pipeline value
        if bone == "_composite":
            f = (pipeline_regions.get("femur") or {})
            t = (pipeline_regions.get("tibia") or {})
            if disp == "MFTC":
                fem = f.get("cMF"); tib = t.get("MT")
            else:
                fem = f.get("cLF"); tib = t.get("LT")
            pv = fem + tib if (fem is not None and tib is not None
                               and np.isfinite(fem) and np.isfinite(tib)) else float("nan")
        else:
            pv = (pipeline_regions.get(bone) or {}).get(pkey, float("nan"))
        # QCart value
        qv = qcart_row.get(f"{disp}_d", float("nan"))
        if qv is None: qv = float("nan")
        labels.append(disp); pipe.append(float(pv)); qcart.append(float(qv))

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, pipe,  width=w, color="#1f77b4", label="pipeline Δ (mm)")
    ax.bar(x + w/2, qcart, width=w, color="#ff7f0e", label="QCart Δ (mm, Eckstein manual GT)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Δ thickness (mm)")
    ax.set_title("Region-wise Δ: pipeline vs OAI Eckstein QCart GT  "
                 "(negative = thinning)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Annotate each bar with its value
    for xi, (pv, qv) in enumerate(zip(pipe, qcart)):
        if np.isfinite(pv):
            ax.text(xi - w/2, pv, f"{pv:+.3f}", ha="center",
                    va="bottom" if pv >= 0 else "top", fontsize=8, color="#1f77b4")
        if np.isfinite(qv):
            ax.text(xi + w/2, qv, f"{qv:+.3f}", ha="center",
                    va="bottom" if qv >= 0 else "top", fontsize=8, color="#ff7f0e")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)


# ---------------------------------------------------------------------------
# Public — pair (PD↔DESS) figures
# ---------------------------------------------------------------------------
def make_pair_figure(case_id: str, cs_save_dir: Path, out_png: Path,
                     vmax_thick: float = 4.0, vabs_delta: float = 1.5,
                     smooth_sigma: float = 1.5) -> Path:
    """4×3 PD vs DESS thickness figure for one pair."""
    cs_save_dir = Path(cs_save_dir)
    arrays: dict = {}
    for bone in ("femur", "tibia"):
        arrays[(bone, "pd")] = np.load(cs_save_dir / f"{case_id}_{bone}_pd.npy")
        arrays[(bone, "dess")] = np.load(cs_save_dir / f"{case_id}_{bone}_dess.npy")
    return _render_grid(
        case_id, ("pd", "dess", "delta"),
        ("PD", "DESS", "Delta (PD - DESS)"),
        arrays, out_png,
        title_suffix="PD vs DESS thickness  [v1_raycast_shared_mesh]",
        vmax_thick=vmax_thick, vabs_delta=vabs_delta, smooth_sigma=smooth_sigma,
    )


# ---------------------------------------------------------------------------
# Public — longitudinal (00m↔48m) figures
# ---------------------------------------------------------------------------
def make_long_figure(case_tag: str, long_save_dir: Path, out_png: Path,
                     vmax_thick: float = 4.0, vabs_delta: float = 1.5,
                     smooth_sigma: float = 1.5,
                     qcart_row: dict | None = None,
                     pipeline_regions: dict[str, dict] | None = None) -> Path:
    """4×3 00m vs 48m thickness figure for one longitudinal pair. If
    `qcart_row` + `pipeline_regions` are provided (typically loaded from the
    JSON report's per-case dict), an extra wide bar-comparison row is appended
    showing pipeline Δ vs QCart Δ for cMF/cLF/MT/LT + MFTC/LFTC composites."""
    long_save_dir = Path(long_save_dir)
    arrays: dict = {}
    for bone in ("femur", "tibia"):
        arrays[(bone, "00m")] = np.load(long_save_dir / f"{case_tag}_{bone}_00m.npy")
        arrays[(bone, "48m")] = np.load(long_save_dir / f"{case_tag}_{bone}_48m.npy")
    return _render_grid(
        case_tag, ("00m", "48m", "delta"),
        ("00m baseline", "48m followup", "Delta (48m - 00m)"),
        arrays, out_png,
        title_suffix="Longitudinal thickness  [v1_raycast_shared_mesh]",
        vmax_thick=vmax_thick, vabs_delta=vabs_delta, smooth_sigma=smooth_sigma,
        qcart_row=qcart_row, pipeline_regions=pipeline_regions,
    )


# ---------------------------------------------------------------------------
# Canonical thickness-panel viz (2x2: 3D patient | 3D template
#                                       2D patient | 2D template)
#
# Used by per-case diagnostics to compare alternative thickness / remap
# methods. Patient and template panels render via PROPORTIONAL projection
# (shared ML/AP bounds, isotropic aspect → relative compartment sizes
# preserved, intercondylar gap visible as empty cells). 3D screenshots use a
# bone-aware oblique camera (tibia: above plateau; femur: below condyles).
# ---------------------------------------------------------------------------

def proportional_projection_2d(pts: np.ndarray, scalar, mask: np.ndarray,
                                grid_size: int = 120, padding_frac: float = 0.02,
                                layout: dict | None = None, agg: str = "mean"):
    """Project (AP, ML) of `pts` onto a 2D grid with isotropic aspect ratio.

    Shared ML/AP bounds across the whole `mask`-True region — relative sizes
    of medial vs lateral plateaus / condyles are preserved AND the intercondylar
    gap appears naturally as empty cells.

    `agg='mean'` averages `scalar` per cell; `agg='any'` returns a 0/1 mask.

    Returns (grid, layout). Pass `layout` back in to keep multiple overlays
    (thickness, subch contours) on the same grid.
    """
    AP = pts[:, 0]; ML = pts[:, 2]
    if layout is None:
        v = mask
        if not v.any():
            return np.full((grid_size, grid_size), np.nan), None
        ml_lo, ml_hi = float(ML[v].min()), float(ML[v].max())
        ap_lo, ap_hi = float(AP[v].min()), float(AP[v].max())
        ml_r = max(ml_hi - ml_lo, 1e-6); ap_r = max(ap_hi - ap_lo, 1e-6)
        ml_lo -= ml_r * padding_frac; ml_hi += ml_r * padding_frac
        ap_lo -= ap_r * padding_frac; ap_hi += ap_r * padding_frac
        ml_r = ml_hi - ml_lo; ap_r = ap_hi - ap_lo
        aspect = ml_r / ap_r
        n_cols = grid_size if aspect >= 1.0 else max(int(round(grid_size * aspect)), 1)
        n_rows = max(int(round(grid_size / aspect)), 1) if aspect >= 1.0 else grid_size
        layout = {"ml_lo": ml_lo, "ap_lo": ap_lo, "ml_r": ml_r, "ap_r": ap_r,
                  "n_rows": n_rows, "n_cols": n_cols}
    n_rows, n_cols = layout["n_rows"], layout["n_cols"]
    v = mask & np.isfinite(scalar) if scalar is not None else mask
    if not v.any():
        return np.full((n_rows, n_cols), np.nan), layout
    ml_n = (ML[v] - layout["ml_lo"]) / layout["ml_r"]
    ap_n = (AP[v] - layout["ap_lo"]) / layout["ap_r"]
    ml_b = np.clip((ml_n * n_cols).astype(int), 0, n_cols - 1)
    ap_b = np.clip((ap_n * n_rows).astype(int), 0, n_rows - 1)
    if agg == "mean":
        sum_g = np.zeros((n_rows, n_cols), dtype=np.float64)
        cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
        np.add.at(sum_g, (ap_b, ml_b), scalar[v] if scalar is not None else 1.0)
        np.add.at(cnt_g, (ap_b, ml_b), 1)
        return np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan), layout
    cnt_g = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(cnt_g, (ap_b, ml_b), 1)
    return np.where(cnt_g > 0, 1.0, np.nan), layout


def bone_oblique_camera(mesh: pv.PolyData, bone: str):
    """Oblique 3D camera that brings the cart-bearing surface into view.

    - Tibia: cart-bearing plateau at SUPERIOR end (low axis 1). Camera placed
      superior to plateau, slight anterior+lateral offset.
    - Femur: cart-bearing condyles at INFERIOR end (high axis 1). Camera
      placed inferior to condyles, slight anterior+lateral offset.

    Returns pyvista camera_position list [(cam), (focus), (view_up)].
    """
    pts = np.asarray(mesh.points)
    bmin = pts.min(axis=0); bmax = pts.max(axis=0)
    center = (bmin + bmax) * 0.5
    diag = float(np.linalg.norm(bmax - bmin))
    d = 1.4 * diag
    if bone == "tibia":
        cam_y = bmin[1] - d * 0.45
        focus = (center[0], bmin[1] + (bmax[1] - bmin[1]) * 0.15, center[2])
    else:  # femur
        cam_y = bmax[1] + d * 0.45
        focus = (center[0], bmax[1] - (bmax[1] - bmin[1]) * 0.15, center[2])
    cam_x = center[0] - d * 0.20
    cam_z = center[2] + d * 0.30
    return [(cam_x, cam_y, cam_z), focus, (-1.0, 0.0, 0.0)]   # anterior up


def screenshot_3d_thickness(mesh: pv.PolyData, scalar: np.ndarray, bone: str,
                             vmax: float = 4.0, window: tuple = (900, 900)):
    """Off-screen 3D thickness render: jet colormap, NaN-masked verts gray.

    Verts with thickness <= 0 are also masked to gray so denuded regions read
    as background, not blue (vmin=0). Camera = `bone_oblique_camera(mesh, bone)`.
    """
    m = mesh.copy()
    sc = np.asarray(scalar, dtype=np.float32).copy()
    sc = np.where(sc > 0, sc, np.nan)
    m["thickness_mm"] = sc
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    pl.add_mesh(m, scalars="thickness_mm", cmap="jet_r", clim=(0, vmax),
                 nan_color=(0.85, 0.85, 0.85), smooth_shading=True,
                 show_scalar_bar=True,
                 scalar_bar_args={"title": "thickness (mm)", "fmt": "%.1f",
                                    "vertical": False, "position_x": 0.25,
                                    "position_y": 0.04, "width": 0.5,
                                    "height": 0.04})
    pl.camera_position = bone_oblique_camera(mesh, bone)
    pl.reset_camera()
    img = pl.screenshot(return_img=True, transparent_background=False)
    pl.close()
    return img


def _close_filled(mask_grid: np.ndarray) -> np.ndarray:
    """Binary closing for a 0/1 mask grid (fills 1-cell pinholes) → float."""
    from scipy.ndimage import binary_closing
    binary = np.where(np.isfinite(mask_grid), 1, 0).astype(np.uint8)
    binary = binary_closing(binary, structure=np.ones((3, 3), dtype=bool),
                             iterations=1).astype(np.float32)
    return binary


def make_thickness_panel(case_id: str, bone: str,
                          patient_mesh: pv.PolyData,
                          patient_thickness: np.ndarray,
                          template_mesh: pv.PolyData,
                          template_thickness: np.ndarray,
                          out_png: Path,
                          vmax: float = 4.0,
                          grid_size: int = 120,
                          smooth_sigma: float = 1.2,
                          patient_subch_mask: np.ndarray | None = None,
                          patient_compartment: np.ndarray | None = None,
                          subtitle: str = "") -> Path:
    """Canonical 2x2 figure for comparing patient vs template thickness.

        [PATIENT 3D]   [TEMPLATE 3D]
        [PATIENT 2D]   [TEMPLATE 2D]

    - 3D panels: oblique cart-bearing view (tibia: from above plateau,
      femur: from below condyles), bone surface colored by thickness (jet,
      0..vmax mm), denuded / unassigned verts in gray.
    - 2D panels: proportional projection (shared ML/AP bounds, isotropic
      aspect), border-preserving Gaussian smooth (σ in grid cells). Subch /
      compartment contours overlaid in black if `patient_subch_mask` is given
      (patient side) and the template has a `compartment` field (template side).

    Args:
        patient_subch_mask: bool (n_patient_verts,). If provided, draw a black
            contour on the patient 2D panel showing the subch zone. Recommended
            source: `subch_prob >= 0.5` from the library's `get_subch`.
        patient_compartment: int (n_patient_verts,), values {0, 1, 2}. If
            provided, split the patient subch contour into medial (1) and
            lateral (2) closed loops, drawn as separate black contours.
        subtitle: extra suptitle text (e.g. "raycast 3D vs 2D-per-slice").
    """
    pat_pts = np.asarray(patient_mesh.points)
    tpl_pts = np.asarray(template_mesh.points)

    has_tpl_compartment = "compartment" in template_mesh.point_data
    tpl_subch = (np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5)
    if has_tpl_compartment:
        tpl_comp = np.asarray(template_mesh.point_data["compartment"])
    else:
        tpl_comp = None

    # 2D layouts (proportional, shared bounds within each side)
    # Patient layout: cover whichever mask the caller gave us, else thickness > 0
    if patient_subch_mask is not None:
        pat_layout_mask = patient_subch_mask | (patient_thickness > 0)
    else:
        pat_layout_mask = patient_thickness > 0

    # CANONICAL 2D projection per bone (femur=unwrap, tibia=proportional;
    # both lateral-left). Reuse the tibia layout across thickness + contours.
    g_pat_th, lay_pat = project_thickness_2d_canonical(
        bone, patient_mesh, patient_thickness, grid_size=grid_size,
        smooth_sigma=smooth_sigma, subch_prob=patient_subch_mask)
    g_tpl_th, lay_tpl = project_thickness_2d_canonical(
        bone, template_mesh, template_thickness, grid_size=grid_size,
        smooth_sigma=smooth_sigma)

    # Subch / compartment contours (same canonical projection so they overlay)
    def _contour(mesh_, mask, layout, sprob=None):
        g, _ = project_thickness_2d_canonical(
            bone, mesh_, mask.astype(np.float32), grid_size=grid_size,
            smooth_sigma=0.0, subch_prob=sprob, tibia_layout=layout, agg="any")
        return _close_filled(g)

    pat_contours, tpl_contours = [], []
    if patient_subch_mask is not None and has_tpl_compartment and patient_compartment is not None:
        for lbl in (1, 2):
            pat_contours.append(_contour(
                patient_mesh, patient_subch_mask & (patient_compartment == lbl),
                lay_pat, sprob=patient_subch_mask))
            tpl_contours.append(_contour(
                template_mesh, tpl_subch & (tpl_comp == lbl), lay_tpl))
    elif patient_subch_mask is not None:
        pat_contours.append(_contour(patient_mesh, patient_subch_mask,
                                      lay_pat, sprob=patient_subch_mask))
        tpl_contours.append(_contour(template_mesh, tpl_subch, lay_tpl))

    # Both bones now display lateral-left (canonical), so M label on the right.
    ml_flip_2d = True

    # 3D screenshots
    img_pat_3d = screenshot_3d_thickness(patient_mesh, patient_thickness, bone, vmax=vmax)
    img_tpl_3d = screenshot_3d_thickness(template_mesh, template_thickness, bone, vmax=vmax)

    # Render
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(img_pat_3d); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[0, 0].set_title("PATIENT 3D", fontsize=12)
    axes[0, 1].imshow(img_tpl_3d); axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    axes[0, 1].set_title("TEMPLATE 3D", fontsize=12)

    # Canonical orientation: lateral-left/medial-right (both bones); femur
    # anterior at TOP (unwrap native), tibia anterior at BOTTOM.
    a_y, p_y = (0.97, 0.03) if bone == "femur" else (0.03, 0.97)

    def _draw_2d(ax, g, contours, title, info):
        im = ax.imshow(g, origin="upper", cmap="jet_r", vmin=0, vmax=vmax,
                        interpolation="bilinear", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
            "thickness (mm)", fontsize=8)
        for c in contours:
            if c.any():
                ax.contour(c, levels=[0.5], colors="black", linewidths=1.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for tx, ty, label, ha in ((0.98, 0.97, "M", "right"),
                                   (0.02, 0.97, "L", "left"),
                                   (0.5, a_y, "A", "center"),
                                   (0.5, p_y, "P", "center")):
            ax.text(tx, ty, label, transform=ax.transAxes, color="white",
                    fontsize=9, va=("top" if ty > 0.5 else "bottom"), ha=ha,
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))
        ax.text(0.5, -0.04, info, transform=ax.transAxes,
                fontsize=8, va="top", ha="center")

    with np.errstate(invalid="ignore"):
        pat_pos = patient_thickness[patient_thickness > 0]
        info_pat = (f"mean(>0)={pat_pos.mean():.2f}mm  n>0={len(pat_pos)}"
                     if len(pat_pos) else "no hits")
        if has_tpl_compartment:
            tpl_med = float(np.nanmean(template_thickness[tpl_comp == 1]))
            tpl_lat = float(np.nanmean(template_thickness[tpl_comp == 2]))
            info_tpl = (f"med={tpl_med:.2f}  lat={tpl_lat:.2f}  "
                         f"total={np.nanmean(template_thickness):.2f}mm")
        else:
            info_tpl = f"mean={np.nanmean(template_thickness):.2f}mm"

    _draw_2d(axes[1, 0], g_pat_th, pat_contours,
              f"PATIENT 2D  (proportional projection, grid={grid_size})", info_pat)
    _draw_2d(axes[1, 1], g_tpl_th, tpl_contours,
              f"TEMPLATE 2D  (proportional projection, grid={grid_size})", info_tpl)

    suptitle = f"{case_id}  {bone}"
    if subtitle:
        suptitle += f"  -  {subtitle}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    return Path(out_png)


# ===========================================================================
# Canonical 2x3 method-comparison figure (template-only, proportional
# projection, bone-aware contours, optional QCart bar row)
# ===========================================================================
def _template_layout(template_mesh: pv.PolyData, grid_size: int):
    tpl_pts = np.asarray(template_mesh.points)
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5
    _, layout = proportional_projection_2d(tpl_pts, None, tpl_subch,
                                              grid_size=grid_size, agg="any")
    return tpl_pts, tpl_subch, layout


def project_thickness_2d_canonical(bone: str, mesh: pv.PolyData,
                                     scalar: np.ndarray,
                                     grid_size: int = 120,
                                     smooth_sigma: float = 1.2,
                                     subch_prob: np.ndarray | None = None,
                                     tibia_layout: dict | None = None,
                                     agg: str = "mean",
                                     subch_threshold: float = 0.5):
    """CANONICAL 2D projection of a per-vertex scalar on a femur/tibia mesh.

    Projection method per bone (FIXED — do not mix):
      - femur : cylindrical UNWRAP (`template_thickness_2d_femur`). Preserves
                the condyle arc length (anterior trochlea → distal →
                posterior). Works on patient OR template meshes; for patient
                meshes (no `subch_prob` point_data) pass `subch_prob=` the
                per-vertex subch probability (falls back to scalar>0).
      - tibia : PROPORTIONAL top-down (`proportional_projection_2d`), shared
                ML/AP bounds → real compartment proportions + intercondylar gap.

    Canonical display orientation (BOTH bones): lateral on the LEFT, medial on
    the RIGHT. Femur anterior at TOP (unwrap native); tibia anterior at BOTTOM.

    Returns (grid_2d, tibia_layout). `tibia_layout` is the reusable layout for
    tibia (None for femur) so multiple overlays snap to the same grid.
    """
    scalar = np.asarray(scalar)
    if bone == "tibia":
        pts = np.asarray(mesh.points)
        if "subch_prob" in mesh.point_data:
            sub = np.asarray(mesh.point_data["subch_prob"]) >= subch_threshold
        elif subch_prob is not None:
            sub = np.asarray(subch_prob) >= subch_threshold
        else:
            sub = np.isfinite(scalar) & (scalar > 0)
        g, tibia_layout = proportional_projection_2d(
            pts, scalar.astype(np.float64),
            sub & np.isfinite(scalar), grid_size=grid_size,
            layout=tibia_layout, agg=agg)
        if smooth_sigma and smooth_sigma > 0 and agg == "mean":
            g = _smooth_border_preserving(g, sigma=smooth_sigma)
        g = g[:, ::-1]                 # lateral-left
        return g, tibia_layout
    # ---- femur unwrap ----
    from cartilage_morphometry import (
        compute_femur_template_subregions, template_thickness_2d_femur,
    )
    m = mesh
    if "subch_prob" not in m.point_data:
        m = mesh.copy()
        sp = subch_prob if subch_prob is not None else (np.asarray(scalar) > 0).astype(np.float32)
        m["subch_prob"] = np.asarray(sp, dtype=np.float32)
    subregions = compute_femur_template_subregions(m, subch_thresh=subch_threshold)
    if agg == "any":
        v = np.where(np.asarray(scalar, dtype=bool), 1.0, np.nan).astype(np.float64)
    else:
        v = scalar.astype(np.float64)
    g, _, _ = template_thickness_2d_femur(m, v, grid_size=grid_size,
                                           subregions=subregions)
    if agg == "any":
        g = np.where(np.isfinite(g) & (g > 0), 1.0, np.nan)
    elif smooth_sigma and smooth_sigma > 0:
        g = _smooth_border_preserving(g, sigma=smooth_sigma)
    g = g[:, ::-1]                     # lateral-left (femur native is medial-left)
    return g, None


def _project_bone_2d(bone: str, template_mesh: pv.PolyData,
                       scalar: np.ndarray, layout_pack,
                       smooth_sigma: float, grid_size: int):
    """Project a per-template-vert scalar to a 2D grid using the correct
    projection per bone:

    - Tibia: proportional top-down (AP, ML) projection with shared bounds
      (`proportional_projection_2d`). Preserves intercondylar gap and
      compartment proportions.
    - Femur: angular unwrap along the condylar cylinder axis (library
      `template_thickness_2d_femur`). The cart-bearing surface wraps from
      anterior trochlea → distal condyle → posterior condyle, so a top-down
      (AP, ML) projection collapses that arc and looks too short. The unwrap
      preserves arc length along the condyle surface.

    `layout_pack` is the per-bone layout cache returned by
    `_bone_layout_pack`. Returns the 2D grid (always rows=AP-like, cols=ML;
    interpolation/aspect handled by the caller).
    """
    if bone == "tibia":
        tpl_pts = np.asarray(template_mesh.points)
        tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5
        g, _ = proportional_projection_2d(
            tpl_pts, scalar.astype(np.float64),
            tpl_subch & np.isfinite(scalar), layout=layout_pack["layout"])
    else:  # femur
        subregions = layout_pack["femur_subregions"]
        from cartilage_morphometry import template_thickness_2d_femur
        g, _, _ = template_thickness_2d_femur(
            template_mesh, scalar.astype(np.float64),
            grid_size=grid_size, subregions=subregions)
    if smooth_sigma and smooth_sigma > 0:
        g = _smooth_border_preserving(g, sigma=smooth_sigma)
    return g


def _bone_layout_pack(bone: str, template_mesh: pv.PolyData, grid_size: int):
    """Per-bone projection state: a layout dict for proportional (tibia) or
    a FemurSubregions object for the unwrap (femur). Cached across the
    multiple panel calls of one figure.
    """
    if bone == "tibia":
        tpl_pts = np.asarray(template_mesh.points)
        tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5
        _, layout = proportional_projection_2d(tpl_pts, None, tpl_subch,
                                                  grid_size=grid_size, agg="any")
        return {"layout": layout}
    # femur unwrap
    from cartilage_morphometry import compute_femur_template_subregions
    return {"femur_subregions": compute_femur_template_subregions(template_mesh)}


def femur_cmf_clf_arc_masks(template_mesh: pv.PolyData,
                              fraction: float = 0.75,
                              subch_threshold: float = 0.5):
    """cMF / cLF region masks — canonical Eckstein AP-distance split.

    Thin wrapper around `compute_femur_eckstein_apsplit_subregions`
    (cartilage_morphometry/subregions.py), which is the definition used in the
    PDvDESS v6 study and the manuscript:

      - START anchor = trochlear notch = the most-posterior subch vert on the
        central ML slice (the deepest point of the trochlear/intercondylar
        groove dip).
      - END anchor = per-compartment posterior tip = max-AP subch vert in
        each medial / lateral hemisphere.
      - cMF / cLF = the anterior `fraction` (default 0.75) of the AP distance
        from notch to posterior tip:
            AP in [ap_notch, ap_notch + fraction*(ap_end - ap_notch)]
        pMF/pLF is the remaining posterior portion; trochlear cart anterior
        to the notch is excluded (LBL_NOTCH_PRE).

    Posterior-island removal: the condyle is C-shaped in sagittal section, so
    the AP band [notch, split] intersects cartilage TWICE — the main
    weight-bearing surface AND a small island where the posterior tip rolls
    back up into the same AP range. We drop that island using the cylindrical
    unwrap arc: keep only verts on the main-surface side of the most-posterior
    subch vert's arc position.

    Returns (cmf_mask, clf_mask) as bool arrays (n_template,).
    """
    from cartilage_morphometry import (
        compute_femur_eckstein_apsplit_subregions,
        compute_femur_template_subregions,
    )
    from cartilage_morphometry.subregions import LBL_CMF_REGION, LBL_CLF_REGION
    sub = compute_femur_eckstein_apsplit_subregions(
        template_mesh, fraction=fraction, subch_thresh=subch_threshold)
    region = np.asarray(sub.region)
    cmf = (region == LBL_CMF_REGION)
    clf = (region == LBL_CLF_REGION)

    # ---- posterior-island removal via the unwrap arc ----
    arc_sub = compute_femur_template_subregions(template_mesh)
    arc = np.asarray(arc_sub.arc_norm)
    is_subch = np.asarray(arc_sub.is_subch)
    pts = np.asarray(template_mesh.points)
    AP = pts[:, 0]; ML = pts[:, 2]
    valid = is_subch & np.isfinite(arc)
    ml_subch = ML[is_subch]
    ml_mid = float((ml_subch.min() + ml_subch.max()) / 2.0)

    def _drop_island(mask: np.ndarray, side_mask: np.ndarray) -> np.ndarray:
        m = mask & valid
        if not m.any():
            return mask
        side = side_mask & valid
        if not side.any():
            return mask
        # Arc of the most-posterior subch vert in THIS compartment
        ap_post_thresh = float(np.percentile(AP[side], 98))
        post_band = side & (AP >= ap_post_thresh)
        if not post_band.any():
            return mask
        arc_tip = float(np.median(arc[post_band]))
        # Which arc side holds the main surface? The side containing the bulk
        # of the AP-split mask (by vert count).
        n_above = int((m & (arc >= arc_tip)).sum())
        n_below = int((m & (arc < arc_tip)).sum())
        if n_above >= n_below:
            keep = arc >= arc_tip      # main surface on the high-arc side
        else:
            keep = arc <= arc_tip
        return mask & keep

    is_med = is_subch & (ML <= ml_mid)
    is_lat = is_subch & (ML > ml_mid)
    cmf = _drop_island(cmf, is_med)
    clf = _drop_island(clf, is_lat)
    return cmf, clf


def _template_contours(template_mesh: pv.PolyData, bone: str,
                        layout_pack, grid_size: int,
                        cmf_fraction: float = 0.75,
                        skip_femur: bool = False):
    """Return list of binary contour grids (closed-filled, float 0/1) to
    overlay in black on every 2D panel of this bone.

    Tibia: medial and lateral subch (compartment field 1/2), proportional projection.
    Femur: arc-based cMF / cLF (anterior 75 % of the cylindrical unwrap;
           excludes the posterior condyle tip — PDvDESS v6 convention).
           Skipped entirely if `skip_femur=True` (e.g. cross-sectional pairs
           where the Eckstein ROI isn't the focus).
    """
    tpl_pts = np.asarray(template_mesh.points)
    tpl_subch = np.asarray(template_mesh.point_data["subch_prob"]) >= 0.5
    contours: list[np.ndarray] = []
    if bone == "tibia" and "compartment" in template_mesh.point_data:
        comp = np.asarray(template_mesh.point_data["compartment"])
        for lbl in (1, 2):
            g, _ = proportional_projection_2d(
                tpl_pts, None, tpl_subch & (comp == lbl),
                layout=layout_pack["layout"], agg="any")
            contours.append(_close_filled(g))
    elif bone == "femur" and not skip_femur:
        cmf_mask, clf_mask = femur_cmf_clf_arc_masks(template_mesh,
                                                      fraction=cmf_fraction)
        from cartilage_morphometry import template_thickness_2d_femur
        femur_subreg = layout_pack["femur_subregions"]
        for mask in (cmf_mask, clf_mask):
            m = mask.astype(np.float32)
            m[m == 0] = np.nan
            g, _, _ = template_thickness_2d_femur(
                template_mesh, m, grid_size=grid_size,
                subregions=femur_subreg)
            contours.append(_close_filled(g))
    return contours


def _femur_arc_region_means(template_mesh: pv.PolyData,
                              thickness: np.ndarray,
                              fraction: float = 0.75,
                              zero_impute: bool = True) -> dict:
    """Per-region (cMF/cLF) mean thickness via the canonical Eckstein
    AP-distance subregions (`femur_eckstein_region_means_3d`).

    `zero_impute=True` matches the manuscript / Eckstein-QCart convention
    (denuded subch verts count as 0 mm). Returns only cMF/cLF (pMF/pLF
    dropped — not used in the figure annotations).
    """
    from cartilage_morphometry import (
        compute_femur_eckstein_apsplit_subregions,
        femur_eckstein_region_means_3d,
    )
    sub = compute_femur_eckstein_apsplit_subregions(template_mesh, fraction=fraction)
    means = femur_eckstein_region_means_3d(thickness, sub, zero_impute=zero_impute)
    return {"cMF": means.get("cMF", float("nan")),
            "cLF": means.get("cLF", float("nan"))}


def _render_grid_proportional(
        case_id: str,
        col_keys: tuple[str, str, str],
        col_labels: tuple[str, str, str],
        arrays: dict[tuple[str, str], np.ndarray],
        out_png: Path,
        title_suffix: str,
        vmax_thick: float = 4.0, vabs_delta: float = 1.5,
        grid_size: int = 120,
        smooth_sigma: float = 1.2,
        cmf_fraction: float = 0.75,
        skip_femur_contours: bool = False,
        regional_means: dict | None = None,
        qcart_row: dict | None = None,
        pipeline_regions: dict[str, dict] | None = None) -> Path:
    """Canonical 2x3 method-comparison figure (femur + tibia rows × col0,
    col1, Δ). Uses proportional projection (shared ML/AP bounds, isotropic
    aspect), Gaussian smoothing, bone-aware black contours, femur M/L flip.

    `col_keys[2]` must be "delta". `arrays[(bone, key)]` are template
    thickness arrays (shape n_template, NaN outside template subch).

    If `qcart_row` + `pipeline_regions` provided, appends a 3rd wide row
    with the standard pipeline-Δ-vs-QCart-Δ bar comparison.
    """
    if col_keys[2] != "delta":
        raise ValueError("col_keys[2] must be 'delta'")

    # Build per-bone projections + contours
    grids: dict[tuple[str, str], np.ndarray] = {}
    contours_per_bone: dict[str, list[np.ndarray]] = {}
    for bone in ("femur", "tibia"):
        template_mesh = _template_pack(bone)["mesh"]
        layout_pack = _bone_layout_pack(bone, template_mesh, grid_size)
        a = _project_bone_2d(bone, template_mesh, arrays[(bone, col_keys[0])],
                              layout_pack, smooth_sigma, grid_size)
        b = _project_bone_2d(bone, template_mesh, arrays[(bone, col_keys[1])],
                              layout_pack, smooth_sigma, grid_size)
        d = (b - a) if col_keys[1] == "48m" else (a - b)
        contours = _template_contours(template_mesh, bone, layout_pack,
                                        grid_size, cmf_fraction=cmf_fraction,
                                        skip_femur=skip_femur_contours)
        # Canonical 2D orientation (BOTH bones): lateral on the LEFT, medial
        # on the RIGHT → flip ML cols on every grid.
        a = a[:, ::-1]; b = b[:, ::-1]; d = d[:, ::-1]
        contours = [c[:, ::-1] for c in contours]
        # Femur AP: library unwrap already places anterior at TOP of the
        # array. No row flip applied — labels A=top, P=bottom match the
        # rendered orientation as-is.
        grids[(bone, col_keys[0])] = a
        grids[(bone, col_keys[1])] = b
        grids[(bone, "delta")] = d
        contours_per_bone[bone] = contours

    show_qcart = qcart_row is not None and qcart_row.get("present") and pipeline_regions is not None
    # Pair scatter mode is enabled when arrays for col_keys[0] and col_keys[1]
    # exist for both bones (true for our pair use case) AND not in long-qcart
    # mode. In long-qcart mode the bottom row is the QCart-comparison bars.
    show_pair_scatter = (not show_qcart) and (col_keys[0] in ("pd",) or col_keys[1] in ("dess",))
    if show_qcart:
        fig = plt.figure(figsize=(15, 13))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.55])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                          for r in range(2)])
        ax_bars = fig.add_subplot(gs[2, :])
        ax_scatter = None
    elif show_pair_scatter:
        fig = plt.figure(figsize=(15, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                          for r in range(2)])
        ax_bars = None
        ax_scatter = (fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax_bars = None
        ax_scatter = None

    rows = [("femur", "Femur"), ("tibia", "Tibia")]
    cols = [(col_keys[0], col_labels[0]),
            (col_keys[1], col_labels[1]),
            ("delta", col_labels[2])]

    # `r_2d` is only meaningful for PD↔DESS (cross-method agreement). In
    # longitudinal (00m vs 48m of the SAME knee) it's near 1.0 by construction
    # and just adds noise — suppress.
    show_r2d = col_keys[0] == "pd" and col_keys[1] == "dess"

    for r, (bone, row_label) in enumerate(rows):
        r_val = _r2d(grids[(bone, col_keys[0])], grids[(bone, col_keys[1])]) if show_r2d else float("nan")
        # Both bones now have M-on-right / L-on-left (lateral side on the
        # left of the image). Femur additionally has A-on-top after the
        # row flip; tibia keeps the original A-on-bottom (which matched
        # the previously-approved tibia layout).
        m_x, l_x = 0.98, 0.02
        m_ha, l_ha = "right", "left"
        if bone == "femur":
            a_y, p_y = 0.97, 0.03
        else:  # tibia
            a_y, p_y = 0.03, 0.97
        for c, (key, col_label) in enumerate(cols):
            ax = axes[r, c]
            g = grids[(bone, key)]
            if key == "delta":
                im = ax.imshow(g, origin="upper", cmap="RdBu",
                                 vmin=-vabs_delta, vmax=vabs_delta,
                                 interpolation="bilinear", aspect="equal")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("Delta thickness (mm)", fontsize=9)
            else:
                im = ax.imshow(g, origin="upper", cmap="jet_r",
                                 vmin=0.0, vmax=vmax_thick,
                                 interpolation="bilinear", aspect="equal")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("thickness (mm)", fontsize=9)
            for contour in contours_per_bone[bone]:
                if contour.any():
                    ax.contour(contour, levels=[0.5], colors="black",
                                linewidths=1.4)
            title = f"{row_label} -- {col_label}"
            if c == 0 and show_r2d and np.isfinite(r_val):
                title += f"     [r_2d={r_val:.3f}]"
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for tx, ty, label, ha in ((m_x, 0.97, "M", m_ha),
                                       (l_x, 0.97, "L", l_ha),
                                       (0.5, a_y, "A", "center"),
                                       (0.5, p_y, "P", "center")):
                ax.text(tx, ty, label, transform=ax.transAxes,
                        color="white", fontsize=9,
                        va=("top" if ty > 0.5 else "bottom"), ha=ha,
                        bbox=dict(facecolor="black", alpha=0.5,
                                   edgecolor="none", pad=2))
            # Per-panel info: overall mean + per-region means (cMF/cLF for
            # femur, MT/LT for tibia). Pulled from `regional_means` dict if
            # provided by the caller, computed on the underlying 3D template
            # arrays (not the smoothed 2D grid).
            with np.errstate(invalid="ignore"):
                if key == "delta":
                    line0 = f"mean Δ = {np.nanmean(g):+.3f} mm"
                else:
                    line0 = f"mean = {np.nanmean(g):.3f} mm"
            lines = [line0]
            if regional_means is not None:
                rm = regional_means.get(bone, {}).get(key)
                if rm:
                    if bone == "femur":
                        lines.append(
                            f"cMF={rm.get('cMF', float('nan')):+.2f}  "
                            f"cLF={rm.get('cLF', float('nan')):+.2f}"
                            if key == "delta" else
                            f"cMF={rm.get('cMF', float('nan')):.2f}  "
                            f"cLF={rm.get('cLF', float('nan')):.2f}")
                    else:  # tibia
                        lines.append(
                            f"MT={rm.get('MT', float('nan')):+.2f}  "
                            f"LT={rm.get('LT', float('nan')):+.2f}"
                            if key == "delta" else
                            f"MT={rm.get('MT', float('nan')):.2f}  "
                            f"LT={rm.get('LT', float('nan')):.2f}")
            ax.text(0.5, -0.06, "\n".join(lines), transform=ax.transAxes,
                    fontsize=9, va="top", ha="center")

    if ax_bars is not None:
        _draw_qcart_bar_row(ax_bars, qcart_row, pipeline_regions)

    if ax_scatter is not None:
        # Per-bone vertex-wise scatter of col0 (e.g. PD) vs col1 (e.g. DESS),
        # one panel per bone. r = Pearson over template subch verts where
        # both methods returned finite thickness.
        for i, (bone, ax_s) in enumerate(zip(("femur", "tibia"), ax_scatter)):
            x = arrays[(bone, col_keys[0])]
            y = arrays[(bone, col_keys[1])]
            valid = np.isfinite(x) & np.isfinite(y)
            xv = x[valid]; yv = y[valid]
            r = _r2d(xv, yv) if len(xv) >= 3 else float("nan")
            mae = float(np.mean(np.abs(xv - yv))) if len(xv) else float("nan")
            bias = float(np.mean(xv - yv)) if len(xv) else float("nan")
            ax_s.scatter(yv, xv, s=2, alpha=0.25, c="#1f77b4", linewidths=0)
            lim = float(vmax_thick)
            ax_s.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
            ax_s.set_xlim(0, lim); ax_s.set_ylim(0, lim)
            ax_s.set_xlabel(f"{col_labels[1]} (mm)", fontsize=10)
            ax_s.set_ylabel(f"{col_labels[0]} (mm)", fontsize=10)
            ax_s.set_title(f"{bone.title()} per-vertex  (n={len(xv)})",
                            fontsize=11)
            # Stats inset (top-left of panel) so the title stays short and
            # nothing overlaps with the axis ticks.
            stats_txt = (f"r = {r:.3f}\n"
                          f"MAE = {mae:.2f} mm\n"
                          f"bias = {bias:+.2f} mm")
            ax_s.text(0.04, 0.96, stats_txt, transform=ax_s.transAxes,
                       fontsize=10, va="top", ha="left",
                       bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="0.6", pad=4))
            ax_s.grid(True, linestyle=":", alpha=0.4)
            ax_s.set_aspect("equal")

    fig.suptitle(f"{case_id}  {title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def make_pair_figure_proportional(case_id: str, cs_save_dir: Path, out_png: Path,
                                    vmax_thick: float = 4.0,
                                    vabs_delta: float = 1.5,
                                    smooth_sigma: float = 1.2,
                                    grid_size: int = 120,
                                    cmf_fraction: float = 0.75) -> Path:
    """Canonical proportional-projection 2x3 PD vs DESS pair figure.

    Cross-sectional: cMF/cLF outlines are NOT drawn on femur (the Eckstein
    ROI is only relevant to longitudinal QCart comparison); full cart pattern
    is shown without ROI restrictions. Tibia medial/lateral compartment
    contours are kept (they're useful for cross-method comparison).
    """
    cs_save_dir = Path(cs_save_dir)
    arrays: dict = {}
    for bone in ("femur", "tibia"):
        arrays[(bone, "pd")] = np.load(cs_save_dir / f"{case_id}_{bone}_pd.npy")
        arrays[(bone, "dess")] = np.load(cs_save_dir / f"{case_id}_{bone}_dess.npy")
    return _render_grid_proportional(
        case_id, ("pd", "dess", "delta"),
        ("PD", "DESS", "Delta (PD - DESS)"),
        arrays, out_png,
        title_suffix="PD vs DESS thickness  [v1.2 raycast_2d, proportional]",
        vmax_thick=vmax_thick, vabs_delta=vabs_delta,
        smooth_sigma=smooth_sigma, grid_size=grid_size,
        cmf_fraction=cmf_fraction,
        skip_femur_contours=True,
    )


def _compute_long_regional_means(arrays: dict, cmf_fraction: float = 0.75) -> dict:
    """For each bone, compute 00m / 48m / Δ regional means.
    Femur uses arc-based cMF/cLF; tibia uses Eckstein AP-split MT/LT.

    Returns: `{bone: {"00m": {<region>: mm}, "48m": {<region>: mm}, "delta": {<region>: mm}}}`
    """
    from .api import _template_pack as _api_template_pack, _region_means as _api_region_means
    out: dict[str, dict[str, dict[str, float]]] = {}
    for bone in ("femur", "tibia"):
        template_mesh = _api_template_pack(bone)["mesh"]
        a = arrays[(bone, "00m")]
        b = arrays[(bone, "48m")]
        if bone == "femur":
            r00 = _femur_arc_region_means(template_mesh, a, fraction=cmf_fraction)
            r48 = _femur_arc_region_means(template_mesh, b, fraction=cmf_fraction)
        else:
            r00 = _api_region_means(bone, a)
            r48 = _api_region_means(bone, b)
        delta = {k: float(r48[k] - r00[k]) for k in r00 if k in r48}
        out[bone] = {"00m": r00, "48m": r48, "delta": delta}
    return out


def make_long_figure_proportional(case_tag: str, long_save_dir: Path,
                                    out_png: Path,
                                    vmax_thick: float = 4.0,
                                    vabs_delta: float = 1.5,
                                    smooth_sigma: float = 1.2,
                                    grid_size: int = 120,
                                    cmf_fraction: float = 0.75,
                                    qcart_row: dict | None = None,
                                    pipeline_regions: dict[str, dict] | None = None) -> Path:
    """Canonical proportional-projection 2x3 longitudinal figure (+ optional
    QCart bar row if `qcart_row` + `pipeline_regions` supplied).

    Per-panel info text shows the region-of-interest means: cMF/cLF for femur
    (arc-based 75 % anterior weight-bearing) and MT/LT for tibia.

    If `qcart_row` is given but `pipeline_regions` isn't, the bar row's
    pipeline values will fall back to whatever was passed.
    """
    long_save_dir = Path(long_save_dir)
    arrays: dict = {}
    for bone in ("femur", "tibia"):
        arrays[(bone, "00m")] = np.load(long_save_dir / f"{case_tag}_{bone}_00m.npy")
        arrays[(bone, "48m")] = np.load(long_save_dir / f"{case_tag}_{bone}_48m.npy")
    regional_means = _compute_long_regional_means(arrays, cmf_fraction=cmf_fraction)
    # If pipeline_regions isn't pre-built, derive from regional_means so the
    # bar comparison row stays in sync with the per-panel annotations.
    if pipeline_regions is None and qcart_row is not None:
        pipeline_regions = {
            "femur": regional_means["femur"]["delta"],
            "tibia": regional_means["tibia"]["delta"],
        }
    return _render_grid_proportional(
        case_tag, ("00m", "48m", "delta"),
        ("00m baseline", "48m followup", "Delta (48m - 00m)"),
        arrays, out_png,
        title_suffix="Longitudinal thickness  [v1.2 raycast_2d, proportional]",
        vmax_thick=vmax_thick, vabs_delta=vabs_delta,
        smooth_sigma=smooth_sigma, grid_size=grid_size,
        cmf_fraction=cmf_fraction,
        regional_means=regional_means,
        qcart_row=qcart_row, pipeline_regions=pipeline_regions,
    )
