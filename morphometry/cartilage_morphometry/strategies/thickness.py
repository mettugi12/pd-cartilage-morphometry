"""Per-vertex cartilage-thickness strategies.

Each strategy has the same signature:
    fn(bone_mesh, bone_mask, cart_mask, spacing, cfg) -> np.ndarray of (n_verts,)

The returned array has thickness in mm at every bone-mesh vertex, with 0 for
verts that don't hit cartilage (or with too few intersections).

Strategies available:
    raycast          DEFAULT — first cart hit → last cart hit along outward normal
    raycast_to_far   Bone vertex → last cart hit (PDvDESS v6 forward raycast variant)
    edt              Per-cart-voxel distance-transform, neighbourhood-max-sampled at verts
                     (the original "PD-vs-DESS canonical EDT" — fast, less faithful)

The CANONICAL raycast definition is `raycast` (first-cart to last-cart): this
measures the true cart slab thickness along the ray, ignoring any bone-cart
gap from sub-resolution segmentation jitter or marginal sclerosis. The
"bone-surface to far-cart-surface" variant (PDvDESS v6's `thickness_raycast`)
overcounts by the bone-cart gap and is provided only for backwards
compatibility / sensitivity analysis.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import (
    binary_erosion, distance_transform_edt, gaussian_filter,
    map_coordinates, maximum_filter,
)
from scipy.spatial import cKDTree

from . import register_thickness


_MAX_THICKNESS_MM_DEFAULT = 6.0
_CART_SMOOTH_ITERS = 15


def _build_cart_surface(cart_mask: np.ndarray, spacing: tuple,
                        smooth_iters: int = _CART_SMOOTH_ITERS) -> pv.PolyData | None:
    """Cart mesh via marching cubes + light smoothing."""
    if cart_mask.sum() == 0:
        return None
    grid = pv.ImageData(dimensions=cart_mask.shape)
    grid.spacing = spacing
    grid.point_data["m"] = cart_mask.astype(np.float32).flatten(order="F")
    mesh = grid.contour([0.5], scalars="m")
    if mesh.n_points == 0:
        return None
    if smooth_iters > 0:
        mesh = mesh.smooth(n_iter=smooth_iters, relaxation_factor=0.1)
    return mesh


@register_thickness("raycast")
def thickness_raycast_first_slab(bone_mesh, bone_mask, cart_mask, spacing, cfg):
    """**Canonical raycast** — first cart intersection to its IMMEDIATE EXIT
    (i.e. the FIRST cart slab along the outward normal). Per-vertex cart
    slab thickness.

    Algorithm:
      - Build the cart surface mesh (marching cubes + smoothing).
      - For each bone vertex, cast a ray of length `2 × max_thickness_mm` along
        the outward normal (computed off a smoothed bone mesh — see
        `build_meshes(bone_smooth_iters=30)`).
      - `multi_ray_trace(first_point=False)` returns ALL cart intersections.
      - Per ray: sort hits by distance. **Use only the first two hits** — the
        entry into the first cart slab (≈ bone-cart interface side) and its
        immediate exit (articular side). Thickness = d[1] - d[0].
      - Discard rays where d[1] - d[0] >= max_thickness_mm (default 4 mm) as
        unphysiologically thick — almost always means the ray crossed joint
        space into a different cart (e.g. femoral → tibial) rather than
        cleanly exiting one slab.
      - Verts with 0 or 1 hit → thickness 0.

    Why this and not first-to-last:
      A single ray can cross MULTIPLE cart slabs (femoral cart → joint space
      → tibial cart, or trochlea → patella cart). first-to-last would sum
      those slabs plus the joint space, giving wildly inflated thickness
      (capped at 6 mm).  first-to-second-hit isolates the local slab.

    The smoothed bone mesh (Laplacian smoothing in `build_meshes`) is
    essential — marching-cubes' axis-aligned voxel faces produce normals that
    point in cardinal directions, which makes rays miss the cart slab on flat
    panels and overshoot on stair diagonals.
    """
    max_thickness_mm = float(getattr(cfg, "max_thickness_mm", _MAX_THICKNESS_MM_DEFAULT))
    smooth_iters = int(getattr(cfg, "cart_smooth_iters", _CART_SMOOTH_ITERS))
    n = bone_mesh.n_points
    if cart_mask.sum() == 0:
        return np.zeros(n, dtype=np.float32)
    cart_surface = _build_cart_surface(cart_mask, spacing, smooth_iters=smooth_iters)
    if cart_surface is None or cart_surface.n_points == 0:
        return np.zeros(n, dtype=np.float32)
    cart_tri = cart_surface.triangulate()

    bone_pts = np.asarray(bone_mesh.points, dtype=np.float64)
    bone_normals = np.asarray(bone_mesh.point_normals, dtype=np.float64)
    nrm = np.linalg.norm(bone_normals, axis=1, keepdims=True)
    bone_normals = bone_normals / np.maximum(nrm, 1e-9)
    # Ray length 2× max_thickness_mm so we capture entry + exit of the first
    # slab. We don't NEED longer rays because we only use the first two hits.
    # 4× max_thickness for very oblique angles on the posterior condyle —
    # we still only use the first 2 hits, so longer rays don't risk crossing
    # joint space.
    ray_length = 4.0 * max_thickness_mm
    directions = bone_normals * ray_length

    out = np.zeros(n, dtype=np.float32)
    try:
        # first_point=False → returns ALL intersections per ray
        ipts, iray, _ = cart_tri.multi_ray_trace(
            bone_pts, directions, first_point=False, retry=False
        )
    except Exception as e:
        print(f"  [raycast] multi_ray_trace failed ({e}); falling back per-vert")
        for i in range(n):
            try:
                pts_i, _ = cart_tri.ray_trace(bone_pts[i], bone_pts[i] + directions[i],
                                              first_point=False)
                if len(pts_i) >= 2:
                    d = np.linalg.norm(pts_i - bone_pts[i], axis=1)
                    d.sort()
                    t = float(d[1] - d[0])
                    if 0 < t < max_thickness_mm:
                        out[i] = t
            except Exception:
                pass
        return out

    if len(iray) == 0:
        return out

    # For each ray, pick the first two hits (sorted by distance from origin)
    d_per_hit = np.linalg.norm(ipts - bone_pts[iray], axis=1)
    order = np.lexsort((d_per_hit, iray))
    iray_s = iray[order]
    d_s = d_per_hit[order]
    diffs = np.diff(iray_s, prepend=iray_s[0] - 1)
    starts = np.where(diffs != 0)[0]
    ends = np.append(starts[1:], len(iray_s))
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue                            # need ≥2 hits for a slab
        first_d = d_s[s]
        exit_d = d_s[s + 1]                     # IMMEDIATE exit (first slab)
        t = float(exit_d - first_d)
        if 0 < t < max_thickness_mm:
            out[iray_s[s]] = t
    return out


@register_thickness("raycast_to_far")
def thickness_raycast_to_far(bone_mesh, bone_mask, cart_mask, spacing, cfg):
    """Legacy v6 variant — bone vertex to FAR cart surface (first hit on the
    articular-side cart mesh after dropping inner cart-bone interface faces).

    Provided for sensitivity analysis only. Use `raycast` (first-to-last) as
    canonical. See PDvDESS v6 `thickness_raycast` for the original.
    """
    max_thickness_mm = float(getattr(cfg, "max_thickness_mm", _MAX_THICKNESS_MM_DEFAULT))
    smooth_iters = int(getattr(cfg, "cart_smooth_iters", _CART_SMOOTH_ITERS))
    min_art_dist_mm = float(getattr(cfg, "min_art_dist_mm", 0.5))
    n = bone_mesh.n_points
    if cart_mask.sum() == 0:
        return np.zeros(n, dtype=np.float32)
    cart_surface = _build_cart_surface(cart_mask, spacing, smooth_iters=smooth_iters)
    if cart_surface is None or cart_surface.n_points == 0:
        return np.zeros(n, dtype=np.float32)

    # Extract only the "articular" cart faces (centroid >= min_art_dist_mm from bone)
    from scipy.spatial import cKDTree
    cart_tri = cart_surface.triangulate()
    faces = cart_tri.faces.reshape(-1, 4)[:, 1:4]
    pts = np.asarray(cart_tri.points)
    centroids = pts[faces].mean(axis=1)
    bone_tree = cKDTree(np.asarray(bone_mesh.points))
    d_to_bone, _ = bone_tree.query(centroids, k=1)
    art_face_idx = np.where(d_to_bone >= min_art_dist_mm)[0]
    if len(art_face_idx) == 0:
        return np.zeros(n, dtype=np.float32)
    art_tri = cart_tri.extract_cells(art_face_idx).extract_surface(
        algorithm="dataset_surface"
    ).triangulate()

    bone_pts = np.asarray(bone_mesh.points, dtype=np.float64)
    bone_normals = np.asarray(bone_mesh.point_normals, dtype=np.float64)
    nrm = np.linalg.norm(bone_normals, axis=1, keepdims=True)
    bone_normals = bone_normals / np.maximum(nrm, 1e-9)
    directions = bone_normals * max_thickness_mm

    out = np.zeros(n, dtype=np.float32)
    try:
        ipts, iray, _ = art_tri.multi_ray_trace(
            bone_pts, directions, first_point=True, retry=False
        )
        if len(iray) > 0:
            d = np.linalg.norm(ipts - bone_pts[iray], axis=1)
            for ri, di in zip(iray, d):
                if 0 < di < max_thickness_mm:
                    out[ri] = float(di)
    except Exception:
        for i in range(n):
            try:
                pts_i, _ = art_tri.ray_trace(bone_pts[i], bone_pts[i] + directions[i],
                                             first_point=True)
                if len(pts_i) > 0:
                    di = float(np.linalg.norm(np.atleast_2d(pts_i)[0] - bone_pts[i]))
                    if 0 < di < max_thickness_mm:
                        out[i] = di
            except Exception:
                pass
    return out


@register_thickness("template_raycast")
def thickness_template_raycast(bone_mesh, bone_mask, cart_mask, spacing, cfg,
                                template_mesh=None, aligned_pts=None):
    """**Template-frame raycast** — thickness measured on the TEMPLATE's smooth
    bone surface, against the patient's cart mesh transformed into template
    frame. Per-vertex result lives natively on TEMPLATE vertices (no IDW).

    Why this beats patient-frame raycast:
        - The template surface is built by averaging 327 patient meshes; its
          per-vertex normals are smooth and anatomy-correct.
        - The patient's single-instance marching-cubes bone mesh has voxel
          stair-step artefacts even after heavy smoothing → ray directions
          are noisy → striped/lumpy thickness.
        - With template normals, every measurement location has a stable,
          smooth ray direction. The only patient-specific input is the cart
          slab itself.

    Signature note: this strategy needs `template_mesh` + `aligned_pts` from
    the anchor step, which the standard `thickness` strategy contract doesn't
    provide. `pipeline.process_one_patient` special-cases it: detects
    `cfg.thickness_method == "template_raycast"` AFTER the anchor and calls
    this function with the extra args, then skips the IDW remap (since output
    is already on template).

    Returns: (template_n_verts,) array of thickness in mm. NaN where ray
    missed cart entirely (preserves denudation analysis); 0 where ray hit
    < 2 cart intersections.
    """
    if template_mesh is None or aligned_pts is None:
        raise RuntimeError(
            "template_raycast requires template_mesh + aligned_pts kwargs "
            "(set by pipeline.process_one_patient after the anchor step)"
        )

    max_thickness_mm = float(getattr(cfg, "max_thickness_mm", _MAX_THICKNESS_MM_DEFAULT))
    smooth_iters = int(getattr(cfg, "cart_smooth_iters", _CART_SMOOTH_ITERS))
    subch_threshold = float(getattr(cfg, "subch_threshold", 0.5))

    if cart_mask.sum() == 0:
        return np.full(template_mesh.n_points, np.nan, dtype=np.float32)

    # 1) Build cart mesh in patient mm
    cart_surface = _build_cart_surface(cart_mask, spacing, smooth_iters=smooth_iters)
    if cart_surface is None or cart_surface.n_points == 0:
        return np.full(template_mesh.n_points, np.nan, dtype=np.float32)
    cart_tri = cart_surface.triangulate()

    # 2) Fit the patient-bone → template-bone affine (4×3) from
    #    (bone_mesh.points, aligned_pts) pairs. This recovers whichever anchor
    #    was used (aniso_rigid, rigid_only, bounded_affine, etc.) as a single
    #    least-squares affine — exact for the rigid+aniso family and a good
    #    approximation for bounded-affine.
    src = np.asarray(bone_mesh.points, dtype=np.float64)
    dst = np.asarray(aligned_pts, dtype=np.float64)
    A = np.hstack([src, np.ones((len(src), 1))])
    T, *_ = np.linalg.lstsq(A, dst, rcond=None)  # T: (4, 3)
    # Apply to cart vertices
    cart_pts = np.asarray(cart_tri.points, dtype=np.float64)
    cart_pts_tpl = np.hstack([cart_pts, np.ones((len(cart_pts), 1))]) @ T
    cart_tri_tpl = cart_tri.copy()
    cart_tri_tpl.points = cart_pts_tpl.astype(np.float32)

    # 3) Template normals (smooth across the 327-patient average mesh)
    if "Normals" not in template_mesh.point_data:
        template_mesh = template_mesh.copy()
        template_mesh.compute_normals(
            point_normals=True, cell_normals=False, inplace=True,
            auto_orient_normals=True,
        )
    tpl_pts = np.asarray(template_mesh.points, dtype=np.float64)
    tpl_normals = np.asarray(template_mesh.point_normals, dtype=np.float64)
    nrm = np.linalg.norm(tpl_normals, axis=1, keepdims=True)
    tpl_normals = tpl_normals / np.maximum(nrm, 1e-9)

    # Only cast from subch verts (where cart is anatomically expected)
    subch_prob = np.asarray(template_mesh.point_data["subch_prob"])
    subch_mask = subch_prob >= subch_threshold

    out = np.full(template_mesh.n_points, np.nan, dtype=np.float32)
    # Default subch verts to 0 (denuded — ray missed cart)
    out[subch_mask] = 0.0
    # Cast rays only from subch verts
    origins = tpl_pts[subch_mask]
    directions = tpl_normals[subch_mask] * (4.0 * max_thickness_mm)

    try:
        ipts, iray, _ = cart_tri_tpl.multi_ray_trace(
            origins, directions, first_point=False, retry=False
        )
    except Exception as e:
        print(f"  [template_raycast] multi_ray_trace failed ({e}); per-vert fallback")
        # Per-vert fallback
        subch_idx = np.where(subch_mask)[0]
        for j, i in enumerate(subch_idx):
            try:
                pts_i, _ = cart_tri_tpl.ray_trace(
                    origins[j], origins[j] + directions[j], first_point=False
                )
                if len(pts_i) >= 2:
                    d = np.linalg.norm(pts_i - origins[j], axis=1)
                    d.sort()
                    t = float(d[1] - d[0])
                    if 0 < t < max_thickness_mm:
                        out[i] = t
            except Exception:
                pass
        return out

    if len(iray) == 0:
        return out

    # First-slab: for each ray, take first two hits
    d_per_hit = np.linalg.norm(ipts - origins[iray], axis=1)
    order = np.lexsort((d_per_hit, iray))
    iray_s = iray[order]
    d_s = d_per_hit[order]
    diffs = np.diff(iray_s, prepend=iray_s[0] - 1)
    starts = np.where(diffs != 0)[0]
    ends = np.append(starts[1:], len(iray_s))
    subch_idx_arr = np.where(subch_mask)[0]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        first_d = d_s[s]
        exit_d = d_s[s + 1]
        t = float(exit_d - first_d)
        if 0 < t < max_thickness_mm:
            out[subch_idx_arr[iray_s[s]]] = t
    return out


@register_thickness("edt")
def thickness_edt(bone_mesh, bone_mask, cart_mask, spacing, cfg):
    """Legacy "PD-vs-DESS canonical" distance-transform thickness.

    Per-cart-voxel: dt_bone = distance to nearest bone voxel.
    Per-bone-vert: max over (2r+1)³ neighborhood of dt_bone, capped at max.

    Fast (vectorised) but underestimates on thick/oblique cart because it
    samples the LOCAL cart depth without following the surface-normal ray.
    Provided as a fallback / sensitivity analysis. Default is `raycast`.
    """
    max_thickness_mm = float(getattr(cfg, "max_thickness_mm", _MAX_THICKNESS_MM_DEFAULT))
    search_radius_voxels = int(getattr(cfg, "edt_search_radius_voxels", 3))
    n = bone_mesh.n_points
    out = np.zeros(n, dtype=np.float32)
    if cart_mask.sum() == 0 or bone_mask.sum() == 0:
        return out

    bone_bool = bone_mask.astype(bool)
    cart_bool = cart_mask.astype(bool)
    dt_bone = distance_transform_edt(~bone_bool, sampling=spacing)
    thickness_volume = np.where(cart_bool, dt_bone, 0.0).astype(np.float32)
    sr = int(search_radius_voxels)
    nbhd_max = maximum_filter(thickness_volume, size=2 * sr + 1, mode="constant", cval=0.0)

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    verts_voxel = np.round(bone_mesh.points / spacing_arr).astype(int)
    H, W, D = cart_mask.shape
    np.clip(verts_voxel[:, 0], 0, H - 1, out=verts_voxel[:, 0])
    np.clip(verts_voxel[:, 1], 0, W - 1, out=verts_voxel[:, 1])
    np.clip(verts_voxel[:, 2], 0, D - 1, out=verts_voxel[:, 2])
    sampled = nbhd_max[verts_voxel[:, 0], verts_voxel[:, 1], verts_voxel[:, 2]]
    np.minimum(sampled, max_thickness_mm, out=sampled)
    out[:] = sampled.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 2D-per-slice raycast (raycast_2d) — anisotropic-PD canonical (v1.2)
# ---------------------------------------------------------------------------
# PD voxels are anisotropic (~0.357 x 0.357 in-plane, 0.75 mm slice). The 3D
# `raycast` casts along the bone-mesh normal and threads through the slice-axis
# voxel staircase, missing ~70 % of the cart-bearing zone. The 2D in-plane
# raycast works slice-by-slice (axis 2 = the slab/slice axis), staying within
# the fine in-plane resolution: ~75-85 % hit rate, anatomically plausible
# thickness. Per-slice 2D measurements are aggregated onto the 3D bone mesh
# verts by nearest-pixel-in-slice lookup. See cartilage-validation v1.2.

def _smooth_bone_normals_2d(bone2d: np.ndarray, sigma: float):
    """Gaussian-smooth a 2D bone mask, return (soft, boundary, ny, nx) where
    (ny,nx) are unit OUTWARD normals from -grad(soft) and `boundary` is the
    1-pixel ring at the soft 0.5 level set."""
    soft = gaussian_filter(bone2d.astype(np.float32), sigma=sigma)
    gy, gx = np.gradient(soft)
    ny = -gy; nx = -gx
    mag = np.sqrt(ny ** 2 + nx ** 2) + 1e-9
    ny /= mag; nx /= mag
    thr = soft >= 0.5
    boundary = thr & ~binary_erosion(thr, structure=np.ones((3, 3)))
    return soft, boundary, ny, nx


def _raycast_2d_batch(ys, xs, ny, nx, cart_soft, spacing_yx,
                       max_mm=6.0, step_mm=0.05, cart_threshold=0.5):
    """Vectorised 2D in-plane raycast for an array of boundary pixels. Returns
    cart slab thickness (mm) per pixel, 0 where no clean 2-edge crossing."""
    if not len(ys):
        return np.zeros(0, dtype=np.float32)
    n_steps = int(np.ceil(max_mm / step_mm))
    ts = np.arange(1, n_steps + 1) * step_mm
    dy_per_mm = ny / spacing_yx[0]
    dx_per_mm = nx / spacing_yx[1]
    ys_grid = ys[None, :] + ts[:, None] * dy_per_mm[None, :]
    xs_grid = xs[None, :] + ts[:, None] * dx_per_mm[None, :]
    v = map_coordinates(cart_soft, np.stack([ys_grid.ravel(), xs_grid.ravel()]),
                         order=1, mode="constant", cval=0.0).reshape(n_steps, -1)
    inside = v > cart_threshold
    has_any = inside.any(axis=0)
    first = np.argmax(inside, axis=0)
    rng = np.arange(n_steps)[:, None]
    out_seq = (~inside) & (rng >= (first[None, :] + 1))
    has_exit = out_seq.any(axis=0)
    exit_idx = np.argmax(out_seq, axis=0)
    last_in_idx = np.where(has_exit, exit_idx - 1, n_steps - 1)
    entry_mm = ts[first]
    exit_mm = ts[np.clip(last_in_idx, 0, n_steps - 1)]
    thickness = (exit_mm - entry_mm).astype(np.float32)
    thickness[~has_any] = 0.0
    thickness[thickness < 0] = 0.0
    thickness[thickness >= max_mm] = 0.0
    return thickness


@register_thickness("raycast_2d")
def thickness_raycast_2d(bone_mesh, bone_mask, cart_mask, spacing, cfg):
    """2D-per-slice in-plane raycast → per-vertex cart thickness (v1.2 canonical
    for anisotropic PD). Loops axis-2 slices: smooth bone+cart masks, find the
    bone-surface 1-pixel ring near cart, cast a 2D ray along the smoothed
    in-plane outward normal, take first-cart-hit → exit. Aggregate onto bone
    mesh verts by nearest-measured-pixel-in-slice lookup.

    cfg knobs (with defaults): `raycast2d_bone_sigma`=1.8, `raycast2d_cart_sigma`
    =0.8, `raycast2d_near_cart_mm`=4.0, `max_thickness_mm`=6.0,
    `raycast2d_max_query_mm`=1.5.
    """
    bone_sigma = float(getattr(cfg, "raycast2d_bone_sigma", 1.8))
    cart_sigma = float(getattr(cfg, "raycast2d_cart_sigma", 0.8))
    near_cart_mm = float(getattr(cfg, "raycast2d_near_cart_mm", 4.0))
    max_mm = float(getattr(cfg, "max_thickness_mm", _MAX_THICKNESS_MM_DEFAULT))
    max_query_mm = float(getattr(cfg, "raycast2d_max_query_mm", 1.5))

    sp_y, sp_x, sp_z = spacing
    n_slices = bone_mask.shape[2]
    per_slice = [None] * n_slices
    n_hit = n_pts = 0
    for z in range(n_slices):
        b2d = bone_mask[:, :, z].astype(bool)
        c2d = cart_mask[:, :, z].astype(bool)
        if not b2d.any() or not c2d.any():
            continue
        soft, boundary, ny, nx = _smooth_bone_normals_2d(b2d, sigma=bone_sigma)
        cart_soft = gaussian_filter(c2d.astype(np.float32), sigma=cart_sigma)
        ys, xs = np.where(boundary)
        if not len(ys):
            continue
        dt_cart = distance_transform_edt(~c2d, sampling=(sp_y, sp_x))
        keep = dt_cart[ys, xs] <= near_cart_mm
        ys = ys[keep]; xs = xs[keep]
        if not len(ys):
            continue
        thick = _raycast_2d_batch(ys.astype(np.float64), xs.astype(np.float64),
                                   ny[ys, xs], nx[ys, xs], cart_soft,
                                   (sp_y, sp_x), max_mm=max_mm)
        per_slice[z] = (ys, xs, thick)
        n_pts += len(ys); n_hit += int((thick > 0).sum())

    pts = np.asarray(bone_mesh.points, dtype=np.float64)
    out = np.zeros(len(pts), dtype=np.float32)
    z_idx = np.clip(np.round(pts[:, 2] / sp_z).astype(int), 0, n_slices - 1)
    for z in range(n_slices):
        if per_slice[z] is None:
            continue
        m = z_idx == z
        if not m.any():
            continue
        ys_s, xs_s, ths_s = per_slice[z]
        tree = cKDTree(np.stack([ys_s * sp_y, xs_s * sp_x], axis=1))
        d, idx = tree.query(pts[m][:, :2], k=1)
        within = d <= max_query_mm
        if within.any():
            out[np.where(m)[0][within]] = ths_s[idx[within]]
    print(f"    [raycast_2d] {n_hit}/{n_pts} pixel hits "
          f"({100 * n_hit / max(n_pts, 1):.0f}%)")
    return out
