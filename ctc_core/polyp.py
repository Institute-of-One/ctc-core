"""Semi-automatic polyp measurement in a spherical volume of interest.

Ported from ``polyp_voi.py`` and ``shape_index.py`` in the in-house prototype (not distributed).

The workflow is deliberately semi-automatic: a reader supplies a point inside
the lesion and everything after that is deterministic. Detection is out of
scope, so no claim about sensitivity or specificity can be made from this module
-- only about measurement, given a correct location.

Conventions as elsewhere: arrays are ``(z, y, x)``, ``spacing_zyx`` matches, and
physical points are ``(x, y, z)`` in millimetres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PolypConfig",
    "PolypResult",
    "extract_sphere_voi",
    "segment_polyp",
    "max_diameter_mm",
    "boundary_iso_level",
    "shape_index_curvedness",
    "measure_polyp",
]


@dataclass(frozen=True)
class PolypConfig:
    """Thresholds for polyp segmentation and measurement. See docs/PARAMETERS.md."""

    radius_mm: float = 18.0
    soft_hu_min: float = -50.0
    soft_hu_max: float = 180.0
    grow_from_center: bool = True
    closing_radius_mm: float = 1.0
    min_component_mm3: float = 4.0
    # Curvature scale for the shape index, in mm.
    sigma_mm: float = 2.0
    # Measure the diameter on the grayscale iso-surface rather than on the
    # binary mask. Marching cubes at level 0.5 on a binary mask places the
    # surface half a voxel outside the outermost included voxel centre, at both
    # ends, so the diameter is biased high by exactly one voxel -- measured as
    # +1.000 mm at 1 mm isotropic for spheres of radius 3, 5 and 8 mm. At the
    # 6 mm reporting threshold that is a 17 % error. Using partial-volume
    # information instead reduces the bias to 0.03-0.11 mm. See
    # tests/test_polyp.py and docs/PARAMETERS.md.
    diameter_from_grayscale: bool = True


@dataclass
class PolypResult:
    ok: bool
    message: str
    crop_origin_zyx: tuple[int, int, int] = (0, 0, 0)
    mask: np.ndarray | None = None
    measurements: dict[str, Any] = field(default_factory=dict)


def extract_sphere_voi(
    ct_zyx: np.ndarray,
    center_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    """Crop a cube around ``center_zyx`` and the inscribed sphere mask.

    Returns ``(crop, sphere_mask, crop_origin_zyx, center_local_zyx)``. Cropping
    keeps every later operation local, which is what makes the measurement cheap
    enough to be interactive.
    """
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    centre = np.asarray(center_zyx, dtype=np.float64)
    r_vox = np.ceil(radius_mm / sp).astype(int)
    shape = np.asarray(ct_zyx.shape, dtype=int)

    lo = np.maximum(np.floor(centre).astype(int) - r_vox, 0)
    hi = np.minimum(np.floor(centre).astype(int) + r_vox + 1, shape)
    if np.any(hi <= lo):
        raise ValueError("VOI falls outside the volume")

    crop = ct_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local = centre - lo

    zz = (np.arange(crop.shape[0]) - local[0]) * sp[0]
    yy = (np.arange(crop.shape[1]) - local[1]) * sp[1]
    xx = (np.arange(crop.shape[2]) - local[2]) * sp[2]
    dist2 = zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2
    sphere = dist2 <= radius_mm**2

    return crop, sphere, tuple(int(v) for v in lo), tuple(float(v) for v in local)


def _ball(radius_vox: int) -> np.ndarray:
    r = int(radius_vox)
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz**2 + yy**2 + xx**2) <= r * r


def segment_polyp(
    crop: np.ndarray,
    sphere_mask: np.ndarray,
    center_local_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    cfg: PolypConfig,
) -> np.ndarray:
    """Soft-tissue lesion mask inside the VOI.

    Soft tissue within the sphere, reduced to the component containing (or
    nearest to) the supplied point, closed to fill pits, then de-speckled. The
    HU window excludes both lumen air below and tagged fluid or bone above.
    """
    from scipy import ndimage

    soft = (crop >= cfg.soft_hu_min) & (crop <= cfg.soft_hu_max) & sphere_mask
    if not soft.any():
        return np.zeros_like(soft, dtype=bool)

    if cfg.grow_from_center:
        labels, n = ndimage.label(soft)
        if n >= 1:
            c = [
                int(np.clip(round(v), 0, s - 1))
                for v, s in zip(center_local_zyx, soft.shape, strict=True)
            ]
            target = int(labels[c[0], c[1], c[2]])
            if target == 0:
                # The point landed off tissue: take the nearest labelled voxel
                # rather than failing, since a reader clicks near, not on.
                idx = np.argwhere(labels > 0)
                d = ((idx - np.array(c)) ** 2).sum(axis=1)
                nz, ny, nx = idx[int(np.argmin(d))]
                target = int(labels[nz, ny, nx])
            soft = labels == target

    sp = np.asarray(spacing_zyx, dtype=np.float64)
    r_vox = max(0, int(round(cfg.closing_radius_mm / max(sp.min(), 1e-6))))
    if r_vox >= 1 and soft.any():
        soft = ndimage.binary_closing(soft, structure=_ball(r_vox))
        soft &= sphere_mask  # closing can spill past the VOI

    vox_mm3 = float(np.prod(sp))
    min_vox = max(1, int(round(cfg.min_component_mm3 / max(vox_mm3, 1e-9))))
    if soft.any():
        labels, n = ndimage.label(soft)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            keep = [i for i in range(1, n + 1) if sizes[i] >= min_vox]
            soft = np.isin(labels, keep) if keep else (labels == int(np.argmax(sizes)))
    return np.asarray(soft, dtype=bool)


def _mesh_volume_mm3(verts_mm: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by a closed triangular surface, by the divergence theorem.

    Used instead of counting mask voxels because a thresholded voxel count is
    biased by where the threshold sits on the partial-volume ramp. The lesion
    window's lower bound of -50 HU sits about 90 % of the way up a lumen-to-soft-
    tissue ramp, so voxel counting *under*-reads volume -- 81 against a true
    113 mm^3 for a 3 mm-radius sphere. Taking volume from the same iso-surface
    that gives the diameter and the area also makes sphericity self-consistent.
    """
    v = verts_mm[faces]
    return float(abs(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)


def boundary_iso_level(
    crop: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    """Iso-level midway between the lesion and what immediately surrounds it.

    Estimated per lesion rather than fixed, because a polyp may border lumen
    air (about -1000 HU) or tagged fluid (well above 0), and the midpoint that
    locates the true boundary differs accordingly.
    """
    from scipy import ndimage

    inner = ndimage.binary_erosion(mask, structure=_ball(1))
    if not inner.any():
        inner = mask
    shell = ndimage.binary_dilation(mask, structure=_ball(2)) & ~mask
    if not shell.any():
        return None
    return float((np.median(crop[inner]) + np.median(crop[shell])) / 2.0)


def max_diameter_mm(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    crop: np.ndarray | None = None,
    iso_level: float | None = None,
) -> tuple[float, np.ndarray | None, np.ndarray | None, float, float]:
    """Largest distance between surface points, plus surface area and volume.

    Returns ``(diameter_mm, p1_zyx, p2_zyx, surface_area_mm2, mesh_volume_mm3)``.

    With ``crop`` and ``iso_level`` supplied the surface is the grayscale
    iso-surface, which uses partial-volume information and is essentially
    unbiased. Without them it is marching cubes on the binary mask, which is
    what the reference implementation did and which **overestimates the
    diameter by one voxel**: the level-0.5 surface sits half a voxel outside the
    outermost included voxel centre at each end. Measured on spheres of radius
    3, 5 and 8 mm at 1 mm isotropic, the binary route gives 7.000, 11.000 and
    17.000 mm against true 6, 10 and 16, while the grayscale route gives 6.029,
    10.074 and 16.107.

    The convex hull reduces the pair search to hull vertices; the answer is
    unchanged, because the most distant pair of any point set lies on its hull.
    """
    if mask is None or int(mask.sum()) < 2:
        return 0.0, None, None, 0.0, 0.0

    sp = np.asarray(spacing_zyx, dtype=np.float64)
    verts_vox: np.ndarray | None = None
    surface_area = 0.0
    mesh_volume = 0.0
    try:
        from skimage.measure import marching_cubes, mesh_surface_area

        if crop is not None and iso_level is not None:
            from scipy import ndimage

            # Restrict to the lesion's neighbourhood so an adjacent structure
            # cannot contribute vertices to the hull.
            near = ndimage.binary_dilation(mask, structure=_ball(3))
            far_value = iso_level - abs(iso_level) - 1000.0
            field = np.where(near, crop.astype(np.float32), far_value)
            padded = np.pad(field, 1, mode="constant", constant_values=far_value)
            verts, faces, _n, _v = marching_cubes(padded, level=float(iso_level))
        else:
            # Pad so a mask touching the crop border still yields a closed surface.
            padded = np.pad(mask.astype(np.float32), 1, mode="constant")
            verts, faces, _n, _v = marching_cubes(padded, level=0.5)
        verts = verts - 1.0
        verts_vox = verts
        surface_area = float(mesh_surface_area(verts * sp, faces))
        mesh_volume = _mesh_volume_mm3(verts * sp, faces)
    except Exception as exc:  # pragma: no cover - degenerate masks
        LOGGER.debug("marching cubes failed (%s); using voxel coordinates", exc)
        verts_vox = None

    if verts_vox is None or len(verts_vox) < 2:
        verts_vox = np.argwhere(mask).astype(np.float64)
        if len(verts_vox) < 2:
            return 0.0, None, None, surface_area, mesh_volume

    pts_mm = verts_vox * sp
    hull_mm, hull_vox = pts_mm, verts_vox
    if len(pts_mm) > 8:
        try:
            from scipy.spatial import ConvexHull

            sel = ConvexHull(pts_mm).vertices
            hull_mm, hull_vox = pts_mm[sel], verts_vox[sel]
        except Exception:  # pragma: no cover - coplanar point sets
            pass

    d2 = ((hull_mm[:, None, :] - hull_mm[None, :, :]) ** 2).sum(axis=-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return float(np.sqrt(d2[i, j])), hull_vox[i], hull_vox[j], surface_area, mesh_volume


def shape_index_curvedness(
    crop: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    sigma_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shape index, curvedness (1/mm) and gradient magnitude (HU/mm).

    Curvatures of the iso-intensity surface, from Gaussian derivatives divided
    by the per-axis spacing so the result is in physical units. Sign convention:
    a bright blob in dark surroundings -- soft tissue in an insufflated colon --
    has a negative-definite Hessian, so the mean curvature is negated to make a
    cap read as shape index +1, matching Koenderink's convention as used for
    polyp detection.

    Canonical values: cap +1, ridge +0.5, saddle 0, rut -0.5, cup -1.
    """
    from scipy.ndimage import gaussian_filter

    f = np.asarray(crop, dtype=np.float64)
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    sig = [max(sigma_mm / s, 1e-3) for s in sp]

    def deriv(order: tuple[int, int, int]) -> np.ndarray:
        g = gaussian_filter(f, sigma=sig, order=order, mode="nearest")
        return g / (sp[0] ** order[0] * sp[1] ** order[1] * sp[2] ** order[2])

    fz, fy, fx = deriv((1, 0, 0)), deriv((0, 1, 0)), deriv((0, 0, 1))
    fzz, fyy, fxx = deriv((2, 0, 0)), deriv((0, 2, 0)), deriv((0, 0, 2))
    fyz, fxz, fxy = deriv((1, 1, 0)), deriv((1, 0, 1)), deriv((0, 1, 1))

    eps = 1e-12
    g2 = fx * fx + fy * fy + fz * fz
    gmag = np.sqrt(g2)
    trH = fxx + fyy + fzz
    gHg = (
        fx * fx * fxx + fy * fy * fyy + fz * fz * fzz
        + 2.0 * (fx * fy * fxy + fx * fz * fxz + fy * fz * fyz)
    )
    k_mean = (g2 * trH - gHg) / (2.0 * gmag**3 + eps)

    adj = (
        fx * fx * (fyy * fzz - fyz * fyz)
        + fy * fy * (fxx * fzz - fxz * fxz)
        + fz * fz * (fxx * fyy - fxy * fxy)
        + 2.0
        * (
            fx * fy * (fxz * fyz - fxy * fzz)
            + fy * fz * (fxy * fxz - fyz * fxx)
            + fx * fz * (fxy * fyz - fxz * fyy)
        )
    )
    k_gauss = adj / (g2 * g2 + eps)

    s = np.sqrt(np.clip(k_mean * k_mean - k_gauss, 0.0, None))
    k1, k2 = k_mean + s, k_mean - s

    shape_index = (2.0 / np.pi) * np.arctan2(-k_mean, s)
    curvedness = np.sqrt((k1 * k1 + k2 * k2) / 2.0)

    # Curvature is undefined where the gradient vanishes.
    flat = gmag < 1e-6
    shape_index[flat] = 0.0
    curvedness[flat] = 0.0
    return shape_index, curvedness, gmag


def measure_polyp(
    ct_zyx: np.ndarray,
    center_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    cfg: PolypConfig | None = None,
) -> PolypResult:
    """Segment and measure one polyp from a point inside it.

    Never raises: a failure is reported through ``ok`` so one bad lesion cannot
    abort a batch.
    """
    cfg = cfg or PolypConfig()
    try:
        crop, sphere, origin, local = extract_sphere_voi(
            ct_zyx, center_zyx, spacing_zyx, cfg.radius_mm
        )
        mask = segment_polyp(crop, sphere, local, spacing_zyx, cfg)
        if not mask.any():
            return PolypResult(False, "no soft tissue in the VOI", origin, mask)

        sp = np.asarray(spacing_zyx, dtype=np.float64)
        vox_mm3 = float(np.prod(sp))
        volume_mm3 = float(mask.sum()) * vox_mm3

        level = boundary_iso_level(crop, mask) if cfg.diameter_from_grayscale else None
        diameter, p1, p2, area, mesh_volume = max_diameter_mm(
            mask, spacing_zyx, crop=crop if level is not None else None, iso_level=level
        )
        # Prefer the iso-surface volume: it refers to the same surface as the
        # area and the diameter, and it is not biased by where the segmentation
        # threshold sits on the partial-volume ramp.
        if mesh_volume > 0:
            volume_mm3 = mesh_volume

        # Sphericity: 1 for a sphere, lower for anything else.
        sphericity = (
            float((np.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / area)
            if area > 0
            else None
        )

        si, cv, _ = shape_index_curvedness(crop, spacing_zyx, cfg.sigma_mm)
        surface = mask & ~_erode_once(mask)

        meas: dict[str, Any] = {
            "volume_mm3": round(volume_mm3, 2),
            "volume_voxel_count_mm3": round(float(mask.sum()) * vox_mm3, 2),
            "voxel_count": int(mask.sum()),
            "max_diameter_mm": round(diameter, 3),
            "surface_area_mm2": round(area, 2),
            "sphericity": round(sphericity, 4) if sphericity is not None else None,
            "equivalent_diameter_mm": round(
                float(2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1 / 3)), 3
            ),
            "shape_index_median": round(float(np.median(si[surface])), 4)
            if surface.any()
            else None,
            "curvedness_median_per_mm": round(float(np.median(cv[surface])), 5)
            if surface.any()
            else None,
            "diameter_iso_level_hu": round(level, 1) if level is not None else None,
            "endpoint1_zyx": None if p1 is None else [float(v) for v in p1],
            "endpoint2_zyx": None if p2 is None else [float(v) for v in p2],
        }
        return PolypResult(True, "ok", origin, mask, meas)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("polyp measurement failed")
        return PolypResult(False, f"{type(exc).__name__}: {exc}")


def _erode_once(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_erosion(mask, structure=_ball(1))
