"""Pericolonic fat quantified along the centerline.

Two representations, ported from the in-house prototype (not distributed):

``fat_profile_spherical``
    A spherical ROI centred on each resampled centerline point, excluding lumen.
    This is what the 2026-08 batch measured (``phase0_fat_along_centerline.py``)
    and it is the one used for the per-patient indices, so it is validated
    against that batch's recorded values.

``fat_map_polar``
    The ``(s, theta)`` map from ``vgp_fat_heatmap.py``: at each arc-length
    station, rays are cast outward in a frame carried along the centerline, the
    colon wall is found along each ray, and a ring beyond the wall is sampled.
    This resolves fat *by direction*, which the spherical ROI averages away, and
    it is what supports a left/right asymmetry index.

Conventions, as in :mod:`ctc_core.quality`: arrays are ``(z, y, x)``,
``spacing_zyx`` matches, and physical points are ``(x, y, z)`` in millimetres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FatConfig",
    "resample_centerline",
    "fat_profile_spherical",
    "fat_map_polar",
    "summarise_fat_profile",
    "summarise_fat_map",
    "smooth_centerline",
]


@dataclass(frozen=True)
class FatConfig:
    """Fat-map parameters. Registered in docs/PARAMETERS.md."""

    # Adipose window. Conventional in body-composition CT.
    fat_hu_min: float = -190.0
    fat_hu_max: float = -30.0

    # Spherical-ROI variant (the 2026-08 batch settings).
    step_mm: float = 5.0
    roi_radius_mm: float = 15.0
    min_fat_voxels: int = 50

    # Polar (s, theta) variant: the ring 5-15 mm beyond the detected wall.
    # This is the primary pericolonic measurement since 2026-09-12: a sphere
    # centred on the centerline cannot reach past the wall where the colon is
    # wide (38 % of stations in the illustrated series held almost no tissue),
    # so the spherical profile samples narrow segments only.
    n_theta: int = 180
    sample_step_mm: float = 0.5
    # Long enough for the wall plus the full ring in a distended colon: at
    # 35 mm the ring was truncated on the quarter of rays whose wall lies
    # beyond 20 mm.
    max_ray_mm: float = 60.0
    skin_skip_mm: float = 1.0
    ring_inner_mm: float = 5.0
    ring_outer_mm: float = 15.0
    # Rays within this angle of the patient's left-right axis count as left or
    # right for the asymmetry index; the rest (anterior, posterior, cranial,
    # caudal) are ignored by it.
    lateral_cone_deg: float = 45.0
    # Gaussian smoothing of the centerline (mm) before the ring frames are
    # computed, as the VGP unfold does: voxel-scale kinks otherwise flip the
    # tangent between stations and the frame rotates with it. Endpoints fixed.
    frame_smooth_sigma_mm: float = 2.0
    # Minimum fat samples in a station's ring for the station to count.
    min_ring_fat_samples: int = 50


def smooth_centerline(points_xyz: np.ndarray, sigma_mm: float) -> np.ndarray:
    """Gaussian-smooth a centerline in physical space, endpoints fixed.

    The same operation as the VGP unfold applies before computing its camera
    frames. ``sigma_mm`` is converted to samples with the mean point spacing.
    """
    p = np.asarray(points_xyz, dtype=np.float64)
    if sigma_mm <= 0 or p.shape[0] < 3:
        return p
    from scipy.ndimage import gaussian_filter1d

    step = float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
    if step <= 0:
        return p
    out = np.column_stack([gaussian_filter1d(p[:, k], sigma=sigma_mm / step, mode="nearest")
                           for k in range(3)])
    out[0], out[-1] = p[0], p[-1]
    return out


def resample_centerline(
    points_xyz: np.ndarray,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length resampling of a centerline. Returns ``(s_mm, xyz_mm)``.

    Sampling by arc length rather than by stored point index makes every
    per-station statistic comparable between series, whatever their point
    density.
    """
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.shape[0] < 2:
        raise ValueError("centerline needs at least two points")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_native = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s_native[-1])
    if total <= 0.0:
        raise ValueError("centerline total length must be positive")

    s = np.arange(0.0, total, step_mm, dtype=np.float64)
    if s.size == 0 or s[-1] < total:
        s = np.concatenate((s, [total]))
    xyz = np.column_stack([np.interp(s, s_native, p[:, d]) for d in range(3)])
    return s, xyz


# ---------------------------------------------------------------------------
# Spherical ROI
# ---------------------------------------------------------------------------


def fat_profile_spherical(
    ct: sitk.Image,
    ct_arr: np.ndarray,
    body: np.ndarray,
    air: np.ndarray,
    s_mm: np.ndarray,
    xyz_mm: np.ndarray,
    cfg: FatConfig | None = None,
) -> list[dict[str, Any]]:
    """Fat statistics in a sphere at each station, excluding lumen.

    The ROI is ``sphere AND body AND NOT air``, so it is pericolonic tissue
    rather than colon content. A station with fewer than ``min_fat_voxels`` fat
    voxels reports NaN rather than a noisy value; the reason is recorded so the
    two causes (small pockets vs. centerline outside fat-bearing tissue) stay
    distinguishable.
    """
    cfg = cfg or FatConfig()
    spacing_xyz = np.array(ct.GetSpacing(), dtype=np.float64)
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])
    radius_vox = np.ceil(cfg.roi_radius_mm / spacing_zyx).astype(int)
    shape = np.array(ct_arr.shape, dtype=int)
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0

    rows: list[dict[str, Any]] = []
    for s_val, xyz in zip(s_mm, xyz_mm, strict=True):
        i_f, j_f, k_f = ct.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in xyz)
        )
        centre = np.array([k_f, j_f, i_f], dtype=np.float64)
        lo = np.maximum(np.rint(centre).astype(int) - radius_vox, 0)
        hi = np.minimum(np.rint(centre).astype(int) + radius_vox + 1, shape)
        z0, y0, x0 = (int(v) for v in lo)
        z1, y1, x1 = (int(v) for v in hi)

        zz = (np.arange(z0, z1, dtype=np.float64) - centre[0]) * spacing_zyx[0]
        yy = (np.arange(y0, y1, dtype=np.float64) - centre[1]) * spacing_zyx[1]
        xx = (np.arange(x0, x1, dtype=np.float64) - centre[2]) * spacing_zyx[2]
        dist2 = zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2
        sphere = dist2 <= cfg.roi_radius_mm**2

        candidate = sphere & body[z0:z1, y0:y1, x0:x1] & ~air[z0:z1, y0:y1, x0:x1]
        roi_voxels = int(candidate.sum())
        values = ct_arr[z0:z1, y0:y1, x0:x1][candidate] if roi_voxels else np.empty(0)
        fat = values[(values >= cfg.fat_hu_min) & (values <= cfg.fat_hu_max)]

        row: dict[str, Any] = {
            "s_mm": float(s_val),
            "x_mm": float(xyz[0]),
            "y_mm": float(xyz[1]),
            "z_mm": float(xyz[2]),
            "roi_voxels": roi_voxels,
            "fat_voxels": int(fat.size),
        }
        if fat.size < cfg.min_fat_voxels:
            row.update(
                fat_mean_hu=np.nan,
                fat_median_hu=np.nan,
                fat_fraction=np.nan,
                fat_volume_ml=np.nan,
                invalid_reason="zero-fat" if fat.size == 0 else "below-threshold",
            )
        else:
            row.update(
                fat_mean_hu=float(fat.mean()),
                fat_median_hu=float(np.median(fat)),
                fat_fraction=float(fat.size / max(roi_voxels, 1)),
                fat_volume_ml=float(fat.size * vox_ml),
                invalid_reason="",
            )
        rows.append(row)
    return rows


def summarise_fat_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-series summary of a fat profile, matching the batch's fields."""
    n = len(rows)
    valid = [r for r in rows if not np.isnan(r["fat_mean_hu"])]
    out: dict[str, Any] = {
        "n_points": n,
        "n_valid": len(valid),
        "valid_fraction": (len(valid) / n) if n else 0.0,
        "n_below_threshold": sum(1 for r in rows if r.get("invalid_reason") == "below-threshold"),
        "n_zero_fat": sum(1 for r in rows if r.get("invalid_reason") == "zero-fat"),
    }
    if valid:
        out["fat_mean_hu_median"] = float(np.median([r["fat_mean_hu"] for r in valid]))
        out["fat_fraction_mean"] = float(np.mean([r["fat_fraction"] for r in valid]))
        out["fat_volume_ml_sum"] = float(np.sum([r["fat_volume_ml"] for r in valid]))
    else:
        out["fat_mean_hu_median"] = None
        out["fat_fraction_mean"] = None
        out["fat_volume_ml_sum"] = None
    return out


# ---------------------------------------------------------------------------
# Polar (s, theta) map
# ---------------------------------------------------------------------------


def centerline_frames(xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame at each station: ``(t_hat, n_hat, b_hat)``.

    ``n_hat`` is the anterior direction ``(0, -1, 0)`` projected perpendicular to
    the tangent, so ``theta = 0`` points anteriorly in every series and position.
    The sense of rotation follows the tangent, however, so left and right must be
    read from the ray directions (:func:`polar_asymmetry`), not from ``theta``.
    Where the tangent is nearly anterior, the reference axis falls back to
    ``(0, 0, 1)``.
    """
    p = np.asarray(xyz_mm, dtype=np.float64)
    t = np.gradient(p, axis=0)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.divide(t, norm, out=np.zeros_like(t), where=norm > 1e-9)

    ref = np.tile(np.array([0.0, -1.0, 0.0]), (len(p), 1))
    degenerate = np.abs((t * ref).sum(axis=1)) > 0.99
    ref[degenerate] = np.array([0.0, 0.0, 1.0])

    n = ref - (ref * t).sum(axis=1, keepdims=True) * t
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, n_norm, out=np.zeros_like(n), where=n_norm > 1e-9)
    b = np.cross(t, n)
    return t, n, b


def fat_map_polar(
    ct: sitk.Image,
    ct_arr: np.ndarray,
    body: np.ndarray,
    air: np.ndarray,
    s_mm: np.ndarray,
    xyz_mm: np.ndarray,
    cfg: FatConfig | None = None,
) -> dict[str, np.ndarray]:
    """Fat fraction and mean HU on the ``(s, theta)`` grid.

    Along each ray the first non-lumen sample beyond ``skin_skip_mm`` is taken as
    the colon wall, and the ring ``[wall + ring_inner, wall + ring_outer]``
    restricted to body and non-lumen is the pericolonic sample. Anchoring the
    ring to the wall rather than to the centerline is what makes stations of
    different calibre comparable.

    Returns arrays shaped ``(len(s_mm), n_theta)`` plus the theta axis.
    """
    cfg = cfg or FatConfig()
    _, n_hat, b_hat = centerline_frames(xyz_mm)

    theta = np.linspace(0.0, 2.0 * np.pi, cfg.n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    n_steps = int(np.ceil(cfg.max_ray_mm / cfg.sample_step_mm)) + 1
    dist = np.arange(n_steps, dtype=np.float64) * cfg.sample_step_mm
    skin_steps = int(np.floor(cfg.skin_skip_mm / cfg.sample_step_mm))

    spacing_xyz = np.array(ct.GetSpacing(), dtype=np.float64)
    origin = np.array(ct.GetOrigin(), dtype=np.float64)
    shape = ct_arr.shape

    fat_fraction = np.full((len(s_mm), cfg.n_theta), np.nan)
    fat_mean_hu = np.full((len(s_mm), cfg.n_theta), np.nan)
    wall_mm = np.full((len(s_mm), cfg.n_theta), np.nan)
    # Per-ray sample counts, so stations can be summarised by pooling rays.
    ring_samples = np.zeros((len(s_mm), cfg.n_theta), dtype=np.int32)
    fat_samples = np.zeros((len(s_mm), cfg.n_theta), dtype=np.int32)
    fat_hu_sum = np.zeros((len(s_mm), cfg.n_theta), dtype=np.float64)
    # Patient-frame x component of each ray: +1 points to the patient's left
    # (LPS). Which half of theta faces left depends on the direction of travel,
    # so anything left/right must be read from this, never from theta.
    lateral = np.full((len(s_mm), cfg.n_theta), np.nan)

    for si in range(len(s_mm)):
        # (n_theta, 3) ray directions in physical space
        r_hat = cos_t[:, None] * n_hat[si][None, :] + sin_t[:, None] * b_hat[si][None, :]
        lateral[si] = r_hat[:, 0]
        # (n_theta, n_steps, 3) sample points
        pts = xyz_mm[si][None, None, :] + dist[None, :, None] * r_hat[:, None, :]

        idx = np.rint((pts - origin) / spacing_xyz).astype(int)  # (x, y, z)
        ok = (
            (idx[..., 0] >= 0) & (idx[..., 0] < shape[2])
            & (idx[..., 1] >= 0) & (idx[..., 1] < shape[1])
            & (idx[..., 2] >= 0) & (idx[..., 2] < shape[0])
        )
        zi = np.clip(idx[..., 2], 0, shape[0] - 1)
        yi = np.clip(idx[..., 1], 0, shape[1] - 1)
        xi = np.clip(idx[..., 0], 0, shape[2] - 1)

        hu = np.where(ok, ct_arr[zi, yi, xi], np.nan)
        in_air = np.where(ok, air[zi, yi, xi], False)
        in_body = np.where(ok, body[zi, yi, xi], False)

        # Wall: first non-lumen sample beyond the skin skip.
        beyond = np.zeros_like(in_air, dtype=bool)
        beyond[:, skin_steps:] = True
        candidate = (~in_air) & beyond & ok
        has_wall = candidate.any(axis=1)
        first = np.argmax(candidate, axis=1)
        wall_dist = np.where(has_wall, dist[first], np.nan)
        wall_mm[si] = wall_dist

        ring = (
            (dist[None, :] >= (wall_dist[:, None] + cfg.ring_inner_mm))
            & (dist[None, :] <= (wall_dist[:, None] + cfg.ring_outer_mm))
            & in_body
            & (~in_air)
            & ok
        )
        counts = ring.sum(axis=1)
        is_fat = ring & (hu >= cfg.fat_hu_min) & (hu <= cfg.fat_hu_max)
        fat_counts = is_fat.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(counts > 0, fat_counts / np.maximum(counts, 1), np.nan)
            mean_hu = np.where(
                fat_counts > 0,
                np.nansum(np.where(is_fat, hu, 0.0), axis=1) / np.maximum(fat_counts, 1),
                np.nan,
            )
        fat_fraction[si] = frac
        fat_mean_hu[si] = mean_hu
        ring_samples[si] = counts
        fat_samples[si] = fat_counts
        fat_hu_sum[si] = np.nansum(np.where(is_fat, hu, 0.0), axis=1)

    return {
        "s_mm": np.asarray(s_mm, dtype=np.float64),
        "theta": theta,
        "fat_fraction": fat_fraction,
        "fat_mean_hu": fat_mean_hu,
        "wall_mm": wall_mm,
        "lateral": lateral,
        "ring_samples": ring_samples,
        "fat_samples": fat_samples,
        "fat_hu_sum": fat_hu_sum,
    }


def summarise_fat_map(fat_map: dict[str, np.ndarray], cfg: FatConfig | None = None) -> dict:
    """Per-series ring indices, pooling the rays of each station.

    A station counts when its ring holds at least ``min_ring_fat_samples`` fat
    samples. Fat attenuation is the median over stations of the pooled mean fat
    HU; fat fraction the mean over stations of pooled fat samples over ring
    samples. Both refer to tissue 5-15 mm beyond the colon wall at every
    station, whatever the lumen calibre.
    """
    cfg = cfg or FatConfig()
    ring = fat_map["ring_samples"].sum(axis=1).astype(np.float64)
    fat = fat_map["fat_samples"].sum(axis=1).astype(np.float64)
    hu = fat_map["fat_hu_sum"].sum(axis=1)
    valid = fat >= cfg.min_ring_fat_samples
    n = int(len(ring))
    out: dict[str, Any] = {"ring_n_stations": n, "ring_n_valid": int(valid.sum()),
                           "ring_valid_fraction": float(valid.mean()) if n else 0.0}
    if valid.any():
        out["ring_fat_mean_hu_median"] = float(np.median(hu[valid] / fat[valid]))
        out["ring_fat_fraction_mean"] = float(np.mean(fat[valid] / ring[valid]))
    else:
        out["ring_fat_mean_hu_median"] = None
        out["ring_fat_fraction_mean"] = None
    return out


def polar_asymmetry(
    fat_map: dict[str, np.ndarray], cfg: FatConfig | None = None
) -> dict[str, Any]:
    """Left/right asymmetry of the fat fraction, in the patient frame.

    A ray counts as *left* when it points within ``lateral_cone_deg`` of the
    patient's left (+x in LPS) and as *right* within the same angle of the
    patient's right. Returns ``(left - right) / (left + right)``: positive means
    more pericolonic fat on the patient's left, 0 a symmetric distribution, and
    the value is bounded by +/-1.

    The two halves of ``theta`` do **not** give this. ``theta`` is anchored
    anteriorly, but its sense of rotation follows the tangent, so the half that
    faces the patient's left swaps between a segment traversed cranially and one
    traversed caudally, and with the arbitrary choice of which seed is the start.
    Segments running left-right (the transverse colon) have no left or right side
    and contribute almost no rays.

    The original specification's index, which splits at ``theta = pi``
    (``theta`` in ``[0, pi)`` against ``[pi, 2 pi)``), is also returned, as
    ``fat_asymmetry_theta_halves``, so the two definitions can be compared. It
    is a side-of-travel index, not a left/right one.
    """
    cfg = cfg or FatConfig()
    frac = fat_map["fat_fraction"]
    lateral = fat_map["lateral"]
    c = float(np.cos(np.deg2rad(cfg.lateral_cone_deg)))
    with np.errstate(invalid="ignore"):
        left_rays, right_rays = lateral >= c, lateral <= -c
    left = np.nanmean(frac[left_rays]) if np.isfinite(frac[left_rays]).any() else np.nan
    right = np.nanmean(frac[right_rays]) if np.isfinite(frac[right_rays]).any() else np.nan
    total = left + right

    theta = fat_map["theta"]
    h0 = np.nanmean(frac[:, theta < np.pi]) if np.isfinite(frac[:, theta < np.pi]).any() \
        else np.nan
    h1 = np.nanmean(frac[:, theta >= np.pi]) if np.isfinite(frac[:, theta >= np.pi]).any() \
        else np.nan
    h_total = h0 + h1
    return {
        "fat_fraction_left": float(left) if np.isfinite(left) else None,
        "fat_fraction_right": float(right) if np.isfinite(right) else None,
        "fat_asymmetry": (
            float((left - right) / total) if np.isfinite(total) and total > 0 else None
        ),
        "fat_asymmetry_theta_halves": (
            float((h0 - h1) / h_total) if np.isfinite(h_total) and h_total > 0 else None
        ),
    }
