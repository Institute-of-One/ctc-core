"""Preparation and distension quality indices derived from the centerline.

Ported from ``ctc_analytics.py`` in the in-house prototype (not distributed).

**Every threshold here is heuristic and uncalibrated**, and the manuscript must
say so. They define the indices; they are not validated cut-points. The one
inherited band set that is demonstrably wrong -- the CTI interpretation bands --
is disabled by default rather than reproduced; see :class:`QualityConfig`.

Two conventions matter and are easy to get wrong, so both are stated once here
and tested:

* Array axes are ``(z, y, x)``; ``spacing_zyx`` follows the same order, while
  SimpleITK's ``GetSpacing()`` is ``(x, y, z)``.
* ``centerline_ijk`` follows the columns of ``centerline_points.csv``, i.e.
  ``i`` indexes x, ``j`` indexes y and ``k`` indexes z. Indexing an array with
  it therefore requires ``arr[k, j, i]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)

__all__ = [
    "QualityConfig",
    "QualityResult",
    "smooth_path",
    "path_length_mm",
    "straight_line_mm",
    "gas_volume_ml",
    "collapse_ratio_pct",
    "distension_score",
    "compute_quality",
]


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the quality indices. All registered in docs/PARAMETERS.md."""

    air_hu_max: float = -700.0
    collapse_radius_mm: float = 4.0
    restrict_gas_to_centerline_components: bool = True
    connectivity: int = 6
    # Light de-staircasing before length integration. Applied to a local copy
    # only; the stored centerline is never modified.
    smooth_sigma_points: float = 1.5

    # Predicted *anatomical* colon length for PLRI, cm. Mid-point of the
    # 150-200 cm range. Note the unit mismatch this creates: PLRI divides our
    # traced centerline length by an anatomical prediction, and the two are not
    # the same quantity (see docs/EVALUATION_PRONE_SUPINE.md section 2.3). PLRI
    # is therefore a descriptor of traced extent relative to a nominal colon,
    # not a redundancy measure, and is reported as such.
    predicted_colon_length_cm: float = 160.0

    # Distension score ramps.
    lag_poor_ml_per_cm: float = 2.0
    lag_excellent_ml_per_cm: float = 8.0
    gas_poor_ml: float = 500.0
    gas_good_ml: float = 1500.0
    collapse_zero_pct: float = 0.0
    collapse_full_pct: float = 40.0
    w_collapse: float = 0.45
    w_lag: float = 0.40
    w_gas: float = 0.15

    # Grade cut-points for the composite score.
    grade_bands: tuple[tuple[float, str], ...] = (
        (80.0, "Excellent"),
        (60.0, "Good"),
        (40.0, "Fair"),
    )

    # Interpretation bands for the tortuosity index, as (upper bound, label).
    # Empty by default and deliberately so: the inherited bands were
    # (12, "Low"), (18, "Moderate"), else "High", which cannot fire on a lumen
    # geodesic. Measured on this cohort the index runs 2.3-4.5, so every series
    # would be labelled "Low" and the band would carry no information. Supply
    # calibrated bands here before emitting a label.
    cti_bands: tuple[tuple[float, str], ...] = ()


@dataclass
class QualityResult:
    ok: bool
    message: str
    indices: dict[str, Any] = field(default_factory=dict)


def _clip_0_100(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def smooth_path(xyz_mm: np.ndarray, sigma_points: float) -> np.ndarray:
    """Gaussian-smooth the centerline coordinates, keeping the endpoints fixed.

    Voxel stepping inflates a summed length by several percent; smoothing before
    integration removes that without moving the anatomical ends.
    """
    p = np.asarray(xyz_mm, dtype=np.float64)
    if sigma_points <= 0 or p.shape[0] < 5:
        return p
    from scipy.ndimage import gaussian_filter1d

    out = np.empty_like(p)
    for c in range(p.shape[1]):
        out[:, c] = gaussian_filter1d(p[:, c], sigma=sigma_points, mode="nearest")
    out[0] = p[0]
    out[-1] = p[-1]
    return out


def path_length_mm(xyz_mm: np.ndarray) -> float:
    p = np.asarray(xyz_mm, dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def straight_line_mm(xyz_mm: np.ndarray) -> float:
    p = np.asarray(xyz_mm, dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]))


def _sample_at_centerline(
    arr: np.ndarray,
    centerline_ijk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Values of ``arr`` at the centerline voxels, plus an in-bounds mask.

    ``centerline_ijk`` is ``(i, j, k)`` = ``(x, y, z)``; ``arr`` is ``(z, y, x)``.
    """
    ijk = np.asarray(centerline_ijk, dtype=int)
    if ijk.size == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    nz, ny, nx = arr.shape
    k, j, i = ijk[:, 2], ijk[:, 1], ijk[:, 0]
    inside = (k >= 0) & (k < nz) & (j >= 0) & (j < ny) & (i >= 0) & (i < nx)
    out = np.zeros(len(ijk), dtype=arr.dtype if arr.dtype != bool else np.uint8)
    if inside.any():
        out[inside] = arr[k[inside], j[inside], i[inside]]
    return out, inside


def gas_volume_ml(
    ct_zyx: np.ndarray,
    lumen_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    centerline_ijk: np.ndarray | None,
    cfg: QualityConfig,
) -> tuple[float, np.ndarray]:
    """Gas volume in mL, and the gas mask.

    Gas is lumen below the air threshold. Restricting it to the components the
    centerline actually passes through is what keeps lung and stomach air out of
    a colonic measurement.
    """
    gas = (np.asarray(lumen_zyx) > 0) & (np.asarray(ct_zyx) <= cfg.air_hu_max)

    if cfg.restrict_gas_to_centerline_components and centerline_ijk is not None and len(
        centerline_ijk
    ):
        import cc3d

        labels, _ = cc3d.connected_components(
            gas.astype(np.uint8), connectivity=cfg.connectivity, return_N=True
        )
        at_path, inside = _sample_at_centerline(labels, centerline_ijk)
        keep = {int(v) for v in np.unique(at_path[inside]) if int(v) > 0}
        if keep:
            gas = np.isin(labels, list(keep))
        else:
            LOGGER.warning("centerline touches no gas component; gas volume unrestricted")

    vox_ml = float(np.prod(spacing_zyx)) / 1000.0
    return float(gas.sum()) * vox_ml, gas


def luminal_radius_mm(
    lumen_zyx: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Inscribed luminal radius at each centerline point, in mm."""
    img = sitk.GetImageFromArray((np.asarray(lumen_zyx) > 0).astype(np.uint8))
    img.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    edt = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            img, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
        )
    )
    radii, _ = _sample_at_centerline(edt.astype(np.float64), centerline_ijk)
    return np.asarray(radii, dtype=np.float64)


def collapse_ratio_pct(
    lumen_zyx: np.ndarray,
    centerline_ijk: np.ndarray,
    centerline_xyz_mm: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: QualityConfig,
) -> tuple[float, np.ndarray]:
    """Share of centerline *length* whose luminal radius is below the threshold.

    Length-weighted rather than point-weighted, so a densely sampled stretch
    cannot dominate.
    """
    radii = luminal_radius_mm(lumen_zyx, centerline_ijk, spacing_zyx)

    p = np.asarray(centerline_xyz_mm, dtype=np.float64)
    weight = np.zeros(len(p))
    if len(p) > 1:
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        weight[:-1] += seg / 2.0
        weight[1:] += seg / 2.0

    total = float(weight.sum())
    if total <= 0:
        return 0.0, radii
    n = min(len(weight), len(radii))
    collapsed = float(weight[:n][radii[:n] < cfg.collapse_radius_mm].sum())
    return 100.0 * collapsed / total, radii


def distension_score(
    gas_ml: float,
    length_adjusted_gas: float,
    collapse_pct: float,
    cfg: QualityConfig,
) -> tuple[float, str]:
    """Composite 0-100 distension score and its grade.

    The weights (0.45 / 0.40 / 0.15) have **no empirical basis**; they are
    inherited. Report the score as an arbitrary weighting or replace it.
    """
    s_lag = _clip_0_100(
        (length_adjusted_gas - cfg.lag_poor_ml_per_cm)
        / max(cfg.lag_excellent_ml_per_cm - cfg.lag_poor_ml_per_cm, 1e-6)
        * 100.0
    )
    s_gas = _clip_0_100(
        (gas_ml - cfg.gas_poor_ml) / max(cfg.gas_good_ml - cfg.gas_poor_ml, 1e-6) * 100.0
    )
    s_collapse = _clip_0_100(
        (
            1.0
            - (collapse_pct - cfg.collapse_zero_pct)
            / max(cfg.collapse_full_pct - cfg.collapse_zero_pct, 1e-6)
        )
        * 100.0
    )
    score = float(
        np.clip(cfg.w_collapse * s_collapse + cfg.w_lag * s_lag + cfg.w_gas * s_gas, 0.0, 100.0)
    )
    grade = "Poor"
    for threshold, label in cfg.grade_bands:
        if score >= threshold:
            grade = label
            break
    return score, grade


def _band(value: float, bands: tuple[tuple[float, str], ...]) -> str | None:
    if not bands:
        return None
    for upper, label in bands:
        if value <= upper:
            return label
    return bands[-1][1]


def compute_quality(
    ct_zyx: np.ndarray,
    lumen_zyx: np.ndarray,
    centerline_ijk: np.ndarray,
    centerline_xyz_mm: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: QualityConfig | None = None,
    patient_height_cm: float | None = None,
) -> QualityResult:
    """All preparation / distension / morphology indices for one series.

    Never raises: on failure it returns ``ok=False`` with whatever was computed,
    so one bad series cannot abort a cohort run.
    """
    cfg = cfg or QualityConfig()
    ind: dict[str, Any] = {}
    try:
        xyz = smooth_path(centerline_xyz_mm, cfg.smooth_sigma_points)
        length_mm = path_length_mm(xyz)
        straight_mm = straight_line_mm(xyz)
        length_cm = length_mm / 10.0

        # Deliberately *not* called colon length: this is the extent of the
        # traceable gas-filled centerline, which depends on how much of the
        # colon was distended in this acquisition. Measured prone/supine limits
        # of agreement are +/-61.8 cm on a ~100 cm value, so it is not a
        # patient-level anatomical length. See docs/EVALUATION_PRONE_SUPINE.md.
        ind["traced_centerline_length_cm"] = round(length_cm, 2)
        ind["straight_line_distance_cm"] = round(straight_mm / 10.0, 2)

        cti = length_mm / straight_mm if straight_mm > 1e-6 else None
        ind["centerline_tortuosity_index"] = round(cti, 3) if cti is not None else None
        ind["cti_band"] = _band(cti, cfg.cti_bands) if cti is not None else None

        if patient_height_cm and float(patient_height_cm) > 0:
            ind["patient_height_cm"] = round(float(patient_height_cm), 1)
            ind["height_normalized_colon_length_index"] = round(
                length_cm / float(patient_height_cm), 3
            )
        else:
            ind["patient_height_cm"] = None
            ind["height_normalized_colon_length_index"] = None

        pred = float(cfg.predicted_colon_length_cm)
        ind["predicted_colon_length_cm"] = round(pred, 1)
        ind["predicted_length_redundancy_index"] = (
            round(length_cm / pred, 3) if pred > 1e-6 else None
        )

        gas_ml, _ = gas_volume_ml(ct_zyx, lumen_zyx, spacing_zyx, centerline_ijk, cfg)
        ind["gas_volume_ml"] = round(gas_ml, 1)
        lag = gas_ml / length_cm if length_cm > 1e-6 else None
        ind["length_adjusted_gas_ml_per_cm"] = round(lag, 3) if lag is not None else None

        collapse_pct, radii = collapse_ratio_pct(
            lumen_zyx, centerline_ijk, xyz, spacing_zyx, cfg
        )
        ind["collapse_ratio_pct"] = round(collapse_pct, 1)
        ind["collapse_radius_threshold_mm"] = cfg.collapse_radius_mm
        ind["luminal_radius_mm_median"] = round(float(np.median(radii)), 2) if len(radii) else None

        score, grade = distension_score(gas_ml, lag or 0.0, collapse_pct, cfg)
        ind["distension_quality_score"] = round(score, 1)
        ind["distension_grade"] = grade

        return QualityResult(ok=True, message="ok", indices=ind)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("quality indices failed")
        return QualityResult(ok=False, message=f"{type(exc).__name__}: {exc}", indices=ind)
