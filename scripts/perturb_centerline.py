"""Controlled experiment: how much do the reported indices move when the
centerline wobbles?

Pre-specified in ``docs/PLAN_CENTERLINE_PERTURBATION.md``. The same images,
masks, positions and measurement parameters are reused, and **only the
centerline changes**, so this is a control experiment on the series already
analysed, not an independent validation.

The displacement is added once, to the stored centerline points, i.e. before the
smoothing that the length and the fat map each apply. What comes out is
therefore the sensitivity of the reported indices to wobble in the input path,
after the pipeline's own smoothing.

    python scripts/perturb_centerline.py --limit 2      # correctness and timing
    python scripts/perturb_centerline.py                # the full eligible set

Writes one row per series and condition to
``results/tables/centerline_perturbation.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.centerline import CenterlineConfig  # noqa: E402
from ctc_core.fatmap import (  # noqa: E402
    FatConfig,
    centerline_frames,
    fat_map_polar,
    polar_asymmetry,
    resample_centerline,
    smooth_centerline,
    summarise_fat_map,
)
from ctc_core.masks import build_body_air_masks  # noqa: E402
from ctc_core.quality import QualityConfig, path_length_mm, smooth_path  # noqa: E402

LOGGER = logging.getLogger("perturb")

# Section 4 of the plan: 2 amplitudes x 2 wavelengths x 2 directions, plus the
# unperturbed control. "n" and "b" are the two directions of the local plane
# perpendicular to the path, not anatomical left/right.
AMPLITUDES_MM = (1.0, 2.0)
WAVELENGTHS_MM = (10.0, 20.0)
DIRECTIONS = ("n", "b")
TAPER_MM = 15.0  # the displacement is zero at both end points
# The centerline success criterion of this cohort: extraction without error and
# a traced path of at least this length (scripts/make_tables.py FAILED_BELOW_MM).
FAILED_BELOW_MM = 600.0

FIELDS = [
    "PatientID", "role", "SeriesInstanceUID", "condition",
    "amplitude_mm", "wavelength_mm", "direction",
    "traced_length_mm", "traced_length_rel_change",
    "ring_fat_mean_hu_median", "ring_fat_fraction_mean", "fat_asymmetry",
    "d_fat_hu", "d_fat_fraction", "d_asymmetry",
    "points_outside_lumen_frac", "ring_valid_fraction", "n_stations",
    "residual_rms_after_smoothing_mm", "seconds",
]


def arc_length(xyz: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def taper(s: np.ndarray, width_mm: float = TAPER_MM) -> np.ndarray:
    """Raised-cosine ramp: 0 at both ends, 1 in the middle."""
    total = float(s[-1])
    w = min(width_mm, total / 2.0) if total > 0 else 0.0
    if w <= 0:
        return np.ones_like(s)
    up = np.clip(s / w, 0.0, 1.0)
    down = np.clip((total - s) / w, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * up)) * 0.5 * (1 - np.cos(np.pi * down))


def perturb(xyz: np.ndarray, amplitude_mm: float, wavelength_mm: float,
            direction: str, frame_sigma_mm: float) -> np.ndarray:
    """Sinusoidal displacement in the local plane, end points fixed.

    The frame comes from the smoothed control path, which is smooth along the
    path, so the displacement direction cannot flip between neighbouring points
    and no artificial kink is introduced by the frame itself.
    """
    s = arc_length(xyz)
    _t, n_hat, b_hat = centerline_frames(smooth_centerline(xyz, frame_sigma_mm))
    u = n_hat if direction == "n" else b_hat
    d = (amplitude_mm * taper(s) * np.sin(2.0 * np.pi * s / wavelength_mm))[:, None]
    return xyz + d * u


def outside_lumen_fraction(xyz: np.ndarray, ct: sitk.Image, lumen: np.ndarray) -> float:
    idx = np.array([ct.TransformPhysicalPointToIndex(tuple(float(v) for v in p))
                    for p in xyz])
    shape = lumen.shape
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < shape[2]) & (idx[:, 1] >= 0)
          & (idx[:, 1] < shape[1]) & (idx[:, 2] >= 0) & (idx[:, 2] < shape[0]))
    inside = np.zeros(len(xyz), dtype=bool)
    i = idx[ok]
    inside[ok] = lumen[i[:, 2], i[:, 1], i[:, 0]]
    return float(1.0 - inside.mean())


def measure(xyz: np.ndarray, ct: sitk.Image, ct_arr: np.ndarray, body: np.ndarray,
            air: np.ndarray, lumen: np.ndarray, q_cfg: QualityConfig,
            f_cfg: FatConfig) -> dict:
    """The reported quantities, computed exactly as the manuscript computes them."""
    row: dict = {}
    row["traced_length_mm"] = round(path_length_mm(
        smooth_path(xyz, q_cfg.smooth_sigma_points)), 2)

    s_r, sampled = resample_centerline(
        smooth_centerline(xyz, f_cfg.frame_smooth_sigma_mm), f_cfg.step_mm)
    fmap = fat_map_polar(ct, ct_arr, body, air, s_r, sampled, f_cfg)
    summary = summarise_fat_map(fmap, f_cfg)
    asym = polar_asymmetry(fmap, f_cfg)
    row["ring_fat_mean_hu_median"] = summary["ring_fat_mean_hu_median"]
    row["ring_fat_fraction_mean"] = summary["ring_fat_fraction_mean"]
    row["ring_valid_fraction"] = round(summary["ring_valid_fraction"], 4)
    row["n_stations"] = summary["ring_n_stations"]
    row["fat_asymmetry"] = asym.get("fat_asymmetry")
    row["points_outside_lumen_frac"] = round(outside_lumen_fraction(xyz, ct, lumen), 4)
    return row


def eligible(tables: Path, manifest: Path) -> list[dict]:
    """Series that met the centerline success criterion in the current results."""
    ok_centerline = set()
    with (tables / "centerline_auto.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                long_enough = float(r.get("length_mm") or 0.0) >= FAILED_BELOW_MM
            except ValueError:
                long_enough = False
            if str(r.get("ok", "")).strip().lower() == "true" and long_enough:
                ok_centerline.add((r["PatientID"], r["role"]))
    measurable = set()
    with (tables / "indices.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("ok", "")).strip().lower() == "true":
                measurable.add((r["PatientID"], r["role"]))
    keep = ok_centerline & measurable
    with manifest.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (r["PatientID"], r["role"]) in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/centerline_perturbation.csv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N eligible series only (correctness and timing check)")
    ap.add_argument("--patients", type=int, default=None,
                    help="first N patients of the manifest order (the plan's fallback)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    q_cfg, f_cfg = QualityConfig(), FatConfig()
    cl_cfg = CenterlineConfig.corrected()
    records = eligible(args.tables, args.manifest)
    if args.patients:
        keep = list(dict.fromkeys(r["PatientID"] for r in records))[:args.patients]
        records = [r for r in records if r["PatientID"] in keep]
    if args.limit:
        records = records[:args.limit]
    LOGGER.info("eligible series: %d (%d patients)", len(records),
                len({r["PatientID"] for r in records}))

    conditions = [("control", None, None, None)] + [
        (f"A{a:.0f}_L{w:.0f}_{d}", a, w, d)
        for a in AMPLITUDES_MM for w in WAVELENGTHS_MM for d in DIRECTIONS]

    rows: list[dict] = []
    for rec in records:
        pid, role = rec["PatientID"], rec["role"]
        tag = f"{pid[-4:]}-{1 if role == 'primary' else 2}"
        try:
            ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
            lumen_path = args.centerline_root / pid / role / "lumen_mask.nii.gz"
            pts_path = args.centerline_root / pid / role / "centerline_points.csv"
            ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
            ct_arr = sitk.GetArrayFromImage(ct)
            lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path))) > 0
            with pts_path.open(encoding="utf-8") as fh:
                xyz = np.array([(float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]))
                                for r in csv.DictReader(fh)])
            body, air, _ = build_body_air_masks(
                ct, body_threshold=cl_cfg.body_threshold, air_threshold=cl_cfg.air_threshold,
                body_closing_radius=cl_cfg.body_closing_radius,
                air_closing_radius=cl_cfg.air_closing_radius,
                fill_holes=cl_cfg.fill_holes, connectivity=cl_cfg.connectivity)
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the run
            LOGGER.warning("%s skipped: %s", tag, exc)
            continue

        control: dict = {}
        for name, amp, wav, direction in conditions:
            t0 = time.perf_counter()
            pts = xyz if amp is None else perturb(
                xyz, amp, wav, direction, f_cfg.frame_smooth_sigma_mm)
            m = measure(pts, ct, ct_arr, body, air, lumen, q_cfg, f_cfg)
            if amp is None:
                control = dict(m)
                residual = 0.0
            else:
                sm_ctrl = smooth_centerline(xyz, f_cfg.frame_smooth_sigma_mm)
                sm_pert = smooth_centerline(pts, f_cfg.frame_smooth_sigma_mm)
                residual = float(np.sqrt(
                    (np.linalg.norm(sm_pert - sm_ctrl, axis=1) ** 2).mean()))

            def delta(key: str, m=m, control=control) -> float | None:
                a, b = m.get(key), control.get(key)
                return None if a is None or b is None else round(float(a) - float(b), 4)

            rel = (None if not control.get("traced_length_mm") else
                   round(m["traced_length_mm"] / control["traced_length_mm"] - 1.0, 5))
            rows.append({
                "PatientID": pid, "role": role, "SeriesInstanceUID": rec["SeriesInstanceUID"],
                "condition": name, "amplitude_mm": amp, "wavelength_mm": wav,
                "direction": direction, **m,
                "traced_length_rel_change": rel,
                "d_fat_hu": delta("ring_fat_mean_hu_median"),
                "d_fat_fraction": delta("ring_fat_fraction_mean"),
                "d_asymmetry": delta("fat_asymmetry"),
                "residual_rms_after_smoothing_mm": round(residual, 4),
                "seconds": round(time.perf_counter() - t0, 1),
            })
            LOGGER.info("%s %-12s length %8.1f mm (%+.2f %%) fat %s HU outside %.3f [%.0f s]",
                        tag, name, m["traced_length_mm"], 100 * (rel or 0.0),
                        m["ring_fat_mean_hu_median"], m["points_outside_lumen_frac"],
                        rows[-1]["seconds"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows ({len(records)} series x {len(conditions)} conditions)"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
