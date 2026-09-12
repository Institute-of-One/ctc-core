"""Compute the quality and fat indices for every series with a centerline.

Consumes the centerline stage's outputs (``lumen_mask.nii.gz`` and
``centerline_points.csv``) and writes one row per series.

    python scripts/run_indices.py --raw-root . \\
        --centerline-root data/centerline/corrected \\
        --out results/tables/indices.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.centerline import CenterlineConfig  # noqa: E402
from ctc_core.fatmap import (  # noqa: E402
    FatConfig,
    fat_map_polar,
    fat_profile_spherical,
    polar_asymmetry,
    resample_centerline,
    smooth_centerline,
    summarise_fat_map,
    summarise_fat_profile,
)
from ctc_core.masks import build_body_air_masks  # noqa: E402
from ctc_core.quality import QualityConfig, compute_quality  # noqa: E402

LOGGER = logging.getLogger("run_indices")

QUALITY_FIELDS = [
    "traced_centerline_length_cm", "straight_line_distance_cm", "centerline_tortuosity_index",
    "cti_band", "height_normalized_colon_length_index",
    "predicted_length_redundancy_index", "gas_volume_ml",
    "length_adjusted_gas_ml_per_cm", "collapse_ratio_pct",
    "luminal_radius_mm_median", "distension_quality_score", "distension_grade",
]
FAT_FIELDS = [
    "fat_n_points", "fat_n_valid", "fat_valid_fraction",
    "fat_mean_hu_median", "fat_fraction_mean", "fat_volume_ml_sum",
    "fat_fraction_left", "fat_fraction_right", "fat_asymmetry",
    "fat_asymmetry_theta_halves",
    "ring_n_stations", "ring_n_valid", "ring_valid_fraction",
    "ring_fat_mean_hu_median", "ring_fat_fraction_mean",
]
FIELDS = ["PatientID", "role", "SeriesInstanceUID", "position", "ok", "error"] + (
    QUALITY_FIELDS + FAT_FIELDS
)


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz_mm, ijk)`` from a centerline_points.csv."""
    xyz, ijk = [], []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            xyz.append((float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])))
            ijk.append((int(r["i"]), int(r["j"]), int(r["k"])))
    return np.array(xyz, dtype=np.float64), np.array(ijk, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--position", type=Path, default=Path("results/tables/patient_position.csv"))
    ap.add_argument("--polar", action="store_true",
                    help="also compute the (s, theta) map and its asymmetry index")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    positions: dict[str, str] = {}
    if args.position.exists():
        for r in csv.DictReader(args.position.open(encoding="utf-8")):
            positions[r["SeriesInstanceUID"]] = r["position_combined"]

    cl_cfg = CenterlineConfig.corrected()
    q_cfg = QualityConfig()
    f_cfg = FatConfig()

    rows = []
    for rec in csv.DictReader(args.manifest.open(encoding="utf-8")):
        pid, role = rec["PatientID"], rec["role"]
        row = {
            "PatientID": pid,
            "role": role,
            "SeriesInstanceUID": rec["SeriesInstanceUID"],
            "position": positions.get(rec["SeriesInstanceUID"], ""),
            "ok": False,
            "error": "",
        }
        try:
            ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
            lumen_path = args.centerline_root / pid / role / "lumen_mask.nii.gz"
            pts_path = args.centerline_root / pid / role / "centerline_points.csv"
            for p in (ct_path, lumen_path, pts_path):
                if not p.exists():
                    raise FileNotFoundError(str(p))

            ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
            ct_arr = sitk.GetArrayFromImage(ct)
            spacing_xyz = ct.GetSpacing()
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

            lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path))) > 0
            xyz, ijk = load_points(pts_path)

            q = compute_quality(ct_arr, lumen, ijk, xyz, spacing_zyx, q_cfg)
            if not q.ok:
                raise RuntimeError(q.message)
            for k in QUALITY_FIELDS:
                row[k] = q.indices.get(k)

            # The fat ROI needs body and non-lumen air, from the shared entry point.
            body, air, _ = build_body_air_masks(
                ct,
                body_threshold=cl_cfg.body_threshold,
                air_threshold=cl_cfg.air_threshold,
                body_closing_radius=cl_cfg.body_closing_radius,
                air_closing_radius=cl_cfg.air_closing_radius,
                fill_holes=cl_cfg.fill_holes,
                connectivity=cl_cfg.connectivity,
            )
            s, sampled = resample_centerline(xyz, f_cfg.step_mm)
            profile = fat_profile_spherical(ct, ct_arr, body, air, s, sampled, f_cfg)
            summary = summarise_fat_profile(profile)
            row["fat_n_points"] = summary["n_points"]
            row["fat_n_valid"] = summary["n_valid"]
            row["fat_valid_fraction"] = round(summary["valid_fraction"], 4)
            for k in ("fat_mean_hu_median", "fat_fraction_mean", "fat_volume_ml_sum"):
                v = summary[k]
                row[k] = round(v, 4) if v is not None else None

            if args.polar:
                # The ring map runs on the smoothed centerline (VGP frames);
                # the spherical profile above keeps the 2026-08 batch geometry.
                s_r, sampled_r = resample_centerline(
                    smooth_centerline(xyz, f_cfg.frame_smooth_sigma_mm), f_cfg.step_mm)
                fmap = fat_map_polar(ct, ct_arr, body, air, s_r, sampled_r, f_cfg)
                for k, v in {**polar_asymmetry(fmap, f_cfg),
                             **summarise_fat_map(fmap, f_cfg)}.items():
                    row[k] = round(v, 4) if isinstance(v, float) else v

            row["ok"] = True
            LOGGER.info(
                "%s/%s [%s] length=%.1f cm gas=%.0f mL collapse=%.1f%% fat_frac=%s",
                pid[-4:], role, row["position"] or "?",
                row["traced_centerline_length_cm"] or 0, row["gas_volume_ml"] or 0,
                row["collapse_ratio_pct"] or 0, row["fat_fraction_mean"],
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            LOGGER.error("%s/%s FAILED %s", pid[-4:], role, row["error"])
            LOGGER.debug(traceback.format_exc())
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["ok"])
    print(f"\n{len(rows)} series ({ok} ok) -> {args.out}")


if __name__ == "__main__":
    main()
