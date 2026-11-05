"""Measure the image noise of the cohort, so the phantom noise is not invented.

The phantoms have to carry the noise of a screening acquisition, and the dose
that would set it is not in the retained headers: the acquisition
stratification found tube current recorded consistently and CTDIvol absent.
So the noise is measured from the images themselves rather than inferred from
a dose.

The estimator is the local standard deviation in a 5 x 5 in-plane window, over
windows that lie entirely inside soft tissue (0 to 100 HU). Within such a
window the variation is noise plus whatever
texture the tissue has, so the distribution of local standard deviations is an
upper bound at every percentile; its low percentiles come from the most
homogeneous windows and are the closest thing to noise alone. All three of the
10th, 25th and 50th percentiles are written out, so the choice of which to
inject is visible rather than buried.

Air is measured as well, inside the gas lumen the pipeline segmented and eroded
away from the wall, because the primary phantom's contrast is tissue against
air and both sides of that edge carry noise.

    python scripts/measure_image_noise.py --out results/tables/image_noise.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, uniform_filter

sys.stdout.reconfigure(encoding="utf-8")

# The soft-tissue window for the partial-volume background.
SOFT_HU_MIN = 0.0
SOFT_HU_MAX = 100.0
# In-plane window for the local standard deviation. Odd, and small enough that
# a window inside an organ is plausibly homogeneous.
WINDOW = 5
# Every Nth axial slice, to keep 60 volumes affordable. Noise is a per-voxel
# property, so a sample of slices estimates it; the count of contributing
# windows is written out so a thin sample is visible.
SLICE_STEP = 10
# Percentiles of the local-standard-deviation distribution to report.
PERCENTILES = (10, 25, 50)
# Erosion of the gas lumen before measuring air, in voxels, to keep the wall
# and its partial-volume ramp out of the sample.
AIR_EROSION = 3
# In-plane lags, in voxels, at which the noise autocorrelation is measured.
# Reconstruction correlates neighbouring pixels, so injecting white noise of
# the measured standard deviation would not be the same perturbation; the
# correlated condition is matched to these.
LAGS = (1, 2, 3)


def local_sd_2d(plane: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation in a window x window neighbourhood of each pixel."""
    m1 = uniform_filter(plane, size=window, mode="nearest")
    m2 = uniform_filter(plane * plane, size=window, mode="nearest")
    return np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))


def soft_tissue_sds(
    arr: np.ndarray, window: int, step: int
) -> tuple[np.ndarray, dict[int, float]]:
    """Local standard deviations, and the in-plane autocorrelation of the residual.

    Both are taken over windows lying entirely inside soft tissue. The residual
    is the plane minus its own local mean, which is the noise plus the tissue's
    texture; the autocorrelation of that residual is therefore an upper bound on
    the correlation length of the noise alone, in the same way the standard
    deviation is an upper bound on its size.
    """
    out = []
    num = {k: 0.0 for k in LAGS}
    den = 0.0
    for k in range(0, arr.shape[0], step):
        plane = arr[k].astype(np.float32)
        soft = ((plane >= SOFT_HU_MIN) & (plane <= SOFT_HU_MAX)).astype(np.float32)
        # A window is admitted only when every pixel in it is soft tissue, so
        # organ boundaries and the fat and gas next to them are excluded. This
        # is a homogeneity criterion on the window, not on its variance, so it
        # does not bias the estimate towards small values.
        whole = uniform_filter(soft, size=window, mode="constant", cval=0.0) > 0.999
        if not whole.any():
            continue
        out.append(local_sd_2d(plane, window)[whole])
        resid = plane - uniform_filter(plane, size=window, mode="nearest")
        r = np.where(whole, resid, 0.0)
        den += float((r * r)[whole].sum())
        for lag in LAGS:
            both = whole[:, :-lag] & whole[:, lag:]
            if both.any():
                num[lag] += float((r[:, :-lag] * r[:, lag:])[both].sum())
    if not out:
        return np.empty(0, dtype=np.float32), {}
    acf = {lag: (num[lag] / den if den > 0 else float("nan")) for lag in LAGS}
    return np.concatenate(out), acf


def air_sd(arr: np.ndarray, lumen: np.ndarray | None, window: int, step: int) -> float | None:
    """Local standard deviation inside the segmented gas lumen, eroded."""
    if lumen is None:
        return None
    inner = binary_erosion(lumen, iterations=AIR_EROSION)
    out = []
    for k in range(0, arr.shape[0], step):
        if not inner[k].any():
            continue
        plane = arr[k].astype(np.float32)
        mask = uniform_filter(inner[k].astype(np.float32), size=window,
                              mode="constant", cval=0.0) > 0.999
        if not mask.any():
            continue
        out.append(local_sd_2d(plane, window)[mask])
    if not out:
        return None
    return float(np.median(np.concatenate(out)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for rec in csv.DictReader(args.manifest.open(encoding="utf-8")):
        pid, role = rec["PatientID"], rec["role"]
        ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
        if not ct_path.exists():
            print(f"{pid} {role}: no image")
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path), sitk.sitkFloat32))

        lumen_path = args.centerline_root / pid / role / "lumen_mask.nii.gz"
        lumen = None
        if lumen_path.exists():
            lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path))) > 0

        sds, acf = soft_tissue_sds(arr, WINDOW, SLICE_STEP)
        if sds.size == 0:
            print(f"{pid} {role}: no homogeneous soft-tissue window")
            continue
        row = {
            "PatientID": pid,
            "role": role,
            "windows": int(sds.size),
            "slices_sampled": int(np.ceil(arr.shape[0] / SLICE_STEP)),
        }
        for p in PERCENTILES:
            row["soft_sd_p%d_hu" % p] = round(float(np.percentile(sds, p)), 3)
        for lag in LAGS:
            row["acf_lag%d" % lag] = round(acf[lag], 4)
        a = air_sd(arr, lumen, WINDOW, SLICE_STEP)
        row["air_sd_median_hu"] = None if a is None else round(a, 3)
        rows.append(row)
        print("%s %-9s windows %8d  soft p10 %6.2f  p25 %6.2f  p50 %6.2f  air %s"
              % (pid, role, row["windows"], row["soft_sd_p10_hu"],
                 row["soft_sd_p25_hu"], row["soft_sd_p50_hu"],
                 "n/a" if a is None else "%6.2f" % a))

    if not rows:
        raise SystemExit("no series measured")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary: dict[str, object] = {
        "series": len(rows),
        "window": WINDOW,
        "slice_step": SLICE_STEP,
        "soft_hu_range": [SOFT_HU_MIN, SOFT_HU_MAX],
        "air_erosion_voxels": AIR_EROSION,
    }
    for p in PERCENTILES:
        vals = [r["soft_sd_p%d_hu" % p] for r in rows]
        summary["soft_sd_p%d_median_over_series_hu" % p] = round(statistics.median(vals), 3)
        summary["soft_sd_p%d_min_hu" % p] = round(min(vals), 3)
        summary["soft_sd_p%d_max_hu" % p] = round(max(vals), 3)
    for lag in LAGS:
        vals = [r["acf_lag%d" % lag] for r in rows]
        summary["acf_lag%d_median_over_series" % lag] = round(statistics.median(vals), 4)
        summary["acf_lag%d_min" % lag] = round(min(vals), 4)
        summary["acf_lag%d_max" % lag] = round(max(vals), 4)
    air = [r["air_sd_median_hu"] for r in rows if r["air_sd_median_hu"] is not None]
    summary["air_series"] = len(air)
    if air:
        summary["air_sd_median_over_series_hu"] = round(statistics.median(air), 3)
        summary["air_sd_min_hu"] = round(min(air), 3)
        summary["air_sd_max_hu"] = round(max(air), 3)
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nover %d series, median of the per-series percentile:" % len(rows))
    for p in PERCENTILES:
        print("  soft tissue p%-2d  %6.2f HU  (%.2f to %.2f)"
              % (p, summary["soft_sd_p%d_median_over_series_hu" % p],
                 summary["soft_sd_p%d_min_hu" % p], summary["soft_sd_p%d_max_hu" % p]))
    if air:
        print("  air (lumen)      %6.2f HU  (%.2f to %.2f) over %d series"
              % (summary["air_sd_median_over_series_hu"], summary["air_sd_min_hu"],
                 summary["air_sd_max_hu"], len(air)))
    print("in-plane autocorrelation of the residual:")
    for lag in LAGS:
        print("  lag %d  %+.3f  (%+.3f to %+.3f)"
              % (lag, summary["acf_lag%d_median_over_series" % lag],
                 summary["acf_lag%d_min" % lag], summary["acf_lag%d_max" % lag]))
    print("-> %s\n-> %s" % (args.out, out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
