"""Polyp measurement under conditions a lesion in a patient actually has.

Spheres at 60 HU in a -1000 HU background with a one-voxel partial-volume
transition are easier than any in-vivo lesion. What a lesion in a patient has
instead: image noise at screening dose, a partial-volume interface against 0 to
100 HU rather than against air, and adhesion to a wall or a fold. The error is
reported against the 6 mm and 10 mm reporting thresholds, with the rate at which
measurement error would move a lesion across one.

This script does not replace ``eval_polyp_phantoms.py``: that table is the
noise-free limit and its numbers are cited in the paper, so it is left alone and
reproduced here as the ``air`` condition. What is added is a set of named
conditions, each a single departure from that limit, so that a change in the
error can be attributed to the departure that caused it.

The noise is not invented. ``measure_image_noise.py`` measures it in the cohort
itself -- the local standard deviation of windows lying entirely inside soft
tissue, and the in-plane autocorrelation of the residual -- and the levels and
the correlation used here are read from that file. Nothing in this script
carries a hard-coded noise level.

    python scripts/measure_image_noise.py --out results/tables/image_noise.csv
    python scripts/eval_polyp_realism.py --noise results/tables/image_noise.json \
        --out results/tables/polyp_realism.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from ctc_core.polyp import PolypConfig, measure_polyp  # noqa: E402

AIR_HU = -1000.0
SOFT_HU = 60.0
# The lesion-to-background difference where the background is not air.
# 20 HU is untagged fluid and sits inside the 0 to 100 HU soft-tissue range;
# the lesion is 60 HU, so the contrast falls from 1060 HU to 40 HU.
FLUID_HU = 20.0
# The wall a lesion adheres to is the same tissue as the lesion. Giving it a
# different attenuation would make the adhesion detectable by attenuation
# alone, which is not what adhesion is.
WALL_HU = 60.0
# What lies beyond the wall. Pericolonic fat is outside the segmentation
# window, so it is where a region grown from inside the lesion stops.
FAT_HU = -100.0
# Wall thickness. A distended colonic wall is of this order; the sweep also
# runs a deliberately thick wall as the pessimistic bound, since the thickness
# is what decides how much tissue a leak can reach.
WALL_THICKNESS_MM = 2.0
WALL_THICKNESS_THICK_MM = 10.0

# Reporting classes for a CT colonography size report: diminutive, small, large.
THRESHOLDS = (6.0, 10.0)

# Seeds. One stream per (condition, phantom, repeat) so that adding a condition
# or a size cannot move a number already reported for another one.
SEED = 20261105
REPEATS = 20

# True diameters. The sizes of the noise-free table are kept, and sizes 0.5 mm
# either side of each reporting threshold are added, because the rate at which
# a lesion crosses a threshold can only be measured on lesions near one.
SPHERE_DIAMETERS = (5.5, 6.0, 6.5, 8.0, 9.5, 10.0, 10.5, 12.0, 16.0)


def gauss_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def kernel_rho1(sigma: float) -> float:
    """Lag-1 autocorrelation of white noise after smoothing with this kernel."""
    k = gauss_kernel(sigma)
    return float((k[:-1] * k[1:]).sum() / (k * k).sum())


def sigma_for_rho1(rho: float) -> float:
    """The smoothing scale whose lag-1 autocorrelation is ``rho``."""
    lo, hi = 1e-3, 5.0
    if rho <= kernel_rho1(lo):
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kernel_rho1(mid) < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def noise_field(
    shape: tuple[int, int, int], sd: float, rho1: float, rng: np.random.Generator
) -> np.ndarray:
    """Noise of standard deviation ``sd`` and in-plane lag-1 correlation ``rho1``.

    Correlated in plane and independent between slices, which is how the
    measurement was taken: ``measure_image_noise.py`` measures the
    autocorrelation along a row and says nothing about the axis across slices.
    """
    if rho1 <= 0.0:
        return rng.standard_normal(shape).astype(np.float32) * sd
    k = gauss_kernel(sigma_for_rho1(rho1))
    # Each 1-D convolution multiplies the variance by sum(k^2), so two of them
    # multiply the standard deviation by sum(k^2). Scale the white field before
    # smoothing rather than rescaling afterwards, which would remove the
    # realization-to-realization spread of the standard deviation itself.
    white = rng.standard_normal(shape).astype(np.float32) * (sd / float((k * k).sum()))
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, white)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 2, out)
    return out.astype(np.float32)


def sphere_frac(
    semi: tuple[float, float, float], n: int, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy fraction of the lesion, and the physical coordinate grids."""
    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    zz, yy, xx = (zz - c) * spacing, (yy - c) * spacing, (xx - c) * spacing
    r_norm = np.sqrt((zz / semi[0]) ** 2 + (yy / semi[1]) ** 2 + (xx / semi[2]) ** 2)
    dist = (r_norm - 1.0) * min(semi)
    frac = np.clip(0.5 - dist / spacing, 0.0, 1.0)
    return frac.astype(np.float32), np.stack([zz, yy, xx])


def layered(coord: np.ndarray, base: float, thickness: float, spacing: float) -> np.ndarray:
    """Lumen, then a wall of ``thickness``, then pericolonic fat, along one axis.

    A wall that extends indefinitely is not a colon: a region grown from a
    point inside the lesion would never run out of tissue, and the measurement
    would fail for a reason the patient does not have. Beyond the wall is fat,
    which lies outside the segmentation window and stops the growth where the
    anatomy stops it.
    """
    f_air = np.clip(0.5 + (base - coord) / spacing, 0.0, 1.0)
    f_wall = (np.clip(0.5 + (coord - base) / spacing, 0.0, 1.0)
              * np.clip(0.5 + (base + thickness - coord) / spacing, 0.0, 1.0))
    f_fat = np.clip(0.5 + (coord - (base + thickness)) / spacing, 0.0, 1.0)
    total = np.clip(f_air + f_wall + f_fat, 1e-6, None)
    return (f_air * AIR_HU + f_wall * WALL_HU + f_fat * FAT_HU) / total


def build(
    semi: tuple[float, float, float],
    n: int,
    spacing: float,
    geometry: str,
    wall_mm: float = WALL_THICKNESS_MM,
) -> np.ndarray:
    """A lesion in a background, with the geometry of its surroundings.

    The surroundings are built first and the lesion is laid into them with its
    own partial-volume ramp, so the ramp is against whatever is actually behind
    it -- air, fluid, or, where the lesion is continuous with its wall, nothing
    at all.
    """
    frac, (zz, yy, xx) = sphere_frac(semi, n, spacing)

    if geometry == "fluid":
        bg = np.full_like(frac, FLUID_HU)
    elif geometry == "air":
        bg = np.full_like(frac, AIR_HU)
    else:
        # ``wall_tangent`` puts the wall surface at the lesion's far pole, so
        # the two touch at a single point: the mildest adhesion that is still
        # adhesion. In ``wall_sessile`` the surface passes through the lesion's
        # centre, so the lesion is a hemisphere rising from the wall and
        # continuous with it, which is what a sessile polyp is. The largest
        # diameter of the protruding part is the equatorial one either way, so
        # the truth is the same number in both and the two are comparable.
        base = 0.0 if geometry.startswith("wall_sessile") else semi[0]
        bg = layered(zz, base, wall_mm, spacing)
        if geometry == "fold":
            # A fold: a second wall meeting the first, so the lesion sits in a
            # concave corner and touches tissue on two sides. Wall outranks fat
            # and fat outranks lumen, which is what the maximum selects.
            bg = np.maximum(bg, layered(xx, semi[2], wall_mm, spacing))
    return (frac * SOFT_HU + (1.0 - frac) * bg).astype(np.float32)


def size_class(diameter_mm: float) -> str:
    if diameter_mm < THRESHOLDS[0]:
        return "<6"
    if diameter_mm < THRESHOLDS[1]:
        return "6-9"
    return ">=10"


def conditions(noise: dict) -> list[tuple[str, str, float, float, float]]:
    """(name, geometry, noise sd in HU, in-plane lag-1 correlation, wall mm).

    Every level is read from the measurement file. ``sd_measured`` is the median
    over the sixty series of each series' 10th percentile of local standard
    deviation; the 10th percentile is the most homogeneous windows, and so the
    closest of the three reported percentiles to noise without tissue texture.
    """
    sd = float(noise["soft_sd_p10_median_over_series_hu"])
    rho = float(noise["acf_lag1_median_over_series"])
    t = WALL_THICKNESS_MM
    return [
        ("air", "air", 0.0, 0.0, t),
        ("air_noise", "air", sd, rho, t),
        ("air_noise_white", "air", sd, 0.0, t),
        ("air_noise_double", "air", 2.0 * sd, rho, t),
        ("fluid", "fluid", 0.0, 0.0, t),
        ("fluid_noise", "fluid", sd, rho, t),
        ("wall_tangent", "wall_tangent", 0.0, 0.0, t),
        ("wall_tangent_noise", "wall_tangent", sd, rho, t),
        ("wall_sessile", "wall_sessile", 0.0, 0.0, t),
        ("wall_sessile_noise", "wall_sessile", sd, rho, t),
        ("wall_sessile_thick_noise", "wall_sessile", sd, rho, WALL_THICKNESS_THICK_MM),
        ("fold", "fold", 0.0, 0.0, t),
        ("fold_noise", "fold", sd, rho, t),
    ]


NEIGHBOURHOODS = (1, 2, 3)


def neighbourhood_tradeoff(noise: dict, spacing: float, repeats: int) -> list[dict]:
    """Where the second failure comes from, and what it would cost to remove.

    The grayscale surface is confined to a fixed dilation of the segmentation.
    Anything inside that distance joins the surface, so a neighbour inflates
    the diameter; shrinking the distance keeps the neighbour out but clips the
    lesion's own surface. Both halves are measured here rather than asserted,
    on the sphere that sits exactly on the 10 mm reporting threshold.
    """
    sd = float(noise["soft_sd_p10_median_over_series_hu"])
    rho = float(noise["acf_lag1_median_over_series"])
    phantoms = [("sphere d=%g" % d, (d / 2.0, d / 2.0, d / 2.0))
                for d in SPHERE_DIAMETERS]
    phantoms.append(("ellipsoid 8x8x20", (4.0, 4.0, 10.0)))
    out = []
    for geometry in ("air", "wall_tangent", "fold"):
        for radius in NEIGHBOURHOODS:
            cfg = PolypConfig(surface_neighbourhood_voxels=radius)
            errs, leaks = [], 0
            for label, semi in phantoms:
                n = int(np.ceil(2 * max(semi) / spacing)) + 28
                centre = (n / 2.0, n / 2.0, n / 2.0)
                clean = build(semi, n, spacing, geometry)
                truth = 2.0 * max(semi)
                true_vol = (4.0 / 3.0) * np.pi * semi[0] * semi[1] * semi[2]
                for rep in range(repeats):
                    rng = np.random.default_rng(
                        [SEED, zlib.crc32(b"neighbourhood"),
                         zlib.crc32(label.encode()), radius, rep])
                    r = measure_polyp(
                        clean + noise_field(clean.shape, sd, rho, rng),
                        centre, (spacing,) * 3, cfg)
                    if not r.ok:
                        continue
                    if float(r.measurements["volume_mm3"]) > 2.0 * true_vol:
                        leaks += 1
                        continue
                    errs.append(float(r.measurements["max_diameter_mm"]) - truth)
            a = sorted(abs(e) for e in errs)
            out.append({
                "geometry": geometry,
                "neighbourhood_voxels": radius,
                "shipped": radius == PolypConfig().surface_neighbourhood_voxels,
                "phantoms": len(phantoms),
                "measurements": len(errs),
                "left_the_lesion": leaks,
                "median_diameter_error_mm": round(statistics.median(errs), 3)
                if errs else None,
                "p95_abs_diameter_error_mm": round(float(np.percentile(a, 95)), 3)
                if errs else None,
                "max_abs_diameter_error_mm": round(max(a), 3) if errs else None,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=Path, default=Path("results/tables/image_noise.json"),
                    help="output of measure_image_noise.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=1.0, help="isotropic voxel size, mm")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    noise = json.loads(args.noise.read_text(encoding="utf-8"))
    sp = args.spacing
    cfg = PolypConfig(diameter_from_grayscale=True)
    cfg_bin = PolypConfig(diameter_from_grayscale=False)

    phantoms = [("sphere d=%g" % d, (d / 2.0, d / 2.0, d / 2.0)) for d in SPHERE_DIAMETERS]
    phantoms.append(("ellipsoid 8x8x20", (4.0, 4.0, 10.0)))

    rows = []
    for cname, geometry, sd, rho, wall_mm in conditions(noise):
        reps = 1 if sd == 0.0 else args.repeats
        for label, semi in phantoms:
            n = int(np.ceil(2 * max(semi) / sp)) + 28
            clean = build(semi, n, sp, geometry, wall_mm)
            centre = (n / 2.0, n / 2.0, n / 2.0)
            true_diam = 2.0 * max(semi)
            true_vol = (4.0 / 3.0) * np.pi * semi[0] * semi[1] * semi[2]
            for rep in range(reps):
                if sd == 0.0:
                    ct = clean
                else:
                    rng = np.random.default_rng(
                        [SEED, zlib.crc32(cname.encode()), zlib.crc32(label.encode()), rep]
                    )
                    ct = clean + noise_field(clean.shape, sd, rho, rng)
                g = measure_polyp(ct, centre, (sp, sp, sp), cfg)
                b = measure_polyp(ct, centre, (sp, sp, sp), cfg_bin)
                row = {
                    "condition": cname,
                    "geometry": geometry,
                    "wall_mm": wall_mm,
                    "noise_sd_hu": round(sd, 3),
                    "noise_rho1": round(rho, 4),
                    "phantom": label,
                    "repeat": rep,
                    "true_max_diameter_mm": round(true_diam, 3),
                    "true_class": size_class(true_diam),
                    "ok": bool(g.ok),
                    "message": "" if g.ok else g.message,
                }
                if g.ok:
                    d = float(g.measurements["max_diameter_mm"])
                    v = float(g.measurements["volume_mm3"])
                    row.update({
                        "grayscale_max_diameter_mm": round(d, 3),
                        "grayscale_diameter_error_mm": round(d - true_diam, 3),
                        "measured_class": size_class(d),
                        "class_changed": size_class(d) != size_class(true_diam),
                        "mesh_volume_mm3": round(v, 2),
                        "mesh_volume_error_pct": round(100.0 * (v - true_vol) / true_vol, 2),
                        # A segmentation that has grown out of the lesion is not
                        # a measurement with a large error; it is a different
                        # object. The criterion is stated rather than implied:
                        # more than twice the true volume.
                        "leaked": v > 2.0 * true_vol,
                        "sphericity": g.measurements["sphericity"],
                        # The flag a caller could refuse on, recorded here so
                        # it can be checked against the leak it is meant to
                        # predict rather than assumed to work.
                        "touches_voi_boundary": bool(
                            g.measurements["touches_voi_boundary"]),
                        "voi_boundary_fraction": g.measurements["voi_boundary_fraction"],
                        # The iso-level is where the two diameter routes part
                        # company, so it is recorded rather than inferred.
                        "iso_level_hu": g.measurements["diameter_iso_level_hu"],
                        "voxel_count": g.measurements["voxel_count"],
                    })
                else:
                    row.update({
                        "grayscale_max_diameter_mm": None,
                        "grayscale_diameter_error_mm": None,
                        "measured_class": "",
                        "class_changed": None,
                        "mesh_volume_mm3": None,
                        "mesh_volume_error_pct": None,
                        "leaked": None,
                        "sphericity": None,
                        "touches_voi_boundary": None,
                        "voi_boundary_fraction": None,
                        "iso_level_hu": None,
                        "voxel_count": None,
                    })
                if b.ok:
                    db = float(b.measurements["max_diameter_mm"])
                    row["binary_max_diameter_mm"] = round(db, 3)
                    row["binary_diameter_error_mm"] = round(db - true_diam, 3)
                else:
                    row["binary_max_diameter_mm"] = None
                    row["binary_diameter_error_mm"] = None
                rows.append(row)
        print("%-18s %s" % (cname, "done"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = summarise(rows, args.noise.name, noise)
    summary["neighbourhood_tradeoff"] = neighbourhood_tradeoff(
        noise, sp, args.repeats)
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n%-26s %5s %6s %8s %8s %7s %7s %7s"
          % ("condition", "n", "usable", "med err", "band", "leaked", "flagged", "flips"))
    for c in summary["conditions"]:
        print("%-26s %5d %6d %+8.2f %8.2f %6.0f%% %6.0f%% %6.0f%%"
              % (c["condition"], c["cases"], c["usable"],
                 c["median_error_mm"] if c["usable"] else float("nan"),
                 c["threshold_band_mm"] if c["usable"] else float("nan"),
                 100.0 * c["leak_rate"], 100.0 * c["flag_rate"],
                 100.0 * c["class_change_rate"]))
    f = summary["refusal_flag"]
    print("\nrefusal flag (segmentation reaches the edge of the volume of interest):")
    print("  sensitivity to a leak %s, specificity %s, over %d measured cases"
          % (f["sensitivity_to_leak"], f["specificity_to_leak"], f["cases_measured"]))
    print("  of %d cases it passes, %d still report the wrong size class (%s)"
          % (f["passed_cases"], f["passed_but_wrong_class"], f["passed_but_wrong_class_rate"]))
    print("-> %s\n-> %s" % (args.out, out_json))
    return 0


def summarise(rows: list[dict], noise_file: str, noise: dict) -> dict:
    out: list[dict] = []
    for cname in dict.fromkeys(r["condition"] for r in rows):
        sub = [r for r in rows if r["condition"] == cname]
        # A leaked segmentation is excluded from the error statistics and
        # counted separately: averaging a leak into a diameter error would
        # report a measurement where there was none.
        usable = [r for r in sub if r["ok"] and not r["leaked"]]
        errs = [float(r["grayscale_diameter_error_mm"]) for r in usable]
        near = [r for r in usable
                if min(abs(r["true_max_diameter_mm"] - t) for t in THRESHOLDS) <= 0.5]
        rec = {
            "condition": cname,
            "geometry": sub[0]["geometry"],
            "wall_mm": sub[0]["wall_mm"],
            "noise_sd_hu": sub[0]["noise_sd_hu"],
            "noise_rho1": sub[0]["noise_rho1"],
            "cases": len(sub),
            "usable": len(usable),
            "failed": sum(1 for r in sub if not r["ok"]),
            "leaked": sum(1 for r in sub if r["ok"] and r["leaked"]),
            "leak_rate": round(sum(1 for r in sub if r["ok"] and r["leaked"]) / len(sub), 4),
            "flagged": sum(1 for r in sub if r["ok"] and r["touches_voi_boundary"]),
            "flag_rate": round(
                sum(1 for r in sub if r["ok"] and r["touches_voi_boundary"]) / len(sub), 4),
        }
        if usable:
            a = sorted(abs(e) for e in errs)
            p95 = float(np.percentile(a, 95))
            rec.update({
                "median_error_mm": round(statistics.median(errs), 3),
                "min_error_mm": round(min(errs), 3),
                "max_error_mm": round(max(errs), 3),
                "p95_abs_error_mm": round(p95, 3),
                # The half-width of the band of true diameters around a
                # reporting threshold within which this error can move a lesion
                # to the other side of it. It is the error expressed in the
                # units the report uses.
                "threshold_band_mm": round(p95, 3),
                "class_changes": sum(1 for r in usable if r["class_changed"]),
                "class_change_rate": round(
                    sum(1 for r in usable if r["class_changed"]) / len(usable), 4),
                "near_threshold_cases": len(near),
                "near_threshold_class_change_rate": (
                    round(sum(1 for r in near if r["class_changed"]) / len(near), 4)
                    if near else None),
                "median_volume_error_pct": round(
                    statistics.median(float(r["mesh_volume_error_pct"]) for r in usable), 2),
            })
        else:
            rec.update({"median_error_mm": None, "p95_abs_error_mm": None,
                        "threshold_band_mm": None, "class_changes": None,
                        "class_change_rate": 0.0, "near_threshold_cases": len(near),
                        "near_threshold_class_change_rate": None,
                        "median_volume_error_pct": None})
        out.append(rec)
    # The second signal, and the reason there is one. Where a lesion lies
    # against tissue it is not continuous with, the region does not leak --
    # the binary segmentation is unchanged -- and what moves is the grayscale
    # iso-level, which is estimated from a shell around the segmentation that
    # now contains the neighbour's tissue. So the two diameter routes, which
    # agree to a fifth of a millimetre on a free lesion, disagree; and that
    # disagreement is computable without knowing the truth.
    GAP_MM = 1.0
    # Does the flag predict the leak it is meant to predict? A refusal
    # criterion nobody checked is worse than none, because the paper would
    # recommend it. Counted over every case that produced a measurement.
    ok = [r for r in rows if r["ok"]]
    tp = sum(1 for r in ok if r["touches_voi_boundary"] and r["leaked"])
    fp = sum(1 for r in ok if r["touches_voi_boundary"] and not r["leaked"])
    fn = sum(1 for r in ok if not r["touches_voi_boundary"] and r["leaked"])
    tn = sum(1 for r in ok if not r["touches_voi_boundary"] and not r["leaked"])
    # What the flag cannot see: cases it passes which still report the wrong
    # size class. This is the limit of the criterion, not evidence for it.
    passed = [r for r in ok if not r["touches_voi_boundary"] and not r["leaked"]]
    missed_class = sum(1 for r in passed if r["class_changed"])
    flag = {
        "cases_measured": len(ok),
        "flagged_and_leaked": tp,
        "flagged_not_leaked": fp,
        "not_flagged_but_leaked": fn,
        "neither": tn,
        "sensitivity_to_leak": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity_to_leak": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "passed_cases": len(passed),
        "passed_but_wrong_class": missed_class,
        "passed_but_wrong_class_rate": round(missed_class / len(passed), 4) if passed else None,
    }

    def gap_of(r: dict) -> float | None:
        if r.get("binary_max_diameter_mm") is None:
            return None
        return abs(float(r["grayscale_max_diameter_mm"])
                   - float(r["binary_max_diameter_mm"]))

    with_gap = [(gap_of(r), bool(r["class_changed"])) for r in passed
                if gap_of(r) is not None]
    g_tp = sum(1 for g, wrong in with_gap if g > GAP_MM and wrong)
    g_fn = sum(1 for g, wrong in with_gap if g <= GAP_MM and wrong)
    g_fp = sum(1 for g, wrong in with_gap if g > GAP_MM and not wrong)
    g_tn = sum(1 for g, wrong in with_gap if g <= GAP_MM and not wrong)
    free_gaps = [gap_of(r) for r in passed
                 if r["geometry"] == "air" and gap_of(r) is not None]
    near_gaps = [gap_of(r) for r in passed
                 if r["geometry"] in ("wall_tangent", "fold")
                 and gap_of(r) is not None]
    flag["second_signal"] = {
        "what": ("the disagreement between the grayscale and the binary "
                 "diameter, in mm"),
        "threshold_mm": GAP_MM,
        "threshold_chosen_on": ("these phantoms; it is not validated on any "
                                "other data"),
        "applies_to": "the measurements the volume-of-interest flag passes",
        "cases": len(with_gap),
        "wrong_class": g_tp + g_fn,
        "caught": g_tp,
        "missed": g_fn,
        "sensitivity_to_wrong_class": round(g_tp / (g_tp + g_fn), 4) if (g_tp + g_fn) else None,
        "specificity": round(g_tn / (g_tn + g_fp), 4) if (g_tn + g_fp) else None,
        "median_gap_free_mm": round(float(np.median(free_gaps)), 3) if free_gaps else None,
        "max_gap_free_mm": round(max(free_gaps), 3) if free_gaps else None,
        "median_gap_near_tissue_mm": round(float(np.median(near_gaps)), 3) if near_gaps else None,
        "min_gap_near_tissue_mm": round(min(near_gaps), 3) if near_gaps else None,
    }
    # Could sphericity be the second criterion, for the attached lesions the
    # flag passes? Only if an attached lesion were less spherical than every
    # lesion that is genuinely not a sphere. The three numbers that settle it
    # are computed here rather than argued.
    def sph(rows_in: list[dict]) -> list[float]:
        return [float(r["sphericity"]) for r in rows_in if r.get("sphericity")]

    free_sph = sph([r for r in passed if r["geometry"] == "air"
                    and r["phantom"].startswith("sphere")])
    free_ell = sph([r for r in passed if r["geometry"] == "air"
                    and r["phantom"].startswith("ellipsoid")])
    attached = sph([r for r in passed
                    if r["geometry"] in ("wall_tangent", "fold")])
    return {
        "noise_source": noise_file,
        "refusal_flag": flag,
        "sphericity": {
            "free_sphere_min": round(min(free_sph), 4) if free_sph else None,
            "free_ellipsoid_min": round(min(free_ell), 4) if free_ell else None,
            "free_ellipsoid_max": round(max(free_ell), 4) if free_ell else None,
            "attached_max": round(max(attached), 4) if attached else None,
            "separates_attached_from_elongated": bool(
                free_ell and attached and max(attached) < min(free_ell)),
        },
        "materials_hu": {"lesion": SOFT_HU, "air": AIR_HU, "fluid": FLUID_HU,
                         "wall": WALL_HU, "beyond_the_wall": FAT_HU},
        "wall_thickness_mm": WALL_THICKNESS_MM,
        "wall_thickness_thick_mm": WALL_THICKNESS_THICK_MM,
        "noise_sd_used_hu": float(noise["soft_sd_p10_median_over_series_hu"]),
        "noise_rho1_used": float(noise["acf_lag1_median_over_series"]),
        "repeats_per_noisy_case": max(r["repeat"] for r in rows) + 1,
        "thresholds_mm": list(THRESHOLDS),
        "leak_criterion": "mesh volume above twice the true volume",
        "conditions": out,
    }


if __name__ == "__main__":
    raise SystemExit(main())
