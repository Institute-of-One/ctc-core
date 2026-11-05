"""Which table is the paper's, does it still say what the paper says, and
does the paper still lead with the set it is supposed to lead with?

results/tables/ holds several files per experiment and the plainest names are
not the reported ones. eval_hqcolon.csv is a 2026-09-10 hand run that predates
two columns the evaluator now writes and gives median Dice 0.8781 where the
paper reports 0.882; centerline_batch.csv is the prototype baseline, the
"before" column of the pipeline table, not the result. run_all.py writes
neither name. A reader who opens the directory and takes the obvious file gets
numbers that are not in the manuscript, which is what happened on 2026-11-04.

So this does two things a README alone cannot:

* every CSV in the directory must be accounted for - as a reported table, an
  ablation the paper prints, an input, or an orphan left by a hand run. A file
  nobody has classified is a finding, because that is how the trap was set.
* the reported tables must still produce the numbers the manuscript prints.
  A register that only lists names cannot notice the build being pointed at the
  wrong file; recomputing the headline values can. The expected values below
  were measured first and then written down, not asserted and then chased.

  python scripts/check_tables.py
  python scripts/check_tables.py --write-readme
  python scripts/check_tables.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

import run_all as run_all_module  # noqa: E402  (scripts/ is on sys.path)
from run_all import OUTPUTS, STAGES, commands  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DEVELOPMENT_PATIENTS = {"0001", "0003", "0004", "0007", "0030"}

# role: reported = the manuscript's numbers come from it
#       ablation = printed in the manuscript as a comparison column
#       input    = feeds another step
#       audit    = written by the build as a record; no later step reads it
#       orphan   = left by a hand run; run_all.py does not write it
#
# Every role but "orphan" is a claim that run_all.py writes the file, and
# build_coverage() below holds the register to it. That check found five
# reported files with no stage at all -- the threshold sweep, the gas
# attenuation, the coverage analysis, the acquisition strata and the Table 4
# comparator -- each with its script named right here in the register, and
# none of them in the plan that calls itself the whole reproduction.
REGISTER: dict[str, tuple[str, str]] = {
    "eval_hqcolon_auto.csv": ("reported", "Reference evaluation. Table 3: restricted to the 13 patients not inspected it is the primary column, whole it is the development-tuned column"),
    "centerline_auto.csv": ("reported", "Centerline run, 60 series. The 'after' column of Table 2"),
    "centerline_batch.csv": ("reported", "Prototype baseline, 60 series. The 'before' column of Table 2"),
    "indices.csv": ("reported", "Quantitative indices, 60 series"),
    "prone_supine_agreement.csv": ("reported", "ICC and limits of agreement, 28 pairs"),
    "polyp_phantom_accuracy.csv": ("reported", "Phantom polyp measurement"),
    "centerline_perturbation.csv": ("reported", "Centerline perturbation experiment"),
    "cohort_characteristics.csv": ("reported", "Cohort table"),
    "acquisition.csv": ("reported", "Acquisition parameters"),
    "clustered_statistics.csv": ("reported", "Patient-cluster intervals (scripts/stats_clustered.py)"),
    "clustered_statistics.json": ("input", "The resampling unit, the number of bootstrap resamples, the seed and the criteria the intervals were computed under"),
    "coverage_anatomy.csv": ("reported", "Coverage by position along the reference centerline, and the fluid or collapsed volume the gas mask excludes. Tables 8 and 9 (scripts/analyse_coverage_anatomy.py)"),
    "coverage_anatomy.json": ("input", "Per-series orientation decisions and the settings of the coverage analysis"),
    "stratified_acquisition.csv": ("reported", "The results by manufacturer, kernel, slice thickness, tube voltage and tube current. Supplementary file 6 (scripts/stratify_acquisition.py)"),
    "stratified_acquisition.json": ("input", "The collinearity of the acquisition axes and the recorded Exposure distribution"),
    "threshold_sweep.csv": ("reported", "Each threshold varied one at a time, masks and path rebuilt per setting. Table 10 (scripts/sweep_thresholds.py)"),
    "threshold_sweep.json": ("input", "The settings and range of the threshold sweep"),
    "gas_attenuation.csv": ("reported", "Attenuation of the colonic gas inside the reference mask, and the share each candidate threshold admits. Table 10 and the Methods (scripts/measure_gas_attenuation.py)"),
    "gas_attenuation.json": ("input", "The candidate thresholds and what the gas measurement covers"),
    "eval_hqcolon_batch.csv": ("reported", "The 2026-08 prototype configuration evaluated against the same HQColon reference, from its own masks and paths in data/centerline/batch. Table 4, the comparator"),
    "image_noise.csv": ("reported", "Image noise measured in the cohort itself: the local standard deviation of homogeneous soft-tissue windows and its in-plane autocorrelation. The Methods, and the level injected into the phantoms (scripts/measure_image_noise.py)"),
    "image_noise.json": ("input", "The noise estimator's window, slice sampling and summary over series; read by eval_polyp_realism.py, which has no noise level of its own"),
    "polyp_realism.csv": ("reported", "Polyp measurement under noise, a fluid interface and adhesion to a wall or fold. Table 11 and Supplementary file 8 (scripts/eval_polyp_realism.py)"),
    "polyp_realism.json": ("input", "Per-condition summary of the realism sweep and the behaviour of the refusal flag"),
    "cohort_lesions.csv": ("reported", "Every annotated lesion in the cohort, decoded from the collection's published column key. Table 12 (scripts/report_cohort_lesions.py)"),
    "cohort_lesions.json": ("input", "What the annotation carries and what it lacks, counted"),
    "eval_hqcolon_fill_bbox.csv": ("ablation", "3-D fill only"),
    "eval_hqcolon_fill_slicewise.csv": ("ablation", "Slice-wise fill only"),
    "eval_hqcolon_seed_extremes.csv": ("ablation", "Previous seed rule"),
    "centerline_fill_bbox.csv": ("ablation", "3-D fill only, centerline side"),
    "centerline_fill_slicewise.csv": ("ablation", "Slice-wise fill only, centerline side"),
    "centerline_seed_extremes.csv": ("ablation", "Previous seed rule, centerline side"),
    "hqcolon_overlap.csv": ("input", "Which series have an HQColon reference"),
    "patient_position.csv": ("input", "Prone/supine recovered from DICOM"),
    "eval_hqcolon.csv": ("orphan", "2026-09-10 hand run. Missing colon_within_30mm_frac and colon_beyond_60mm_frac; Dice 0.8781, not the paper's 0.882. run_all.py does not write it"),
    "eval_hqcolon_slicewise.csv": ("orphan", "Hand run, superseded by eval_hqcolon_fill_slicewise.csv"),
    "eval_revert.csv": ("orphan", "Hand run during the coverage diagnosis"),
    "eval_w6.csv": ("orphan", "Core-radius sweep, 6 mm"),
    "eval_w12.csv": ("orphan", "Core-radius sweep, 12 mm"),
    "eval_w25.csv": ("orphan", "Core-radius sweep, 25 mm"),
    "centerline_corrected.csv": ("orphan", "Hand run, superseded by centerline_auto.csv"),
    "centerline_gated.csv": ("orphan", "Hand run during the gate experiment"),
    "centerline_gate_effect.csv": ("orphan", "Gate experiment summary"),
    "centerline_revert.csv": ("orphan", "Hand run, 26 series only"),
    "centerline_slicewise.csv": ("orphan", "Hand run, superseded"),
    "centerline_w6.csv": ("orphan", "Core-radius sweep, 6 mm"),
    "centerline_w12.csv": ("orphan", "Core-radius sweep, 12 mm"),
    "centerline_w25.csv": ("orphan", "Core-radius sweep, 25 mm"),
    "centerline_comparison.csv": ("orphan", "compare_centerline_configs.py output, baseline vs candidate"),
    "pair_reference_status.csv": ("audit", "The per-patient reference and coverage audit behind Table 4, written by the tables stage. No later step reads it: make_tables.py calls pair_reference_status.build() live"),
    "batch_metrics_2026-08.csv": ("orphan", "August batch metrics"),
    "mask_diagnosis.csv": ("orphan", "Mask failure diagnosis"),
    "seed_rule_comparison.csv": ("orphan", "Seed rule diagnosis"),
    "eval_hqcolon_fill_bbox_subset.csv": ("orphan", "Hand run"),
}

# Measured on 2026-11-04 from the files named above, then written here.
EXPECTED: list[tuple[str, str, str, float, float]] = [
    # file, column, statistic, value, tolerance
    ("eval_hqcolon_auto.csv", "dice_lumen_vs_gas", "median", 0.8822, 5e-4),
    ("eval_hqcolon_auto.csv", "colon_within_30mm_frac", "median", 0.8600, 5e-4),
    ("eval_hqcolon_auto.csv", "colon_beyond_60mm_frac", "median", 0.0054, 5e-4),
    ("eval_hqcolon_auto.csv", "length_ratio_gf", "median", 0.9750, 5e-4),
    ("eval_hqcolon_auto.csv", "colon_beyond_60mm_frac", "n_coverage_pass", 15, 0),
    ("eval_hqcolon_auto.csv", "dice_lumen_vs_gas", "n_rows", 26, 0),
    ("centerline_auto.csv", "length_mm", "n_at_least_600", 56, 0),
    ("centerline_auto.csv", "length_mm", "median", 1565.7, 0.05),
    ("centerline_auto.csv", "length_mm", "n_rows", 60, 0),
    ("centerline_batch.csv", "length_mm", "n_at_least_600", 41, 0),
    ("centerline_batch.csv", "length_mm", "median", 908.5, 0.05),
]

# The cohort-wide figures restricted to the 25 patients not inspected, which
# the Results now give beside the 60-series ones: the seed rule was chosen with
# the reference in view, so those are development-tuned too.
EXPECTED_COHORT_SUBSET: list[tuple[str, str, float, float]] = [
    ("length_mm", "n_at_least_600", 46, 0),
    ("length_mm", "median", 1387.7, 0.05),
    ("length_mm", "n_rows", 50, 0),
]

# The same, restricted to the 18 series of the 13 patients not inspected.
EXPECTED_SUBSET: list[tuple[str, str, float, float]] = [
    ("dice_lumen_vs_gas", "median", 0.9187, 5e-4),
    ("colon_within_30mm_frac", "median", 0.6954, 5e-4),
    ("length_ratio_gf", "median", 0.9095, 5e-4),
    ("colon_beyond_60mm_frac", "n_coverage_pass", 8, 0),
    # Headline since the swap: this is the number that moves most between the
    # two sets, from 0.5 % to 19.3 %.
    ("colon_beyond_60mm_frac", "median", 0.1928, 5e-4),
    ("dice_lumen_vs_gas", "n_rows", 18, 0),
]

# What the paper must still be doing with those two sets. The values here are
# the ones the manuscript prints; they are checked against the tables by
# EXPECTED and EXPECTED_SUBSET above, so this part is only about which one the
# paper leads with.
PRIMARY_DICE = "0.919"
TUNED_DICE = "0.882"
PRIMARY_OPENS_RESULTS = "The primary analysis is the 18 series"
TUNED_FOLLOWS_IN_RESULTS = "Over all 26 reference series"
NO_HELD_OUT = "no held-out test set exists"
TUNED_LABEL = "development-tuned"

# The 2026-08 prototype against the same reference: Table 4, the comparator.
# Measured on 2026-11-04 from eval_hqcolon_batch.csv, over the 17 of the 18
# primary series it produced a centerline for.
EXPECTED_PROTOTYPE: list[tuple[str, str, float, float]] = [
    ("dice_lumen_vs_gas", "n_rows", 18, 0),
    ("dice_lumen_vs_gas", "n_evaluated", 17, 0),
    ("dice_lumen_vs_gas", "median", 0.9830, 5e-4),
    ("dice_lumen_vs_gas", "n_segmentation_failures", 5, 0),
    ("colon_within_30mm_frac", "median", 0.3881, 5e-4),
    ("colon_beyond_60mm_frac", "median", 0.5032, 5e-4),
    ("colon_beyond_60mm_frac", "n_coverage_pass", 0, 0),
    ("length_ratio_gf", "median", 0.5390, 5e-4),
]

# The traced length against the published range, which is the external half of
# the comparison. Measured the same day from centerline_auto.csv and
# centerline_batch.csv, in centimetres.
PUBLISHED_RANGE_CM = (109.7, 195.9)
EXPECTED_LENGTH_BAND: list[tuple[str, str, int]] = [
    # file, what, count
    ("centerline_auto.csv", "inside", 30),
    ("centerline_auto.csv", "shorter", 18),
    ("centerline_auto.csv", "longer", 12),
    ("centerline_batch.csv", "inside", 14),
    ("centerline_batch.csv", "shorter", 41),
    ("centerline_batch.csv", "longer", 0),
    ("centerline_batch.csv", "no_path", 5),
]

# Comments 2 and 3, from coverage_anatomy.csv over the 18 primary series.
# Measured on 2026-11-04 and then written down.
COVERAGE_WHOLE: list[tuple[str, str, float, float]] = [
    ("fluid_or_collapsed_frac", "median", 0.0413, 5e-4),
    ("dice_lumen_vs_gas_in_part", "median", 0.9187, 5e-4),
    ("beyond_60mm_frac", "n_rows", 18, 0),
]
# Over the series that have any colon beyond 60 mm of the path.
COVERAGE_MISSED: list[tuple[str, str, float, float]] = [
    ("beyond_60mm_that_was_gas_frac", "median", 0.9708, 5e-4),
    ("beyond_60mm_that_was_gas_frac", "n_rows", 12, 0),
]
# The finding: where along the colon the coverage fails. Part 1 is the rectal
# end. A change that flattened this profile would show up nowhere else.
COVERAGE_BY_PART: list[tuple[int, int]] = [
    (1, 0), (2, 1), (3, 7), (4, 7), (5, 6), (6, 8)]

# Comment 6. The sweep, by setting: (setting, column, statistic, value, tol).
# The coverage counts are the finding; the failures at body -400 are what the
# medians of that row step over.
THRESHOLD_SWEEP: list[tuple[str, str, str, float, float]] = [
    ("reported", "colon_beyond_60mm_frac", "n_coverage_pass", 15, 0),
    ("gas -800", "colon_beyond_60mm_frac", "n_coverage_pass", 20, 0),
    ("gas -600", "colon_beyond_60mm_frac", "n_coverage_pass", 10, 0),
    ("body -200", "colon_beyond_60mm_frac", "n_coverage_pass", 15, 0),
    ("body -400", "colon_beyond_60mm_frac", "n_coverage_pass", 14, 0),
    ("body -400", "dice_lumen_vs_gas", "n_segmentation_failures", 4, 0),
    ("reported", "dice_lumen_vs_gas", "n_segmentation_failures", 0, 0),
    ("gas -800", "colon_within_30mm_frac", "median", 0.9521, 5e-4),
    ("reported", "colon_within_30mm_frac", "median", 0.8600, 5e-4),
    ("gas -600", "colon_within_30mm_frac", "median", 0.6081, 5e-4),
    ("reported", "dice_lumen_vs_gas", "n_rows", 26, 0),
]

# The measurement that refused the expected explanation: the colonic gas sits
# far below every candidate threshold, so the sensitivity is not the threshold
# missing gas. The paragraph in the Results says so in those terms.
GAS_ATTENUATION: list[tuple[str, str, float, float]] = [
    ("gas_hu_median", "median", -988.35, 0.005),
    ("frac_below_800", "median", 0.9827, 5e-4),
    ("frac_below_700", "median", 0.9941, 5e-4),
    ("frac_below_600", "median", 0.9985, 5e-4),
    ("gas_hu_median", "n_rows", 26, 0),
]

# Image noise, measured in the cohort on 2026-11-05 and then written here.
# The phantom sweep reads these from the JSON rather than carrying a level of
# its own, so an unnoticed change here would silently change the experiment.
IMAGE_NOISE: list[tuple[str, str, float, float]] = [
    ("soft_sd_p10_hu", "median", 15.056, 5e-3),
    ("soft_sd_p50_hu", "median", 19.553, 5e-3),
    ("acf_lag1", "median", 0.1946, 5e-4),
    ("air_sd_median_hu", "median", 18.122, 5e-3),
    ("soft_sd_p10_hu", "n_rows", 60, 0),
]

# The realism sweep. What the manuscript says about it, recomputed from the
# rows: noise of the measured size changes nothing, and what the lesion
# touches changes everything.
REALISM: list[tuple[str, str, float, float]] = [
    # condition, quantity, value, tolerance
    ("air", "band_mm", 0.283, 5e-4),
    ("air_noise", "band_mm", 0.291, 5e-4),
    ("air_noise_white", "band_mm", 0.278, 5e-4),
    ("air_noise_double", "band_mm", 0.261, 5e-4),
    ("air_noise", "leak_rate", 0.0, 0),
    ("air_noise_double", "class_change_rate", 0.0, 0),
    ("fluid_noise", "leak_rate", 1.0, 0),
    ("wall_tangent_noise", "band_mm", 3.037, 5e-4),
    ("wall_tangent_noise", "leak_rate", 0.0, 0),
    ("wall_tangent_noise", "class_change_rate", 0.30, 5e-3),
    ("wall_sessile_noise", "leak_rate", 0.90, 5e-3),
    ("fold_noise", "band_mm", 5.208, 5e-4),
    ("fold_noise", "class_change_rate", 0.4444, 5e-4),
    ("fold_noise", "near_threshold_class_change_rate", 0.60, 5e-3),
    ("wall_tangent_noise", "near_threshold_class_change_rate", 0.3333, 5e-4),
]

# The refusal flag the Discussion recommends. A criterion a paper recommends
# and nobody measured is the worst kind of recommendation, so both what it
# catches and what it misses are pinned.
REALISM_FLAG: list[tuple[str, float, float]] = [
    ("cases_measured", 1650, 0),
    ("sensitivity_to_leak", 0.9657, 5e-4),
    ("specificity_to_leak", 0.9721, 5e-4),
    ("passed_but_wrong_class_rate", 0.1457, 5e-4),
    # The second signal, which the Discussion recommends alongside the first.
    ("second_signal_sensitivity", 1.0, 0),
    ("second_signal_specificity", 0.674, 5e-4),
    ("second_signal_wrong_class", 147, 0),
]

# The annotated lesions of the cohort, and the fields the annotation lacks.
COHORT_LESIONS: list[tuple[str, object]] = [
    ("lesions_in_cohort", 3),
    ("with_in_plane_coordinate", 0),
    ("with_supine_slice", 2),
    ("with_prone_slice", 1),
    ("with_no_slice_at_all", 1),
    ("reported_sizes_mm", [6, 9, 9]),
]
COHORT_LESIONS_LARGE_IN_COHORT = 0
# The cut the Discussion recommends for the second signal. It lives here as
# well as in the sweep because the manuscript quotes it.
SECOND_SIGNAL_GAP_MM = 1.0

COVERAGE_BEYOND_60_MAX = 0.05
LENGTH_MIN_MM = 600.0
# The same cut as scripts/make_tables.py: observed values leave a gap from
# 0.002 to 0.721, so nothing depends on where in that gap it sits.
SEG_FAILURE_DICE = 0.5


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def realism_quantity(rows: list[dict], quantity: str) -> float | None:
    """One summary of one condition of the realism sweep, from its own rows.

    A leaked segmentation is excluded from the error statistics and counted
    instead: averaging a leak into a diameter error would report a measurement
    where there was none. The script that writes the table makes the same
    split, and this recomputes it rather than reading its answer.
    """
    measured = [r for r in rows if _truthy(r.get("ok"))]
    usable = [r for r in measured if not _truthy(r.get("leaked"))]
    if quantity == "leak_rate":
        return sum(1 for r in measured if _truthy(r.get("leaked"))) / len(rows)
    if quantity == "flag_rate":
        return sum(1 for r in measured if _truthy(r.get("touches_voi_boundary"))) / len(rows)
    if not usable:
        return None
    if quantity == "band_mm":
        errs = sorted(abs(float(r["grayscale_diameter_error_mm"])) for r in usable)
        import numpy as _np
        return float(_np.percentile(errs, 95))
    if quantity == "class_change_rate":
        return sum(1 for r in usable if _truthy(r.get("class_changed"))) / len(usable)
    if quantity == "near_threshold_class_change_rate":
        near = [r for r in usable
                if min(abs(float(r["true_max_diameter_mm"]) - t)
                       for t in (6.0, 10.0)) <= 0.5]
        if not near:
            return None
        return sum(1 for r in near if _truthy(r.get("class_changed"))) / len(near)
    raise KeyError(quantity)


def flag_quantity(rows: list[dict], name: str) -> float | None:
    """How well the refusal flag predicts the leak, from the rows themselves."""
    ok = [r for r in rows if _truthy(r.get("ok"))]
    if not ok:
        return None
    tp = sum(1 for r in ok if _truthy(r["touches_voi_boundary"]) and _truthy(r["leaked"]))
    fp = sum(1 for r in ok if _truthy(r["touches_voi_boundary"]) and not _truthy(r["leaked"]))
    fn = sum(1 for r in ok if not _truthy(r["touches_voi_boundary"]) and _truthy(r["leaked"]))
    tn = sum(1 for r in ok if not _truthy(r["touches_voi_boundary"])
             and not _truthy(r["leaked"]))
    passed = [r for r in ok if not _truthy(r["touches_voi_boundary"])
              and not _truthy(r["leaked"])]
    if name == "cases_measured":
        return float(len(ok))
    if name == "sensitivity_to_leak":
        return tp / (tp + fn) if (tp + fn) else None
    if name == "specificity_to_leak":
        return tn / (tn + fp) if (tn + fp) else None
    if name == "passed_but_wrong_class_rate":
        if not passed:
            return None
        return sum(1 for r in passed if _truthy(r.get("class_changed"))) / len(passed)

    # The second signal: the two diameter routes disagree where the grayscale
    # surface has reached a neighbour. Recomputed from the two columns rather
    # than read from the summary.
    def gap(r: dict) -> float | None:
        a, b = r.get("grayscale_max_diameter_mm"), r.get("binary_max_diameter_mm")
        if not a or not b:
            return None
        return abs(float(a) - float(b))

    pairs = [(gap(r), _truthy(r.get("class_changed"))) for r in passed
             if gap(r) is not None]
    wrong = sum(1 for _g, w in pairs if w)
    caught = sum(1 for g, w in pairs if w and g > SECOND_SIGNAL_GAP_MM)
    tn2 = sum(1 for g, w in pairs if not w and g <= SECOND_SIGNAL_GAP_MM)
    fp2 = sum(1 for g, w in pairs if not w and g > SECOND_SIGNAL_GAP_MM)
    if name == "second_signal_wrong_class":
        return float(wrong)
    if name == "second_signal_sensitivity":
        return caught / wrong if wrong else None
    if name == "second_signal_specificity":
        return tn2 / (tn2 + fp2) if (tn2 + fp2) else None
    raise KeyError(name)


def as_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def compute(rows: list[dict], column: str, statistic: str):
    if statistic == "n_rows":
        return float(len(rows))
    values = [as_float(r.get(column)) for r in rows]
    present = [v for v in values if v is not None]
    if statistic == "median":
        return statistics.median(present) if present else None
    if statistic == "n_coverage_pass":
        return float(sum(1 for v in present if v < COVERAGE_BEYOND_60_MAX))
    if statistic == "n_at_least_600":
        # a series with no length did not produce a centerline, so it fails
        return float(sum(1 for v in values if v is not None and v >= LENGTH_MIN_MM))
    if statistic == "n_evaluated":
        # Rows that carry a value for this column at all. For the prototype
        # the difference from n_rows is the series it produced no centerline
        # for, which must not be counted as evaluated.
        return float(len(present))
    if statistic == "n_segmentation_failures":
        return float(sum(1 for v in present if v < SEG_FAILURE_DICE))
    raise ValueError("unknown statistic %r" % statistic)


def subset(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if (r.get("PatientID") or "").rsplit(".", 1)[-1]
            not in DEVELOPMENT_PATIENTS]


def check(tables: Path = TABLES, register=None, expected=None,
          expected_subset=None) -> list[str]:
    register = REGISTER if register is None else register
    expected = EXPECTED if expected is None else expected
    expected_subset = (EXPECTED_SUBSET if expected_subset is None
                       else expected_subset)
    bad: list[str] = []

    # JSON as well as CSV: the sidecars carry the settings a result was
    # produced under (the seed of the bootstrap, the thresholds of the sweep),
    # and one of them sat unclassified because this glob only saw tables.
    on_disk = {p.name for p in tables.glob("*.csv")} | {p.name for p in tables.glob("*.json")}
    for name in sorted(on_disk - set(register)):
        bad.append("%s is in results/tables but not classified in the register; "
                   "an unclassified file is how the wrong table gets read" % name)
    for name, (role, _) in sorted(register.items()):
        if role in ("reported", "ablation") and name not in on_disk:
            bad.append("%s is registered as %s but is not on disk" % (name, role))

    cache: dict[str, list[dict]] = {}
    for name, column, statistic, want, tol in expected:
        path = tables / name
        if not path.exists():
            bad.append("%s is missing, so the paper's %s cannot be checked"
                       % (name, statistic))
            continue
        rows = cache.setdefault(name, read(path))
        got = compute(rows, column, statistic)
        if got is None:
            bad.append("%s has no usable values in %r" % (name, column))
        elif abs(got - want) > tol:
            bad.append("%s %s of %s is %.4f, the manuscript reports %.4f"
                       % (name, statistic, column, got, want))

    hq = tables / "eval_hqcolon_auto.csv"
    if hq.exists():
        sub = subset(cache.setdefault(hq.name, read(hq)))
        for column, statistic, want, tol in expected_subset:
            got = compute(sub, column, statistic)
            if got is None:
                bad.append("the 18-series subset has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("18-series subset: %s of %s is %.4f, the manuscript "
                           "reports %.4f" % (statistic, column, got, want))

    sweep = tables / "threshold_sweep.csv"
    if sweep.exists():
        allrows = cache.setdefault(sweep.name, read(sweep))
        for setting, column, statistic, want, tol in THRESHOLD_SWEEP:
            here = [r for r in allrows if r.get("setting") == setting]
            if not here:
                bad.append("the threshold sweep has no setting %r" % setting)
                continue
            got = compute(here, column, statistic)
            if got is None:
                bad.append("the sweep setting %r has no usable %r"
                           % (setting, column))
            elif abs(got - want) > tol:
                bad.append("the threshold sweep (Table 10): %s of %s at %r is "
                           "%.4f, the manuscript reports %.4f"
                           % (statistic, column, setting, got, want))

    gasf = tables / "gas_attenuation.csv"
    if gasf.exists():
        rows_g = [r for r in cache.setdefault(gasf.name, read(gasf))
                  if (r.get("gas_hu_median") or "").strip()]
        for column, statistic, want, tol in GAS_ATTENUATION:
            got = compute(rows_g, column, statistic)
            if got is None:
                bad.append("the gas attenuation table has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("gas attenuation: %s of %s is %.4f, the manuscript "
                           "reports %.4f" % (statistic, column, got, want))

    noisef = tables / "image_noise.csv"
    if noisef.exists():
        rows_n = cache.setdefault(noisef.name, read(noisef))
        for column, statistic, want, tol in IMAGE_NOISE:
            got = compute(rows_n, column, statistic)
            if got is None:
                bad.append("the image noise table has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("image noise: %s of %s is %.4f, the manuscript "
                           "reports %.4f" % (statistic, column, got, want))

    realf = tables / "polyp_realism.csv"
    if realf.exists():
        rows_r = cache.setdefault(realf.name, read(realf))
        # Recomputed from the rows, not read from the JSON summary: a summary
        # that disagreed with its own rows would otherwise check out.
        for cond, quantity, want, tol in REALISM:
            sub = [r for r in rows_r if r.get("condition") == cond]
            if not sub:
                bad.append("the realism sweep has no condition %r" % cond)
                continue
            got = realism_quantity(sub, quantity)
            if got is None:
                bad.append("the realism condition %r has no usable %r"
                           % (cond, quantity))
            elif abs(got - want) > tol:
                bad.append("the realism sweep (Table 11): %s at %r is %.4f, "
                           "the manuscript reports %.4f"
                           % (quantity, cond, got, want))
        for name, want, tol in REALISM_FLAG:
            got = flag_quantity(rows_r, name)
            if got is None:
                bad.append("the refusal flag has no usable %r" % name)
            elif abs(got - want) > tol:
                bad.append("the refusal flag: %s is %.4f, the manuscript "
                           "reports %.4f" % (name, got, want))

    lesf = tables / "cohort_lesions.csv"
    if lesf.exists():
        rows_l = cache.setdefault(lesf.name, read(lesf))
        got_l = {
            "lesions_in_cohort": len(rows_l),
            "with_in_plane_coordinate": 0,
            "with_supine_slice": sum(1 for r in rows_l if r.get("slice_supine")),
            "with_prone_slice": sum(1 for r in rows_l if r.get("slice_prone")),
            "with_no_slice_at_all": sum(
                1 for r in rows_l
                if not r.get("slice_supine") and not r.get("slice_prone")),
            "reported_sizes_mm": sorted(
                int(r["reported_size_mm"]) for r in rows_l if r.get("reported_size_mm")),
        }
        for name, want in COHORT_LESIONS:
            if got_l[name] != want:
                bad.append("the cohort lesions (Table 12): %s is %r, the "
                           "manuscript reports %r" % (name, got_l[name], want))
        large = sum(1 for r in rows_l if r.get("sheet_size_band") == ">=10 mm")
        if large != COHORT_LESIONS_LARGE_IN_COHORT:
            bad.append("the cohort lesions: %d are in the >=10 mm sheet, the "
                       "manuscript reports %d -- the claim that the cohort "
                       "cannot test the threshold that matters most turns on "
                       "this being zero"
                       % (large, COHORT_LESIONS_LARGE_IN_COHORT))

    cov = tables / "coverage_anatomy.csv"
    if cov.exists():
        rows = subset(cache.setdefault(cov.name, read(cov)))
        whole = [r for r in rows if r.get("segment") == "whole colon"]
        for column, statistic, want, tol in COVERAGE_WHOLE:
            got = compute(whole, column, statistic)
            if got is None:
                bad.append("the coverage analysis has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("coverage (Tables 8 and 9): %s of %s is %.4f, the "
                           "manuscript reports %.4f"
                           % (statistic, column, got, want))
        missed = [r for r in whole
                  if (as_float(r.get("beyond_60mm_frac")) or 0.0) > 0]
        for column, statistic, want, tol in COVERAGE_MISSED:
            got = compute(missed, column, statistic)
            if got is None:
                bad.append("no series with colon beyond 60 mm carries %r" % column)
            elif abs(got - want) > tol:
                bad.append("coverage (Table 9): %s of %s over the series with "
                           "a shortfall is %.4f, the manuscript reports %.4f"
                           % (statistic, column, got, want))
        for part, want in COVERAGE_BY_PART:
            seg = "part %d of 6 from the rectum" % part
            here = [r for r in rows if r.get("segment") == seg]
            if not here:
                bad.append("the coverage analysis has no %s" % seg)
                continue
            got = sum(1 for r in here
                      if (as_float(r.get("beyond_60mm_frac")) or 0.0)
                      >= COVERAGE_BEYOND_60_MAX)
            if got != want:
                bad.append("coverage (Table 8): %d of %d series fail the "
                           "criterion in part %d, the manuscript reports %d"
                           % (got, len(here), part, want))

    proto = tables / "eval_hqcolon_batch.csv"
    if proto.exists():
        sub = subset(cache.setdefault(proto.name, read(proto)))
        for column, statistic, want, tol in EXPECTED_PROTOTYPE:
            got = compute(sub, column, statistic)
            if got is None:
                bad.append("the prototype's primary set has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("the prototype (Table 4): %s of %s is %.4f, the "
                           "manuscript reports %.4f"
                           % (statistic, column, got, want))

    lo, hi = PUBLISHED_RANGE_CM
    for name, what, want in EXPECTED_LENGTH_BAND:
        path = tables / name
        if not path.exists():
            continue
        lengths = [as_float(r.get("length_mm")) for r in
                   cache.setdefault(name, read(path))]
        cm = [v / 10.0 for v in lengths if v is not None]
        counts = {"inside": sum(1 for v in cm if lo <= v <= hi),
                  "shorter": sum(1 for v in cm if v < lo),
                  "longer": sum(1 for v in cm if v > hi),
                  "no_path": sum(1 for v in lengths if v is None)}
        if counts[what] != want:
            bad.append("%s: %d series are %s the published range %.1f-%.1f cm, "
                       "the manuscript reports %d"
                       % (name, counts[what], what, lo, hi, want))

    cl = tables / "centerline_auto.csv"
    if cl.exists():
        sub = subset(cache.setdefault(cl.name, read(cl)))
        for column, statistic, want, tol in EXPECTED_COHORT_SUBSET:
            got = compute(sub, column, statistic)
            if got is None:
                bad.append("the 50-series cohort subset has no usable %r" % column)
            elif abs(got - want) > tol:
                bad.append("50-series cohort subset: %s of %s is %.4f, the "
                           "manuscript reports %.4f"
                           % (statistic, column, got, want))
    return bad


def build_coverage(register=None, outputs=None, results: Path = RESULTS) -> list[str]:
    """Count the shipped artefacts one by one against the steps that make them.

    A register that names a script is not a build. Until 2026-11-05 five
    reported files -- threshold_sweep, gas_attenuation, coverage_anatomy,
    stratified_acquisition and eval_hqcolon_batch -- were described here, cited
    in the manuscript, and written by nothing that ``run_all.py`` runs. The
    reader who followed the README got the committed copies back unchanged and
    no sign that four of the tables had never been recomputed.

    So: every registered file that is not declared an orphan must be claimed by
    exactly one stage, every orphan by none, every file claimed at the top
    level of results/tables must be in the register, and every file shipped
    under results/tables/manuscript and results/figures, at any depth, must be
    claimed by some stage. The register classifies the top level of
    results/tables only -- the manuscript tables below it are named by the
    stage that writes them and checked against the shipped tree instead.

    All of this is a check on declarations. That a stage's commands really
    produce what the stage claims is settled by running them:
    scripts/check_rerun.py deletes the claimed files and requires them back.
    """
    register = REGISTER if register is None else register
    outputs = OUTPUTS if outputs is None else outputs
    bad: list[str] = []

    claims: dict[str, list[str]] = {}
    for stage, files in outputs.items():
        for f in files:
            claims.setdefault(f, []).append(stage)

    for stage in STAGES:
        if stage not in outputs:
            bad.append("the %s stage has no entry in run_all.OUTPUTS, so "
                       "nothing knows what it writes" % stage)
    for stage in outputs:
        if stage not in STAGES:
            bad.append("run_all.OUTPUTS names %r, which is not a stage" % stage)

    for name, (role, _) in sorted(register.items()):
        where = claims.get("tables/%s" % name, [])
        if role == "orphan":
            if where:
                bad.append("%s is registered as an orphan, but the %s stage "
                           "writes it" % (name, ", ".join(where)))
        elif not where:
            bad.append("%s is registered as %s and no stage of run_all.py "
                       "writes it" % (name, role))
        elif len(where) > 1:
            bad.append("%s is written by more than one stage (%s)"
                       % (name, ", ".join(where)))

    for path, where in sorted(claims.items()):
        head, _, tail = path.partition("/")
        if head == "tables" and "/" not in tail and tail not in register:
            bad.append("the %s stage writes %s, which the register does not "
                       "classify" % (where[0], path))

    # Every shipped file anywhere under results/, not two chosen folders at
    # their first level: a result in a directory nobody thought of is exactly
    # the one no stage writes. A file at the top of results/tables may instead
    # be accounted for by the register, which is what classifies that level.
    if results.is_dir():
        for p in sorted(results.rglob("*")):
            if not p.is_file() or p.name == "README.md":
                continue
            rel = p.relative_to(results).as_posix()
            registered = (rel.count("/") == 1 and rel.startswith("tables/")
                          and p.name in register)
            if rel not in claims and not registered:
                bad.append("results/%s is shipped but no stage claims to "
                           "write it" % rel)

    # A claim is a declaration. This much of it can be checked without running
    # anything: the stage must at least be pointed at the directory it says it
    # writes into. Whether the commands really produce the files is settled by
    # scripts/check_rerun.py, which deletes them and requires them back.
    plan = commands(".", "results")
    for stage, files_ in sorted(outputs.items()):
        # Only the arguments that say where output goes count as evidence of
        # routing. An input path under results/ (--overlap, --position, the
        # noise summary the phantom sweep reads) says nothing about where the
        # stage writes, and counting it let a stage claim a file simply
        # because it reads another file from the same directory.
        # --tables is an input flag for most scripts that take it
        # (perturb_centerline, stats_clustered, stratify_acquisition,
        # make_vgp, make_case_figures all read from it), so it is evidence of
        # where output goes only for the two that write into that directory.
        writes_into_tables = ("make_tables.py", "pair_reference_status.py")
        paths = []
        for cmd in plan.get(stage, []):
            for flag, value in zip(cmd, cmd[1:], strict=False):
                out_flag = flag in ("--out", "--summary", "--results") or (
                    flag == "--tables" and cmd[0] in writes_into_tables)
                if out_flag:
                    value = value.replace("\\", "/")
                    if value.startswith("results"):
                        paths.append(value)
        for f in files_:
            target = "results/%s" % f
            stem = target.rsplit(".", 1)[0]
            # A command names the file, or the directory it goes in, or the
            # file it is the sidecar of: several scripts write their settings
            # beside the table with out.with_suffix(".json").
            if not any(target == a or target.startswith(a.rstrip("/") + "/")
                       or stem == a.rsplit(".", 1)[0] for a in paths):
                bad.append("the %s stage claims %s, but none of its commands "
                           "is pointed at that path" % (stage, f))

    # Both stage tables a reader is given. They were four and six stages short
    # of the build when this was written, which is how a reader concludes that
    # a table nothing regenerates is not part of the reproduction. The stage
    # has to open a row of the table: a bare "| strata |" anywhere, in a code
    # block or in some other column, was enough before.
    # run_all.py's own docstring is the third stage table, and the one a
    # reader sees first: --help prints it. It was two stages short while the
    # other two were complete.
    doc = run_all_module.__doc__ or ""
    for stage in STAGES:
        if not re.search(r"(?m)^%s\s" % re.escape(stage), doc):
            bad.append("the docstring run_all.py --help prints has no row "
                       "for the %s stage" % stage)

    for doc in ("README.md", "docs/USER_MANUAL.md"):
        path = results.parent / doc
        if not path.is_file():
            bad.append("%s is missing, so the stages are documented nowhere" % doc)
            continue
        # Fenced code blocks are stripped first: a command line showing the
        # stage name is not the reader's table of stages, and the earlier
        # substring test counted it.
        # Fenced blocks of both kinds, and HTML comments, are removed first.
        # What is left is a heuristic about Markdown rather than a renderer:
        # a row whose first cell is a stage name, in any table, satisfies it.
        text = path.read_text(encoding="utf-8")
        for pattern in (r"(?ms)^```.*?^```", r"(?ms)^~~~.*?^~~~", r"(?s)<!--.*?-->"):
            text = re.sub(pattern, "", text)
        for stage in STAGES:
            if not re.search(r"(?m)^\|\s*%s\s*\|" % re.escape(stage), text):
                bad.append("%s has no table row for the %s stage" % (doc, stage))
    return bad


def names_the_committed_tree(text: str) -> bool:
    """Is this string a path with "results" as a segment of its own?

    "results", "results/tables/x", "../results", "./results", "x/results".
    One rule for every spelling, and in one place: the default-override rule
    and the walk over the rest of the module each had their own copy, the
    second was made segment-aware and the first was not, and a default that
    no stage overrides went unreported when it said "results" on its own.
    """
    return "results" in [p for p in text.replace("\\", "/").split("/") if p]


def stage_inputs(root: Path = ROOT) -> list[str]:
    """No stage may read the committed results when it is told to use another copy.

    ``run_all.py --results <dir>`` is how a reader re-runs without overwriting
    anything, and ``check_rerun.py`` is built on it. A script that resolves an
    input through a module-level constant pointing into ``results/`` ignores
    that flag: it is then computed from the committed tables and compared with
    them, and the comparison cannot fail. stats_clustered.py did exactly this
    until 2026-11-05, for four of its inputs.

    So: in any script a stage runs, a path built from such a constant may
    appear only as an argparse default whose flag the stage passes, or as a
    helper's parameter default with every call passing the directory.

    What it does not reach, deliberately: a helper called on a receiver other
    than ``self`` or this script's own module, and a path that arrives from
    another module or through a variable. Following every attribute call by
    name reported honest code -- ``csv.read()`` is not this script's
    ``read()`` -- and the alternative to the trade is a type inference this
    does not have. The run-time half of the guarantee is scripts/check_rerun.py.
    """
    bad: list[str] = []
    plan = commands(".", "results")
    scripts = sorted({cmd[0] for cmds in plan.values() for cmd in cmds})
    for name in scripts:
        path = root / "scripts" / name
        if not path.exists():
            bad.append("%s is run by a stage but is not in scripts/" % name)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        committed = {
            node.targets[0].id
            for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and "results" in ast.unparse(node.value)
        }
        # Where a committed path may legitimately appear: in the statement
        # that defines the constant, in an argparse default (the command line
        # overrides it, if the stage passes the flag), and as a helper's
        # parameter default (every call must pass the directory). Anywhere
        # else is a read of the committed tree.
        # Only the statement that defines a committed constant is exempt. The
        # earlier version exempted every module-level assignment, so
        # SUMMARY = TABLES / "x.csv" at the top of a script -- a read that
        # happens on import -- was not a finding.
        defining = {id(child) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in committed
                            for t in node.targets)
                    for child in ast.walk(node.value)}
        argparse_calls = [node for node in ast.walk(tree)
                          if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute)
                          and node.func.attr == "add_argument"]
        # The default only, not the whole call: a committed path reached
        # through type= or any other argument of add_argument is not
        # overridable by the command line, and exempting the call exempted it.
        in_argparse = {id(child) for node in argparse_calls
                       for kw in node.keywords if kw.arg == "default"
                       for child in ast.walk(kw.value)}
        param_defaults = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for default in list(node.args.defaults) + list(node.args.kw_defaults)
            if default is not None
            for child in ast.walk(default)
        }

        # Per invocation, not pooled: a script a stage runs twice, with the
        # flag on one command line only, leaves the other reading the
        # committed tree.
        invocations = [cmd for cmds in plan.values() for cmd in cmds
                       if cmd[0] == name]
        passed = {a for a in (invocations[0] if invocations else [])
                  if a.startswith("--")}
        for cmd in invocations[1:]:
            passed &= {a for a in cmd if a.startswith("--")}

        def points_at_committed(node: ast.AST, names: set[str] = committed) -> bool:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in names:
                    return True
                if (isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                        and names_the_committed_tree(child.value)):
                    return True
            return False

        # A default -- built from the constant or spelled out as text -- is
        # only overridable if the stage passes the flag.
        for node in argparse_calls:
            flags = {a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            if points_at_committed(node) and not (flags & passed):
                bad.append("%s defaults %s to the committed tree and no "
                           "stage passes that flag, so --results cannot move it"
                           % (name, "/".join(sorted(flags)) or "an argument"))

        for node in ast.walk(tree):
            if id(node) in in_argparse or id(node) in defining:
                continue
            if (isinstance(node, ast.Name) and node.id in committed
                    and isinstance(node.ctx, ast.Load)
                    and id(node) not in param_defaults):
                bad.append("%s uses %s outside an argparse default, so "
                           "--results cannot move it off the committed tree"
                           % (name, node.id))
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and names_the_committed_tree(node.value)):
                bad.append("%s names %r, which is the committed tree "
                           "whatever --results says" % (name, node.value[:50]))

        # The same hole one level down: a helper whose parameter defaults to
        # the committed directory is safe only while every call passes the
        # directory it should read. This is how the corrected stats_clustered
        # is written -- def read(name, tables=TABLES) -- so the rule has to
        # follow the calls rather than trust the shape.
        bad += _defaults_through_helpers(name, tree, committed)
    return sorted(set(bad))


def _defaults_through_helpers(script: str, tree: ast.AST,
                              committed: set[str]) -> list[str]:
    bad: list[str] = []
    # A method's first parameter is the receiver, which an attribute call
    # supplies before its arguments; a module-level function reached as
    # module.read(...) has no such shift. Counting positions without that
    # distinction reports self.read(name, tables) as if the directory were
    # missing.
    methods = {child.name
               for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               for child in node.body
               if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = node.args.posonlyargs + node.args.args
        risky: dict[str, int] = {}
        offset = len(params) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            if any(isinstance(c, ast.Name) and c.id in committed
                   for c in ast.walk(default)):
                risky[params[offset + i].arg] = offset + i
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults,
                                strict=False):
            if default is not None and any(isinstance(c, ast.Name)
                                           and c.id in committed
                                           for c in ast.walk(default)):
                risky[arg.arg] = -1
        if not risky:
            continue
        # The call sites are only checkable while the helper is called by its
        # own name. A reference to it that is not a call -- an alias, a
        # callback, an entry in a table of functions -- takes it somewhere
        # this rule cannot follow, so that is reported rather than passed.
        for ref in ast.walk(tree):
            if (isinstance(ref, ast.Name) and ref.id == node.name
                    and isinstance(ref.ctx, ast.Load)
                    and not any(isinstance(parent, ast.Call)
                                and parent.func is ref
                                for parent in ast.walk(tree))):
                bad.append("%s passes %s around rather than calling it, so "
                           "its committed default cannot be followed"
                           % (script, node.name))
                break
        for call in ast.walk(tree):
            # By bare name, and also as an attribute: self.read(...) and
            # module.read(...) are the same helper reached another way.
            # By bare name, by self.<name>, or through this script's own
            # module. Any attribute whose name happened to match was followed
            # before, so csv.read() would have been read as this helper and
            # reported for not passing an argument it does not have.
            own = {"self", script.removesuffix(".py")}
            if not (isinstance(call, ast.Call)
                    and ((isinstance(call.func, ast.Name)
                          and call.func.id == node.name)
                         or (isinstance(call.func, ast.Attribute)
                             and call.func.attr == node.name
                             and isinstance(call.func.value, ast.Name)
                             and call.func.value.id in own))):
                continue
            given = {kw.arg for kw in call.keywords}
            shift = 1 if (isinstance(call.func, ast.Attribute)
                          and node.name in methods) else 0
            for arg, index in risky.items():
                if arg in given or (index >= 0 and len(call.args) > index - shift):
                    continue
                bad.append("%s calls %s() without %s, so it reads the "
                           "committed tree whatever --results says"
                           % (script, node.name, arg))
    return bad


def write_readme(tables: Path = TABLES) -> Path:
    out = tables / "README.md"
    lines = ["# results/tables — which file is which",
             "",
             "Generated by `scripts/check_tables.py --write-readme`. Do not edit by",
             "hand: the register lives in that script, and the same register is what",
             "`check_tables.py` enforces.",
             "",
             "**The plainest names are not the reported ones.** `eval_hqcolon.csv` is a",
             "hand run from 2026-09-10 and `centerline_batch.csv` is the prototype",
             "baseline. The manuscript's numbers come from the `_auto` pair.",
             ""]
    for role, heading in (("reported", "Reported — the manuscript's numbers"),
                          ("ablation", "Ablations the manuscript prints"),
                          ("input", "Inputs to other steps"),
                          ("audit", "Records the build writes and nothing reads"),
                          ("orphan", "Orphans — left by hand runs, not written by run_all.py")):
        names = sorted(n for n, (r, _) in REGISTER.items() if r == role)
        if not names:
            continue
        lines += ["## %s" % heading, "", "| file | what it is |", "|---|---|"]
        for n in names:
            mark = "**%s**" % n if role == "reported" else n
            lines.append("| %s | %s |" % (mark, REGISTER[n][1]))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out


def selftest() -> int:
    """Plant the confusions this gate exists to catch and require each to fire."""
    ok = True
    baseline = check()
    print("baseline (%d finding(s))" % len(baseline))
    for b in baseline:
        print("   - %s" % b)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "tables"
        shutil.copytree(TABLES, work)
        clean = check(work)
        if clean != baseline:
            print("  NOISY    an untouched copy differs from the live tree")
            ok = False
        else:
            print("  SILENT   an untouched copy (negative control)")

        cases = [
            ("the build is pointed at the 2026-09-10 hand run",
             lambda d: shutil.copy(d / "eval_hqcolon.csv",
                                   d / "eval_hqcolon_auto.csv"),
             "median of dice_lumen_vs_gas"),
            ("the prototype baseline is served as the result",
             lambda d: shutil.copy(d / "centerline_batch.csv",
                                   d / "centerline_auto.csv"),
             "n_at_least_600"),
            ("a reported table disappears",
             lambda d: (d / "eval_hqcolon_auto.csv").unlink(),
             "is not on disk"),
            ("a new file appears that nobody classified",
             lambda d: (d / "eval_hqcolon_v2_final.csv").write_text(
                 "PatientID\n", encoding="utf-8"),
             "not classified in the register"),
            ("two series are dropped from the reference set",
             lambda d: _drop_rows(d / "eval_hqcolon_auto.csv", 2),
             "n_rows"),
            # The comparator in Table 4. The first two
            # are the mistakes the first draft of that table made.
            ("the series the prototype never produced is filled in, so its "
             "failure disappears",
             lambda d: _fill_missing_dice(d / "eval_hqcolon_batch.csv"),
             "n_evaluated"),
            ("the prototype's total failures are quietly raised above the cut",
             lambda d: _raise_failed_dice(d / "eval_hqcolon_batch.csv"),
             "n_segmentation_failures"),
            ("the prototype is served as the comparator's reference run",
             lambda d: shutil.copy(d / "eval_hqcolon_auto.csv",
                                   d / "eval_hqcolon_batch.csv"),
             "the prototype (Table 4)"),
            ("a series is lengthened past any published colon",
             lambda d: _lengthen(d / "centerline_auto.csv", 2),
             "the published range"),
            # Comments 2 and 3. The first two are the defects this analysis
            # made and had to undo.
            ("the per-part Dice is measured inside the reference colon again",
             lambda d: _set_column(d / "coverage_anatomy.csv",
                                   "dice_lumen_vs_gas_in_part", "0.996"),
             "dice_lumen_vs_gas_in_part"),
            ("the failures are flattened along the colon",
             lambda d: _flatten_parts(d / "coverage_anatomy.csv"),
             "fail the criterion in part"),
            ("the colon the path missed is said to have been fluid",
             lambda d: _set_column(d / "coverage_anatomy.csv",
                                   "beyond_60mm_that_was_gas_frac", "0.10"),
             "beyond_60mm_that_was_gas_frac"),
            ("the coverage analysis loses a series",
             lambda d: _drop_segment_rows(d / "coverage_anatomy.csv",
                                          "whole colon", 1),
             "n_rows of beyond_60mm_frac"),
            # Comment 6.
            ("the body -400 failures are lifted over the cut",
             lambda d: _set_where(d / "threshold_sweep.csv", "setting",
                                  "body -400", "dice_lumen_vs_gas", "0.88",
                                  only_below=0.5),
             "n_segmentation_failures of dice_lumen_vs_gas at 'body -400'"),
            ("the gas threshold stops mattering",
             lambda d: _flatten_sweep(d / "threshold_sweep.csv"),
             "n_coverage_pass of colon_beyond_60mm_frac at"),
            ("the colonic gas is said to sit above the threshold",
             lambda d: _set_column(d / "gas_attenuation.csv",
                                   "frac_below_800", "0.60"),
             "frac_below_800"),
            ("a scanner's gas attenuation drifts",
             lambda d: _set_column(d / "gas_attenuation.csv",
                                   "gas_hu_median", "-880.0"),
             "gas_hu_median"),
            # Comment 5. The first is the one that would flatter the paper
            # most: a noise level quietly lowered until noise stops mattering,
            # which is the finding the Results lead with.
            ("the noise injected into the phantoms is quietly lowered",
             lambda d: _set_column(d / "image_noise.csv", "soft_sd_p10_hu", "4.0"),
             "soft_sd_p10_hu"),
            ("the noise is said to be white, so the correlated condition "
             "means nothing",
             lambda d: _set_column(d / "image_noise.csv", "acf_lag1", "0.0"),
             "acf_lag1"),
            ("a series is dropped from the noise measurement",
             lambda d: _drop_rows(d / "image_noise.csv", 1),
             "n_rows of soft_sd_p10_hu"),
            ("adhesion to a wall stops changing the diameter",
             lambda d: _set_column(d / "polyp_realism.csv",
                                   "grayscale_diameter_error_mm", "0.1",
                                   where=("condition", "wall_tangent_noise")),
             "band_mm at 'wall_tangent_noise'"),
            ("the lesions in a fold stop crossing the reporting threshold",
             lambda d: _set_column(d / "polyp_realism.csv", "class_changed",
                                   "False", where=("condition", "fold_noise")),
             "class_change_rate at 'fold_noise'"),
            ("the sessile lesions are said to have been measured after all",
             lambda d: _set_column(d / "polyp_realism.csv", "leaked", "False",
                                   where=("condition", "wall_sessile_noise")),
             "leak_rate at 'wall_sessile_noise'"),
            ("the refusal flag is made to fire on everything, so it looks "
             "sensitive",
             lambda d: _set_column(d / "polyp_realism.csv",
                                   "touches_voi_boundary", "True"),
             "specificity_to_leak"),
            ("what the refusal flag misses is hidden",
             lambda d: _set_column(d / "polyp_realism.csv", "class_changed",
                                   "False",
                                   where=("touches_voi_boundary", "False")),
             "passed_but_wrong_class_rate"),
            ("the second signal is blinded by making the two diameter "
             "routes agree",
             lambda d: _copy_column(d / "polyp_realism.csv",
                                    "grayscale_max_diameter_mm",
                                    "binary_max_diameter_mm"),
             "second_signal_sensitivity"),
            ("the cohort is given a lesion above 10 mm",
             lambda d: _promote_lesion(d / "cohort_lesions.csv"),
             ">=10 mm sheet"),
            ("the annotation is said to carry a slice number for every lesion",
             lambda d: _fill_empty(d / "cohort_lesions.csv", "slice_supine", "100"),
             "with_no_slice_at_all"),
        ]
        for name, plant, expect in cases:
            case = Path(tmp) / name.replace(" ", "_")[:40]
            shutil.copytree(TABLES, case)
            before = _stamp(case)
            plant(case)
            after = _stamp(case)
            if before == after:
                print("  NO-OP    %s (the injection changed nothing)" % name)
                ok = False
                continue
            found = [f for f in check(case) if f not in baseline]
            hit = [f for f in found if expect in f]
            print("  %-8s %s" % ("DETECTED" if hit else "MISSED", name))
            if hit:
                print("           -> %s" % hit[0][:120])
            else:
                ok = False
                print("           expected a finding containing %r, got %s"
                      % (expect, [f[:80] for f in found] or "nothing"))

    ok = _selftest_build_coverage() and ok
    print("selftest %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _selftest_build_coverage() -> bool:
    """The same treatment for the register-against-the-build check.

    These injections are structural, not file-level: they break the map between
    what the register classifies and what the stages claim to write. The
    no-op guard is the equality of the dictionaries, since nothing on disk
    moves.
    """
    ok = True
    baseline = build_coverage()
    print("build coverage baseline (%d finding(s))" % len(baseline))
    for b in baseline:
        print("   - %s" % b)

    def without(stage: str, path: str) -> dict:
        out = {k: tuple(v) for k, v in OUTPUTS.items()}
        out[stage] = tuple(f for f in out[stage] if f != path)
        return out

    reported = "tables/threshold_sweep.csv"
    orphan_claim = {k: tuple(v) for k, v in OUTPUTS.items()}
    orphan_claim["tables"] = orphan_claim["tables"] + ("tables/eval_hqcolon.csv",)
    unregistered = {k: tuple(v) for k, v in OUTPUTS.items()}
    unregistered["tables"] = unregistered["tables"] + ("tables/eval_hqcolon_v2.csv",)
    no_figures = {k: tuple(v) for k, v in OUTPUTS.items()}
    no_figures["figures"] = ()
    dropped_stage = {k: tuple(v) for k, v in OUTPUTS.items() if k != "strata"}
    twice = {k: tuple(v) for k, v in OUTPUTS.items()}
    twice["cohort"] = twice["cohort"] + ("tables/indices.csv",)

    cases = [
        ("a reported table loses the stage that writes it",
         without("sweep", reported), None, "no stage of run_all.py"),
        ("a hand run is claimed as a build product",
         orphan_claim, None, "registered as an orphan"),
        ("a stage writes a table nobody classified",
         unregistered, None, "the register does not classify"),
        ("the figures stop being produced by anything",
         no_figures, None, "shipped but no stage claims"),
        ("a stage disappears from the output map",
         dropped_stage, None, "has no entry in run_all.OUTPUTS"),
        ("two stages claim the same table",
         twice, None, "written by more than one stage"),
        ("a registered file is renamed out of the register",
         None, {k: v for k, v in REGISTER.items() if k != "indices.csv"},
         "which the register does not classify"),
    ]
    for name, outputs, register, expect in cases:
        if outputs == OUTPUTS and register == REGISTER:
            print("  NO-OP    %s (the injection changed nothing)" % name)
            ok = False
            continue
        found = [f for f in build_coverage(register=register, outputs=outputs)
                 if f not in baseline]
        hit = [f for f in found if expect in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED", name))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           expected a finding containing %r, got %s"
                  % (expect, [f[:80] for f in found] or "nothing"))

    claim_elsewhere = {k: tuple(v) for k, v in OUTPUTS.items()}
    # The strata stage is pointed only at the tables directory, so a claim on
    # a figure is one no command of it could satisfy.
    claim_elsewhere["strata"] = ("figures/stratified_acquisition.png",)
    for name, outputs, register, expect in [
        ("a stage claims a file no command of it is pointed at",
         claim_elsewhere, None, "none of its commands is pointed at that path"),
    ]:
        found = [f for f in build_coverage(register=register, outputs=outputs)
                 if f not in baseline]
        hit = [f for f in found if expect in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED", name))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           expected %r, got %s"
                  % (expect, [f[:80] for f in found] or "nothing"))

    # The two documents a reader follows, and the shipped tree itself.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        manual = (ROOT / "docs" / "USER_MANUAL.md").read_text(encoding="utf-8")

        def build_docs(target: Path, readme_text: str, manual_text: str | None):
            (target / "results").mkdir(parents=True)
            (target / "README.md").write_text(readme_text, encoding="utf-8")
            if manual_text is not None:
                (target / "docs").mkdir()
                (target / "docs" / "USER_MANUAL.md").write_text(
                    manual_text, encoding="utf-8")

        doc_cases = [
            ("a stage is dropped from the README's table",
             readme.replace("| strata |", "| x |", 1), manual,
             "no table row for the strata stage"),
            ("a stage is named only inside a code block",
             readme.replace("| strata |", "| x |", 1).replace(
                 "## Tests", "```\n| strata |\n```\n\n## Tests", 1), manual,
             "no table row for the strata stage"),
            ("the manual is gone altogether",
             readme, None, "USER_MANUAL.md is missing"),
        ]
        for i, (name, readme_text, manual_text, expect) in enumerate(doc_cases):
            target = root / ("doc%d" % i)
            build_docs(target, readme_text, manual_text)
            found = [f for f in build_coverage(results=target / "results")
                     if f not in baseline]
            hit = [f for f in found if expect in f]
            print("  %-8s %s" % ("DETECTED" if hit else "MISSED", name))
            if hit:
                print("           -> %s" % hit[0][:120])
            else:
                ok = False
                print("           expected %r, got %s"
                      % (expect, [f[:80] for f in found] or "nothing"))

        # A file shipped inside a sub-directory of the figures tree.
        nested = root / "nested"
        build_docs(nested, readme, manual)
        (nested / "results" / "figures" / "panels").mkdir(parents=True)
        (nested / "results" / "figures" / "panels" / "a.png").write_bytes(b"x")
        found = [f for f in build_coverage(results=nested / "results")
                 if f not in baseline]
        hit = [f for f in found if "figures/panels/a.png is shipped" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "an artefact appears in a sub-directory"))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False

    ok = _selftest_stage_inputs() and ok
    return ok


def _selftest_stage_inputs() -> bool:
    """Plant a stage script that reads the committed results regardless of --results."""
    ok = True
    baseline = stage_inputs()
    print("stage inputs baseline (%d finding(s))" % len(baseline))
    for b in baseline:
        print("   - %s" % b)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(ROOT / "scripts", root / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        victim = root / "scripts" / "stats_clustered.py"
        text = victim.read_text(encoding="utf-8")
        victim.write_text(text + "\n\ndef _planted():\n"
                                 "    return (TABLES / 'indices.csv').read_text()\n",
                          encoding="utf-8")
        found = [f for f in stage_inputs(root) if f not in baseline]
        hit = [f for f in found if "outside an argparse default" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "a stage script reads the committed tables directly"))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           got %s" % ([f[:80] for f in found] or "nothing"))

        # A default that points at the committed tree and that no stage ever
        # overrides: the flag exists, so the path looks parameterised, and
        # nothing passes it.
        unpassed = root / "unpassed"
        shutil.copytree(ROOT / "scripts", unpassed / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        target = unpassed / "scripts" / "stats_clustered.py"
        text = target.read_text(encoding="utf-8")
        marker = '    args = ap.parse_args()'
        text = text.replace(
            marker,
            '    ap.add_argument("--extra", type=Path, default=TABLES / "indices.csv")\n'
            + marker, 1)
        target.write_text(text, encoding="utf-8")
        found = [f for f in stage_inputs(unpassed) if f not in baseline]
        hit = [f for f in found if "no stage passes that flag" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "an input defaults to the committed tree and "
                             "nothing overrides it"))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           got %s" % ([f[:80] for f in found] or "nothing"))

        # The committed path written out as text instead of built.
        literal = root / "literal"
        shutil.copytree(ROOT / "scripts", literal / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        spelled = literal / "scripts" / "make_cohort_table.py"
        text = spelled.read_text(encoding="utf-8")
        spelled.write_text(
            text + '\n\ndef _planted():\n'
                   '    return open("results/tables/indices.csv").read()\n',
            encoding="utf-8")
        found = [f for f in stage_inputs(literal) if f not in baseline]
        hit = [f for f in found if "which is the committed tree" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "a committed path is spelled out as text"))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           got %s" % ([f[:80] for f in found] or "nothing"))

        # The same defect one level down: the helper keeps its committed
        # default and a call stops passing the directory.
        helper = root / "helper"
        shutil.copytree(ROOT / "scripts", helper / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        target = helper / "scripts" / "stats_clustered.py"
        text = target.read_text(encoding="utf-8")
        text = text.replace('read("indices.csv", args.tables)', 'read("indices.csv")', 1)
        target.write_text(text, encoding="utf-8")
        found = [f for f in stage_inputs(helper) if f not in baseline]
        hit = [f for f in found if "without tables" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "a helper keeps its committed default and a call "
                             "stops passing the directory"))
        if hit:
            print("           -> %s" % hit[0][:120])
        else:
            ok = False
            print("           got %s" % ([f[:80] for f in found] or "nothing"))

        gone = root / "gone"
        (gone / "scripts").mkdir(parents=True)
        found = [f for f in stage_inputs(gone)]
        hit = [f for f in found if "is not in scripts/" in f]
        print("  %-8s %s" % ("DETECTED" if hit else "MISSED",
                             "a stage names a script that is not there"))
        if not hit:
            ok = False
    return ok


def _stamp(directory: Path) -> list[tuple[str, str]]:
    """(name, sha256) for every file, so a same-width edit is still a change.

    The earlier version of this used the byte size, which cannot tell 21.1
    from 22.2 and reported a real injection as a no-op.
    """
    return sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in directory.iterdir() if p.is_file())

def _set_column(path: Path, column: str, value: str,
                where: tuple[str, str] | None = None) -> None:
    """Give every row the same value in one column, or every matching row."""
    rows = read(path)
    for r in rows:
        if where is not None and r.get(where[0]) != where[1]:
            continue
        if (r.get(column) or "").strip():
            r[column] = value
    _write(path, rows)


def _fill_empty(path: Path, column: str, value: str) -> None:
    """Fill in the cells that are empty, which is where the claim lives.

    ``_set_column`` only overwrites cells that already carry a value, so using
    it here would change the file without changing what the rule reads: the
    lesion with no slice number would still have none. The selftest caught
    that, which is what it is for.
    """
    rows = read(path)
    for r in rows:
        if not (r.get(column) or "").strip():
            r[column] = value
    _write(path, rows)


def _copy_column(path: Path, src: str, dst: str) -> None:
    """Make one column equal to another, which removes a disagreement."""
    rows = read(path)
    for r in rows:
        if (r.get(src) or "").strip():
            r[dst] = r[src]
    _write(path, rows)


def _promote_lesion(path: Path) -> None:
    """Move one annotated lesion into the size class the cohort does not have."""
    rows = read(path)
    if rows:
        rows[0]["sheet_size_band"] = ">=10 mm"
        rows[0]["reported_size_mm"] = "12"
    _write(path, rows)


def _flatten_parts(path: Path) -> None:
    """Make every part equally covered, erasing the profile along the colon."""
    rows = read(path)
    for r in rows:
        if str(r.get("segment", "")).startswith("part "):
            r["beyond_60mm_frac"] = "0.0"
            r["within_30mm_frac"] = "0.95"
    _write(path, rows)


def _drop_segment_rows(path: Path, segment: str, n: int) -> None:
    rows = read(path)
    seen = 0
    kept = []
    for r in rows:
        # From a patient that is in the primary set: dropping one of the five
        # inspected patients leaves that set at 18 and tests nothing.
        inspected = (r.get("PatientID") or "").rsplit(".", 1)[-1]             in DEVELOPMENT_PATIENTS
        if r.get("segment") == segment and not inspected and seen < n:
            seen += 1
            continue
        kept.append(r)
    _write(path, kept)

def _set_where(path: Path, key: str, value: str, column: str, to: str,
               only_below: float | None = None) -> None:
    """Set a column on the rows whose `key` equals `value`."""
    rows = read(path)
    for r in rows:
        if r.get(key) != value:
            continue
        if only_below is not None:
            v = as_float(r.get(column))
            if v is None or v >= only_below:
                continue
        r[column] = to
    _write(path, rows)


def _flatten_sweep(path: Path) -> None:
    """Give every setting the reported setting's coverage, erasing the finding."""
    rows = read(path)
    base = {(r["PatientID"], r["role"]): r for r in rows
            if r.get("setting") == "reported"}
    for r in rows:
        b = base.get((r["PatientID"], r["role"]))
        if b is not None:
            r["colon_beyond_60mm_frac"] = b["colon_beyond_60mm_frac"]
            r["colon_within_30mm_frac"] = b["colon_within_30mm_frac"]
            r["met_coverage"] = b["met_coverage"]
    _write(path, rows)

def _fill_missing_dice(path: Path) -> None:
    """Give every row a Dice, so a series with no centerline looks evaluated."""
    rows = read(path)
    for r in rows:
        if not (r.get("dice_lumen_vs_gas") or "").strip():
            r["dice_lumen_vs_gas"] = "0.9"
    _write(path, rows)


def _raise_failed_dice(path: Path) -> None:
    """Lift the total failures just over the cut, leaving everything else."""
    rows = read(path)
    for r in rows:
        v = as_float(r.get("dice_lumen_vs_gas"))
        if v is not None and v < SEG_FAILURE_DICE:
            r["dice_lumen_vs_gas"] = "0.55"
    _write(path, rows)


def _lengthen(path: Path, n: int) -> None:
    """Push n series past the longest published colon."""
    rows = read(path)
    done = 0
    for r in rows:
        v = as_float(r.get("length_mm"))
        if v is not None and v / 10.0 <= PUBLISHED_RANGE_CM[1] and done < n:
            r["length_mm"] = "%.2f" % (PUBLISHED_RANGE_CM[1] * 10.0 + 100.0)
            done += 1
    _write(path, rows)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def _drop_rows(path: Path, n: int) -> None:
    _write(path, read(path)[:-n])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-readme", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.write_readme:
        print("wrote %s" % write_readme())
    bad = check() + build_coverage() + stage_inputs()
    for b in bad:
        print("FAIL %s" % b)
    if not bad:
        print("OK   every table is classified, every shipped result is written "
              "by a stage, and the reported tables still give the "
              "manuscript's numbers")
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
