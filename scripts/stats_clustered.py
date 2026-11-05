"""Cluster-aware intervals for every headline quantity.

The objection this answers: the 60 series sit in 30 patients and the 26 reference
series in 18, so the observations are not independent, and the manuscript gives
medians and proportions with no intervals at all. ICC alone was offered for
prone-supine agreement, without limits of agreement intervals, and the fat
result was given as a median only.

Everything here resamples **patients**, not series, which is the only part that
matters: a patient contributes a prone and a supine series that fail or succeed
together far more often than two unrelated series would. The point estimates are
unchanged; what is added is how wide they are once that is admitted.

  python scripts/stats_clustered.py
  python scripts/stats_clustered.py --boot 20000 --out results/tables/clustered_statistics.csv

The number of clusters, not the number of series, sets the precision: 18
patients for the reference set and 13 for the subset. Intervals that look
embarrassing are the finding, not a defect in the method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import zlib
from pathlib import Path

import numpy as np

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

# The five series inspected against the reference while the seed rule and core
# radius were chosen (manuscript section 2). Their patients are excluded to form
# the subset that is reported as primary.
INSPECTED_PATIENTS = {"0001", "0003", "0004", "0007", "0030"}

COVERAGE_BEYOND_60_MAX = 0.05   # a series meets the coverage criterion below this
LENGTH_MIN_MM = 600.0           # the length criterion


def patient_key(row: dict) -> str:
    return row.get("PatientID") or row.get("patient") or ""


def short_patient(pid: str) -> str:
    return pid.rsplit(".", 1)[-1]


def read(name: str, tables: Path = TABLES) -> list[dict]:
    with (tables / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def stream(seed: int, label: str) -> np.random.Generator:
    """One independent draw sequence per quantity.

    A single shared generator makes every interval depend on how many
    quantities were computed before it: adding one row moved the published
    interval of the coverage criterion from 22.2-68.8 % to 21.1-68.8 % without
    any data changing. Deriving the stream from the quantity's own name means a
    quantity added later cannot disturb a number already in the manuscript.
    """
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def cluster_bootstrap(groups: dict[str, list[float]], statistic, boot: int,
                      rng: np.random.Generator) -> tuple[float, float, float]:
    """Resample whole patients with replacement; return (estimate, lo, hi)."""
    keys = sorted(groups)
    pooled = [v for k in keys for v in groups[k]]
    if not pooled:
        return (float("nan"),) * 3
    estimate = statistic(pooled)
    draws = np.empty(boot)
    n = len(keys)
    for b in range(boot):
        picks = rng.integers(0, n, n)
        sample: list[float] = []
        for i in picks:
            sample.extend(groups[keys[i]])
        draws[b] = statistic(sample) if sample else float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(estimate), float(lo), float(hi)


def group_by_patient(rows: list[dict], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        v = as_float(r.get(field))
        if v is not None:
            out.setdefault(patient_key(r), []).append(v)
    return out


def loa_interval(sd_diff: float, n_pairs: int) -> tuple[float, float]:
    """Half-width of the 95% CI on each limit of agreement (Bland & Altman).

    var(LoA) ~= (1/n + 1.96^2 / (2(n-1))) * sd^2, so the standard error is
    sd * sqrt(that), and the interval uses the t quantile on n-1 df.
    """
    if n_pairs < 3 or not sd_diff:
        return (float("nan"), float("nan"))
    var_factor = 1.0 / n_pairs + (1.96**2) / (2.0 * (n_pairs - 1))
    se = sd_diff * math.sqrt(var_factor)
    # t(0.975, n-1) without scipy; Student's t via the Cornish-Fisher-free route
    t = _t_quantile_975(n_pairs - 1)
    return (se, t * se)


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t_quantile_975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    return 1.96 + 2.4 / max(df, 1)   # adequate beyond 30 df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261104)
    # The inputs come from a directory. Until 2026-11-05 they were read from
    # the committed results no matter what --results said, so a re-run into
    # another directory was computed from the committed tables and compared
    # with them: the comparison could not fail.
    ap.add_argument("--tables", type=Path, default=TABLES)
    ap.add_argument("--out", type=Path, default=None)
    # The plainly named eval_hqcolon.csv and centerline_batch.csv are NOT the
    # manuscript's numbers. eval_hqcolon.csv is a 2026-09-10 run that predates
    # the colon_within_30mm_frac and colon_beyond_60mm_frac columns and gives
    # median Dice 0.8781 where the paper reports 0.882; centerline_batch.csv is
    # the prototype baseline (41 of 60 at 600 mm, median 908.5 mm), which is the
    # "before" column of Table 2. The paper's figures come from the _auto pair:
    # eval_hqcolon_auto.csv (Dice 0.8822, 86.0 % within 30 mm, 15 of 26) and
    # centerline_auto.csv (56 of 60, median 1565.7 mm). Naming them on the
    # command line keeps that choice visible instead of buried in a default.
    ap.add_argument("--hq", default="eval_hqcolon_auto.csv")
    ap.add_argument("--batch", default="centerline_auto.csv")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.tables / "clustered_statistics.csv"

    hq = read(args.hq, args.tables)
    batch = read(args.batch, args.tables)
    # pair_reference_status.csv is deliberately NOT read here: it is written by
    # the "tables" stage, which runs after this one, and it carries a beyond-60
    # mm value for only 24 of the 26 reference series. The coverage block below
    # takes that quantity from the evaluation table instead.
    indices = read("indices.csv", args.tables)
    agree = read("prone_supine_agreement.csv", args.tables)

    hq_sub = [r for r in hq
              if short_patient(patient_key(r)) not in INSPECTED_PATIENTS]

    rows: list[dict] = []

    def record(quantity: str, setname: str, rowsrc: list[dict], field: str,
               statistic, stat_name: str, unit: str = "") -> None:
        groups = group_by_patient(rowsrc, field)
        if not groups:
            print("  (skipped %s on %s: no values for %r)"
                  % (quantity, setname, field))
            return
        est, lo, hi = cluster_bootstrap(
            groups, statistic, args.boot,
            stream(args.seed, f"{quantity}|{setname}|{stat_name}"))
        n_series = sum(len(v) for v in groups.values())
        rows.append({
            "quantity": quantity, "set": setname, "statistic": stat_name,
            # The source column travels with the row so that a consumer joins
            # on it rather than on a prose label spelled out twice. make_tables
            # prints the interval column of the reference table by this key.
            "column": field,
            "unit": unit, "n_series": n_series, "n_patients": len(groups),
            "estimate": round(est, 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
        })
        print("  %-38s %-22s %-7s n=%2d/%2d  %8.4f  [%.4f, %.4f]"
              % (quantity, setname, stat_name, n_series, len(groups),
                 est, lo, hi))

    median = statistics.median

    print("Medians with patient-cluster bootstrap intervals "
          "(%d resamples, seed %d)" % (args.boot, args.seed))
    for setname, src in (("26 series / 18 patients", hq),
                         ("18 series / 13 patients", hq_sub)):
        record("Dice, lumen vs gas reference", setname, src,
               "dice_lumen_vs_gas", median, "median")
        record("Jaccard vs gas reference", setname, src,
               "jaccard_lumen_vs_gas", median, "median")
        record("Recall vs gas reference", setname, src,
               "lumen_recall_vs_gas", median, "median")
        record("Precision vs gas reference", setname, src,
               "lumen_precision_vs_gas", median, "median")
        record("Mean symmetric surface distance", setname, src,
               "msd_lumen_vs_gas_mm", median, "median", "mm")
        record("95th-percentile Hausdorff distance", setname, src,
               "hd95_lumen_vs_gas_mm", median, "median", "mm")
        record("Centerline points inside reference colon", setname, src,
               "centerline_inside_gas_frac", median, "median", "fraction")
        record("Reference colon within 30 mm of path", setname, src,
               "colon_within_30mm_frac", median, "median", "fraction")
        record("Reference colon beyond 60 mm of path", setname, src,
               "colon_beyond_60mm_frac", median, "median", "fraction")
        record("Centerline length ratio", setname, src,
               "length_ratio_gf", median, "median", "ratio")
        record("Reference centerline covered", setname, src,
               "ref_gf_covered_frac", median, "median", "fraction")
        record("Centerline length ratio, gas-only reference", setname, src,
               "length_ratio", median, "median", "ratio")

    print("\nProportions with patient-cluster bootstrap intervals")

    # Coverage criterion, from the per-series flags that carry the 60 mm share.
    # From the evaluation table itself, not from pair_reference_status.csv:
    # that file carries a beyond-60 mm value for only 24 of the 26 reference
    # series and would give 13 of 24 where the paper reports 15 of 26.
    cov: dict[str, list[float]] = {}
    for r in hq:
        beyond = as_float(r.get("colon_beyond_60mm_frac"))
        if beyond is None:
            continue
        cov.setdefault(patient_key(r), []).append(
            1.0 if beyond < COVERAGE_BEYOND_60_MAX else 0.0)
    cov_sub = {k: v for k, v in cov.items()
               if short_patient(k) not in INSPECTED_PATIENTS}
    for setname, groups in (("26 series / 18 patients", cov),
                            ("18 series / 13 patients", cov_sub)):
        est, lo, hi = cluster_bootstrap(
            groups, lambda x: sum(x) / len(x), args.boot,
            stream(args.seed, f"coverage criterion|{setname}|proportion"))
        n_series = sum(len(v) for v in groups.values())
        n_pass = int(round(est * n_series))
        rows.append({
            "quantity": "Met the coverage criterion (<5% beyond 60 mm)",
            "set": setname, "statistic": "proportion",
            "column": "colon_beyond_60mm_frac", "unit": "",
            "n_series": n_series, "n_patients": len(groups),
            "estimate": round(est, 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
        })
        print("  %-38s %-22s %-7s n=%2d/%2d  %8.4f  [%.4f, %.4f]  (%d of %d)"
              % ("Met the coverage criterion", setname, "prop", n_series,
                 len(groups), est, lo, hi, n_pass, n_series))

    # Length criterion over all 60 series.
    length: dict[str, list[float]] = {}
    for r in batch:
        v = as_float(r.get("length_mm"))
        ok = (r.get("ok") or "").strip().lower() in ("true", "1", "yes")
        if v is None and not ok:
            v = 0.0
        if v is None:
            continue
        length.setdefault(patient_key(r), []).append(
            1.0 if v >= LENGTH_MIN_MM else 0.0)
    # The seed rule was chosen with the reference in hand, so the cohort-wide
    # length criterion is development-tuned in the same way the reference
    # measures are: the five inspected patients contribute ten of the 60
    # series. The restricted set is given beside it for the same reason the
    # the 18-series subset is the primary one.
    length_sub = {k: v for k, v in length.items()
                  if short_patient(k) not in INSPECTED_PATIENTS}
    for setname, groups in (("60 series / 30 patients", length),
                            ("50 series / 25 patients", length_sub)):
        est, lo, hi = cluster_bootstrap(
            groups, lambda x: sum(x) / len(x), args.boot,
            stream(args.seed, f"length criterion|{setname}|proportion"))
        n_series = sum(len(v) for v in groups.values())
        rows.append({
            "quantity": "Met the length criterion (>=600 mm)",
            "set": setname, "statistic": "proportion",
            "column": "length_mm", "unit": "",
            "n_series": n_series, "n_patients": len(groups),
            "estimate": round(est, 4), "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
        })
        print("  %-38s %-22s %-7s n=%2d/%2d  %8.4f  [%.4f, %.4f]  (%d of %d)"
              % ("Met the length criterion", setname, "prop",
                 n_series, len(groups), est, lo, hi,
                 int(round(est * n_series)), n_series))

    batch_sub = [r for r in batch
                 if short_patient(patient_key(r)) not in INSPECTED_PATIENTS]
    for setname, src in (("60 series / 30 patients", batch),
                         ("50 series / 25 patients", batch_sub)):
        record("Traced centerline length", setname, src, "length_mm",
               median, "median", "mm")

    print("\nFat attenuation: the whole distribution, not the median alone")
    fat_rows = [r for r in indices
                if as_float(r.get("fat_mean_hu_median")) is not None]
    fat_sets = {
        "60 series / 30 patients": fat_rows,
        "18-patient-excluded subset": [
            r for r in fat_rows
            if short_patient(patient_key(r)) not in INSPECTED_PATIENTS],
    }
    for setname, src in fat_sets.items():
        vals = sorted(as_float(r["fat_mean_hu_median"]) for r in src)
        if not vals:
            continue
        q = np.percentile(vals, [0, 5, 25, 50, 75, 95, 100])
        est, lo, hi = cluster_bootstrap(
            group_by_patient(src, "fat_mean_hu_median"), median, args.boot,
            stream(args.seed, f"fat attenuation|{setname}|median"))
        rows.append({
            "quantity": "Fat attenuation (mean HU per series)", "set": setname,
            "statistic": "median", "column": "fat_mean_hu_median",
            "unit": "HU", "n_series": len(vals),
            "n_patients": len({patient_key(r) for r in src}),
            "estimate": round(est, 4), "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
        })
        # The distribution matters, not the median alone, so the
        # percentiles are written out too rather than printed and lost.
        for name, value in zip(("min", "p5", "q1", "q3", "p95", "max"),
                               (q[0], q[1], q[2], q[4], q[5], q[6]),
                               strict=True):
            rows.append({
                "quantity": "Fat attenuation (mean HU per series)",
                "set": setname, "statistic": name,
                "column": "fat_mean_hu_median", "unit": "HU",
                "n_series": len(vals),
                "n_patients": len({patient_key(r) for r in src}),
                "estimate": round(float(value), 4),
                "ci_lower": "", "ci_upper": "",
            })
        print("  %-26s n=%d  min %.1f  p5 %.1f  q1 %.1f  median %.1f "
              "[%.1f, %.1f]  q3 %.1f  p95 %.1f  max %.1f"
              % (setname, len(vals), q[0], q[1], q[2], q[3], lo, hi,
                 q[4], q[5], q[6]))

    print("\nPRONE-SUPINE: limits of agreement with their own intervals")
    for r in agree:
        n_pairs = int(as_float(r.get("n_pairs")) or 0)
        sd = as_float(r.get("sd_diff"))
        lo_loa, hi_loa = as_float(r.get("loa_lower")), as_float(r.get("loa_upper"))
        if not (n_pairs and sd and lo_loa is not None and hi_loa is not None):
            continue
        se, half = loa_interval(sd, n_pairs)
        rows.append({
            "quantity": "Lower limit of agreement: %s" % r.get("index", "?"),
            "set": "%d pairs" % n_pairs, "statistic": "LoA",
            "column": r.get("index", ""), "unit": r.get("unit", ""),
            "n_series": 2 * n_pairs, "n_patients": n_pairs,
            "estimate": round(lo_loa, 4), "ci_lower": round(lo_loa - half, 4),
            "ci_upper": round(lo_loa + half, 4),
        })
        rows.append({
            "quantity": "Upper limit of agreement: %s" % r.get("index", "?"),
            "set": "%d pairs" % n_pairs, "statistic": "LoA",
            "column": r.get("index", ""), "unit": r.get("unit", ""),
            "n_series": 2 * n_pairs, "n_patients": n_pairs,
            "estimate": round(hi_loa, 4), "ci_lower": round(hi_loa - half, 4),
            "ci_upper": round(hi_loa + half, 4),
        })
        print("  %-34s LoA %8.3f [%8.3f, %8.3f] .. %8.3f [%8.3f, %8.3f]  n=%d"
              % (str(r.get("index"))[:34], lo_loa, lo_loa - half, lo_loa + half,
                 hi_loa, hi_loa - half, hi_loa + half, n_pairs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({
        "bootstrap_resamples": args.boot,
        "seed": args.seed,
        "resampling_unit": "patient",
        "inspected_patients_excluded_from_subset": sorted(INSPECTED_PATIENTS),
        "coverage_criterion": "fraction of reference colon volume beyond 60 mm "
                              "of the path < %.2f" % COVERAGE_BEYOND_60_MAX,
        "length_criterion_mm": LENGTH_MIN_MM,
        "rows": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s (%d rows) and %s" % (args.out, len(rows), meta.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
