"""Manuscript tables and every in-text number, from the committed result tables.

Nothing here touches image data. Each manuscript table is written as a CSV under
``results/tables/manuscript/`` and every number quoted in the text goes into
``numbers.json`` there, keyed by name, so that a sentence in the manuscript can
be traced to the line that computed it.

Definitions fixed here (and registered in ``docs/PARAMETERS.md``):

* A centerline is **failed** if the stage raised or the path is shorter than
  600 mm (about a third of a colon). No length threshold for completeness is
  used: the former 900 mm one was calibrated on truncated reference paths and
  is withdrawn. Completeness is measured directly, as whole-colon coverage,
  where a reference exists.
* Whole-colon coverage: share of the reference colon (gas and fluid) within
  30 mm of the path; the colon is **reached** when less than 5 % of it lies
  beyond 60 mm.
* A lumen segmentation **failed** against HQColon if its Dice with the
  gas-filled reference is below 0.5. Observed values leave a gap from 0.002 to
  0.721, so the result does not depend on where in that gap the cut sits.
* A patient is **usable in both positions** if neither series failed.

    python scripts/make_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

FAILED_BELOW_MM = 600.0
REACHED_FAR_MAX = 0.05
# Series inspected case by case while the pipeline was being developed, with the
# reference in view: the extent diagnosis and the seed-rule and core-radius
# choices were made on them (scripts/diagnose_centerline_extent.py,
# diagnose_diameter_seeds.py). The additional column excludes their patients.
# It is not an independent test set: the candidate seed rules were scored
# against all 26 reference series before the pipeline was changed, so the
# reference informed the choice for every series in the set.
DEVELOPMENT_SERIES = ["0007-1", "0003-2", "0001-1", "0004-2", "0030-1"]
DEVELOPMENT_PATIENTS = sorted({x.split("-")[0] for x in DEVELOPMENT_SERIES})
SEG_FAILURE_DICE = 0.5
COVERAGE_FLOOR = 0.6


# The printed reference table's two quoted columns, and the names the clustered
# statistics give those same two sets. They are spelled once here because the
# manuscript, the table legend and check_tables.py all have to agree with them.
# Short enough to typeset as a column head; the legend carries the rest.
PRIMARY_COLUMN = "Primary: 18 series, patients not inspected"
TUNED_COLUMN = "Development-tuned: all 26 series"
CI_LABEL = {PRIMARY_COLUMN: "95% CI (primary)",
            TUNED_COLUMN: "95% CI (development-tuned)"}
SET_PRIMARY = "18 series / 13 patients"
SET_TUNED = "26 series / 18 patients"


def intervals(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(source column, set, statistic) -> that quantity's 95% interval.

    The key is the column the statistic was computed from, which travels in
    ``clustered_statistics.csv`` for exactly this reason: joining on a prose
    label would mean spelling each quantity out in two files and letting them
    drift apart silently. The statistic belongs in the key as well, because one
    column carries two different quantities -- ``colon_beyond_60mm_frac`` gives
    both the median share of colon beyond 60 mm and the proportion of series
    meeting the coverage criterion, and their intervals are nothing like each
    other (0 to 20 % against 21 to 69 %). Keying on the column alone returned
    the first of those under the second one's label.
    """
    if not path.exists():
        print("  WARNING: %s is missing, so the table will carry no intervals; "
              "run the clustered stage of run_all.py" % path.name)
        return {}
    d = pd.read_csv(path)
    d = d[d["column"].notna()]
    out = {}
    for r in d.itertuples():
        lo, hi = pd.to_numeric(r.ci_lower, errors="coerce"), pd.to_numeric(
            r.ci_upper, errors="coerce")
        if pd.notna(lo) and pd.notna(hi):
            key = (str(r.column), str(r.set), str(r.statistic))
            out[key] = (float(lo), float(hi))
    return out


def fmt_ci(bounds, digits: int = 3, percent: bool = False) -> str:
    if not bounds:
        return ""
    lo, hi = bounds
    if percent:
        return f"{lo * 100:.1f} to {hi * 100:.1f} %"
    return f"{lo:.{digits}f} to {hi:.{digits}f}"


def med_iqr(x: pd.Series, digits: int = 3) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return {"median": round(float(q2), 4), "q1": round(float(q1), 4),
            "q3": round(float(q3), 4), "n": int(len(x))}


def outcome_summary(cl: pd.DataFrame) -> dict:
    ok = cl["ok"].astype(str).str.lower() == "true"
    length = pd.to_numeric(cl["length_mm"], errors="coerce")
    failed = ~ok | (length < FAILED_BELOW_MM)
    pid = cl["PatientID"]
    usable_both = (~failed).groupby(pid).sum().eq(2).sum()
    produced_both = ok.groupby(pid).sum().eq(2).sum()
    out = {
        "n_series": int(len(cl)),
        "produced": int(ok.sum()),
        "failed": int(failed.sum()),
        "length_mm": stats(length[ok]),
        "patients_usable_both": int(usable_both),
        "patients_produced_both": int(produced_both),
        "series_with_bridges": int((pd.to_numeric(cl["n_bridges"], errors="coerce") > 0).sum()),
        "runtime_sec": stats(cl.loc[ok, "total_sec"]),
    }
    if "fill_holes_used" in cl:
        out["slicewise_fallback"] = int((cl["fill_holes_used"] == "slicewise").sum())
    return out


# Composites whose weights are inherited and uncalibrated. They are not
# descriptors of anything measured here, so they are reported in a
# supplementary table of their own and their agreement is not interpreted
# (they are exploratory).
EXPLORATORY_COMPOSITES = [
    ("Distension state and traced path", "distension_quality_score",
     "Distension score (exploratory composite)", "", 1),
]


PILLAR1 = [
    ("Colon within 30 mm of the path (whole colon, gas and fluid)", "colon_within_30mm_frac", 3),
    ("Colon beyond 60 mm of the path (never visited)", "colon_beyond_60mm_frac", 3),
    ("Dice, lumen vs gas-filled reference", "dice_lumen_vs_gas", 3),
    ("Jaccard", "jaccard_lumen_vs_gas", 3),
    ("Recall", "lumen_recall_vs_gas", 3),
    ("Precision", "lumen_precision_vs_gas", 3),
    ("Mean symmetric surface distance, mm", "msd_lumen_vs_gas_mm", 2),
    ("95th-percentile Hausdorff distance, mm", "hd95_lumen_vs_gas_mm", 1),
    ("Centerline points inside reference colon", "centerline_inside_gas_frac", 3),
    # The gas-only reference centerline stops wherever the colon is collapsed or
    # fluid-filled, so the whole-colon comparison uses the gas+fluid reference
    # ("_gf"). The gas-only columns stay in the CSV, labelled for what they are.
    ("Centerline length / reference length (whole colon, gas and fluid)",
     "length_ratio_gf", 3),
    ("Whole-colon reference centerline within 10 mm of the path", "ref_gf_covered_frac", 3),
    ("Centerline length / reference length (gas-filled reference)", "length_ratio", 3),
]

AGREEMENT = [
    # (construct, index, label, unit, digits). Fat is measured in the ring
    # 5-15 mm beyond the wall (primary); the spherical ROI of the prototype
    # batch is kept as the previous definition.
    ("Fat composition", "ring_fat_mean_hu_median", "Pericolonic fat attenuation (ring)", "HU", 1),
    ("Fat amount and distribution", "ring_fat_fraction_mean", "Fat fraction (ring)", "", 3),
    ("Fat amount and distribution", "fat_asymmetry", "Left-right fat asymmetry (ring)", "", 3),
    ("Previous definition (15 mm sphere)", "fat_mean_hu_median", "Fat attenuation (sphere)",
     "HU", 1),
    ("Previous definition (15 mm sphere)", "fat_fraction_mean", "Fat fraction (sphere)", "", 3),
    ("Previous definition (15 mm sphere)", "fat_volume_ml_sum", "Fat volume (sphere)", "mL", 0),
    ("Distension state and traced path", "gas_volume_ml", "Gas volume", "mL", 0),
    ("Distension state and traced path", "length_adjusted_gas_ml_per_cm",
     "Length-adjusted gas", "mL/cm", 1),
    ("Distension state and traced path", "collapse_ratio_pct", "Collapse ratio", "%", 1),
    ("Distension state and traced path", "luminal_radius_mm_median", "Luminal radius", "mm", 2),
    ("Distension state and traced path", "traced_centerline_length_cm",
     "Traced centerline length", "cm", 1),
    ("Distension state and traced path", "centerline_tortuosity_index",
     "Centerline tortuosity index", "", 2),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <tables>/manuscript")
    args = ap.parse_args()
    t = args.tables
    out = args.out or t / "manuscript"
    out.mkdir(parents=True, exist_ok=True)
    N: dict = {}

    # ---- Table 1: cohort and acquisition ---------------------------------
    cohort = pd.read_csv(t / "cohort_characteristics.csv")
    acq = pd.read_csv(t / "acquisition.csv")
    pos = pd.read_csv(t / "patient_position.csv")
    rows = [dict(r) for r in cohort.to_dict("records")]
    maker = {"GE MEDICAL SYSTEMS": "GE", "SIEMENS": "Siemens", "TOSHIBA": "Toshiba"}
    vendor = acq["manufacturer"].map(lambda m: maker.get(m, m)) + " " + acq["model"]
    for name, n in vendor.value_counts().items():
        rows.append({"characteristic": "Scanner (series)", "value": name, "n": int(n),
                     "percent": round(100 * n / len(acq), 1)})
    for col, label in (("kvp", "Tube voltage, kVp"),
                       ("slice_thickness_mm", "Slice thickness, mm"),
                       ("reconstruction_interval_mm", "Reconstruction interval, mm"),
                       ("pixel_spacing_mm", "In-plane pixel spacing, mm"),
                       ("tube_current_ma", "Tube current, mA"),
                       ("n_slices", "Slices per series")):
        v = pd.to_numeric(acq[col], errors="coerce")
        rows.append({"characteristic": label, "value": f"{med_iqr(v, 2)} (range "
                     f"{v.min():.2f}-{v.max():.2f})", "n": int(v.notna().sum()), "percent": ""})
    pd.DataFrame(rows).to_csv(out / "table1_cohort.csv", index=False)
    scored = pos[pos["hqcolon_position"].notna()]
    N["position_rule"] = {
        "n_scored": int(len(scored)),
        "tag_correct": int((scored["tag_agrees"].astype(str) == "True").sum()),
        "orientation_correct": int((scored["orientation_agrees"].astype(str) == "True").sum()),
    }
    N["n_vendors"] = int(acq["manufacturer"].nunique())

    # ---- Printed Table 3: the reference comparison -------------------------
    # The primary set is the 18 series of the 13 patients that were not
    # inspected case by case. The reference informed the extent diagnosis, the
    # seed rule and the core radius, so every value over all 26 series is
    # development-tuned, and that emphasis belongs the right way round. Each
    # ablation is therefore
    # restricted to the primary set as well -- a comparison between a tuned
    # column and an untuned one would not be a comparison of components -- and
    # the reported pipeline over all 26 series stands beside them under the
    # development-tuned label. The whole 26-series table, ablations included,
    # is written as a supplementary file so that nothing is withheld.
    evals = {"Adaptive (reported)": t / "eval_hqcolon_auto.csv",
             "3-D fill only": t / "eval_hqcolon_fill_bbox.csv",
             "Slice-wise fill only": t / "eval_hqcolon_fill_slicewise.csv",
             "Previous seed rule": t / "eval_hqcolon_seed_extremes.csv"}
    ev = {k: pd.read_csv(p) for k, p in evals.items() if p.exists()}

    def not_inspected(d: pd.DataFrame) -> pd.DataFrame:
        return d[~d["PatientID"].str[-4:].isin(DEVELOPMENT_PATIENTS)]

    ci = intervals(t / "clustered_statistics.csv")
    cols = {PRIMARY_COLUMN: not_inspected(ev["Adaptive (reported)"])}
    cols.update({k: not_inspected(d) for k, d in ev.items()
                 if k != "Adaptive (reported)"})
    cols[TUNED_COLUMN] = ev["Adaptive (reported)"]
    # Only the two columns that the text quotes carry intervals; an ablation is
    # read against the primary column beside it, not on its own.
    ci_for = {PRIMARY_COLUMN: SET_PRIMARY, TUNED_COLUMN: SET_TUNED}
    rows, missing = [], []
    for label, col, digits in PILLAR1:
        row = {"Metric": label}
        for k, d in cols.items():
            row[k] = med_iqr(d[col], digits)
            if k in ci_for:
                text = fmt_ci(ci.get((col, ci_for[k], "median")), digits)
                row[CI_LABEL[k]] = text
                if not text:
                    missing.append((label, k))
        rows.append(row)
    rows.append({"Metric": f"Segmentation failures (Dice < {SEG_FAILURE_DICE})",
                 **{k: f"{int((d['dice_lumen_vs_gas'] < SEG_FAILURE_DICE).sum())} / {len(d)}"
                    for k, d in cols.items()}})
    rows.append({"Metric": "Met the coverage criterion",
                 **{k: f"{int((d['colon_beyond_60mm_frac'] < REACHED_FAR_MAX).sum())}"
                       f" / {len(d)}" for k, d in cols.items()},
                 **{CI_LABEL[k]: fmt_ci(
                        ci.get(("colon_beyond_60mm_frac", s, "proportion")),
                        3, percent=True)
                    for k, s in ci_for.items()}})
    rows.append({"Metric": "Series / patients",
                 **{k: f"{len(d)} / {d['PatientID'].nunique()}" for k, d in cols.items()}})
    order = ["Metric", PRIMARY_COLUMN, CI_LABEL[PRIMARY_COLUMN]]
    order += [k for k in cols if k not in (PRIMARY_COLUMN, TUNED_COLUMN)]
    order += [TUNED_COLUMN, CI_LABEL[TUNED_COLUMN]]
    # fillna: a row with no interval (the counts) must leave the cell blank,
    # not print "NaN" into the typeset table.
    pd.DataFrame(rows).reindex(columns=order).fillna("").to_csv(
        out / "table2_reference.csv", index=False)
    if missing:
        # A blank interval cell is the one defect this table can carry without
        # looking wrong, so it is named here and checked in check_tables.py.
        print("  WARNING: no clustered interval for %d cell(s): %s"
              % (len(missing), "; ".join(f"{m} [{k}]" for m, k in missing)))

    # The same table over all 26 series, every column development-tuned, as a
    # supplementary file: the primary table above drops those eight series from
    # the ablation columns and this is where they remain visible.
    sup = {k: d for k, d in ev.items()}
    srows = []
    for label, col, digits in PILLAR1:
        srows.append({"Metric": label,
                      **{k: med_iqr(d[col], digits) for k, d in sup.items()}})
    srows.append({"Metric": f"Segmentation failures (Dice < {SEG_FAILURE_DICE})",
                  **{k: f"{int((d['dice_lumen_vs_gas'] < SEG_FAILURE_DICE).sum())} / {len(d)}"
                     for k, d in sup.items()}})
    srows.append({"Metric": "Met the coverage criterion",
                  **{k: f"{int((d['colon_beyond_60mm_frac'] < REACHED_FAR_MAX).sum())}"
                        f" / {len(d)}" for k, d in sup.items()}})
    srows.append({"Metric": "Series / patients",
                  **{k: f"{len(d)} / {d['PatientID'].nunique()}" for k, d in sup.items()}})
    pd.DataFrame(srows).to_csv(
        out / "table_s5_reference_development_tuned.csv", index=False)
    auto = ev["Adaptive (reported)"]
    N["pillar1"] = {col: stats(auto[col]) for _, col, _ in PILLAR1}
    N["pillar1"]["n_series"] = int(len(auto))
    N["pillar1"]["n_patients"] = int(auto["PatientID"].nunique())
    N["pillar1"]["seg_failures"] = {k: int((d["dice_lumen_vs_gas"] < SEG_FAILURE_DICE).sum())
                                    for k, d in ev.items()}
    N["pillar1"]["reference_centerline_mm"] = stats(auto["ref_gf_centerline_len_mm"])
    N["pillar1"]["reference_centerline_gas_only_mm"] = stats(auto["ref_centerline_len_mm"])
    low = auto[auto["colon_within_30mm_frac"] < COVERAGE_FLOOR].sort_values(
        "colon_within_30mm_frac")
    N["pillar1"]["low_coverage_series"] = [
        {"series": f"{r.PatientID[-4:]}-{1 if r.role == 'primary' else 2}",
         "coverage": round(r.colon_within_30mm_frac, 3), "beyond_60mm": round(
             r.colon_beyond_60mm_frac, 3), "dice": round(r.dice_lumen_vs_gas, 3)}
        for r in low.itertuples()]
    N["pillar1"]["coverage_criterion_met"] = int(
        (auto["colon_beyond_60mm_frac"] < REACHED_FAR_MAX).sum())
    # The reference entered development twice over: five series were inspected
    # case by case, and the candidate seed rules were scored on all 26. The
    # column below only removes the first of those, so it is an additional
    # analysis, not an independent validation.
    ho = cols[PRIMARY_COLUMN]
    # Named for what it is: the primary set of the manuscript, and still not an
    # independent test set, because the candidate seed rules were scored
    # against all 26 reference series before the pipeline was changed.
    N["pillar1"]["primary_set_patients_not_inspected"] = {
        "inspected_series": DEVELOPMENT_SERIES,
        "independent": False,
        "held_out_test_set_exists": False,
        "n_series": int(len(ho)), "n_patients": int(ho["PatientID"].nunique()),
        "dice_lumen_vs_gas": stats(ho["dice_lumen_vs_gas"]),
        "colon_within_30mm_frac": stats(ho["colon_within_30mm_frac"]),
        "colon_beyond_60mm_frac": stats(ho["colon_beyond_60mm_frac"]),
        "centerline_inside_gas_frac": stats(ho["centerline_inside_gas_frac"]),
        "length_ratio_gf": stats(ho["length_ratio_gf"]),
        "coverage_criterion_met": int((ho["colon_beyond_60mm_frac"] < REACHED_FAR_MAX).sum()),
    }
    N["pillar1"]["colon_mostly_missed"] = int((auto["colon_beyond_60mm_frac"] > 0.3).sum())
    for label, d in ev.items():
        N["pillar1"].setdefault("by_configuration", {})[label] = {
            "colon_within_30mm": stats(d["colon_within_30mm_frac"]),
            "coverage_criterion_met": int((d["colon_beyond_60mm_frac"] < REACHED_FAR_MAX).sum()),
            "inside": stats(d["centerline_inside_gas_frac"]),
        }
    if "3-D fill only" in ev:
        m = auto.merge(ev["3-D fill only"], on=["PatientID", "role"], suffixes=("", "_3d"))
        delta = m["dice_lumen_vs_gas"] - m["dice_lumen_vs_gas_3d"]
        N["pillar1"]["adaptive_vs_3d"] = {"improved": int((delta > 0.02).sum()),
                                          "degraded": int((delta < -0.02).sum()),
                                          "unchanged_within_0.02": int((delta.abs() <= 0.02).sum())}
    if "3-D fill only" in ev:
        d3 = ev["3-D fill only"]
        lost = d3[d3["dice_lumen_vs_gas"] < SEG_FAILURE_DICE]
        N["pillar1"]["3d_failures"] = {
            "n": int(len(lost)),
            "reference_gas_ml_range": [round(float(lost["ref_gas_ml"].min()), 0),
                                       round(float(lost["ref_gas_ml"].max()), 0)],
            "retained_air_ml_range": [round(float(lost["our_air_ml"].min()), 0),
                                      round(float(lost["our_air_ml"].max()), 0)],
            "dice_max": round(float(lost["dice_lumen_vs_gas"].max()), 3),
            "dice_min_of_the_rest": round(float(
                d3.loc[d3["dice_lumen_vs_gas"] >= SEG_FAILURE_DICE, "dice_lumen_vs_gas"].min()), 3),
        }
    # The gas-volume index against the reference's gas-filled colon volume.
    idx = pd.read_csv(t / "indices.csv")[["PatientID", "role", "gas_volume_ml"]]
    gv = auto.merge(idx, on=["PatientID", "role"])

    def gas_volume_agreement(g: pd.DataFrame) -> dict:
        rel = (g["gas_volume_ml"] - g["ref_gas_ml"]) / g["ref_gas_ml"]
        return {
            "n": int(len(g)),
            "pearson_r": round(float(np.corrcoef(
                g["gas_volume_ml"], g["ref_gas_ml"])[0, 1]), 3),
            "relative_error": stats(rel),
        }

    # On the primary set as well as on all 26, so that the Results paragraph
    # which is now primary does not have to borrow a development-tuned number.
    N["pillar1"]["gas_volume_vs_reference"] = gas_volume_agreement(gv)
    N["pillar1"]["primary_set_patients_not_inspected"][
        "gas_volume_vs_reference"] = gas_volume_agreement(not_inspected(gv))
    if "Slice-wise fill only" in ev:
        m = auto.merge(ev["Slice-wise fill only"], on=["PatientID", "role"], suffixes=("", "_2d"))
        delta = m["dice_lumen_vs_gas_2d"] - m["dice_lumen_vs_gas"]
        N["pillar1"]["slicewise_vs_adaptive"] = {
            "worse_by_more_than_0.02": int((delta < -0.02).sum()),
            "better_by_more_than_0.02": int((delta > 0.02).sum()),
            "within_0.02": int((delta.abs() <= 0.02).sum())}

    # ---- Table 3: cohort outcome, prototype batch vs this pipeline ---------
    before = outcome_summary(pd.read_csv(t / "centerline_batch.csv"))
    after = outcome_summary(pd.read_csv(t / "centerline_auto.csv"))
    N["cohort_outcome"] = {"batch_2026_08": before, "reported": after}

    def fmt(s: dict, key: str) -> str:
        v = s[key]
        if isinstance(v, dict):
            return f"{v['median']:.1f} [{v['q1']:.1f}, {v['q3']:.1f}]"
        return f"{v} / {s['n_series']}" if key != "patients_usable_both" and \
            key != "patients_produced_both" else f"{v} / 30"
    keys = [("Centerline produced", "produced"),
            (f"Failed (error or < {FAILED_BELOW_MM:.0f} mm)", "failed"),
            ("Centerline length, mm", "length_mm"),
            ("Patients usable in both positions", "patients_usable_both"),
            ("Series with at least one bridge", "series_with_bridges"),
            ("Run time per series, s", "runtime_sec")]
    rows = [{"outcome": lab, "prototype batch (2026-08)": fmt(before, k),
             "this pipeline": fmt(after, k)} for lab, k in keys]
    rows.append({"outcome": "Slice-wise fill fallback", "prototype batch (2026-08)": "n/a",
                 "this pipeline": f"{after['slicewise_fallback']} / {after['n_series']}"})
    pd.DataFrame(rows).to_csv(out / "table3_cohort_outcome.csv", index=False)

    # ---- Printed Table 4: comparison with something else --------------------
    # This pipeline has to be set beside something that is not itself: a
    # published open centerline or segmentation approach, or published values
    # from another group on the same collection.
    #
    # Printed Table 2 already sets the prototype beside this pipeline, but on
    # cohort outcomes only: both of its columns are this pipeline's own output
    # measured against itself, so it compares two configurations and not either
    # of them with anything external. Here the prototype is measured against
    # the same HQColon reference, on the same metrics and the same series as
    # the reported pipeline, and the traced length is placed beside published
    # colon lengths.
    #
    # The published values are literature constants and live here rather than
    # in the prose, so that the table and the sentences that quote it resolve
    # to one place. Read from Crossref and from each paper's own text on
    # 2026-11-04; the citations are in the manuscript's reference file.
    PUBLISHED_COLON_LENGTH_CM = [
        ("Utano 2021", "CT colonography, computer-generated centerline, "
                       "295 patients", "150.3 +/- 18.5", "109.7-195.9",
         (109.7, 195.9)),
        ("Alqarni 2024", "Meta-analysis of 31 studies, all techniques, "
                         "58 study means", "148.3 +/- 27.1", "93.5-210.8",
         (93.5, 210.8)),
    ]

    proto_eval = t / "eval_hqcolon_batch.csv"
    if not proto_eval.exists():
        raise SystemExit(
            "%s is missing, so Table 4 cannot be built. It is the prototype "
            "configuration evaluated against the same reference:\n"
            "  python scripts/eval_hqcolon.py --overlap %s/hqcolon_overlap.csv "
            "--centerline-root data/centerline/batch --position "
            "%s/patient_position.csv --out %s"
            % (proto_eval.name, t, t, proto_eval))

    proto_all = not_inspected(pd.read_csv(proto_eval))
    rep_primary = cols[PRIMARY_COLUMN]
    # A row with no Dice is a series for which the prototype produced no
    # centerline, so there was nothing to evaluate. Those rows must not be
    # counted as evaluated and must not be dropped silently either: the count
    # is a row of the table.
    proto = proto_all[proto_all["dice_lumen_vs_gas"].notna()]
    produced_nothing = [
        "%s-%s" % (r.PatientID[-4:], 1 if r.role == "primary" else 2)
        for r in proto_all[proto_all["dice_lumen_vs_gas"].isna()].itertuples()]

    PROTO = "Prototype (2026-08 batch)"
    REPORTED = "Reported pipeline"
    PUBLISHED = "Published"
    SOURCE = "Source"

    def row(measure, proto_cell, rep_cell, published="", source=""):
        return {"Measure": measure, PROTO: proto_cell, REPORTED: rep_cell,
                PUBLISHED: published, SOURCE: source}

    def fraction(d: pd.DataFrame, mask) -> str:
        return "%d / %d" % (int(mask(d).sum()), len(d))

    rows = [row("Reference series evaluated, of %d" % len(rep_primary),
                "%d / %d" % (len(proto), len(rep_primary)),
                "%d / %d" % (len(rep_primary), len(rep_primary))),
            row("No centerline produced",
                "%d / %d" % (len(produced_nothing), len(rep_primary)),
                "0 / %d" % len(rep_primary))]
    for label, col, digits in PILLAR1:
        rows.append(row(label, med_iqr(proto[col], digits),
                        med_iqr(rep_primary[col], digits)))
    # The row that keeps the Dice row from being read on its own: the
    # prototype's median overlap is the higher of the two and five of its
    # series lost the colon completely.
    rows.append(row("Segmentation failures (Dice < %s)" % SEG_FAILURE_DICE,
                    fraction(proto, lambda d: d["dice_lumen_vs_gas"]
                             < SEG_FAILURE_DICE),
                    fraction(rep_primary, lambda d: d["dice_lumen_vs_gas"]
                             < SEG_FAILURE_DICE)))
    rows.append(row("Met the coverage criterion",
                    fraction(proto, lambda d: d["colon_beyond_60mm_frac"]
                             < REACHED_FAR_MAX),
                    fraction(rep_primary, lambda d: d["colon_beyond_60mm_frac"]
                             < REACHED_FAR_MAX)))

    # The traced length over the whole cohort, which is the quantity the
    # published studies measure. Centimetres, because that is their unit.
    cl_proto = pd.read_csv(t / "centerline_batch.csv")["length_mm"] / 10.0
    cl_rep = pd.read_csv(t / "centerline_auto.csv")["length_mm"] / 10.0
    key, _what, value, rng, (lo, hi) = PUBLISHED_COLON_LENGTH_CM[0]
    rows.append(row("Traced centerline length over the cohort, cm",
                    med_iqr(cl_proto, 1), med_iqr(cl_rep, 1), value, key))
    for k2, _w2, v2, _r2, _b2 in PUBLISHED_COLON_LENGTH_CM[1:]:
        rows.append(row("Total colon length, cm", "", "", v2, k2))
    rows.append(row("Range of the traced length, cm",
                    "%.1f-%.1f" % (cl_proto.min(), cl_proto.max()),
                    "%.1f-%.1f" % (cl_rep.min(), cl_rep.max()), rng, key))
    # How many series the published range actually contains. A median inside a
    # published range says nothing about how many series are.
    def inside(x):
        return "%d / %d (%d shorter, %d longer)" % (
            int(((x >= lo) & (x <= hi)).sum()), len(x),
            int((x < lo).sum()), int((x > hi).sum()))
    rows.append(row("Series inside the published range", inside(cl_proto),
                    inside(cl_rep), rng, key))
    pd.DataFrame(rows).reindex(
        columns=["Measure", PROTO, REPORTED, PUBLISHED, SOURCE]).fillna("").to_csv(
        out / "table4_comparators.csv", index=False)

    N["comparators"] = {
        "prototype_reference_evaluation": {
            "n_series_evaluated": int(len(proto)),
            "n_patients": int(proto["PatientID"].nunique()),
            "produced_no_centerline": produced_nothing,
            **{col: stats(proto[col]) for _, col, _ in PILLAR1
               if col in proto.columns},
            "segmentation_failures": int(
                (proto["dice_lumen_vs_gas"] < SEG_FAILURE_DICE).sum()),
            "coverage_criterion_met": int(
                (proto["colon_beyond_60mm_frac"] < REACHED_FAR_MAX).sum()),
        },
        "traced_length_cm": {
            "prototype": stats(cl_proto), "reported": stats(cl_rep),
            "prototype_range": [round(float(cl_proto.min()), 1),
                                round(float(cl_proto.max()), 1)],
            "reported_range": [round(float(cl_rep.min()), 1),
                               round(float(cl_rep.max()), 1)],
            "inside_published_range": {
                "bounds_cm": [lo, hi], "source": key,
                "prototype": int(((cl_proto >= lo) & (cl_proto <= hi)).sum()),
                "prototype_shorter": int((cl_proto < lo).sum()),
                "prototype_longer": int((cl_proto > hi).sum()),
                # A series the prototype produced no path for has no length,
                # so it is neither inside the range nor shorter than it, and
                # the three counts do not add up to 60 without it.
                "prototype_no_path": int(cl_proto.isna().sum()),
                "reported": int(((cl_rep >= lo) & (cl_rep <= hi)).sum()),
                "reported_shorter": int((cl_rep < lo).sum()),
                "reported_longer": int((cl_rep > hi).sum()),
                "reported_no_path": int(cl_rep.isna().sum()),
                "n": int(len(cl_rep)),
            },
        },
        "published_total_colon_length_cm": [
            {"source": k, "cohort": w, "value": v, "range": r}
            for k, w, v, r, _b in PUBLISHED_COLON_LENGTH_CM],
        "searched_for_published_results_on_this_collection": {
            "date": "2026-11-04",
            "database": "Europe PMC REST",
            "found": ["Finocchiaro 2026"],
            "note": "the only published work on this collection is the "
                    "reference segmentation itself, so a comparison with it "
                    "is Evaluation 1 and not an independent comparator",
        },
    }

    # ---- Printed Tables 8 and 9: where coverage fails, and what is excluded --
    # Coverage by position and the excluded volume, from
    # scripts/analyse_coverage_anatomy.py.
    cov_path = t / "coverage_anatomy.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        cov_primary = not_inspected(cov)
        parts = sorted(s for s in cov_primary["segment"].dropna().unique()
                       if s.startswith("part "))

        def q(x, digits=1, scale=100.0):
            x = pd.to_numeric(x, errors="coerce").dropna() * scale
            if x.empty:
                return ""
            lo, mid, hi = np.percentile(x, [25, 50, 75])
            return f"{mid:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

        # Table 8: the positional profile. POORLY_COVERED is the criterion the
        # paper already uses for a series, applied to a part.
        rows = []
        for seg in ["whole colon", *parts]:
            x = cov_primary[cov_primary["segment"] == seg]
            if x.empty:
                continue
            beyond = pd.to_numeric(x["beyond_60mm_frac"], errors="coerce")
            rows.append({
                "Region": seg,
                "Colon volume, mL": q(x["colon_ml"], 0, 1.0),
                "Within 30 mm of the path, %": q(x["within_30mm_frac"]),
                "Beyond 60 mm of the path, %": q(x["beyond_60mm_frac"]),
                "Series with any colon beyond 60 mm":
                    f"{int((beyond > 0).sum())} / {len(x)}",
                "Series failing the coverage criterion here":
                    f"{int((beyond >= REACHED_FAR_MAX).sum())} / {len(x)}",
                "Dice, lumen vs gas reference":
                    q(x["dice_lumen_vs_gas_in_part"], 3, 1.0),
            })
        pd.DataFrame(rows).to_csv(out / "table8_coverage_by_position.csv",
                                  index=False)

        # Table 9: comment 3. The colon outside the gas mask by construction,
        # and the share of the never-visited colon that it accounts for.
        whole = cov_primary[cov_primary["segment"] == "whole colon"]
        beyond = pd.to_numeric(whole["beyond_60mm_frac"], errors="coerce")
        has_miss = whole[beyond > 0]
        rows = [
            {"Quantity": "Colon that is fluid-filled or collapsed, % of colon",
             "Value": q(whole["fluid_or_collapsed_frac"]),
             "Series": len(whole)},
            {"Quantity": "Of that volume, the share above 100 HU (tagged), %",
             "Value": q(whole["excluded_tagged_frac"]),
             "Series": len(whole)},
            {"Quantity": "Attenuation of that volume, HU",
             "Value": q(whole["excluded_hu_median"], 0, 1.0),
             "Series": len(whole)},
            {"Quantity": "Series with any colon beyond 60 mm of the path",
             "Value": f"{len(has_miss)} / {len(whole)}", "Series": len(whole)},
            {"Quantity": "Of the colon beyond 60 mm, the share that was gas, %",
             "Value": q(has_miss["beyond_60mm_that_was_gas_frac"]),
             "Series": len(has_miss)},
            {"Quantity": "Of the colon beyond 60 mm, the share excluded by "
                         "construction, %",
             "Value": q(has_miss["beyond_60mm_that_was_excluded_frac"]),
             "Series": len(has_miss)},
        ]
        pd.DataFrame(rows).to_csv(out / "table9_excluded_volume.csv", index=False)

        N["coverage_by_position"] = {
            "n_series": int(len(whole)), "parts": len(parts),
            "fluid_or_collapsed_frac": stats(whole["fluid_or_collapsed_frac"]),
            "series_with_any_colon_beyond_60mm": int(len(has_miss)),
            "beyond_60mm_that_was_gas_frac": stats(
                has_miss["beyond_60mm_that_was_gas_frac"]) if len(has_miss) else {},
            "excluded_tagged_frac": stats(whole["excluded_tagged_frac"]),
            "anatomical_segments_available": False,
        }
    else:
        print("  WARNING: coverage_anatomy.csv is missing, so Tables 8 and 9 "
              "were not built; run scripts/analyse_coverage_anatomy.py")

    # ---- Printed Table 10: acquisition and thresholds ----------------------
    strat_path = t / "stratified_acquisition.csv"
    sweep_path = t / "threshold_sweep.csv"
    if strat_path.exists():
        strat = pd.read_csv(strat_path, keep_default_na=False)
        keep = ["axis", "stratum", "series", "patients",
                "met the length criterion", "Dice, lumen vs gas reference",
                "Colon within 30 mm", "Colon beyond 60 mm",
                "met the coverage criterion", "note"]
        strat.reindex(columns=[c for c in keep if c in strat.columns]).to_csv(
            out / "table_s6_acquisition.csv", index=False)
    if sweep_path.exists():
        sw = pd.read_csv(sweep_path)
        sw = sw[sw.get("error", pd.Series(dtype=object)).isna()] \
            if "error" in sw.columns else sw
        rows = []
        for name, g in sw.groupby("setting", sort=False):
            rows.append({
                "Setting": name,
                "Body threshold, HU": int(g["body_threshold_hu"].iloc[0]),
                "Gas threshold, HU": int(g["gas_threshold_hu"].iloc[0]),
                "Series": len(g),
                "Gas mask, mL": med_iqr(g["air_ml"], 0),
                "Dice, lumen vs gas reference": med_iqr(g["dice_lumen_vs_gas"], 3),
                "Within 30 mm, %": med_iqr(g["colon_within_30mm_frac"] * 100, 1),
                "Beyond 60 mm, %": med_iqr(g["colon_beyond_60mm_frac"] * 100, 1),
                "Traced length, mm": med_iqr(g["traced_length_mm"], 0),
                # The median steps over a total loss: body -400 keeps a
                # median Dice of 0.882 while four series lose the colon
                # completely, which is the same way a median hid five total
                # failures in Table 4.
                "Segmentation failures (Dice < 0.5)":
                    f"{int((g['dice_lumen_vs_gas'] < SEG_FAILURE_DICE).sum())} / {len(g)}",
                "Below the length criterion":
                    f"{int((g['traced_length_mm'] < FAILED_BELOW_MM).sum())} / {len(g)}",
                "Met the coverage criterion":
                    f"{int(g['met_coverage'].sum())} / {len(g)}",
            })
        pd.DataFrame(rows).to_csv(out / "table10_threshold_sweep.csv", index=False)
        N["threshold_sweep"] = {
            "settings": int(sw["setting"].nunique()),
            "series": int(sw.groupby(["PatientID", "role"]).ngroups),
        }
    else:
        print("  WARNING: threshold_sweep.csv is missing, so the sweep table "
              "was not built; run scripts/sweep_thresholds.py")

    # ---- Printed Table 11: polyp measurement under realistic conditions ----
    real_path = t / "polyp_realism.csv"
    if real_path.exists():
        rl = pd.read_csv(real_path)
        measured = rl[rl["ok"]]
        rows = []
        for name, g in rl.groupby("condition", sort=False):
            gm = g[g["ok"]]
            usable = gm[~gm["leaked"].astype(bool)]
            err = usable["grayscale_diameter_error_mm"].astype(float)
            near = usable[
                usable["true_max_diameter_mm"].apply(
                    lambda d: min(abs(d - 6.0), abs(d - 10.0)) <= 0.5)]
            rows.append({
                "Condition": name,
                "Background": {"air": "air", "fluid": "fluid, 20 HU"}.get(
                    g["geometry"].iloc[0], "air, with wall"),
                "Contact": {"air": "none", "fluid": "none",
                            "wall_tangent": "wall, at a point",
                            "wall_sessile": "wall, through the centre",
                            "fold": "fold, two sides"}[g["geometry"].iloc[0]],
                "Noise, HU": round(float(g["noise_sd_hu"].iloc[0]), 1),
                "Measurements": len(g),
                # A segmentation that left the lesion is counted, never
                # averaged into the error: the median of a mixture of
                # measurements and escapes is neither.
                "Left the lesion": "%d / %d" % (
                    int(gm["leaked"].astype(bool).sum()), len(g)),
                "Usable": len(usable),
                "Diameter error, mm":
                    med_iqr(err, 2) if len(usable) else "not measurable",
                "95th percentile |error|, mm":
                    "%.2f" % float(np.percentile(err.abs(), 95))
                    if len(usable) else "-",
                "Wrong size class": "%d / %d" % (
                    int(usable["class_changed"].astype(bool).sum()), len(usable))
                    if len(usable) else "-",
                "Wrong size class within 0.5 mm of a threshold": "%d / %d" % (
                    int(near["class_changed"].astype(bool).sum()), len(near))
                    if len(near) else "-",
            })
        pd.DataFrame(rows).to_csv(out / "table11_polyp_realism.csv", index=False)
        # The per-measurement rows behind it, as a supplementary file.
        rl.to_csv(out / "table_s8_polyp_realism_rows.csv", index=False)
        N["polyp_realism"] = {
            "conditions": int(rl["condition"].nunique()),
            "measurements": int(len(rl)),
            "measured": int(len(measured)),
            "repeats": int(rl["repeat"].max()) + 1,
        }
    else:
        print("  WARNING: polyp_realism.csv is missing, so Table 11 was not "
              "built; run scripts/eval_polyp_realism.py")

    # ---- Printed Table 12: the annotated lesions of the cohort -------------
    les_path = t / "cohort_lesions.csv"
    if les_path.exists():
        les = pd.read_csv(les_path, keep_default_na=False)
        printed = pd.DataFrame({
            "Patient": les["PatientID"].str[-4:],
            "Segment": les["segment"],
            "Reported size, mm": les["reported_size_mm"],
            "Size taken at": les["size_source"],
            "Histology": les["histology"],
            "Slice, supine": les["slice_supine"].replace("", "not given"),
            "Slice, prone": les["slice_prone"].replace("", "not given"),
            "In-plane coordinate": "not given",
        })
        printed.to_csv(out / "table12_cohort_lesions.csv", index=False)
        N["cohort_lesions"] = {"lesions": int(len(les))}
    else:
        print("  WARNING: cohort_lesions.csv is missing, so Table 12 was not "
              "built; run scripts/report_cohort_lesions.py")

    # ---- Table 4: prone/supine agreement ------------------------------------
    agr = pd.read_csv(t / "prone_supine_agreement.csv").set_index("index")
    rows = []
    for construct, idx, label, unit, d in AGREEMENT:
        r = agr.loc[idx]
        rows.append({
            "construct": construct, "index": label, "unit": unit,
            "prone mean (SD)": f"{r.prone_mean:.{d}f} ({r.prone_sd:.{d}f})",
            "supine mean (SD)": f"{r.supine_mean:.{d}f} ({r.supine_sd:.{d}f})",
            "ICC(2,1) [95% CI]": f"{r.icc21:.3f} [{r.icc_lower:.3f}, {r.icc_upper:.3f}]",
            "bias [95% CI]": f"{r.bias_prone_minus_supine:.{d + 1}f} "
                             f"[{r.bias_ci_lower:.{d + 1}f}, {r.bias_ci_upper:.{d + 1}f}]",
            "95% limits of agreement": f"{r.loa_lower:.{d}f} to {r.loa_upper:.{d}f}",
        })
    pd.DataFrame(rows).to_csv(out / "table4_agreement.csv", index=False)

    # The exploratory composites, in a table of their own. Same columns, so a
    # reader can compare if they wish; the paper does not.
    erows = []
    for construct, idx, label, unit, d in EXPLORATORY_COMPOSITES:
        if idx not in agr.index:
            continue
        r = agr.loc[idx]
        erows.append({
            "construct": construct, "index": label, "unit": unit,
            "prone mean (SD)": f"{r.prone_mean:.{d}f} ({r.prone_sd:.{d}f})",
            "supine mean (SD)": f"{r.supine_mean:.{d}f} ({r.supine_sd:.{d}f})",
            "ICC(2,1) [95% CI]": f"{r.icc21:.3f} [{r.icc_lower:.3f}, {r.icc_upper:.3f}]",
            "bias [95% CI]": f"{r.bias_prone_minus_supine:.{d + 1}f} "
                             f"[{r.bias_ci_lower:.{d + 1}f}, {r.bias_ci_upper:.{d + 1}f}]",
            "95% limits of agreement": f"{r.loa_lower:.{d}f} to {r.loa_upper:.{d}f}",
        })
    if erows:
        pd.DataFrame(erows).to_csv(
            out / "table_s7_exploratory_composites.csv", index=False)
    N["exploratory_composites"] = [idx for _c, idx, *_ in EXPLORATORY_COMPOSITES]
    # The original theta-half asymmetry is not a left/right measure (it follows the
    # direction of travel); it is kept out of Table 4 but reported here so the two
    # definitions can be compared.
    N["asymmetry_definitions"] = {
        k: {"icc21": round(float(agr.loc[k, "icc21"]), 3),
            "ci": [round(float(agr.loc[k, "icc_lower"]), 3),
                   round(float(agr.loc[k, "icc_upper"]), 3)]}
        for k in ("fat_asymmetry", "fat_asymmetry_theta_halves") if k in agr.index
    }
    idx_all = pd.read_csv(t / "indices.csv")
    N["fat_sampling"] = {
        "ring_valid_fraction": stats(idx_all["ring_valid_fraction"]),
        "sphere_valid_fraction": stats(idx_all["fat_valid_fraction"]),
        "ring_valid_fraction_min": round(float(idx_all["ring_valid_fraction"].min()), 3),
        "ring_all_stations_valid": int((idx_all["ring_valid_fraction"] >= 1.0).sum()),
        "sphere_valid_fraction_min": round(float(idx_all["fat_valid_fraction"].min()), 3),
    }
    # How much of the paired cohort could be checked against the reference at
    # all (scripts/pair_reference_status.py writes the per-patient audit). The
    # earlier sensitivity analysis ("whole colon reached, 19 pairs") counted
    # series with no reference as adequate; on the reference itself only 3 pairs
    # qualify, too few for agreement statistics, so none is reported.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pair_reference_status import build as pair_audit

    pairs = pair_audit(t)
    paired = pairs[pairs["paired"]]
    N["agreement"] = {"n_pairs": int(agr["n_pairs"].iloc[0]),
                      "n_patients_two_reference_series": int(
                          (auto.groupby("PatientID").size() == 2).sum()),
                      "n_pairs_reference_both_positions": int(paired["reference_both"].sum()),
                      "n_pairs_coverage_criterion_both_positions": int(
                          paired["criterion_both"].sum()),
                      "n_pairs_reference_one_position": int(
                          (paired["prone_has_reference"] ^ paired["supine_has_reference"]).sum()),
                      "n_pairs_no_reference": int((~paired["prone_has_reference"]
                                                   & ~paired["supine_has_reference"]).sum()),
                      # Eight patients have two reference series, but 0003's two
                      # series are both recorded supine, so only seven of them
                      # form a prone/supine pair.
                      "patients_two_references_not_a_pair": sorted(
                          set(auto.groupby("PatientID").size()[lambda x: x == 2].index.str[-4:])
                          - set(paired[paired["reference_both"]]["patient"])),
                      "sensitivity_analysis_reported": False,
                      "bias_ci_excludes_zero": [i for i in agr.index
                                                if agr.loc[i, "bias_ci_lower"] > 0
                                                or agr.loc[i, "bias_ci_upper"] < 0]}

    # ---- Table 5: polyp phantoms --------------------------------------------
    ph = pd.read_csv(t / "polyp_phantom_accuracy.csv")
    cols = ["phantom", "edge", "true_max_diameter_mm", "grayscale_diameter_error_mm",
            "binary_diameter_error_mm", "true_volume_mm3", "mesh_volume_error_pct",
            "voxel_count_volume_error_pct", "sphericity"]
    ph[cols].round(3).to_csv(out / "table5_phantoms.csv", index=False)
    spheres = ph[ph["phantom"].str.startswith("sphere") & (ph["edge"] == "ramp")]
    step = ph[ph["edge"] == "step"]
    N["phantoms"] = {
        "grayscale_abs_error_max_mm": round(float(spheres["grayscale_diameter_error_mm"]
                                                  .abs().max()), 3),
        "binary_abs_error_max_mm": round(float(spheres["binary_diameter_error_mm"]
                                               .abs().max()), 3),
        "mesh_volume_error_pct_range": [round(float(spheres["mesh_volume_error_pct"].min()), 1),
                                        round(float(spheres["mesh_volume_error_pct"].max()), 1)],
        "voxel_volume_error_pct_range": [
            round(float(spheres["voxel_count_volume_error_pct"].min()), 1),
            round(float(spheres["voxel_count_volume_error_pct"].max()), 1)],
        "sphericity_range": [round(float(spheres["sphericity"].min()), 3),
                             round(float(spheres["sphericity"].max()), 3)],
        "grayscale_error_range_mm": [round(float(spheres["grayscale_diameter_error_mm"].min()), 3),
                                     round(float(spheres["grayscale_diameter_error_mm"].max()), 3)],
        "binary_error_range_mm": [round(float(spheres["binary_diameter_error_mm"].min()), 3),
                                  round(float(spheres["binary_diameter_error_mm"].max()), 3)],
        "step_edge_error_mm": {
            "grayscale": sorted({round(float(v), 3) for v in step["grayscale_diameter_error_mm"]}),
            "binary": sorted({round(float(v), 3) for v in step["binary_diameter_error_mm"]}),
        },
    }

    (out / "numbers.json").write_text(json.dumps(N, indent=2), encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
