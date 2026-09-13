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
    ("Distension state and traced path", "distension_quality_score",
     "Exploratory distension score", "", 1),
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

    # ---- Table 2: pillar 1, with the hole-fill ablation --------------------
    evals = {"Adaptive (reported)": t / "eval_hqcolon_auto.csv",
             "3-D fill only": t / "eval_hqcolon_fill_bbox.csv",
             "Slice-wise fill only": t / "eval_hqcolon_fill_slicewise.csv",
             "Previous seed rule": t / "eval_hqcolon_seed_extremes.csv"}
    ev = {k: pd.read_csv(p) for k, p in evals.items() if p.exists()}
    held = {"Reported pipeline, inspected patients excluded":
            ev["Adaptive (reported)"][~ev["Adaptive (reported)"]["PatientID"]
                                      .str[-4:].isin(DEVELOPMENT_PATIENTS)]}
    cols = {**ev, **held}
    rows = []
    for label, col, digits in PILLAR1:
        rows.append({"metric": label, **{k: med_iqr(d[col], digits) for k, d in cols.items()}})
    rows.append({"metric": f"Segmentation failures (Dice < {SEG_FAILURE_DICE})",
                 **{k: f"{int((d['dice_lumen_vs_gas'] < SEG_FAILURE_DICE).sum())} / {len(d)}"
                    for k, d in cols.items()}})
    rows.append({"metric": "Series / patients",
                 **{k: f"{len(d)} / {d['PatientID'].nunique()}" for k, d in cols.items()}})
    pd.DataFrame(rows).to_csv(out / "table2_reference.csv", index=False)
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
    ho = held["Reported pipeline, inspected patients excluded"]
    N["pillar1"]["excluding_inspected_patients"] = {
        "inspected_series": DEVELOPMENT_SERIES,
        "independent": False,
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
    rel = (gv["gas_volume_ml"] - gv["ref_gas_ml"]) / gv["ref_gas_ml"]
    N["pillar1"]["gas_volume_vs_reference"] = {
        "n": int(len(gv)),
        "pearson_r": round(float(np.corrcoef(gv["gas_volume_ml"], gv["ref_gas_ml"])[0, 1]), 3),
        "relative_error": stats(rel),
    }
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
