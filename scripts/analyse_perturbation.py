"""Summarise the centerline perturbation experiment.

Follows section 5 and 6 of ``docs/PLAN_CENTERLINE_PERTURBATION.md``: paired
per series against its own control, medians with interquartile ranges over
series, and percentile intervals from resampling **patients**, so that the two
positions of a patient and the eight conditions of a series are never treated
as independent observations.

    python scripts/analyse_perturbation.py

Writes ``results/tables/manuscript/table5_perturbation_path.csv`` and
``table6_perturbation_fat.csv`` (one row per
condition) and the numbers for the manuscript to
``results/tables/manuscript/perturbation_numbers.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP = 2000
SEED = 20260912

# Two tables: the traced path and where it runs, then the ring fat indices.
# One table of all ten outcomes needed fourteen columns and ran off the page.
PATH_OUTCOMES = [
    ("traced_length_rel_change", "Traced length, relative change", 100.0, "%", 2),
    ("points_outside_lumen_frac", "Centerline points outside the lumen", 100.0, "%", 1),
    ("ring_valid_fraction", "Stations with a valid ring measurement", 100.0, "%", 1),
    ("residual_rms_after_smoothing_mm", "Displacement between the smoothed paths (RMS)",
     1.0, "mm", 3),
]
FAT_OUTCOMES = [
    ("d_fat_hu", "Ring fat attenuation, change", 1.0, "HU", 2),
    ("abs_d_fat_hu", "Ring fat attenuation, absolute change", 1.0, "HU", 2),
    ("d_fat_fraction", "Ring fat fraction, change", 1.0, "", 4),
    ("abs_d_fat_fraction", "Ring fat fraction, absolute change", 1.0, "", 4),
    ("d_asymmetry", "Asymmetry index, change", 1.0, "", 4),
    ("abs_d_asymmetry", "Asymmetry index, absolute change", 1.0, "", 4),
]
OUTCOMES = PATH_OUTCOMES + FAT_OUTCOMES


def med_iqr(x: pd.Series, scale: float, digits: int) -> str:
    v = pd.to_numeric(x, errors="coerce").dropna() * scale
    if v.empty:
        return "n/a"
    return (f"{v.median():.{digits}f} [{v.quantile(0.25):.{digits}f}, "
            f"{v.quantile(0.75):.{digits}f}]")


def patient_bootstrap(df: pd.DataFrame, column: str, scale: float,
                      rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval for the median, resampling patients with replacement."""
    by_patient = {p: pd.to_numeric(g[column], errors="coerce").dropna().to_numpy() * scale
                  for p, g in df.groupby("PatientID")}
    patients = [p for p, v in by_patient.items() if v.size]
    if len(patients) < 3:
        return float("nan"), float("nan")
    meds = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(patients, size=len(patients), replace=True)
        vals = np.concatenate([by_patient[p] for p in pick])
        meds.append(np.median(vals))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perturbation", type=Path,
                    default=Path("results/tables/centerline_perturbation.csv"))
    ap.add_argument("--centerline", type=Path,
                    default=Path("results/tables/centerline_auto.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/manuscript"))
    args = ap.parse_args()

    df = pd.read_csv(args.perturbation)
    # The plan restricts the experiment to series meeting the centerline success
    # criterion. The run itself covered every series with a centerline, because
    # the first version of the filter missed the length criterion; the four
    # series below 600 mm are dropped here and their rows stay in the CSV,
    # unused (see docs/PLAN_CENTERLINE_PERTURBATION.md section 8).
    cl = pd.read_csv(args.centerline)
    eligible = {(r.PatientID, r.role) for r in cl.itertuples()
                if str(r.ok) == "True" and float(r.length_mm or 0) >= 600.0}
    before = df[["PatientID", "role"]].drop_duplicates().shape[0]
    df = df[[(p, r) in eligible for p, r in zip(df["PatientID"], df["role"], strict=True)]]
    after = df[["PatientID", "role"]].drop_duplicates().shape[0]
    if after != before:
        print(f"restricted to the success criterion: {after} of {before} series")
    df["abs_d_fat_hu"] = df["d_fat_hu"].abs()
    df["abs_d_fat_fraction"] = df["d_fat_fraction"].abs()
    df["abs_d_asymmetry"] = df["d_asymmetry"].abs()
    perturbed = df[df["condition"] != "control"]
    control = df[df["condition"] == "control"]
    rng = np.random.default_rng(SEED)

    n_series = df[["PatientID", "role"]].drop_duplicates().shape[0]
    n_patients = df["PatientID"].nunique()

    rows = []
    for cond, g in df.groupby("condition", sort=False):
        # One identifier column, written out: the coded name is the key of the
        # per-condition file and is given in the legend.
        if cond == "control":
            label = "Control"
        else:
            label = (f"{g['amplitude_mm'].iloc[0]:.0f} mm / "
                     f"{g['wavelength_mm'].iloc[0]:.0f} mm / {g['direction'].iloc[0]}")
        row = {"condition": cond, "Condition": label,
               "n_series": int(g[["PatientID", "role"]].drop_duplicates().shape[0])}
        for key, lbl, scale, unit, digits in OUTCOMES:
            row[f"{lbl}{f' ({unit})' if unit else ''}"] = med_iqr(g[key], scale, digits)
        rows.append(row)
    order = ["control"] + sorted(c for c in df["condition"].unique() if c != "control")
    table = pd.DataFrame(rows).set_index("condition").loc[order].reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    assert table["n_series"].nunique() == 1, "n differs by condition"
    table = table.drop(columns=["condition", "n_series"])

    def columns(outcomes: list) -> list[str]:
        return ["Condition"] + [f"{lbl}{f' ({unit})' if unit else ''}"
                                for _, lbl, _, unit, _ in outcomes]

    table[columns(PATH_OUTCOMES)].to_csv(
        args.out / "table5_perturbation_path.csv", index=False)
    table[columns(FAT_OUTCOMES)].to_csv(
        args.out / "table6_perturbation_fat.csv", index=False)

    N: dict = {"n_series": n_series, "n_patients": n_patients,
               "n_conditions": int(df["condition"].nunique() - 1),
               "bootstrap_samples": BOOTSTRAP, "bootstrap_unit": "patient",
               "by_condition": {}, "by_amplitude": {}, "pooled": {}}
    for cond, g in perturbed.groupby("condition"):
        lo, hi = patient_bootstrap(g, "traced_length_rel_change", 100.0, rng)
        N["by_condition"][cond] = {
            "length_pct_median": round(float(pd.to_numeric(
                g["traced_length_rel_change"]).median() * 100), 2),
            "length_pct_ci": [round(lo, 2), round(hi, 2)],
            "abs_d_fat_hu_median": round(float(g["abs_d_fat_hu"].median()), 3),
            "abs_d_fat_fraction_median": round(float(g["abs_d_fat_fraction"].median()), 4),
            "abs_d_asymmetry_median": round(float(g["abs_d_asymmetry"].median()), 4),
            "outside_lumen_pct_median": round(float(
                g["points_outside_lumen_frac"].median() * 100), 2),
            "ring_valid_pct_median": round(float(g["ring_valid_fraction"].median() * 100), 1),
            "residual_rms_mm_median": round(float(
                g["residual_rms_after_smoothing_mm"].median()), 3),
        }
    for amp, g in perturbed.groupby("amplitude_mm"):
        lo, hi = patient_bootstrap(g, "traced_length_rel_change", 100.0, rng)
        N["by_amplitude"][f"{amp:.0f}mm"] = {
            "length_pct_median": round(float(pd.to_numeric(
                g["traced_length_rel_change"]).median() * 100), 2),
            "length_pct_ci": [round(lo, 2), round(hi, 2)],
            "length_pct_range": [round(float(pd.to_numeric(
                g["traced_length_rel_change"]).min() * 100), 2),
                round(float(pd.to_numeric(g["traced_length_rel_change"]).max() * 100), 2)],
            "abs_d_fat_hu_median": round(float(g["abs_d_fat_hu"].median()), 3),
            "abs_d_fat_hu_max": round(float(g["abs_d_fat_hu"].max()), 3),
            "abs_d_fat_fraction_median": round(float(g["abs_d_fat_fraction"].median()), 4),
            "abs_d_fat_fraction_max": round(float(g["abs_d_fat_fraction"].max()), 4),
            "abs_d_asymmetry_median": round(float(g["abs_d_asymmetry"].median()), 4),
            "abs_d_asymmetry_max": round(float(g["abs_d_asymmetry"].max()), 4),
            "outside_lumen_pct_median": round(float(
                g["points_outside_lumen_frac"].median() * 100), 2),
            "residual_rms_mm_median": round(float(
                g["residual_rms_after_smoothing_mm"].median()), 3),
        }
    lo, hi = patient_bootstrap(perturbed, "traced_length_rel_change", 100.0, rng)
    lo_hu, hi_hu = patient_bootstrap(perturbed, "abs_d_fat_hu", 1.0, rng)
    N["pooled"] = {
        "length_pct_median": round(float(pd.to_numeric(
            perturbed["traced_length_rel_change"]).median() * 100), 2),
        "length_pct_ci": [round(lo, 2), round(hi, 2)],
        "length_increased_in": int((perturbed["traced_length_rel_change"] > 0).sum()),
        "length_n": int(perturbed["traced_length_rel_change"].notna().sum()),
        "abs_d_fat_hu_median": round(float(perturbed["abs_d_fat_hu"].median()), 3),
        "abs_d_fat_hu_ci": [round(lo_hu, 3), round(hi_hu, 3)],
        "abs_d_fat_hu_q95": round(float(perturbed["abs_d_fat_hu"].quantile(0.95)), 3),
        "control_ring_valid_pct_median": round(float(
            control["ring_valid_fraction"].median() * 100), 1),
        "perturbed_ring_valid_pct_median": round(float(
            perturbed["ring_valid_fraction"].median() * 100), 1),
        "control_outside_lumen_pct_median": round(float(
            control["points_outside_lumen_frac"].median() * 100), 2),
    }
    (args.out / "perturbation_numbers.json").write_text(
        json.dumps(N, indent=2), encoding="utf-8")

    print(f"series {n_series} ({n_patients} patients), conditions {N['n_conditions']}")
    print(f"pooled length change {N['pooled']['length_pct_median']:+.2f} % "
          f"[{N['pooled']['length_pct_ci'][0]:+.2f}, {N['pooled']['length_pct_ci'][1]:+.2f}], "
          f"increased in {N['pooled']['length_increased_in']}/{N['pooled']['length_n']}")
    for amp, v in N["by_amplitude"].items():
        print(f"  A={amp}: length {v['length_pct_median']:+.2f} %, |d fat| "
              f"{v['abs_d_fat_hu_median']:.2f} HU (max {v['abs_d_fat_hu_max']:.2f}), "
              f"|d fraction| {v['abs_d_fat_fraction_median']:.4f}, outside lumen "
              f"{v['outside_lumen_pct_median']:.2f} %, residual "
              f"{v['residual_rms_mm_median']:.3f} mm")
    print(f"-> {args.out / 'table5_perturbation_path.csv'} and "
          f"{args.out / 'table6_perturbation_fat.csv'}")


if __name__ == "__main__":
    main()
