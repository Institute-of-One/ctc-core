"""Tabulate and classify per-series results of a headless cohort batch run.

Reads every ``<root>/<PatientID>/<role>/metrics.json`` produced by the batch
pipeline and emits one row per series with the stage status and the diagnostic
fields needed to classify centerline failure modes (docs/FAILURE_ANALYSIS.md).

The script is stdlib-only and read-only with respect to ``--root`` so it can be
pointed at the in-house prototype's batch output without touching it.

Usage
-----
    python scripts/analyze_batch_metrics.py \
        --root <prototype batch output> \
        --out results/tables/batch_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Nominal total colon length is 150-200 cm (Hounsfield et al.; standard anatomy).
# A centerline is called "complete" when it covers most of that range and
# "partial" when it clearly stops early. Thresholds are reported in
# docs/PARAMETERS.md and are deliberately conservative.
COMPLETE_LENGTH_MM = 1200.0
PARTIAL_LENGTH_MM = 600.0

# Below this the largest air component cannot plausibly be an insufflated colon
# (the batch used the same value as an abort criterion in auto_seed).
MIN_INSUFFLATION_VOXELS = 10_000

# Largest contiguous intra-abdominal gas component, in voxels. At 1.0 mm
# isotropic spacing 1 voxel = 1 mm^3 = 0.001 mL, so this is 150 mL. An
# adequately insufflated colon holds 1-2 L of gas and the distension score
# already places its lower anchor at 500 mL of gas; below 150 mL contiguous
# there is no colon to trace. The value also falls inside an empty gap in the
# observed 2026-08 distribution (93 mL -> 241 mL), so no series sits near it.
POOR_INSUFFLATION_VOXELS = 150_000

STAGES = ("resample", "auto_seed", "centerline", "dc", "phase0")

FIELDS = [
    "PatientID",
    "role",
    "SeriesInstanceUID",
    "iso_spacing_mm",
    "all_stages_ok",
    # per-stage ok flags and times
    *[f"{s}_ok" for s in STAGES],
    *[f"{s}_sec" for s in STAGES],
    # auto_seed diagnostics
    "air_cc_count",
    "largest_air_cc_voxels",
    "skeleton_voxels",
    "seed_source",
    "seed_distance_mm",
    # centerline diagnostics
    "centerline_n_points",
    "centerline_length_mm",
    "n_ccs_total",
    "n_ccs_visited",
    "n_bridges",
    "used_kimimaro",
    # phase0 (fat map) diagnostics
    "phase0_valid_fraction",
    "phase0_fat_fraction_mean",
    "phase0_fat_volume_ml_sum",
    # derived
    "first_failed_stage",
    "error",
    "outcome",
    "failure_mode",
]


def _stage(rec: dict, name: str) -> dict:
    return rec.get("stages", {}).get(name, {}) or {}


def classify(row: dict) -> tuple[str, str]:
    """Return ``(outcome, failure_mode)`` for one series.

    ``outcome`` is one of ``complete`` / ``partial`` / ``failed`` and is the
    definition proposed for the manuscript. ``failure_mode`` is empty for
    complete cases and otherwise names the mechanism.
    """
    largest = row["largest_air_cc_voxels"]
    n_total = row["n_ccs_total"]
    n_visited = row["n_ccs_visited"]
    length = row["centerline_length_mm"]

    # --- hard failures -----------------------------------------------------
    if not row["centerline_ok"]:
        if largest is not None and largest < POOR_INSUFFLATION_VOXELS:
            return "failed", "insufficient-distension"
        return "failed", "fmm-unreachable-end-seed"

    # --- the centerline ran; judge its extent ------------------------------
    if length is None:
        return "failed", "no-length-reported"

    if length >= COMPLETE_LENGTH_MM:
        outcome = "complete"
    elif length >= PARTIAL_LENGTH_MM:
        outcome = "partial"
    else:
        outcome = "failed"

    if outcome == "complete":
        return outcome, ""

    # Attribute the truncation. The batch ran with air-mask closing radius 0
    # and no bridging, so a colon split into many air components can only be
    # traversed within the single component holding both seeds.
    if largest is not None and largest < POOR_INSUFFLATION_VOXELS:
        return outcome, "insufficient-distension"
    if n_total is not None and n_visited is not None and n_total > 1 and n_visited <= 1:
        return outcome, "single-component-traversal"
    return outcome, "early-termination-other"


def read_case(path: Path) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    seed = _stage(rec, "auto_seed")
    cl = _stage(rec, "centerline")
    p0 = _stage(rec, "phase0")

    row: dict = {
        "PatientID": rec.get("PatientID"),
        "role": rec.get("role"),
        "SeriesInstanceUID": rec.get("SeriesInstanceUID"),
        "iso_spacing_mm": rec.get("iso_spacing_mm"),
        "all_stages_ok": rec.get("all_stages_ok"),
        "air_cc_count": seed.get("air_cc_count"),
        "largest_air_cc_voxels": seed.get("largest_air_cc_voxels"),
        "skeleton_voxels": seed.get("skeleton_voxels"),
        "seed_source": seed.get("seed_source"),
        "seed_distance_mm": seed.get("seed_distance_mm"),
        "centerline_n_points": cl.get("n_points"),
        "centerline_length_mm": cl.get("length_mm"),
        "n_ccs_total": cl.get("n_ccs_total"),
        "n_ccs_visited": cl.get("n_ccs_visited"),
        "n_bridges": cl.get("n_bridges"),
        "used_kimimaro": cl.get("used_kimimaro"),
        "phase0_valid_fraction": p0.get("valid_fraction"),
        "phase0_fat_fraction_mean": p0.get("fat_fraction_mean"),
        "phase0_fat_volume_ml_sum": p0.get("fat_volume_ml_sum"),
    }

    first_failed, error = "", ""
    for s in STAGES:
        st = _stage(rec, s)
        row[f"{s}_ok"] = st.get("ok")
        row[f"{s}_sec"] = st.get("elapsed_sec")
        if not first_failed and st.get("ok") is False:
            first_failed = s
            error = str(st.get("error", ""))
    row["first_failed_stage"] = first_failed
    row["error"] = error

    row["outcome"], row["failure_mode"] = classify(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path,
                    help="batch output root containing <PatientID>/<role>/metrics.json")
    ap.add_argument("--out", required=True, type=Path, help="destination CSV")
    args = ap.parse_args()

    paths = sorted(args.root.glob("*/*/metrics.json"))
    if not paths:
        raise SystemExit(f"no metrics.json found under {args.root}")

    rows = [read_case(p) for p in paths]
    rows.sort(key=lambda r: (r["PatientID"] or "", r["role"] or ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} series -> {args.out}")
    for key in ("outcome", "failure_mode"):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[key] or "-"] = counts.get(r[key] or "-", 0) + 1
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
