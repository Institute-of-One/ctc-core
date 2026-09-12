"""Prone/supine agreement of the fat and quality indices.

Evaluation pillar 3. Pairs the two acquisitions of each patient and reports
ICC(2,1) with a 95 % confidence interval and Bland-Altman bias with limits of
agreement, for every index.

Pairing is by **recovered acquisition position**, not by the manifest's
primary/secondary role. The role comes from SeriesNumber and carries no position
information: against HQColon it splits 8 prone / 7 supine for primary and
6 prone / 5 supine for secondary. Pairing on it would give a repeatability
analysis mislabelled as a positional one.

    python scripts/eval_prone_supine.py --indices results/tables/indices.csv \\
        --out results/tables/prone_supine_agreement.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.agreement import bland_altman, icc21  # noqa: E402

INDICES = [
    ("traced_centerline_length_cm", "cm"),
    ("centerline_tortuosity_index", ""),
    ("gas_volume_ml", "mL"),
    ("length_adjusted_gas_ml_per_cm", "mL/cm"),
    ("collapse_ratio_pct", "%"),
    ("luminal_radius_mm_median", "mm"),
    ("distension_quality_score", ""),
    ("ring_fat_mean_hu_median", "HU"),
    ("ring_fat_fraction_mean", ""),
    ("fat_mean_hu_median", "HU"),
    ("fat_fraction_mean", ""),
    ("fat_volume_ml_sum", "mL"),
    ("fat_asymmetry", ""),
    ("fat_asymmetry_theta_halves", ""),
]

FIELDS = [
    "index", "unit", "n_pairs",
    "prone_mean", "prone_sd", "supine_mean", "supine_sd",
    "icc21", "icc_lower", "icc_upper",
    "bias_prone_minus_supine", "bias_ci_lower", "bias_ci_upper",
    "sd_diff", "loa_lower", "loa_upper",
]


def num(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", "None", None):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


def select_covered(
    rows: list[dict],
    coverage: dict[tuple[str, str], float | None],
    max_unreached: float,
) -> tuple[list[dict], list[str], list[str]]:
    """Keep series whose coverage against the reference is known and adequate.

    A series with no reference is dropped, not kept: its coverage was never
    measured, and treating it as adequate is what made the earlier "whole colon
    reached" subset meaningless -- most of its pairs had no reference at all.
    """
    keep: list[dict] = []
    low: list[str] = []
    no_ref: list[str] = []
    for r in rows:
        c = coverage.get((r["PatientID"], r["role"]))
        tag = f"{r['PatientID'][-4:]}/{r['role']}"
        if c is None:
            no_ref.append(tag)
        elif c > max_unreached:
            low.append(f"{tag} ({c:.3f})")
        else:
            keep.append(r)
    return keep, low, no_ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indices", type=Path, default=Path("results/tables/indices.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-unreached", type=float, default=None,
                    help="keep only series whose reference colon lies beyond 60 mm of the "
                         "path by no more than this share. Needs --coverage. A series "
                         "without a reference is DROPPED: its coverage is unknown and "
                         "cannot be assumed adequate")
    ap.add_argument("--coverage", type=Path,
                    default=Path("results/tables/eval_hqcolon_auto.csv"))
    ap.add_argument("--min-pairs", type=int, default=3,
                    help="refuse to report agreement below this many complete pairs")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.indices.open(encoding="utf-8"))
            if str(r.get("ok")).lower() == "true"]

    dropped_low_coverage: list[str] = []
    dropped_no_reference: list[str] = []
    if args.max_unreached is not None:
        if not args.coverage.exists():
            raise SystemExit(f"--max-unreached needs --coverage; {args.coverage} not found")
        cov = {
            # Share of the reference colon (gas and fluid) beyond 60 mm of the
            # path, from eval_hqcolon.py. A definition, not a tuned cut.
            (r["PatientID"], r["role"]): num(r, "colon_beyond_60mm_frac")
            for r in csv.DictReader(args.coverage.open(encoding="utf-8"))
        }
        rows, dropped_low_coverage, dropped_no_reference = select_covered(
            rows, cov, args.max_unreached)

    by_patient: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pos = (r.get("position") or "").strip()
        if pos in ("prone", "supine"):
            by_patient[r["PatientID"]][pos].append(r)

    paired: list[tuple[dict, dict]] = []
    excluded: list[str] = []
    for pid, d in sorted(by_patient.items()):
        pr, su = d.get("prone", []), d.get("supine", [])
        if len(pr) == 1 and len(su) == 1:
            paired.append((pr[0], su[0]))
        else:
            excluded.append(f"{pid[-4:]} (prone={len(pr)}, supine={len(su)})")

    print(f"indices rows usable        : {len(rows)}")
    if dropped_no_reference:
        print(f"dropped, no reference      : {len(dropped_no_reference)}")
        print(f"    {', '.join(dropped_no_reference)}")
    if dropped_low_coverage:
        print(f"dropped for low coverage   : {len(dropped_low_coverage)}")
        for d in dropped_low_coverage:
            print(f"    {d}")
    print(f"patients paired prone/supine: {len(paired)}")
    if excluded:
        print(f"patients excluded          : {len(excluded)}")
        for e in excluded:
            print(f"    {e}")
    if len(paired) < args.min_pairs:
        raise SystemExit(f"{len(paired)} pairs is below --min-pairs {args.min_pairs}: "
                         "agreement is not reported on this subset")

    out_rows = []
    print(f"\n{'index':32s} {'n':>3s} {'ICC(2,1) [95% CI]':>26s} "
          f"{'bias':>10s} {'95% LoA':>22s}")
    for key, unit in INDICES:
        pairs = [
            (num(a, key), num(b, key))
            for a, b in paired
            if num(a, key) is not None and num(b, key) is not None
        ]
        if len(pairs) < 3:
            print(f"  {key:30s} {len(pairs):3d}  (too few complete pairs)")
            continue
        prone = np.array([p[0] for p in pairs])
        supine = np.array([p[1] for p in pairs])

        icc = icc21(np.column_stack([prone, supine]))
        ba = bland_altman(prone, supine)

        out_rows.append(
            {
                "index": key,
                "unit": unit,
                "n_pairs": len(pairs),
                "prone_mean": round(float(prone.mean()), 4),
                "prone_sd": round(float(prone.std(ddof=1)), 4),
                "supine_mean": round(float(supine.mean()), 4),
                "supine_sd": round(float(supine.std(ddof=1)), 4),
                "icc21": round(icc.icc, 4),
                "icc_lower": round(icc.lower, 4),
                "icc_upper": round(icc.upper, 4),
                "bias_prone_minus_supine": round(ba.bias, 4),
                "bias_ci_lower": round(ba.bias_ci[0], 4),
                "bias_ci_upper": round(ba.bias_ci[1], 4),
                "sd_diff": round(ba.sd_diff, 4),
                "loa_lower": round(ba.lower_loa, 4),
                "loa_upper": round(ba.upper_loa, 4),
            }
        )
        print(
            f"  {key:30s} {len(pairs):3d}  {icc.icc:6.3f} [{icc.lower:6.3f}, {icc.upper:6.3f}] "
            f"{ba.bias:10.3f} [{ba.lower_loa:9.3f}, {ba.upper_loa:9.3f}]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
