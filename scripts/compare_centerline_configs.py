"""Compare two centerline configurations over the cohort.

Reads two summary CSVs written by ``scripts/run_centerline.py`` (typically the
2026-08 batch reproduction and the corrected pipeline) and reports, per series
and in aggregate:

* the outcome distribution under the definition of ``docs/FAILURE_ANALYSIS.md``
  section 1 (complete >= 1200 mm, partial 600-1200 mm, failed below or raised),
* paired per-patient availability, which is what evaluation pillar 3 needs,
* whether the CC-bridge graph actually ran,
* the per-series length change.

    python scripts/compare_centerline_configs.py \\
        --baseline results/tables/centerline_batch.csv \\
        --candidate results/tables/centerline_corrected.csv \\
        --out results/tables/centerline_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

COMPLETE_LENGTH_MM = 1200.0
PARTIAL_LENGTH_MM = 600.0


def num(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def outcome(row: dict) -> str:
    if str(row.get("ok")).lower() != "true":
        return "failed"
    length = num(row, "length_mm")
    if length is None:
        return "failed"
    if length >= COMPLETE_LENGTH_MM:
        return "complete"
    if length >= PARTIAL_LENGTH_MM:
        return "partial"
    return "failed"


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {
        (r["PatientID"], r["role"]): r
        for r in csv.DictReader(path.open(encoding="utf-8"))
    }


def outcome_counts(rows: list[dict]) -> Counter:
    return Counter(outcome(r) for r in rows)


def paired_availability(rows: list[dict]) -> dict[str, int]:
    by_patient: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_patient[r["PatientID"]][r["role"]] = r
    out = {}
    for label, ok in (
        ("both complete", lambda r: outcome(r) == "complete"),
        ("both complete or partial", lambda r: outcome(r) in ("complete", "partial")),
        ("both produced a centerline", lambda r: str(r.get("ok")).lower() == "true"),
    ):
        out[label] = sum(
            1 for p in by_patient.values() if len(p) == 2 and all(ok(r) for r in p.values())
        )
    return out


def describe(name: str, rows: list[dict]) -> None:
    n = len(rows)
    counts = outcome_counts(rows)
    print(f"\n--- {name}  (n={n}) ---")
    for key in ("complete", "partial", "failed"):
        c = counts.get(key, 0)
        print(f"  {key:9s} {c:3d}  ({100 * c / n:5.1f} %)")

    lengths = [x for x in (num(r, "length_mm") for r in rows) if x is not None]
    if lengths:
        q1, q3 = st.quantiles(lengths, n=4)[0], st.quantiles(lengths, n=4)[2]
        print(f"  length_mm  median {st.median(lengths):7.1f}  IQR {q1:.0f}-{q3:.0f}"
              f"  max {max(lengths):.0f}   (n={len(lengths)})")

    tort = [x for x in (num(r, "tortuosity") for r in rows) if x is not None]
    if tort:
        print(f"  tortuosity median {st.median(tort):.2f}  max {max(tort):.2f}")
    walls = [x for x in (num(r, "max_bridge_wall_frac") for r in rows) if x is not None]
    if walls:
        over = sum(1 for w in walls if w > 0.5)
        print(f"  worst bridge wall_frac: median {st.median(walls):.3f}, "
              f"{over} series above 0.5")
    fb = sum(1 for r in rows if str(r.get("bridge_fallback", "")).strip())
    if fb:
        print(f"  fell back to largest component: {fb}")
    implausible = sum(1 for r in rows if (num(r, "length_mm") or 0) > 2000)
    print(f"  length > 2000 mm (above anatomical range): {implausible}")

    bridged = sum(1 for r in rows if (num(r, "n_bridges") or 0) > 0)
    visited = [x for x in (num(r, "n_ccs_visited") for r in rows) if x is not None]
    print(f"  series with >=1 bridge: {bridged}/{n}")
    if visited:
        print(f"  components visited: median {st.median(visited):.0f}, max {max(visited):.0f}")

    for label, v in paired_availability(rows).items():
        print(f"  {label:28s} {v:2d} patients")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    describe(f"baseline: {args.baseline.name}", list(base.values()))
    describe(f"candidate: {args.candidate.name}", list(cand.values()))

    rows = []
    for key in sorted(set(base) | set(cand)):
        b, c = base.get(key, {}), cand.get(key, {})
        lb, lc = num(b, "length_mm"), num(c, "length_mm")
        rows.append(
            {
                "PatientID": key[0],
                "role": key[1],
                "baseline_ok": b.get("ok", ""),
                "candidate_ok": c.get("ok", ""),
                "baseline_length_mm": lb,
                "candidate_length_mm": lc,
                "delta_mm": round(lc - lb, 2) if (lb is not None and lc is not None) else None,
                "ratio": round(lc / lb, 3) if (lb and lc) else None,
                "baseline_outcome": outcome(b) if b else "",
                "candidate_outcome": outcome(c) if c else "",
                "baseline_n_ccs_visited": b.get("n_ccs_visited", ""),
                "candidate_n_ccs_visited": c.get("n_ccs_visited", ""),
                "candidate_n_bridges": c.get("n_bridges", ""),
                "candidate_error": c.get("error", ""),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n--- transitions (baseline -> candidate) ---")
    trans = Counter((r["baseline_outcome"], r["candidate_outcome"]) for r in rows)
    for (a, b), n in sorted(trans.items(), key=lambda kv: -kv[1]):
        mark = "  " if a == b else ("+ " if b == "complete" or a == "failed" else "- ")
        print(f"{mark}{a:9s} -> {b:9s}  {n:3d}")

    deltas = [r["delta_mm"] for r in rows if r["delta_mm"] is not None]
    if deltas:
        print(f"\nlength delta over {len(deltas)} paired series: "
              f"median {st.median(deltas):+.1f} mm, "
              f"improved {sum(1 for d in deltas if d > 0)}, "
              f"worsened {sum(1 for d in deltas if d < 0)}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
