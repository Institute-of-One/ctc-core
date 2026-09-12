"""Per-patient audit of the prone/supine pairs and their reference coverage.

Written after the sensitivity analysis of 2026-09-12 was found to be invalid:
its filter kept series whose coverage against HQColon was never measured, so the
"whole colon reached" subset of 19 pairs was mostly pairs with no reference.
This script states, per patient and per Series Instance UID, what is actually
known, so the numbers in the manuscript can be checked against the data.

    python scripts/pair_reference_status.py

Writes ``results/tables/pair_reference_status.csv`` (one row per patient) and
prints the counts the manuscript quotes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REACHED_FAR_MAX = 0.05  # share of the reference colon beyond 60 mm of the path


def build(tables: Path) -> pd.DataFrame:
    idx = pd.read_csv(tables / "indices.csv")
    idx = idx[idx["ok"].astype(str).str.lower() == "true"]
    cov = pd.read_csv(tables / "eval_hqcolon_auto.csv")[
        ["PatientID", "role", "colon_beyond_60mm_frac", "colon_within_30mm_frac"]]
    m = idx.merge(cov, on=["PatientID", "role"], how="left")

    rows = []
    for pid, g in m.groupby("PatientID"):
        rec: dict[str, object] = {"PatientID": pid, "patient": pid[-4:]}
        for pos in ("prone", "supine"):
            sub = g[g["position"] == pos]
            one = sub.iloc[0] if len(sub) == 1 else None
            beyond = None if one is None else one["colon_beyond_60mm_frac"]
            rec[f"{pos}_uid"] = "" if one is None else one["SeriesInstanceUID"]
            rec[f"{pos}_has_reference"] = bool(pd.notna(beyond)) if one is not None else False
            rec[f"{pos}_beyond_60mm"] = None if beyond is None or pd.isna(beyond) else round(
                float(beyond), 4)
            rec[f"{pos}_meets_criterion"] = (
                bool(pd.notna(beyond) and beyond < REACHED_FAR_MAX) if one is not None else False)
        rec["n_series_this_patient"] = int(len(g))
        rec["paired"] = bool(rec["prone_uid"] and rec["supine_uid"])
        rec["reference_both"] = bool(rec["prone_has_reference"] and rec["supine_has_reference"])
        rec["criterion_both"] = bool(rec["prone_meets_criterion"] and rec["supine_meets_criterion"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("patient")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    args = ap.parse_args()

    df = build(args.tables)
    out = args.tables / "pair_reference_status.csv"
    df.to_csv(out, index=False)

    paired = df[df["paired"]]
    ref_all = df[df["reference_both"]]
    print(f"patients with a usable centerline in any position : {len(df)}")
    print(f"prone/supine pairs                                : {len(paired)}")
    print(f"  reference in both positions                     : {int(paired.reference_both.sum())}")
    print(f"  coverage criterion met in both                  : {int(paired.criterion_both.sum())}")
    print(f"  reference in one position only                  : "
          f"{int((paired.prone_has_reference ^ paired.supine_has_reference).sum())}")
    print(f"  no reference in either position                 : "
          f"{int((~paired.prone_has_reference & ~paired.supine_has_reference).sum())}")
    print(f"patients with a reference in both series (paired or not): {len(ref_all)}"
          f"  [{', '.join(ref_all.patient)}]")
    unpaired_ref = ref_all[~ref_all["paired"]]
    if len(unpaired_ref):
        print("  of which not a prone/supine pair: "
              f"{', '.join(unpaired_ref.patient)} -- these have two references but no pair")
    print("\nthe former sensitivity subset (series without a reference kept):")
    kept = paired[(paired.prone_meets_criterion | ~paired.prone_has_reference)
                  & (paired.supine_meets_criterion | ~paired.supine_has_reference)]
    print(f"  pairs kept: {len(kept)}  [{', '.join(kept.patient)}]")
    print(f"  of these, pairs with a reference in both positions: {int(kept.reference_both.sum())}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
