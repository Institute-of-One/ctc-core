"""Cohort characteristics table for the Methods section.

Joins the manifest to the collection's clinical-data table on PatientID
(``Blinded_ID``) and tabulates the fields a Methods cohort description needs.

Two things about the source are easy to get wrong, so both are stated here:

* The machine-readable data dictionary lists *numeric* codes (``1`` = Male,
  ``2`` = Female, ...), but this release of the table stores the decoded
  *labels* as strings. Mapping the codes onto it silently matches nothing and
  makes every field look empty.
* The release carries **no age, height, weight or BMI**. The height-normalised
  length index therefore cannot be computed from this collection and is left
  null rather than imputed.

The table also carries bowel-preparation variables -- regimen, cathartic
compliance, barium and iodinated tagging compliance. They are tabulated here
for completeness but deliberately **not** analysed here: whether the quality
indices track preparation is a different research question from this study's.

    python scripts/make_cohort_table.py --out results/tables/cohort_characteristics.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

FIELDS = [
    ("a0e11", "Sex"),
    ("a0e9", "Ethnicity"),
    ("a0e23", "Race: White"),
    ("a0e21", "Race: Black or African American"),
    ("a0e20", "Race: Asian"),
    ("i1e3", "Indication: screening, no symptoms"),
    ("i1e19", "Preparation regimen"),
    ("i1e21", "Cathartic taken as directed"),
    ("i1e25", "Barium sulfate taken as directed"),
    ("i1e30", "Iodinated oral contrast taken as directed"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument(
        "--clinical", type=Path,
        default=Path("data/tcia/CT-Colonography_clinical-data_v01_20260824.tsv"),
    )
    ap.add_argument("--position", type=Path, default=Path("results/tables/patient_position.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cohort = sorted({r["PatientID"] for r in csv.DictReader(args.manifest.open(encoding="utf-8"))})
    with args.clinical.open(encoding="utf-8") as fh:
        clinical = {r["Blinded_ID"].strip(): r for r in csv.DictReader(fh, delimiter="\t")}

    missing = [p for p in cohort if p not in clinical]
    rows = [clinical[p] for p in cohort if p in clinical]
    n = len(rows)

    out: list[dict] = [{"characteristic": "Patients", "value": "", "n": n, "percent": ""}]
    for key, label in FIELDS:
        counts = Counter((r.get(key) or "(blank)").strip() or "(blank)" for r in rows)
        for value, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append({
                "characteristic": label,
                "value": value,
                "n": c,
                "percent": round(100.0 * c / n, 1) if n else "",
            })

    if args.position.exists():
        def label(r: dict) -> str:
            if r["position_combined"]:
                return r["position_combined"]
            # Neither prone nor supine: name what the tag says rather than
            # calling it unidentifiable (the cohort's one case is FFDR).
            tag = (r.get("patient_position_tag") or "").upper()
            return "decubitus" if tag.endswith(("DL", "DR")) else "unidentifiable"

        pos = Counter(label(r) for r in csv.DictReader(args.position.open(encoding="utf-8")))
        for value, c in sorted(pos.items(), key=lambda kv: -kv[1]):
            out.append({
                "characteristic": "Series by recovered position",
                "value": value,
                "n": c,
                "percent": round(100.0 * c / sum(pos.values()), 1),
            })

    for label in ("Age", "Height", "Weight / BMI"):
        out.append({"characteristic": label, "value": "not provided by the collection",
                    "n": "", "percent": ""})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["characteristic", "value", "n", "percent"])
        w.writeheader()
        w.writerows(out)

    print(f"{n} of {len(cohort)} cohort patients found in the clinical table")
    if missing:
        print(f"  missing: {missing}")
    for r in out:
        pct = f" ({r['percent']} %)" if r["percent"] != "" else ""
        print(f"  {r['characteristic']:44s} {r['value']:52s} {r['n']}{pct}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
