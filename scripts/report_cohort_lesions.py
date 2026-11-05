"""Report every annotated lesion in the cohort, and what the annotation is.

The three annotated in-vivo lesions in this cohort are reported even at
n = 3, decoded from the key the collection
publishes beside the spreadsheets ("Key to CT Colon XL Lesion Feature
Spreadsheet", ``data/tcia/Polyp-description-key-table.png``), rather than
described in a sentence.

Decoding them also settles what they can be used for. The size the collection
publishes is the size at colonoscopy or at pathology, not a measurement on the
CT, and the sheet carries no in-plane coordinate at all -- so a diameter
measured on the image could be compared against an optical or a specimen size,
which is a different measurand, and only after a reader had found the lesion
inside the named slice. Both facts come out of the table rather than out of an
argument.

    python scripts/report_cohort_lesions.py --out results/tables/cohort_lesions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# Transcribed from the published key image, which is the only documentation of
# these columns; the machine-readable data dictionary shipped with the
# collection covers the clinical sheet and not this one.
KEY_IMAGE = "Polyp-description-key-table.png"
SEGMENT = {1: "rectum", 2: "sigmoid", 3: "descending", 4: "transverse",
           5: "ascending", 6: "cecum"}
SIZE_SOURCE = {1: "colonoscopy", 2: "pathology"}
PIECEMEAL = {1: "no", 2: "yes"}
HISTOLOGY = {
    1: "adenocarcinoma", 2: "medullary carcinoma", 3: "mucinous carcinoma",
    4: "signet ring cell carcinoma", 5: "squamous cell carcinoma",
    6: "adenosquamous carcinoma", 7: "small cell carcinoma",
    8: "undifferentiated carcinoma", 9: "carcinoma, NOS", 10: "hyperplastic",
    11: "lipomatous", 12: "adenomatous", 13: "tubular adenoma",
    14: "tubulovillous adenoma", 15: "villous adenoma",
    16: "tubulovillous adenoma with dysplasia", 17: "normal mucosa",
    88: "other, specified", 98: "not applicable",
}
SHEETS = {
    "TCIA-CTC-large-10-mm-polyps.xls": ">=10 mm",
    "TCIA-CTC-6-to-9-mm-polyps.xls": "6-9 mm",
}
# The sheet lists up to twenty lesions per row, as five columns each, after
# the identifier and the two slice-number columns.
MAX_LESIONS = 20
FIRST_LESION_COL = 3


def lookup(table: dict[int, str], value) -> str:
    if pd.isna(value):
        return ""
    return table.get(int(value), "code %d, not in the key" % int(value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/tcia"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not (args.annotations / KEY_IMAGE).exists():
        raise SystemExit(
            "the published column key is missing: %s. Without it the lesion "
            "columns are unlabelled numbers." % (args.annotations / KEY_IMAGE))

    cohort = {r["PatientID"] for r in csv.DictReader(args.manifest.open(encoding="utf-8"))}

    rows = []
    counts = {}
    for sheet, band in SHEETS.items():
        x = pd.read_excel(args.annotations / sheet)
        ids = x.iloc[:, 0].astype(str)
        counts[band] = {"patients_in_sheet": int(len(x)),
                        "patients_in_cohort": int(ids.isin(cohort).sum())}
        # The five fields of a lesion are five consecutive columns after the
        # identifier and the two slice numbers. They are addressed by position:
        # their labels are a mixture of strings and floats ("LESION 1.1" then
        # 1.2, 1.3) because the header is a spreadsheet's own numbering, and
        # looking them up by name silently matches nothing.
        if str(x.columns[FIRST_LESION_COL]) != "LESION 1.1":
            raise SystemExit("%s: column %d is %r, not 'LESION 1.1'; the layout "
                             "this script assumes has changed"
                             % (sheet, FIRST_LESION_COL, x.columns[FIRST_LESION_COL]))
        for _, rec in x[ids.isin(cohort)].iterrows():
            for i in range(1, MAX_LESIONS + 1):
                first = FIRST_LESION_COL + 5 * (i - 1)
                if first + 4 >= len(x.columns):
                    break
                seg, size, src, pieces, hist = (rec.iloc[first + j] for j in range(5))
                if pd.isna(seg) and pd.isna(size):
                    continue
                rows.append({
                    "PatientID": str(rec.iloc[0]),
                    "sheet_size_band": band,
                    "lesion": i,
                    "segment": lookup(SEGMENT, seg),
                    "reported_size_mm": "" if pd.isna(size) else int(size),
                    "size_source": lookup(SIZE_SOURCE, src),
                    "specimen_removed_in_pieces": lookup(PIECEMEAL, pieces),
                    "histology": lookup(HISTOLOGY, hist),
                    "slice_supine": "" if pd.isna(rec.iloc[1]) else str(rec.iloc[1]),
                    "slice_prone": "" if pd.isna(rec.iloc[2]) else str(rec.iloc[2]),
                })

    if not rows:
        raise SystemExit("no annotated lesion in the cohort; expected three")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "key_image": KEY_IMAGE,
        "cohort_patients": len(cohort) // 2 if len(cohort) else 0,
        "sheets": counts,
        "lesions_in_cohort": len(rows),
        "lesions_by_band": {b: sum(1 for r in rows if r["sheet_size_band"] == b)
                            for b in SHEETS.values()},
        "reported_sizes_mm": sorted(r["reported_size_mm"] for r in rows
                                    if r["reported_size_mm"] != ""),
        "size_sources": sorted({r["size_source"] for r in rows}),
        # What is missing is the point of the table, so it is counted.
        "with_supine_slice": sum(1 for r in rows if r["slice_supine"]),
        "with_prone_slice": sum(1 for r in rows if r["slice_prone"]),
        "with_no_slice_at_all": sum(1 for r in rows
                                    if not r["slice_supine"] and not r["slice_prone"]),
        "with_in_plane_coordinate": 0,
        "in_plane_coordinate_note": (
            "the sheet has no in-plane coordinate column for any lesion in any "
            "of the three sheets"
        ),
    }
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("%-30s %-8s %-11s %5s %-12s %-22s %8s %8s"
          % ("PatientID", "band", "segment", "mm", "size from", "histology",
             "supine", "prone"))
    for r in rows:
        print("%-30s %-8s %-11s %5s %-12s %-22s %8s %8s"
              % (r["PatientID"], r["sheet_size_band"], r["segment"],
                 r["reported_size_mm"], r["size_source"], r["histology"],
                 r["slice_supine"] or "-", r["slice_prone"] or "-"))
    for band, c in counts.items():
        print("%s: %d patients in the sheet, %d in the cohort"
              % (band, c["patients_in_sheet"], c["patients_in_cohort"]))
    print("slice numbers: supine %d of %d, prone %d of %d, neither %d; "
          "in-plane coordinates 0"
          % (summary["with_supine_slice"], len(rows), summary["with_prone_slice"],
             len(rows), summary["with_no_slice_at_all"]))
    print("-> %s\n-> %s" % (args.out, out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
