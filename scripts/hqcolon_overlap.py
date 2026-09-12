"""Map HQColon reference masks onto our cohort.

HQColon's ``meta-data.json`` is JSON Lines, one record per scan, keyed by
``InstanceUID`` -- the TCIA SeriesInstanceUID. That is the same identifier as
the ``SeriesInstanceUID`` column of ``cohort/manifest.csv``, so the two can be
joined directly with no name-guessing.

Writes one row per cohort series saying whether an HQColon mask exists for it
and, if so, which mask file inside the archives holds it.

    python scripts/hqcolon_overlap.py --hqcolon data/hqcolon \\
        --out results/tables/hqcolon_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = [
    "PatientID", "role", "SeriesInstanceUID",
    "in_hqcolon", "hqcolon_subject_id", "hqcolon_number",
    "hqcolon_position", "hqcolon_sex", "hqcolon_label_file",
]


def load_hqcolon(meta_path: Path) -> dict[str, dict]:
    """SeriesInstanceUID -> HQColon record."""
    recs = {}
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[str(r["InstanceUID"])] = r
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--hqcolon", type=Path, default=Path("data/hqcolon"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hq = load_hqcolon(args.hqcolon / "meta-data.json")
    cohort = list(csv.DictReader(args.manifest.open(encoding="utf-8")))

    rows = []
    for rec in cohort:
        suid = str(rec["SeriesInstanceUID"])
        m = hq.get(suid)
        rows.append(
            {
                "PatientID": rec["PatientID"],
                "role": rec["role"],
                "SeriesInstanceUID": suid,
                "in_hqcolon": bool(m),
                "hqcolon_subject_id": m["subject_id"] if m else "",
                "hqcolon_number": m["number"] if m else "",
                "hqcolon_position": m["Position"] if m else "",
                "hqcolon_sex": m["Sex"] if m else "",
                "hqcolon_label_file": m["nnunet_label_file"] if m else "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_hit = sum(1 for r in rows if r["in_hqcolon"])
    by_patient: dict[str, int] = Counter()
    for r in rows:
        if r["in_hqcolon"]:
            by_patient[r["PatientID"]] += 1

    print(f"HQColon records          : {len(hq)}")
    print(f"cohort series            : {len(rows)}")
    print(f"series with a mask       : {n_hit}")
    print(f"patients with >=1 mask   : {len(by_patient)} / "
          f"{len({r['PatientID'] for r in rows})}")
    print(f"patients with both roles : {sum(1 for v in by_patient.values() if v == 2)}")
    print("\nby role: " + ", ".join(
        f"{k}={v}" for k, v in sorted(
            Counter(r["role"] for r in rows if r["in_hqcolon"]).items()
        )
    ))
    print("HQColon Position of matched series: " + ", ".join(
        f"{k}={v}" for k, v in sorted(
            Counter(r["hqcolon_position"] for r in rows if r["in_hqcolon"]).items()
        )
    ))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
