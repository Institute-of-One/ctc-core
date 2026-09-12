"""Acquisition parameters of the cohort series, for the Methods section.

Reads DICOM headers only (no pixel data): the first slice of each series for the
scanner and exposure fields, and the two first slices in position order for the
reconstruction interval. Writes one row per series; the manuscript reports the
distribution (``scripts/make_tables.py``).

    python scripts/summarise_acquisition.py --raw-root . \\
        --out results/tables/acquisition.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import SimpleITK as sitk

TAGS = {
    "manufacturer": "0008|0070",
    "model": "0008|1090",
    "kvp": "0018|0060",
    "slice_thickness_mm": "0018|0050",
    "pixel_spacing_mm": "0028|0030",
    "tube_current_ma": "0018|1151",
    "exposure_mas": "0018|1152",
    "kernel": "0018|1210",
    "rows": "0028|0010",
    "columns": "0028|0011",
}


def header(path: str) -> sitk.ImageFileReader:
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r


def meta(r: sitk.ImageFileReader, tag: str) -> str:
    return r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--raw-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for rec in csv.DictReader(args.manifest.open(encoding="utf-8")):
        d = args.raw_root / rec["download_dir"]
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), rec["SeriesInstanceUID"])
        first, second = header(files[0]), header(files[1])
        z0 = float(meta(first, "0020|0032").split("\\")[2])
        z1 = float(meta(second, "0020|0032").split("\\")[2])
        row = {"PatientID": rec["PatientID"], "role": rec["role"],
               "SeriesInstanceUID": rec["SeriesInstanceUID"], "n_slices": len(files)}
        row.update({k: meta(first, t) for k, t in TAGS.items()})
        row["pixel_spacing_mm"] = row["pixel_spacing_mm"].split("\\")[0]
        row["reconstruction_interval_mm"] = round(abs(z1 - z0), 3)
        rows.append(row)
        print(f"{rec['PatientID'][-4:]}/{rec['role']}: {row['manufacturer']} {row['model']} "
              f"{row['kvp']} kVp, {row['slice_thickness_mm']} mm / "
              f"{row['reconstruction_interval_mm']} mm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} series -> {args.out}")


if __name__ == "__main__":
    main()
