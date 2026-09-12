"""Recover the true acquisition position (prone / supine) for each cohort series.

This is needed because the ``role`` column of ``cohort/manifest.csv``
(primary / secondary) is derived from SeriesNumber, not from how the patient
lay: against the HQColon reference it splits 8 prone / 7 supine for primary and
6 prone / 5 supine for secondary, i.e. it carries no position information. Any
analysis described as prone/supine agreement therefore needs the position
recovered from the images.

Two DICOM signals are available and they do not always agree:

``PatientPosition`` (0018,5100)
    ``HFS``/``FFS`` = head/feet-first supine, ``HFP``/``FFP`` = prone. Directly
    meaningful, but observed to be wrong in this collection: 0003/primary is
    prone in HQColon yet tagged ``FFS``.

``ImageOrientationPatient`` (0020,0037)
    A prone acquisition reconstructed in the patient frame often carries a
    180-degree in-plane flip of the row and column direction cosines where a
    supine one does not.

Scored against HQColon on the 26 series it covers, **the tag wins**: 25/26
(96.2 %) against 21/26 (80.8 %) for the orientation. Preferring the
orientation, which seemed the safer choice because it is geometric rather than
operator-entered, scored worse overall (23/26): three prone series tagged
FFP/HFP were reconstructed without the in-plane flip, and only one supine-tagged
series (0003/primary, prone in HQColon) is caught by the orientation. Since
disagreement between the two does not resolve reliably in either direction, the
tag is used and that single known error is reported rather than patched.

The tag-first split is also the plausible one -- 28 prone / 31 supine / 1
decubitus against an expected near-balance -- where orientation-first gives a
skewed 21 / 38.

Both signals are written per series so the choice stays auditable.

    python scripts/derive_patient_position.py --raw-root . \\
        --overlap results/tables/hqcolon_overlap.csv \\
        --out results/tables/patient_position.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

FIELDS = [
    "PatientID", "role", "SeriesInstanceUID",
    "patient_position_tag", "image_orientation",
    "position_from_tag", "position_from_orientation", "position_combined",
    "hqcolon_position", "tag_agrees", "orientation_agrees", "combined_agrees",
]


def read_tags(dicom_dir: Path) -> tuple[str, str]:
    files = sorted(glob.glob(str(dicom_dir / "*")))
    if not files:
        return "", ""
    rd = sitk.ImageFileReader()
    rd.SetFileName(files[len(files) // 2])
    rd.ReadImageInformation()
    pp = rd.GetMetaData("0018|5100").strip() if rd.HasMetaDataKey("0018|5100") else ""
    io = rd.GetMetaData("0020|0037").strip() if rd.HasMetaDataKey("0020|0037") else ""
    return pp, io


def position_from_tag(pp: str) -> str:
    if not pp:
        return ""
    u = pp.upper()
    if u.endswith("P"):
        return "prone"
    if u.endswith("S"):
        return "supine"
    return ""


def position_from_orientation(io: str) -> str:
    """Prone if the in-plane axes are flipped by 180 degrees."""
    if not io:
        return ""
    try:
        v = [float(x) for x in io.replace("\\", " ").split()]
    except ValueError:
        return ""
    if len(v) < 6:
        return ""
    row, col = np.array(v[:3]), np.array(v[3:6])
    # Compare against the canonical supine axes (+x right, +y posterior).
    if row[0] < -0.5 and col[1] < -0.5:
        return "prone"
    if row[0] > 0.5 and col[1] > 0.5:
        return "supine"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--overlap", type=Path, default=None,
                    help="hqcolon_overlap.csv, to validate the rules")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    truth: dict[str, str] = {}
    if args.overlap and args.overlap.exists():
        for r in csv.DictReader(args.overlap.open(encoding="utf-8")):
            if r["in_hqcolon"] == "True":
                truth[r["SeriesInstanceUID"]] = r["hqcolon_position"]

    rows = []
    for rec in csv.DictReader(args.manifest.open(encoding="utf-8")):
        pp, io = read_tags(args.raw_root / rec["download_dir"])
        p_tag = position_from_tag(pp)
        p_ori = position_from_orientation(io)
        # Tag first: validated at 25/26 against HQColon versus 21/26 for the
        # orientation. Orientation is only a fallback for an absent or
        # non-committal tag (e.g. the single FFDR decubitus series).
        combined = p_tag or p_ori
        gt = truth.get(rec["SeriesInstanceUID"], "")
        rows.append(
            {
                "PatientID": rec["PatientID"],
                "role": rec["role"],
                "SeriesInstanceUID": rec["SeriesInstanceUID"],
                "patient_position_tag": pp,
                "image_orientation": io,
                "position_from_tag": p_tag,
                "position_from_orientation": p_ori,
                "position_combined": combined,
                "hqcolon_position": gt,
                "tag_agrees": (p_tag == gt) if gt else "",
                "orientation_agrees": (p_ori == gt) if gt else "",
                "combined_agrees": (combined == gt) if gt else "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} series -> {args.out}\n")
    print("derived position (all series): " + ", ".join(
        f"{k or 'unknown'}={v}"
        for k, v in sorted(Counter(r["position_combined"] for r in rows).items())
    ))
    print("PatientPosition tag values  : " + ", ".join(
        f"{k or 'absent'}={v}"
        for k, v in sorted(Counter(r["patient_position_tag"] for r in rows).items())
    ))

    scored = [r for r in rows if r["hqcolon_position"]]
    if scored:
        print(f"\nvalidation against HQColon (n={len(scored)}):")
        for rule in ("tag", "orientation", "combined"):
            key = f"{rule}_agrees"
            ok = sum(1 for r in scored if r[key] is True)
            print(f"  {rule:12s} {ok:2d}/{len(scored)}  ({100 * ok / len(scored):5.1f} %)")
        bad = [r for r in scored if r["combined_agrees"] is not True]
        if bad:
            print("  combined rule disagreements:")
            for r in bad:
                print(f"    {r['PatientID'][-4:]}/{r['role']}: "
                      f"HQColon={r['hqcolon_position']} tag={r['patient_position_tag']} "
                      f"orientation={r['position_from_orientation'] or '-'}")

    print("\nrole vs derived position (role carries no position information):")
    for k, v in sorted(Counter((r["role"], r["position_combined"]) for r in rows).items()):
        print(f"  {k[0]:10s} <-> {k[1] or 'unknown':8s} {v}")

    paired = Counter()
    by_patient: dict[str, set[str]] = {}
    for r in rows:
        by_patient.setdefault(r["PatientID"], set()).add(r["position_combined"])
    for pos in by_patient.values():
        paired["prone+supine" if {"prone", "supine"} <= pos else "not both"] += 1
    print(f"\npatients with both positions: {paired['prone+supine']} / {len(by_patient)}")


if __name__ == "__main__":
    main()
