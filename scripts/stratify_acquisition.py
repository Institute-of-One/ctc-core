"""The main results by scanner, kernel, slice thickness and exposure.

A fixed pipeline applied across a four-vendor collection should be shown to
behave the same way on each of them, or shown not to. Two properties of this
collection decide what stratifying by manufacturer, kernel, slice thickness and
exposure can mean, and both are reported rather than worked around:

* **the first three are the same stratification.** Each manufacturer in the
  collection contributed exactly one kernel and one slice thickness -- Siemens
  B30f at 1.0 mm, GE STANDARD at 1.25 mm, Philips B at 1.0 mm, Toshiba FC13 at
  1.0 mm -- so stratifying by kernel is stratifying by manufacturer. Reporting
  them as three independent axes would show the same split three times and
  imply the collection can separate them.
* **the recorded exposure is not a dose axis at all.** Within one model from
  one manufacturer, at one tube current, the Exposure field takes three
  clusters: 6 mAs on six series, 152 to 193 on four, and 2254 to 2824 on
  twelve, all at 140 mA on a LightSpeed16. Three orders of magnitude at the
  same stated current is a difference in what the field records, not in what
  the scanner delivered, so the results are not split by it; splitting by it
  anyway gives a difference in Dice of 0.14 between the halves, which is a
  difference between recording conventions. The distribution is reported as
  the evidence, and tube current -- recorded consistently -- is stratified in
  its place. CTDIvol is not in the retained headers, so no better dose axis
  can be recovered here.

The strata are small -- two patients each for Philips and Toshiba -- so this is
a description of where the results sit, not a test. No interval is given for a
stratum of two patients.

    python scripts/stratify_acquisition.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

COVERAGE_BEYOND_60_MAX = 0.05
LENGTH_MIN_MM = 600.0
INSPECTED_PATIENTS = {"0001", "0003", "0004", "0007", "0030"}

# The acquisition axes, in the order they are reported.
# (field, label, parse). Numeric fields are parsed, because "1", "1.0" and
# "1.00" are one slice thickness and three strings.
AXES = [("manufacturer", "manufacturer", str),
        ("kernel", "reconstruction kernel", str),
        ("slice_thickness_mm", "slice thickness, mm", float),
        ("kvp", "tube voltage, kVp", float),
        ("tube_current_ma", "tube current, mA", float)]

REFERENCE_MEASURES = [("dice_lumen_vs_gas", "Dice, lumen vs gas reference", 3),
                      ("colon_within_30mm_frac", "Colon within 30 mm", 3),
                      ("colon_beyond_60mm_frac", "Colon beyond 60 mm", 3)]


def read(name: str, tables: Path = TABLES) -> list[dict]:
    with (tables / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def med(values) -> str:
    vals = [v for v in values if v is not None]
    return "%.3f" % statistics.median(vals) if vals else "n/a"


def collinear(rows: list[dict], a: str, b: str) -> bool:
    """Does each value of a go with exactly one value of b, and the reverse?"""
    forward: dict[str, set] = {}
    back: dict[str, set] = {}
    for r in rows:
        forward.setdefault(r[a], set()).add(r[b])
        back.setdefault(r[b], set()).add(r[a])
    return (all(len(v) == 1 for v in forward.values())
            and all(len(v) == 1 for v in back.values()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # The inputs come from a directory, not from the committed one by
    # construction: run_all.py --results <dir> must be able to stratify a
    # re-run's own tables rather than silently reading the committed ones.
    ap.add_argument("--tables", type=Path, default=TABLES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = args.tables / "stratified_acquisition.csv"

    acq = read("acquisition.csv", args.tables)
    hq = {(r["PatientID"], r["role"]): r
          for r in read("eval_hqcolon_auto.csv", args.tables)}
    cl = {(r["PatientID"], r["role"]): r
          for r in read("centerline_auto.csv", args.tables)}

    by_key = {(r["PatientID"], r["role"]): r for r in acq}
    print("acquisition rows: %d" % len(acq))

    # Which of the named axes are the same split?
    pairs = [("manufacturer", "kernel"), ("manufacturer", "slice_thickness_mm"),
             ("kernel", "slice_thickness_mm")]
    collinearity = {}
    for a, b in pairs:
        same = collinear(acq, a, b)
        collinearity["%s~%s" % (a, b)] = same
        print("  %-38s %s" % ("%s and %s" % (a, b),
                              "the same split" if same else "independent"))

    rows: list[dict] = []
    for field, label, parse in AXES:
        def key_of(row, _f=field, _p=parse):
            try:
                return _p(row[_f])
            except (TypeError, ValueError):
                return row[_f]
        values = sorted({key_of(r) for r in acq}, key=str)
        for value in values:
            keys = [k for k, r in by_key.items() if key_of(r) == value]
            series = len(keys)
            patients = len({k[0] for k in keys})
            row = {"axis": label, "stratum": value,
                   "series": series, "patients": patients}
            lengths = [num(cl[k]["length_mm"]) for k in keys if k in cl]
            ok = [(cl[k]["ok"] or "").strip().lower() in ("true", "1", "yes")
                  for k in keys if k in cl]
            row["centerline produced"] = "%d / %d" % (sum(ok), len(ok))
            row["met the length criterion"] = "%d / %d" % (
                sum(1 for v in lengths if v is not None and v >= LENGTH_MIN_MM),
                len(lengths))
            row["traced length, mm"] = (
                "%.0f" % statistics.median([v for v in lengths if v is not None])
                if any(v is not None for v in lengths) else "n/a")
            ref = [hq[k] for k in keys if k in hq]
            row["reference series"] = len(ref)
            for col, name, _digits in REFERENCE_MEASURES:
                row[name] = med([num(r[col]) for r in ref]) if ref else "n/a"
            if ref:
                flags = [num(r["colon_beyond_60mm_frac"]) for r in ref]
                row["met the coverage criterion"] = "%d / %d" % (
                    sum(1 for v in flags
                        if v is not None and v < COVERAGE_BEYOND_60_MAX),
                    sum(1 for v in flags if v is not None))
            else:
                row["met the coverage criterion"] = "n/a"
            rows.append(row)
            print("  %-22s %-20s %2d series / %2d patients  Dice %s  "
                  "within30 %s  coverage %s"
                  % (label, str(value)[:20], series, patients,
                     row["Dice, lumen vs gas reference"],
                     row["Colon within 30 mm"],
                     row["met the coverage criterion"]))

    # The Exposure field, reported as a distribution rather than used as an
    # axis. Splitting the results by it would be splitting them by how the
    # value was recorded.
    print("\nthe recorded Exposure, by manufacturer and model "
          "(not used as an axis)")
    for maker in sorted({r["manufacturer"] for r in acq}):
        sel = [r for r in acq if r["manufacturer"] == maker]
        vals = sorted(v for v in (num(r["exposure_mas"]) for r in sel)
                      if v is not None)
        currents = sorted({num(r["tube_current_ma"]) for r in sel})
        models = sorted({r["model"] for r in sel})
        spread = (vals[-1] / vals[0]) if vals and vals[0] else float("nan")
        rows.append({"axis": "recorded Exposure, mAs (not used as an axis)",
                     "stratum": maker, "series": len(sel),
                     "patients": len({r["PatientID"] for r in sel}),
                     "note": "%s; %s mAs; %s mA; max/min %.0f"
                             % ("/".join(models),
                                "-".join(str(int(v)) for v in (vals[0], vals[-1]))
                                if vals else "n/a",
                                "/".join(str(int(c)) for c in currents if c),
                                spread)})
        print("  %-20s %-16s %s mAs at %s mA -> max/min %.0f"
              % (maker, "/".join(models),
                 "%d-%d" % (vals[0], vals[-1]) if vals else "n/a",
                 "/".join(str(int(c)) for c in currents if c), spread))

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    exposures = {m: sorted(num(r["exposure_mas"]) for r in acq
                           if r["manufacturer"] == m)
                 for m in sorted({r["manufacturer"] for r in acq})}
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({
        "collinear_axes": collinearity,
        "note": "manufacturer, kernel and slice thickness are the same split "
                "in this collection, so they are reported once",
        "exposure_mas_by_manufacturer": {
            m: {"min": v[0], "max": v[-1], "n": len(v),
                "distinct": sorted({int(x) for x in v})}
            for m, v in exposures.items()},
        "exposure_usable_as_a_dose_axis": False,
        "exposure_why_not": "within one model at one tube current the field "
                            "takes three clusters three orders of magnitude "
                            "apart, so it records different quantities on "
                            "different series",
        "ctdivol_available": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s (%d rows) and %s" % (args.out, len(rows), meta.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
