"""Where the colonic gas actually sits in HU, per series and per scanner.

The threshold sweep (scripts/sweep_thresholds.py) shows the coverage moving a
long way when the gas threshold moves: a median 0.955 of the reference colon
within 30 mm of the path at -800 HU against 0.820 at the reported -700 and
0.561 at -600. That is a result about this pipeline. It does not say why, and
without a why a reader cannot tell whether -700 is a bad choice or a choice
that means different things on different scanners.

CT numbers are not absolute. They depend on the scanner's calibration, the
tube voltage, the reconstruction kernel and the patient, so a fixed threshold
in HU is not a fixed physical boundary. That is a statement about CT, and this
script turns it into a measurement on these images rather than an appeal to
it: the attenuation of the colonic gas itself, inside the HQColon gas mask,
per series, with the share of that gas each candidate threshold would admit.

Two quantities per series:

* **colonic gas** -- the HU distribution inside the reference gas mask, which
  is the material the threshold is meant to select, and the share of it below
  -800, -700 and -600 HU;
* **external air** -- the HU of the air outside the patient, in the corners of
  the field of view, which a calibrated scanner puts at -1000. Its departure
  from -1000 is the calibration offset, measured rather than assumed.

    python scripts/measure_gas_attenuation.py
    python scripts/measure_gas_attenuation.py --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
# This tree is read on a cp932 console, where one em dash in a print ends a
# three-hour run at its last line.
sys.stdout.reconfigure(encoding="utf-8")

import eval_hqcolon as E  # noqa: E402

TABLES = ROOT / "results" / "tables"

# The thresholds the sweep varies, so the two tables can be read together.
CANDIDATES = (-800.0, -700.0, -600.0)
# A corner column of this many voxels in-plane, taken at all four corners and
# over the middle half of the slices: outside the patient and, at these fields
# of view, outside the table as well.
CORNER = 24


def external_air_hu(hu: np.ndarray) -> tuple[float, int]:
    """Median HU of the field-of-view corners, and how many voxels that was.

    A calibrated scanner reconstructs air at -1000 HU. Corners are used rather
    than a mask of everything outside the body because the table, the blanket
    and the arms sit beside the patient and would drag the value up.
    """
    nz = hu.shape[0]
    z0, z1 = nz // 4, max(nz // 4 + 1, 3 * nz // 4)
    sl = hu[z0:z1]
    corners = np.concatenate([
        sl[:, :CORNER, :CORNER].ravel(), sl[:, :CORNER, -CORNER:].ravel(),
        sl[:, -CORNER:, :CORNER].ravel(), sl[:, -CORNER:, -CORNER:].ravel()])
    if corners.size == 0:
        return float("nan"), 0
    return float(np.median(corners)), int(corners.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", type=Path, default=TABLES / "hqcolon_overlap.csv")
    ap.add_argument("--hqcolon", type=Path, default=ROOT / "data" / "hqcolon")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data" / "work")
    ap.add_argument("--acquisition", type=Path, default=TABLES / "acquisition.csv")
    ap.add_argument("--out", type=Path, default=TABLES / "gas_attenuation.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with args.acquisition.open(newline="", encoding="utf-8") as fh:
        acq = {(r["PatientID"], r["role"]): r for r in csv.DictReader(fh)}
    with args.overlap.open(newline="", encoding="utf-8") as fh:
        overlap = [r for r in csv.DictReader(fh)
                   if (r.get("hqcolon_label_file") or "").strip()]
    if args.limit:
        overlap = overlap[:args.limit]
    print("%d series with an HQColon reference" % len(overlap))

    rows: list[dict] = []
    for rec in overlap:
        pid, role = rec["PatientID"], rec["role"]
        tag = "%s/%s" % (pid[-4:], role)
        try:
            ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
            if not ct_path.exists():
                raise FileNotFoundError(str(ct_path))
            ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
            hu = sitk.GetArrayFromImage(ct)
            with tempfile.TemporaryDirectory(prefix="hqcolon_") as td:
                zname, folder = E.ARCHIVES["gas"]
                mp = E.extract_one(args.hqcolon / zname, folder,
                                   rec["hqcolon_label_file"], Path(td))
                ref_gas = E.to_our_grid(mp, ct)
                mp.unlink(missing_ok=True)
            gas = hu[ref_gas]
            if gas.size == 0:
                raise RuntimeError("the reference gas mask is empty")
            air_hu, air_n = external_air_hu(hu)
            a = acq.get((pid, role), {})
            q = np.percentile(gas, [5, 25, 50, 75, 95])
            row = {
                "PatientID": pid, "role": role,
                "manufacturer": a.get("manufacturer", ""),
                "model": a.get("model", ""), "kernel": a.get("kernel", ""),
                "kvp": a.get("kvp", ""),
                "gas_voxels": int(gas.size),
                "gas_hu_p5": round(float(q[0]), 1),
                "gas_hu_q1": round(float(q[1]), 1),
                "gas_hu_median": round(float(q[2]), 1),
                "gas_hu_q3": round(float(q[3]), 1),
                "gas_hu_p95": round(float(q[4]), 1),
                "external_air_hu": round(air_hu, 1),
                "external_air_voxels": air_n,
                # How much of the true colonic gas each candidate threshold
                # would admit. This is the quantity the sweep moves.
                **{"frac_below_%d" % abs(int(t)): round(float((gas < t).mean()), 4)
                   for t in CANDIDATES},
            }
            rows.append(row)
            print("  %-16s %-18s gas HU %7.1f [%7.1f, %7.1f]  air %7.1f  "
                  "admitted: -800 %5.1f%%  -700 %5.1f%%  -600 %5.1f%%"
                  % (tag, row["manufacturer"][:18], row["gas_hu_median"],
                     row["gas_hu_p5"], row["gas_hu_p95"], row["external_air_hu"],
                     100 * row["frac_below_800"], 100 * row["frac_below_700"],
                     100 * row["frac_below_600"]), flush=True)
        except Exception as exc:                     # noqa: BLE001
            print("  %-16s FAILED %s: %s" % (tag, type(exc).__name__, exc),
                  flush=True)
            rows.append({"PatientID": pid, "role": role,
                         "error": "%s: %s" % (type(exc).__name__, exc)})

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
    args.out.with_suffix(".json").write_text(json.dumps({
        "candidate_thresholds_hu": list(CANDIDATES),
        "corner_voxels_in_plane": CORNER,
        "calibrated_air_hu": -1000,
        "note": "gas HU is measured inside the HQColon gas mask, which is the "
                "material the threshold selects; external air is the corners "
                "of the field of view, where a calibrated scanner reads -1000",
        "n_series": len([r for r in rows if "error" not in r]),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
