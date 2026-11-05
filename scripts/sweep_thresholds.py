"""Does the output move when the two thresholds move?

The body threshold of -300 HU and the gas threshold of -700 HU are applied
uniformly across a four-vendor collection, so a sensitivity analysis over a
plausible range is what shows whether that uniformity costs anything. Thresholds are not a tabulation:
each value changes the masks, so the seeds, the path and the coverage all have
to be recomputed.

The expensive parts of a run are loading the resampled volume and extracting
the reference masks from the 62 GB archives, and neither depends on the
threshold, so both happen once per series and every setting is evaluated
inside that. A sweep of five settings therefore costs about one and a half
evaluations, not five.

What is measured per setting, against the same HQColon reference as the paper:
the gas mask volume, Dice of the lumen with the gas reference, the share of
the reference colon within 30 mm and beyond 60 mm of the path, the traced
length, and whether the series still meets the coverage and length criteria.
A threshold that changes the mask without changing the coverage is a different
finding from one that changes both, so the mask and the path are reported
separately.

    python scripts/sweep_thresholds.py --limit 2
    python scripts/sweep_thresholds.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_hqcolon as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

TABLES = ROOT / "results" / "tables"

# The range is centred on the reported setting and kept to values a reader
# would call plausible: gas thresholds between the lung and the colon wall,
# body thresholds between air and fat.
# A wider range at fewer points rather than fine steps: a sensitivity
# analysis has to reach the ends of what a reader would call plausible, and
# each setting costs a full mask, seed and path rebuild (90 s per series), so
# nine settings over 26 series is six and a half hours and five is three.
GAS_THRESHOLDS = (-800.0, -700.0, -600.0)
BODY_THRESHOLDS = (-400.0, -300.0, -200.0)
REPORTED_GAS = -700.0
REPORTED_BODY = -300.0

COVERAGE_BEYOND_60_MAX = 0.05
LENGTH_MIN_MM = 600.0


def settings() -> list[tuple[str, float, float]]:
    """One axis at a time through the reported setting, which appears once."""
    out = [("reported", REPORTED_BODY, REPORTED_GAS)]
    for g in GAS_THRESHOLDS:
        if g != REPORTED_GAS:
            out.append(("gas %.0f" % g, REPORTED_BODY, g))
    for b in BODY_THRESHOLDS:
        if b != REPORTED_BODY:
            out.append(("body %.0f" % b, b, REPORTED_GAS))
    return out


def one_setting(ct: sitk.Image, cfg, ref_gas: np.ndarray, ref_gf: np.ndarray,
                vox_ml: float) -> dict:
    """Masks, seeds, path and the reference comparison at one threshold pair."""
    from ctc_core.centerline import (
        detect_seed_endpoints,
        extract_centerline,
    )
    from ctc_core.masks import build_body_air_masks

    spacing_xyz = ct.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    t0 = time.perf_counter()
    _body, air, _diag = build_body_air_masks(
        ct, body_threshold=cfg.body_threshold, air_threshold=cfg.air_threshold,
        body_closing_radius=cfg.body_closing_radius,
        air_closing_radius=cfg.air_closing_radius,
        fill_holes=cfg.fill_holes, connectivity=cfg.connectivity)
    start, end, _s = detect_seed_endpoints(air, spacing_zyx, cfg)
    res = extract_centerline(ct, start, end, cfg)
    secs = time.perf_counter() - t0

    lumen = sitk.GetArrayFromImage(res.lumen) > 0
    path = np.array(res.path_kji, dtype=int)
    shape = lumen.shape
    pmask = np.zeros(shape, dtype=np.uint8)
    pmask[path[:, 0], path[:, 1], path[:, 2]] = 1
    pimg = sitk.GetImageFromArray(pmask)
    pimg.CopyInformation(ct)
    dist = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(
        pimg, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)))
    d = dist[ref_gf]

    from ctc_core.centerline import compute_path_length_mm
    length_mm = compute_path_length_mm(res.path_kji, spacing_xyz)
    beyond = float((d > E.FAR_MM).mean()) if d.size else float("nan")
    return {
        "air_ml": round(int(air.sum()) * vox_ml, 1),
        "lumen_ml": round(int(lumen.sum()) * vox_ml, 1),
        "dice_lumen_vs_gas": round(E.dice(lumen, ref_gas), 4),
        "colon_within_30mm_frac": round(float((d <= E.NEAR_MM).mean()), 4)
        if d.size else "",
        "colon_beyond_60mm_frac": round(beyond, 4) if d.size else "",
        "traced_length_mm": round(float(length_mm), 1),
        "met_coverage": int(beyond < COVERAGE_BEYOND_60_MAX) if d.size else "",
        "met_length": int(length_mm >= LENGTH_MIN_MM),
        "seconds": round(secs, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", type=Path, default=TABLES / "hqcolon_overlap.csv")
    ap.add_argument("--hqcolon", type=Path, default=ROOT / "data" / "hqcolon")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data" / "work")
    ap.add_argument("--out", type=Path, default=TABLES / "threshold_sweep.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from ctc_core.centerline import CenterlineConfig
    base = CenterlineConfig.corrected()
    plan = settings()
    print("%d settings per series: %s" % (len(plan), ", ".join(p[0] for p in plan)))

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
            vox_ml = float(np.prod(ct.GetSpacing())) / 1000.0
            refs = {}
            with tempfile.TemporaryDirectory(prefix="hqcolon_") as td:
                for key, (zname, folder) in E.ARCHIVES.items():
                    mp = E.extract_one(args.hqcolon / zname, folder,
                                       rec["hqcolon_label_file"], Path(td))
                    refs[key] = E.to_our_grid(mp, ct)
                    mp.unlink(missing_ok=True)
            for name, body, gas in plan:
                cfg = replace(base, body_threshold=body, air_threshold=gas)
                try:
                    got = one_setting(ct, cfg, refs["gas"], refs["gas_fluid"], vox_ml)
                    rows.append({"PatientID": pid, "role": role, "setting": name,
                                 "body_threshold_hu": body, "gas_threshold_hu": gas,
                                 **got})
                    print("  %-16s %-12s air %7.1f mL  Dice %.3f  within30 %s  "
                          "beyond60 %s  len %7.1f mm  %5.1f s"
                          % (tag, name, got["air_ml"], got["dice_lumen_vs_gas"],
                             got["colon_within_30mm_frac"],
                             got["colon_beyond_60mm_frac"],
                             got["traced_length_mm"], got["seconds"]), flush=True)
                except Exception as exc:                 # noqa: BLE001
                    rows.append({"PatientID": pid, "role": role, "setting": name,
                                 "body_threshold_hu": body, "gas_threshold_hu": gas,
                                 "error": "%s: %s" % (type(exc).__name__, exc)})
                    print("  %-16s %-12s FAILED %s: %s"
                          % (tag, name, type(exc).__name__, exc), flush=True)
        except Exception as exc:                          # noqa: BLE001
            print("  %-16s SERIES FAILED %s: %s"
                  % (tag, type(exc).__name__, exc), flush=True)
            rows.append({"PatientID": pid, "role": role, "setting": "",
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
        "reported_body_threshold_hu": REPORTED_BODY,
        "reported_gas_threshold_hu": REPORTED_GAS,
        "gas_thresholds_hu": list(GAS_THRESHOLDS),
        "body_thresholds_hu": list(BODY_THRESHOLDS),
        "one_axis_at_a_time": True,
        "n_series": len({(r["PatientID"], r["role"]) for r in rows}),
        "n_settings": len(plan),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
