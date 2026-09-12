"""Run the centerline stage over cohort series under a chosen configuration.

Two configurations are selectable:

``--config batch``
    Reproduces the 2026-08 reference batch exactly (26-connected fill,
    ``air_closing_radius=0``, 26-connected labelling, seeds from the largest
    component only). Used as the regression baseline -- 0001/primary must come
    out at 936.33 mm with 0 bridges.

``--config corrected``
    The three fixes of ``docs/FAILURE_ANALYSIS.md`` applied.

Per-series results go to ``<out>/<PatientID>/<role>/centerline.json`` (gitignored)
and a summary row per series to the ``--summary`` CSV (committed).

    python scripts/run_centerline.py --config batch --case 0001:primary \\
        --raw-root . --summary results/tables/centerline_batch.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.centerline import (  # noqa: E402
    CenterlineConfig,
    detect_seed_endpoints,
    extract_centerline,
    path_to_physical_mm,
)
from ctc_core.io import get_or_make_resampled_ct  # noqa: E402
from ctc_core.masks import build_body_air_masks  # noqa: E402

LOGGER = logging.getLogger("run_centerline")

SUMMARY_FIELDS = [
    "PatientID", "role", "SeriesInstanceUID", "config", "iso_spacing_mm",
    "ok", "error",
    "seed_sec", "centerline_sec", "total_sec",
    "air_cc_count", "largest_air_cc_voxels", "skeleton_voxels",
    "seed_source", "seed_distance_mm", "seeds_cross_components",
    "n_points", "length_mm", "n_ccs_total", "n_ccs_visited",
    "n_bridges", "bridge_length_mm_total", "used_kimimaro",
    "max_bridge_wall_frac", "max_bridge_wall_mm", "bridge_fallback",
    "traversal", "traversal_n_clusters", "traversal_cluster_components",
    "traversal_path_volume_frac_of_cluster",
    "straight_line_mm", "tortuosity",
    "fill_holes_used", "mask_fallback_reason",
    "seed_rule", "seed_rule_chosen",
    "cand_provisional_length_mm", "cand_provisional_lumen_coverage",
    "cand_rectal_length_mm", "cand_rectal_lumen_coverage",
    "cand_rectal_core_length_mm", "cand_rectal_core_lumen_coverage",
]

CONFIGS = {
    "batch": CenterlineConfig.batch_2026_08,
    "corrected": CenterlineConfig.corrected,
}


def run_one(ct: sitk.Image, cfg: CenterlineConfig, case_out: Path) -> dict:
    spacing_xyz = ct.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    t0 = time.perf_counter()
    _body, air, _mdiag = build_body_air_masks(
        ct,
        body_threshold=cfg.body_threshold,
        air_threshold=cfg.air_threshold,
        body_closing_radius=cfg.body_closing_radius,
        air_closing_radius=cfg.air_closing_radius,
        fill_holes=cfg.fill_holes,
        connectivity=cfg.connectivity,
    )
    start, end, sdiag = detect_seed_endpoints(air, spacing_zyx, cfg)
    seed_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = extract_centerline(ct, start, end, cfg)
    centerline_sec = time.perf_counter() - t1

    # Persist the path and the lumen mask the geodesic ran on: the fat map and
    # the quality indices consume the path, and the HQColon comparison needs
    # both. Binary masks compress to a few MB. Gitignored.
    pts = path_to_physical_mm(res.path_kji, ct)
    case_out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(res.lumen, str(case_out / "lumen_mask.nii.gz"), useCompression=True)
    with (case_out / "centerline_points.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_mm", "y_mm", "z_mm", "i", "j", "k"])
        for (k, j, i), (x, y, z) in zip(res.path_kji, pts, strict=True):
            w.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", i, j, k])

    # Tortuosity is the CTI of docs/PARAMETERS.md: path length over the
    # end-to-end straight distance. A physiological colon sits well above 1;
    # an implausibly large value with a large length suggests the path doubled
    # back or wandered out of the colon.
    straight = float(((pts[-1] - pts[0]) ** 2).sum() ** 0.5) if len(pts) >= 2 else 0.0

    return {
        "straight_line_mm": round(straight, 2),
        "tortuosity": round(res.length_mm / straight, 3) if straight > 1e-6 else None,
        "seed_sec": round(seed_sec, 2),
        "centerline_sec": round(centerline_sec, 2),
        "total_sec": round(seed_sec + centerline_sec, 2),
        "start_kji": list(start),
        "end_kji": list(end),
        **{k: v for k, v in sdiag.items()},
        "n_points": len(res.path_kji),
        "length_mm": round(res.length_mm, 2),
        "bridge_length_mm_total": round(
            sum(b["length_mm"] for b in res.diag["bridges"]), 2
        ),
        **res.diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--case", action="append",
                    help="<PatientID suffix>:<role>; repeatable. Omit for the whole manifest.")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/centerline"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--iso-spacing", type=float, default=1.0)
    ap.add_argument("--max-bridge-wall-mm", type=float, default=None,
                    help="override CenterlineConfig.max_bridge_wall_mm (calibration sweep)")
    ap.add_argument("--traversal", choices=("max_coverage", "shortest_path"), default=None,
                    help="override CenterlineConfig.traversal")
    ap.add_argument("--fill-holes", choices=("auto", "bbox", "slicewise"), default=None,
                    help="override CenterlineConfig.fill_holes (hole-fill ablation)")
    ap.add_argument("--seed-rule", choices=("skeleton_extremes", "whole_colon"), default=None,
                    help="override CenterlineConfig.seed_rule (seed-rule ablation)")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="rewrite --summary from the per-case centerline.json files of an "
                         "earlier run, computing nothing")
    ap.add_argument("--tag", default=None,
                    help="per-case output subdirectory; defaults to --config. Set it with any "
                         "override so the reported run's outputs are not overwritten")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = CONFIGS[args.config]()
    overrides = {}
    if args.max_bridge_wall_mm is not None:
        overrides["max_bridge_wall_mm"] = args.max_bridge_wall_mm
    if args.traversal is not None:
        overrides["traversal"] = args.traversal
    if args.fill_holes is not None:
        overrides["fill_holes"] = args.fill_holes
    if args.seed_rule is not None:
        overrides["seed_rule"] = args.seed_rule
    if overrides and args.tag in (None, args.config):
        raise SystemExit("config overrides need a --tag distinct from --config")
    tag = args.tag or args.config
    if overrides:
        cfg = replace(cfg, **overrides)
        LOGGER.info("config overrides: %s", overrides)

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))

    if args.rebuild_summary:
        rows = []
        for rec in manifest:
            f = args.out / tag / rec["PatientID"] / rec["role"] / "centerline.json"
            if f.exists():
                rows.append(json.loads(f.read_text(encoding="utf-8")))
        with args.summary.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} series rebuilt -> {args.summary}")
        return
    wanted = {tuple(c.split(":", 1)) for c in args.case} if args.case else None

    rows = []
    for rec in manifest:
        key = (rec["PatientID"].rsplit(".", 1)[-1], rec["role"])
        if wanted is not None and key not in wanted:
            continue
        dicom_dir = args.raw_root / rec["download_dir"]
        if not dicom_dir.is_dir():
            LOGGER.warning("missing DICOM dir, skipped: %s", dicom_dir)
            continue

        row = {
            "PatientID": rec["PatientID"],
            "role": rec["role"],
            "SeriesInstanceUID": rec["SeriesInstanceUID"],
            "config": tag,
            "iso_spacing_mm": args.iso_spacing,
        }
        LOGGER.info("=== %s / %s [%s]", rec["PatientID"], rec["role"], tag)
        case_out = args.out / tag / rec["PatientID"] / rec["role"]
        try:
            ct = get_or_make_resampled_ct(
                dicom_dir=dicom_dir,
                out_ct_path=args.work_dir / rec["PatientID"] / rec["role"] / "ct_lps_iso.nii.gz",
                iso_spacing=args.iso_spacing,
            )
            result = run_one(ct, cfg, case_out)
            row.update({"ok": True, "error": "", **result})
            LOGGER.info(
                "    length=%.2f mm  points=%d  CCs %d/%d  bridges=%d  %.1fs",
                result["length_mm"], result["n_points"],
                result["n_ccs_visited"], result["n_ccs_total"],
                result["n_bridges"], result["total_sec"],
            )
        except Exception as exc:
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.error("    FAILED %s", row["error"])
            LOGGER.debug(traceback.format_exc())

        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / "centerline.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        rows.append(row)

    if not rows:
        raise SystemExit("no cases processed")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n{len(rows)} series ({ok} ok) -> {args.summary}")


if __name__ == "__main__":
    main()
