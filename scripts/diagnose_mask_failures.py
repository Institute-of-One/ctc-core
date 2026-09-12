"""Settle the two untested failure mechanisms of docs/FAILURE_ANALYSIS.md.

For each requested series this reproduces the 2026-08 batch input (DICOM ->
1.0 mm LPS isotropic) and then measures, on that one volume:

Experiment A -- body-mask hole filling (section 2.2)
    Build the body mask three ways: ``image26`` (what the centerline stage did
    -- BinaryFillhole with a 26-connected background), ``image6`` (the same
    filter with a 6-connected background) and ``bbox`` (what the fat-map stage
    did). Report body volume, air-in-body volume and the largest air component
    under each. If they disagree by orders of magnitude, the "insufficient
    distension" label was a body-mask artefact, not anatomy; comparing
    ``image26`` with ``image6`` isolates the background-connectivity cause.

Experiment B -- air-mask closing radius
    Under the better body mask, compare ``closing_radius`` 0 (batch centerline
    stage) and 1 (batch fat-map stage) for the number of surviving components
    and the size of the largest.

Experiment C -- labelling connectivity vs. the Fast Marching stencil (section 2.3)
    Reproduce the automatic seed pair, then label the air mask with
    ``connectivity=26`` and with ``connectivity=6`` and report whether the two
    seeds share a component under each. cc3d labels 26-connected while
    ``sitk.FastMarchingImageFilter`` propagates 6-connected, so seeds in one
    26-component but two 6-components explain "End seed is unreachable".

Read-only with respect to the raw data. Usage:

    python scripts/diagnose_mask_failures.py --manifest cohort/manifest.csv \\
        --raw-root . --case 0022:primary --case 0026:primary \\
        --out results/tables/mask_diagnosis.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.io import get_or_make_resampled_ct  # noqa: E402
from ctc_core.masks import (  # noqa: E402
    AIR_THRESHOLD_HU,
    BODY_THRESHOLD_HU,
    build_air_mask,
    build_body_mask,
    label_air_components,
    touches_image_border,
)

DUST_VOXELS = 500
MIN_AIR_CC_VOXELS = 10_000


def ml(n_vox: int, spacing_mm: float) -> float:
    """Voxel count -> millilitres."""
    return n_vox * (spacing_mm**3) / 1000.0


def largest_cc_voxels(air: np.ndarray, connectivity: int = 26) -> int:
    import cc3d

    if not air.any():
        return 0
    labels, _ = cc3d.connected_components(
        air.astype(np.uint8), connectivity=connectivity, return_N=True
    )
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return int(sizes.max()) if sizes.size else 0


def detect_seeds(air: np.ndarray, spacing_mm: float) -> tuple[tuple, tuple, dict]:
    """Reproduce ``run_cohort_batch.auto_detect_seed_endpoints`` seed selection."""
    import cc3d
    from scipy import ndimage
    from skimage.morphology import skeletonize

    labels, _ = cc3d.connected_components(air.astype(np.uint32), return_N=True)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest = labels == largest_label
    diag = {"largest_air_cc_voxels": int(sizes[largest_label])}

    skel = skeletonize(largest, method="lee")
    diag["skeleton_voxels"] = int(skel.sum())

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    endpoints = np.argwhere(skel & (degree == 1))

    if len(endpoints) >= 2:
        phys = endpoints.astype(np.float64) * spacing_mm
        d2 = ((phys[:, None, :] - phys[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        start = tuple(int(v) for v in endpoints[i])
        end = tuple(int(v) for v in endpoints[j])
        diag["seed_source"] = "skeleton_endpoints"
        diag["seed_distance_mm"] = float(np.sqrt(d2[i, j]))
    else:
        coords = np.argwhere(largest)
        start = tuple(int(v) for v in coords[coords[:, 0].argmin()])
        end = tuple(int(v) for v in coords[coords[:, 0].argmax()])
        diag["seed_source"] = "cc_z_extremes"
        diag["seed_distance_mm"] = float(abs(end[0] - start[0]) * spacing_mm)

    return start, end, diag


def diagnose(ct: sitk.Image, spacing: float) -> dict:
    ct_arr = sitk.GetArrayFromImage(ct)
    row: dict = {"size_zyx": "x".join(str(s) for s in ct_arr.shape)}

    # --- Experiment A: body-mask hole filling ------------------------------
    for variant in ("image26", "image6", "bbox"):
        body = build_body_mask(ct, BODY_THRESHOLD_HU, closing_radius=2, fill_holes=variant)
        air0 = build_air_mask(ct_arr, body, AIR_THRESHOLD_HU, closing_radius=0, reference=ct)
        row[f"body_ml_{variant}"] = round(ml(int(body.sum()), spacing), 1)
        row[f"air_ml_{variant}"] = round(ml(int(air0.sum()), spacing), 1)
        row[f"largest_cc_ml_{variant}"] = round(ml(largest_cc_voxels(air0), spacing), 1)
        if variant == "bbox":
            body_bbox, air_bbox = body, air0
        elif variant == "image26":
            air_image26 = air0

    row["largest_cc_ratio_bbox_over_image26"] = round(
        row["largest_cc_ml_bbox"] / max(row["largest_cc_ml_image26"], 1e-9), 1
    )
    # Does switching the background to 6-connected alone recover the gas?
    row["largest_cc_ratio_image6_over_image26"] = round(
        row["largest_cc_ml_image6"] / max(row["largest_cc_ml_image26"], 1e-9), 1
    )
    # Gas recovered by "bbox" but not by "image26": does it reach the volume face?
    recovered = air_bbox & ~air_image26
    row["recovered_gas_ml"] = round(ml(int(recovered.sum()), spacing), 1)
    row["recovered_gas_touches_border"] = touches_image_border(recovered)

    # --- Experiment B: air-mask closing radius -----------------------------
    for r in (0, 1):
        air = build_air_mask(ct_arr, body_bbox, AIR_THRESHOLD_HU, closing_radius=r, reference=ct)
        labels, sizes = label_air_components(air, connectivity=26, dust_threshold_voxels=0)
        kept = int((sizes >= DUST_VOXELS).sum())
        row[f"n_cc_kept_close{r}"] = kept
        row[f"largest_cc_ml_close{r}"] = round(ml(int(sizes.max()), spacing), 1)
        if r == 1:
            air_final = air

    # --- Experiment C: connectivity vs Fast Marching stencil ---------------
    # Run under two configurations. "batch" reproduces the mask the 2026-08 run
    # actually failed on, and so is the one that can explain the historical
    # error; "fixed" uses the corrected body mask and air closing, and says
    # whether the connectivity mismatch still bites after the section 2.2 fix.
    configs = {"batch": air_image26, "fixed": air_final}
    for tag, air in configs.items():
        try:
            start, end, sdiag = detect_seeds(air, spacing)
            row[f"seed_distance_mm_{tag}"] = round(sdiag["seed_distance_mm"], 1)
            row[f"seed_source_{tag}"] = sdiag["seed_source"]
            row[f"largest_cc_ml_seedstage_{tag}"] = round(
                ml(sdiag["largest_air_cc_voxels"], spacing), 1
            )
            for conn in (26, 6):
                labels, _ = label_air_components(
                    air, connectivity=conn, dust_threshold_voxels=DUST_VOXELS
                )
                same = int(labels[start]) == int(labels[end]) and int(labels[start]) != 0
                row[f"same_cc_{conn}_{tag}"] = bool(same)
            row[f"connectivity_mismatch_{tag}"] = bool(
                row[f"same_cc_26_{tag}"]
            ) and not bool(row[f"same_cc_6_{tag}"])
        except Exception as exc:  # pragma: no cover - diagnostic path
            row[f"seed_error_{tag}"] = f"{type(exc).__name__}: {exc}"

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="directory the manifest's download_dir column is relative to")
    ap.add_argument("--case", action="append", required=True,
                    help="<PatientID suffix>:<role>, e.g. 0022:primary (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iso-spacing", type=float, default=1.0)
    ap.add_argument("--work-dir", type=Path,
                    default=Path("data/work"),
                    help="scratch for resampled volumes (gitignored)")
    args = ap.parse_args()

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    wanted = {tuple(c.split(":", 1)) for c in args.case}

    rows = []
    for rec in manifest:
        key = (rec["PatientID"].rsplit(".", 1)[-1], rec["role"])
        if key not in wanted:
            continue
        dicom_dir = args.raw_root / rec["download_dir"]
        if not dicom_dir.is_dir():
            print(f"!! missing DICOM dir, skipped: {dicom_dir}")
            continue

        print(f"== {rec['PatientID']} / {rec['role']}", flush=True)
        t0 = time.perf_counter()
        ct = get_or_make_resampled_ct(
            dicom_dir=dicom_dir,
            out_ct_path=args.work_dir / rec["PatientID"] / rec["role"] / "ct_lps_iso.nii.gz",
            iso_spacing=args.iso_spacing,
        )
        print(f"   loaded/resampled in {time.perf_counter() - t0:.1f}s", flush=True)

        row = {"PatientID": rec["PatientID"], "role": rec["role"]}
        row.update(diagnose(ct, args.iso_spacing))
        rows.append(row)
        print("   " + json.dumps({k: v for k, v in row.items() if k not in ("PatientID", "role")},
                                 default=str), flush=True)

    if not rows:
        raise SystemExit("no cases processed")

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} series -> {args.out}")


if __name__ == "__main__":
    main()
