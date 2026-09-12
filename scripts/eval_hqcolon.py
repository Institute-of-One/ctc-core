"""Evaluate our lumen segmentation and centerline against the HQColon reference.

Evaluation pillar 1, and the precondition for any accuracy claim: centerline
length alone distinguishes neither a partially covered colon from a fully
covered one nor a valid path from one bridged through solid tissue.

HQColon masks sit on the native DICOM grid but share the physical frame of our
resampled volume (identical origin, identity direction), so they are brought
onto our grid with an identity-transform nearest-neighbour resample. The
archives are 62 GB each uncompressed, so each mask is extracted on demand and
deleted immediately; peak extra disk is about 350 MB.

Metrics, per series:

*Segmentation.* Dice and Jaccard of our air mask (every intra-abdominal gas
voxel, so recall-oriented — does our mask contain the colon at all?) and of our
lumen mask (the components the geodesic actually ran on, which is our real claim
about where the colon is) against both HQColon variants. Mean and 95th-percentile
symmetric surface distance for the lumen.

*Centerline precision.* Fraction of centerline points falling inside the
reference colon. This is the direct test of bridge validity: a bridge tunnelled
through tissue puts points outside the colon, which no length-based criterion can
detect.

*Centerline coverage.* Primary, and independent of any reference centerline:
the share of the whole reference colon (gas and fluid) within 30 mm of our path,
and the share beyond 60 mm -- colon the path never visits, since no lumen is that
wide. Secondary: a whole-colon reference centerline, a Fast Marching geodesic
between the two ends of the reference colon's uniform-speed geodesic diameter;
our length over its length, and the share of its points within 10 mm of our
path. The reference mask holds colon only, so its diameter ends are the colon's
ends. (Until 2026-09-11 the reference centerline was seeded like ours, by the
most distant skeleton end points by Euclidean distance; both were truncated in
the same way. See docs/EVALUATION_HQCOLON.md section 8.)

    python scripts/eval_hqcolon.py --overlap results/tables/hqcolon_overlap.csv \\
        --centerline-root data/centerline/corrected \\
        --out results/tables/eval_hqcolon.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ARCHIVES = {
    "gas": ("gas-filled-colon-segmentation.zip", "Segmentation Air"),
    "gas_fluid": ("gas-and-fluid-filled-colon-segmentation.zip", "Segmentation Air and Fluid"),
}

# Radius within which reference colon volume counts as "near" our centerline.
# 30 mm is a little over the radius of a well distended colon, so a point of the
# reference lumen within 30 mm of the path is plausibly served by it.
NEAR_MM = 30.0
# Beyond this distance a reference voxel cannot belong to a lumen the path runs
# through, so it marks colon the path never visits.
FAR_MM = 60.0

# Tolerance for calling a reference centerline point covered by our path. A well
# distended colon has a radius of roughly 15-25 mm, so 10 mm keeps a covered
# point comfortably within the same lumen cross-section.
COVER_TOL_MM = 10.0

FIELDS = [
    "PatientID", "role", "SeriesInstanceUID", "hqcolon_subject_id", "hqcolon_position",
    "ref_gas_ml", "ref_gas_fluid_ml", "our_air_ml", "our_lumen_ml",
    "dice_air_vs_gas", "dice_air_vs_gas_fluid",
    "dice_lumen_vs_gas", "jaccard_lumen_vs_gas", "dice_lumen_vs_gas_fluid",
    "lumen_recall_vs_gas", "lumen_precision_vs_gas",
    "msd_lumen_vs_gas_mm", "hd95_lumen_vs_gas_mm",
    "n_centerline_points", "centerline_len_mm",
    "centerline_inside_gas_frac", "centerline_inside_gas_fluid_frac",
    "ref_centerline_len_mm", "length_ratio",
    "ref_covered_frac", "ref_to_our_mean_mm", "our_to_ref_mean_mm",
    "ref_gf_centerline_len_mm", "length_ratio_gf", "ref_gf_covered_frac",
    f"ref_vol_within_{int(NEAR_MM)}mm_frac",
    f"colon_within_{int(NEAR_MM)}mm_frac", f"colon_beyond_{int(FAR_MM)}mm_frac",
    "error",
]


def extract_one(archive: Path, folder: str, label_file: str, dest_dir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        member = f"{folder}/{label_file}"
        if member not in zf.namelist():
            cands = [n for n in zf.namelist() if n.endswith("/" + label_file)]
            if not cands:
                raise FileNotFoundError(f"{label_file} not in {archive.name}")
            member = cands[0]
        out = dest_dir / label_file
        with zf.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    return out


def to_our_grid(mask_path: Path, reference: sitk.Image) -> np.ndarray:
    """Nearest-neighbour resample of a reference mask onto our grid."""
    m = sitk.ReadImage(str(mask_path))
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    rf.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(rf.Execute(m)) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    denom = int(a.sum()) + int(b.sum())
    return 2.0 * inter / denom if denom else float("nan")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else float("nan")


def surface_distances(a: np.ndarray, b: np.ndarray, ref: sitk.Image) -> tuple[float, float]:
    """Mean and 95th-percentile symmetric surface distance in mm."""
    if not a.any() or not b.any():
        return float("nan"), float("nan")

    def contour_and_dist(m: np.ndarray):
        img = sitk.GetImageFromArray(m.astype(np.uint8))
        img.CopyInformation(ref)
        contour = sitk.LabelContour(img, fullyConnected=True)
        dist = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
            )
        )
        # GetArrayFromImage copies. GetArrayViewFromImage would return a view
        # into the image buffer, and these images are freed on return, leaving
        # a dangling pointer that segfaults on first access.
        return sitk.GetArrayFromImage(contour) > 0, sitk.GetArrayFromImage(dist)

    ca, da = contour_and_dist(a)
    cb, db = contour_and_dist(b)
    d_ab = db[ca]
    d_ba = da[cb]
    both = np.concatenate([d_ab, d_ba])
    if both.size == 0:
        return float("nan"), float("nan")
    return float(both.mean()), float(np.percentile(both, 95))


def reference_centerline(
    ref_mask: np.ndarray,
    grid: sitk.Image,
    cfg,
) -> tuple[np.ndarray, float]:
    """Whole-colon centerline of the reference mask.

    Seeds at the two ends of the uniform-speed geodesic diameter of the mask's
    largest component (double sweep: farthest voxel from anywhere, then farthest
    from that), joined by the same wall-weighted Fast Marching path as ours. The
    reference holds colon only, so the diameter ends are the colon's ends.

    Returns the path as an ``(N, 3)`` array of ``(k, j, i)`` and its length in mm.
    """
    from ctc_core.centerline import (
        _largest_component,
        backtrack_centerline,
        compute_arrival_time,
        compute_path_length_mm,
        uniform_arrival,
    )

    largest = _largest_component(ref_mask, connectivity=6)
    if not largest.any():
        raise RuntimeError("reference mask is empty")
    pts = np.argwhere(largest)
    t = uniform_arrival(largest, grid, tuple(int(v) for v in pts[len(pts) // 2]))
    start = tuple(int(v) for v in np.unravel_index(int(np.argmax(t)), t.shape))
    t = uniform_arrival(largest, grid, start)
    end = tuple(int(v) for v in np.unravel_index(int(np.argmax(t)), t.shape))

    lumen = sitk.GetImageFromArray(largest.astype(np.uint8))
    lumen.CopyInformation(grid)
    arrival, lumen_arr = compute_arrival_time(lumen, start, cfg)
    path = backtrack_centerline(arrival, lumen_arr, start, end)
    return np.array(path, dtype=int), compute_path_length_mm(path, grid.GetSpacing())


def path_distance_stats(
    query_kji: np.ndarray,
    target_kji: np.ndarray,
    grid: sitk.Image,
    tolerance_mm: float,
) -> tuple[float, float]:
    """``(fraction of query within tolerance of target, mean distance)`` in mm.

    Implemented as a distance transform of the target path so the cost does not
    grow with the product of the two point counts.
    """
    shape = sitk.GetArrayFromImage(grid).shape
    tmask = np.zeros(shape, dtype=np.uint8)
    tmask[target_kji[:, 0], target_kji[:, 1], target_kji[:, 2]] = 1
    timg = sitk.GetImageFromArray(tmask)
    timg.CopyInformation(grid)
    dist = sitk.GetArrayFromImage(
        sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                timg, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
            )
        )
    )
    d = dist[query_kji[:, 0], query_kji[:, 1], query_kji[:, 2]]
    return float((d <= tolerance_mm).mean()), float(d.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", type=Path, default=Path("results/tables/hqcolon_overlap.csv"))
    ap.add_argument("--hqcolon", type=Path, default=Path("data/hqcolon"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--position", type=Path, default=Path("results/tables/patient_position.csv"))
    ap.add_argument("--fill-holes", choices=("auto", "bbox", "slicewise"), default=None,
                    help="override the air mask's hole fill; must match the centerline run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    overlap = [
        r for r in csv.DictReader(args.overlap.open(encoding="utf-8"))
        if r["in_hqcolon"] == "True"
    ]
    print(f"{len(overlap)} series with an HQColon reference\n")

    rows = []
    for rec in overlap:
        pid, role = rec["PatientID"], rec["role"]
        tag = f"{pid[-4:]}/{role}"
        row = {
            "PatientID": pid,
            "role": role,
            "SeriesInstanceUID": rec["SeriesInstanceUID"],
            "hqcolon_subject_id": rec["hqcolon_subject_id"],
            "hqcolon_position": rec["hqcolon_position"],
            "error": "",
        }
        try:
            ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
            lumen_path = args.centerline_root / pid / role / "lumen_mask.nii.gz"
            pts_path = args.centerline_root / pid / role / "centerline_points.csv"
            for p in (ct_path, lumen_path, pts_path):
                if not p.exists():
                    raise FileNotFoundError(str(p))

            ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
            spacing_xyz = ct.GetSpacing()
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
            vox_ml = float(np.prod(spacing_xyz)) / 1000.0

            # Our masks. The air mask and the lumen the geodesic ran on are
            # evaluated separately: the first is recall-oriented, the second is
            # the actual claim about where the colon is.
            from ctc_core.centerline import CenterlineConfig
            from ctc_core.masks import build_body_air_masks

            cfg = CenterlineConfig.corrected()
            if args.fill_holes is not None:
                cfg = replace(cfg, fill_holes=args.fill_holes)
            _body, our_air, _mdiag = build_body_air_masks(
                ct,
                body_threshold=cfg.body_threshold,
                air_threshold=cfg.air_threshold,
                body_closing_radius=cfg.body_closing_radius,
                air_closing_radius=cfg.air_closing_radius,
                fill_holes=cfg.fill_holes,
                connectivity=cfg.connectivity,
            )
            our_lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path))) > 0

            # Reference masks, extracted on demand.
            refs = {}
            with tempfile.TemporaryDirectory(prefix="hqcolon_") as td:
                for key, (zname, folder) in ARCHIVES.items():
                    mp = extract_one(
                        args.hqcolon / zname, folder, rec["hqcolon_label_file"], Path(td)
                    )
                    refs[key] = to_our_grid(mp, ct)
                    mp.unlink(missing_ok=True)

            ref_gas, ref_gf = refs["gas"], refs["gas_fluid"]
            row["ref_gas_ml"] = round(int(ref_gas.sum()) * vox_ml, 1)
            row["ref_gas_fluid_ml"] = round(int(ref_gf.sum()) * vox_ml, 1)
            row["our_air_ml"] = round(int(our_air.sum()) * vox_ml, 1)
            row["our_lumen_ml"] = round(int(our_lumen.sum()) * vox_ml, 1)

            row["dice_air_vs_gas"] = round(dice(our_air, ref_gas), 4)
            row["dice_air_vs_gas_fluid"] = round(dice(our_air, ref_gf), 4)
            row["dice_lumen_vs_gas"] = round(dice(our_lumen, ref_gas), 4)
            row["jaccard_lumen_vs_gas"] = round(jaccard(our_lumen, ref_gas), 4)
            row["dice_lumen_vs_gas_fluid"] = round(dice(our_lumen, ref_gf), 4)

            inter = int(np.logical_and(our_lumen, ref_gas).sum())
            row["lumen_recall_vs_gas"] = round(inter / max(int(ref_gas.sum()), 1), 4)
            row["lumen_precision_vs_gas"] = round(inter / max(int(our_lumen.sum()), 1), 4)

            msd, hd95 = surface_distances(our_lumen, ref_gas, ct)
            row["msd_lumen_vs_gas_mm"] = round(msd, 2)
            row["hd95_lumen_vs_gas_mm"] = round(hd95, 2)

            # Centerline precision: are the path voxels inside the colon?
            kji = []
            with pts_path.open(encoding="utf-8") as fh:
                for p in csv.DictReader(fh):
                    kji.append((int(p["k"]), int(p["j"]), int(p["i"])))
            kji_arr = np.array(kji, dtype=int)
            row["n_centerline_points"] = len(kji)
            steps = np.diff(kji_arr.astype(float) * np.array(spacing_zyx), axis=0)
            row["centerline_len_mm"] = round(
                float(np.linalg.norm(steps, axis=1).sum()), 2
            )
            k, j, i = kji_arr[:, 0], kji_arr[:, 1], kji_arr[:, 2]
            row["centerline_inside_gas_frac"] = round(float(ref_gas[k, j, i].mean()), 4)
            row["centerline_inside_gas_fluid_frac"] = round(float(ref_gf[k, j, i].mean()), 4)

            # Coverage against the whole-colon reference centerline (gas mask).
            ref_path, ref_len = reference_centerline(ref_gas, ct, cfg)
            row["ref_centerline_len_mm"] = round(ref_len, 1)
            row["length_ratio"] = (
                round(row["centerline_len_mm"] / ref_len, 3) if ref_len else None
            )
            covered, ref_to_our = path_distance_stats(
                ref_path, kji_arr, ct, COVER_TOL_MM
            )
            _, our_to_ref = path_distance_stats(kji_arr, ref_path, ct, COVER_TOL_MM)
            row["ref_covered_frac"] = round(covered, 4)
            row["ref_to_our_mean_mm"] = round(ref_to_our, 2)
            row["our_to_ref_mean_mm"] = round(our_to_ref, 2)

            # The gas-only reference is itself interrupted wherever the colon
            # is collapsed or fluid-filled, so the gas+fluid mask gives the
            # whole-colon reference length and the harder coverage target.
            try:
                ref_gf_path, ref_gf_len = reference_centerline(ref_gf, ct, cfg)
                row["ref_gf_centerline_len_mm"] = round(ref_gf_len, 1)
                row["length_ratio_gf"] = (
                    round(row["centerline_len_mm"] / ref_gf_len, 3) if ref_gf_len else None
                )
                cov_gf, _ = path_distance_stats(ref_gf_path, kji_arr, ct, COVER_TOL_MM)
                row["ref_gf_covered_frac"] = round(cov_gf, 4)
            except Exception as exc:
                row["ref_gf_centerline_len_mm"] = None
                row["ref_gf_covered_frac"] = None
                print(f"    (gas+fluid reference unavailable: {type(exc).__name__}: {exc})",
                      flush=True)

            path_mask = np.zeros_like(ref_gas, dtype=np.uint8)
            path_mask[k, j, i] = 1
            pm = sitk.GetImageFromArray(path_mask)
            pm.CopyInformation(ct)
            dpath = sitk.GetArrayFromImage(
                sitk.Abs(
                    sitk.SignedMaurerDistanceMap(
                        pm, insideIsPositive=False, squaredDistance=False,
                        useImageSpacing=True,
                    )
                )
            )
            near = float((dpath[ref_gas] <= NEAR_MM).mean()) if ref_gas.any() else float("nan")
            row[f"ref_vol_within_{int(NEAR_MM)}mm_frac"] = round(near, 4)
            # Primary coverage: the whole colon, gas and fluid, independent of
            # any reference centerline.
            if ref_gf.any():
                row[f"colon_within_{int(NEAR_MM)}mm_frac"] = round(
                    float((dpath[ref_gf] <= NEAR_MM).mean()), 4)
                row[f"colon_beyond_{int(FAR_MM)}mm_frac"] = round(
                    float((dpath[ref_gf] > FAR_MM).mean()), 4)

            print(
                f"  {tag:16s} Dice(lumen,gas)={row['dice_lumen_vs_gas']:.3f} "
                f"inside={row['centerline_inside_gas_frac']:.3f} "
                f"len/ref={row['length_ratio']} "
                f"covered={covered:.3f}/{row.get('ref_gf_covered_frac')}",
                flush=True,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {tag:16s} FAILED {row['error']}", flush=True)

        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    print(f"\n{len(ok)}/{len(rows)} evaluated -> {args.out}")
    if ok:
        def summ(key: str, fmt: str = "{:.3f}") -> str:
            v = [r[key] for r in ok if r.get(key) is not None and np.isfinite(r[key])]
            if not v:
                return "n/a"
            return (f"median {fmt} (IQR {fmt}-{fmt})").format(
                float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))
            )

        print("\n--- summary over evaluated series ---")
        for key in (
            "dice_lumen_vs_gas", "jaccard_lumen_vs_gas", "dice_air_vs_gas",
            "lumen_recall_vs_gas", "lumen_precision_vs_gas",
            "centerline_inside_gas_frac",
            "length_ratio", "ref_covered_frac",
            "length_ratio_gf", "ref_gf_covered_frac",
            f"ref_vol_within_{int(NEAR_MM)}mm_frac",
            f"colon_within_{int(NEAR_MM)}mm_frac", f"colon_beyond_{int(FAR_MM)}mm_frac",
        ):
            print(f"  {key:34s} {summ(key)}")
        for key in ("msd_lumen_vs_gas_mm", "hd95_lumen_vs_gas_mm",
                    "ref_to_our_mean_mm", "our_to_ref_mean_mm"):
            print(f"  {key:34s} {summ(key, '{:.2f}')} mm")


if __name__ == "__main__":
    main()
