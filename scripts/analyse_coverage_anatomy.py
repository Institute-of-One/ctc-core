"""Where along the colon the coverage fails, and how much of it was excluded.

Two questions a proportion cannot answer, resolved in one pass over the
reference masks because extracting them is the expensive part: where along the
colon the coverage fails, and how much of the colon the gas mask excludes by
construction.

**Where the coverage fails.** The share of reference volume within 30 mm of
the path and beyond 60 mm is wanted per anatomical segment, with a segment-wise
Dice beside it, because one proportion over the whole colon does not say which
cases the pipeline is usable on. HQColon
ships two masks, gas and gas-and-fluid, and **no anatomical labels**, so the
six named colon segments cannot be read off the reference. Two divisions
are reported instead, and the Methods say which is which:

six equal parts of the reference centerline's arc length from the rectal end.

Flexure detection was tried and rejected. Taking each flexure as the superior
extreme of the reference centerline on its own side of the patient put the
hepatic flexure of 0001/primary on the patient's left, in the middle of the
transverse colon, because the transverse colon is as high as the flexure and
because the median x of the path -- used as the midline -- sits 26 mm from the
body's. The failure is obvious in a drawing of the path and invisible in a
table of per-segment numbers, which is why the paths are drawn. Nothing
anatomically named is reported here, and inventing boundaries for "sigmoid" and
"descending" from arc length would have produced six anatomical names attached
to geometry.

**What the gas mask excludes.** No electronic cleansing is applied, so fluid
and tagged segments are outside the gas mask by construction and the bridges
and minimal paths are built on gas alone. How much of the colon that is, and
how much of the coverage shortfall it explains, decides whether the choice is
a bounded limitation or the cause. The gas-and-fluid mask minus the gas mask
is exactly that volume, so the share of the colon the path never visits is
split into the part that was gas -- reachable in principle, and a genuine miss
-- and the part that was not.

Untagged fluid and collapsed lumen are not separable by attenuation: both sit
near soft tissue. Tagged fluid is, so its share is reported as a subset. What
cannot be measured without a reader -- residual stool against tagged fluid,
the rectal catheter's position, prior surgery or herniation -- is not estimated
here; this study has no reader by design, and the manuscript says so.

    python scripts/analyse_coverage_anatomy.py
    python scripts/analyse_coverage_anatomy.py --limit 2 --figures
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_hqcolon as E  # noqa: E402  the extraction and reference-path helpers

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

TABLES = ROOT / "results" / "tables"

NEAR_MM = E.NEAR_MM          # 30 mm: served by the path
FAR_MM = E.FAR_MM            # 60 mm: never visited
TAGGED_HU = 100.0            # oral contrast; untagged fluid and wall sit below
RECTUM_MM = 150.0            # only used to report where the rectum ends

N_PARTS = 6


ORIENTATION_MARGIN_MM = 20.0


def orient_from_rectum(path_kji: np.ndarray, grid: sitk.Image) -> tuple[np.ndarray, dict]:
    """Return the path running from the rectal end, and why that end was chosen.

    The two ends of the colon are the rectum and the caecum. The rectal end is
    the more inferior of the two: the rectum lies on the pelvic floor and the
    caecum in the iliac fossa above it.

    A posterior rule was tried first -- the rectum is the posterior midline
    structure -- and it is the wrong one. On 0003/primary it chose the caecal
    end on a margin of 33 mm and reversed the whole path; drawn on the image
    the error is immediate, and in a table of per-segment numbers it is
    invisible. The inferior margin on the same series is 171 mm.

    A series whose two ends are within ORIENTATION_MARGIN_MM in z is recorded
    as ambiguous rather than oriented on a coin toss.
    """
    ends = np.array([
        grid.TransformContinuousIndexToPhysicalPoint(
            (float(p[2]), float(p[1]), float(p[0])))
        for p in (path_kji[0], path_kji[-1])])
    margin = float(ends[0][2] - ends[1][2])
    first_is_rectum = margin < 0
    why = {
        "start_xyz_mm": [round(float(v), 1) for v in ends[0]],
        "end_xyz_mm": [round(float(v), 1) for v in ends[1]],
        "rule": "the more inferior end (smaller z in LPS) is the rectum",
        "reversed": not first_is_rectum,
        "inferior_margin_mm": round(abs(margin), 1),
        "ambiguous": bool(abs(margin) < ORIENTATION_MARGIN_MM),
    }
    return (path_kji if first_is_rectum else path_kji[::-1]), why


def arc_length_mm(points_xyz: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def analyse_series(ct: sitk.Image, our_lumen: np.ndarray, our_path_kji: np.ndarray,
                   ref_gas: np.ndarray, ref_gf: np.ndarray, cfg) -> dict:
    """Everything comments 2 and 3 ask for, for one series."""
    spacing = ct.GetSpacing()
    vox_ml = float(np.prod(spacing)) / 1000.0

    ref_path, orientation = orient_from_rectum(
        E.reference_centerline(ref_gf, ct, cfg)[0], ct)
    ref_xyz = np.array([
        ct.TransformContinuousIndexToPhysicalPoint(
            (float(p[2]), float(p[1]), float(p[0]))) for p in ref_path])
    arc = arc_length_mm(ref_xyz)

    # Distance from our path to every voxel, once.
    shape = sitk.GetArrayFromImage(ct).shape
    pmask = np.zeros(shape, dtype=np.uint8)
    pmask[our_path_kji[:, 0], our_path_kji[:, 1], our_path_kji[:, 2]] = 1
    pimg = sitk.GetImageFromArray(pmask)
    pimg.CopyInformation(ct)
    dist = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(
        pimg, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)))

    # Every colon voxel is assigned to the nearest point of the reference
    # centerline, so a segment of the path owns a volume of colon.
    gf_idx = np.argwhere(ref_gf)
    gf_xyz = np.array([
        ct.TransformContinuousIndexToPhysicalPoint(
            (float(i), float(j), float(k)))
        for k, j, i in gf_idx]) if len(gf_idx) else np.zeros((0, 3))
    nearest = cKDTree(ref_xyz).query(gf_xyz, k=1)[1] if len(gf_idx) else np.zeros(0, int)

    hu = sitk.GetArrayFromImage(ct)
    # The comparison set for Dice: every voxel either mask claims. Each is
    # assigned to the nearest point of the reference centerline, so a part of
    # the path owns a region of space and the false positives near it count
    # against it.
    cmp_ours, cmp_ref = our_lumen, ref_gas
    cmp_idx = np.argwhere(cmp_ours | cmp_ref)
    if len(cmp_idx):
        cmp_xyz = np.array([
            ct.TransformContinuousIndexToPhysicalPoint(
                (float(i), float(j), float(k))) for k, j, i in cmp_idx])
        cmp_part = arc[cKDTree(ref_xyz).query(cmp_xyz, k=1)[1]] / arc[-1]
    else:
        cmp_part = np.zeros(0)
    gf_gas = ref_gas[tuple(gf_idx.T)]
    gf_dist = dist[tuple(gf_idx.T)]
    gf_hu = hu[tuple(gf_idx.T)]
    gf_arcfrac = arc[nearest] / arc[-1]

    def band(sel: np.ndarray, label: str) -> dict:
        n = int(sel.sum())
        if n == 0:
            return {"segment": label, "colon_ml": 0.0, "n_voxels": 0}
        gas = gf_gas[sel]
        d = gf_dist[sel]
        beyond = d > FAR_MM
        excluded = ~gas                      # fluid-filled or collapsed
        out = {
            "segment": label,
            "n_voxels": n,
            "colon_ml": round(n * vox_ml, 1),
            "gas_ml": round(int(gas.sum()) * vox_ml, 1),
            "fluid_or_collapsed_ml": round(int(excluded.sum()) * vox_ml, 1),
            "fluid_or_collapsed_frac": round(float(excluded.mean()), 4),
            "within_30mm_frac": round(float((d <= NEAR_MM).mean()), 4),
            "beyond_60mm_frac": round(float(beyond.mean()), 4),
        }
        # Comment 3: the colon the path never visited, split by whether it was
        # gas at all. The second part could not have been reached by a method
        # that runs on gas, whatever the path did.
        if beyond.any():
            out["beyond_60mm_that_was_gas_frac"] = round(
                float(gas[beyond].mean()), 4)
            out["beyond_60mm_that_was_excluded_frac"] = round(
                float((~gas[beyond]).mean()), 4)
        else:
            out["beyond_60mm_that_was_gas_frac"] = ""
            out["beyond_60mm_that_was_excluded_frac"] = ""
        # Tagging, as a subset of what the gas mask excludes. Untagged fluid
        # and collapsed lumen both sit near soft tissue and are not separable
        # by attenuation, so they are not separated.
        if excluded.any():
            ex_hu = gf_hu[sel][excluded]
            out["excluded_hu_median"] = round(float(np.median(ex_hu)), 1)
            out["excluded_tagged_frac"] = round(
                float((ex_hu > TAGGED_HU).mean()), 4)
        else:
            out["excluded_hu_median"] = ""
            out["excluded_tagged_frac"] = ""
        return out

    def part_dice(part_of_voxel: np.ndarray, sel_parts: np.ndarray) -> str:
        """Dice of our lumen against the gas reference within one part.

        The part owns a region of space, not a subset of the reference colon.
        Restricting the comparison to reference-colon voxels would drop every
        false positive our lumen has outside the colon, which is most of what
        separates the whole-colon Dice of 0.919 from 1.0, and it returned
        0.995 to 0.997 in all six parts when it was written that way.
        """
        a = cmp_ours & sel_parts
        b = cmp_ref & sel_parts
        denom = int(a.sum()) + int(b.sum())
        return round(2.0 * int((a & b).sum()) / denom, 4) if denom else ""

    def region(a0: float, a1: float) -> np.ndarray:
        """The voxels of the comparison set belonging to an arc-length band."""
        inside = np.zeros(shape, dtype=bool)
        if len(cmp_idx):
            keep = (cmp_part >= a0) & (cmp_part < a1)
            inside[tuple(cmp_idx[keep].T)] = True
        return inside

    whole = band(np.ones(len(gf_idx), bool), "whole colon")
    whole["dice_lumen_vs_gas_in_part"] = part_dice(
        cmp_part, np.ones(shape, dtype=bool))
    rows = [whole]
    for p in range(N_PARTS):
        a0, a1 = p / N_PARTS, (p + 1) / N_PARTS + (1e-9 if p == N_PARTS - 1 else 0)
        r = band((gf_arcfrac >= a0) & (gf_arcfrac < a1),
                 "part %d of %d from the rectum" % (p + 1, N_PARTS))
        r["dice_lumen_vs_gas_in_part"] = part_dice(cmp_part, region(a0, a1))
        rows.append(r)
    return {"rows": rows, "orientation": orientation,
            "reference_length_mm": round(float(arc[-1]), 1),
            "rectum_ends_at_arc_frac": round(float(min(RECTUM_MM / arc[-1], 1.0)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", type=Path, default=TABLES / "hqcolon_overlap.csv")
    ap.add_argument("--hqcolon", type=Path, default=ROOT / "data" / "hqcolon")
    ap.add_argument("--centerline-root", type=Path,
                    default=ROOT / "data" / "centerline" / "corrected")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data" / "work")
    ap.add_argument("--out", type=Path, default=TABLES / "coverage_anatomy.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many series (for a trial run)")
    args = ap.parse_args()

    from ctc_core.centerline import CenterlineConfig
    cfg = CenterlineConfig.corrected()

    with args.overlap.open(newline="", encoding="utf-8") as fh:
        overlap = [r for r in csv.DictReader(fh)
                   if (r.get("hqcolon_label_file") or "").strip()]
    print("%d series with an HQColon reference" % len(overlap))

    out_rows: list[dict] = []
    detail: dict[str, dict] = {}
    done = 0
    for rec in overlap:
        pid, role = rec["PatientID"], rec["role"]
        tag = "%s/%s" % (pid[-4:], role)
        if args.limit is not None and done >= args.limit:
            break
        try:
            ct_path = args.work_dir / pid / role / "ct_lps_iso.nii.gz"
            lumen_path = args.centerline_root / pid / role / "lumen_mask.nii.gz"
            pts_path = args.centerline_root / pid / role / "centerline_points.csv"
            for p in (ct_path, lumen_path, pts_path):
                if not p.exists():
                    raise FileNotFoundError(str(p))
            ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
            our_lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path))) > 0
            with pts_path.open(newline="", encoding="utf-8") as fh:
                pts = np.array([[int(float(r["k"])), int(float(r["j"])),
                                 int(float(r["i"]))] for r in csv.DictReader(fh)])
            refs = {}
            with tempfile.TemporaryDirectory(prefix="hqcolon_") as td:
                for key, (zname, folder) in E.ARCHIVES.items():
                    mp = E.extract_one(args.hqcolon / zname, folder,
                                       rec["hqcolon_label_file"], Path(td))
                    refs[key] = E.to_our_grid(mp, ct)
                    mp.unlink(missing_ok=True)
            res = analyse_series(ct, our_lumen, pts, refs["gas"],
                                 refs["gas_fluid"], cfg)
            for r in res["rows"]:
                out_rows.append({"PatientID": pid, "role": role, **r})
            detail[tag] = {k: v for k, v in res.items() if k != "rows"}
            whole = res["rows"][0]
            print("  %-16s colon %7.1f mL  fluid/collapsed %5.1f%%  "
                  "within30 %5.1f%%  beyond60 %5.1f%%  of which excluded %s"
                  % (tag, whole["colon_ml"],
                     100 * whole["fluid_or_collapsed_frac"],
                     100 * whole["within_30mm_frac"],
                     100 * whole["beyond_60mm_frac"],
                     ("%5.1f%%" % (100 * whole["beyond_60mm_that_was_excluded_frac"]))
                     if whole["beyond_60mm_that_was_excluded_frac"] != "" else "   n/a"),
                  flush=True)
        except Exception as exc:                      # noqa: BLE001
            print("  %-16s FAILED %s: %s" % (tag, type(exc).__name__, exc),
                  flush=True)
            out_rows.append({"PatientID": pid, "role": role, "segment": "whole colon",
                             "error": "%s: %s" % (type(exc).__name__, exc)})
        done += 1

    fields: list[str] = []
    for r in out_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(out_rows)
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({
        "near_mm": NEAR_MM, "far_mm": FAR_MM, "tagged_hu": TAGGED_HU,
        "n_parts": N_PARTS,
        "anatomical_labels_in_reference": False,
        "flexures_detected": False,
        "note": "HQColon provides a gas mask and a gas-and-fluid mask and no "
                "anatomical labels. Flexure detection from the reference "
                "centerline was tried and rejected on inspection of the drawn "
                "paths, so no segment is named anatomically; the parts are "
                "equal arc length from the rectal end",
        "per_series": detail,
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s (%d rows) and %s" % (args.out, len(out_rows), meta.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
