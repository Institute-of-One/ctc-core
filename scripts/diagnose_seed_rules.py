"""Compare seed rules for a whole-colon centerline, on the lumen already produced.

Each rule places a start and an end seed; the path between them is the usual
wall-weighted Fast Marching minimal path in the same lumen.

  current   most distant skeleton end points by Euclidean distance (pipeline)
  rectal    lowest lumen voxel (the rectal end, where the colon is insufflated)
            and the voxel geodesically farthest from it (uniform speed)
  core3     the same rectal start; the end is the voxel geodesically farthest
            from it within the largest component of the lumen's wide core
            (wall distance >= 3 mm), so thin contacts between touching loops
            and narrow small-bowel links cannot redirect the search
  select    whichever of the three covers most of the lumen within 30 mm --
            a rule that needs no reference

Scores against HQColon (gas and fluid): share of the reference colon within
30 mm of the path, share beyond 60 mm (colon never visited), share of path
points inside the reference colon. HQColon is used only for scoring.

    python scripts/diagnose_seed_rules.py --all --out data/diag/seed_rules.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_hqcolon as ev  # noqa: E402

from ctc_core.centerline import (  # noqa: E402
    CenterlineConfig,
    backtrack_centerline,
    compute_arrival_time,
    compute_path_length_mm,
)

CORE_MM = 3.0
NEAR_MM = 30.0
FAR_MM = 60.0


def uniform_arrival(mask, grid, seed):
    img = sitk.GetImageFromArray(mask.astype(np.float32))
    img.CopyInformation(grid)
    fm = sitk.FastMarchingImageFilter()
    fm.SetTrialPoints([(int(seed[2]), int(seed[1]), int(seed[0]))])
    fm.SetStoppingValue(1e8)
    t = sitk.GetArrayFromImage(fm.Execute(img)).astype(np.float64)
    t[~mask] = -1
    t[t > 1e7] = -1
    return t


def farthest(mask, grid, seed):
    t = uniform_arrival(mask, grid, seed)
    idx = np.unravel_index(np.argmax(t), t.shape)
    return tuple(int(v) for v in idx)


def path_between(mask, grid, a, b, cfg):
    lumen = sitk.GetImageFromArray(mask.astype(np.uint8))
    lumen.CopyInformation(grid)
    arr, lum = compute_arrival_time(lumen, a, cfg)
    return np.array(backtrack_centerline(arr, lum, a, b), dtype=int)


def distance_to(path, grid, shape):
    m = np.zeros(shape, np.uint8)
    m[path[:, 0], path[:, 1], path[:, 2]] = 1
    img = sitk.GetImageFromArray(m)
    img.CopyInformation(grid)
    return sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))


def lowest_voxel(mask):
    pts = np.argwhere(mask)
    cand = pts[pts[:, 0] == pts[:, 0].min()]
    return tuple(int(v) for v in cand[len(cand) // 2])


def candidates(lum, grid, cur, cfg):
    import cc3d

    out = {"current": cur}
    r0 = lowest_voxel(lum)
    out["rectal"] = path_between(lum, grid, r0, farthest(lum, grid, r0), cfg)
    img = sitk.GetImageFromArray(lum.astype(np.uint8))
    img.CopyInformation(grid)
    dwall = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        img, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))
    core = lum & (dwall >= CORE_MM)
    labc = cc3d.connected_components(core.astype(np.uint8), connectivity=6)
    if labc.max() > 0:
        comp = labc == np.argmax(np.bincount(labc.ravel())[1:]) + 1
        pts = np.argwhere(comp)
        near = tuple(int(v) for v in pts[np.argmin(((pts - np.array(r0)) ** 2).sum(1))])
        out["core3"] = path_between(lum, grid, r0, farthest(comp, grid, near), cfg)
    return out


def main() -> None:
    import cc3d

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="<pid4>:<role>")
    ap.add_argument("--all", action="store_true", help="every series with a reference")
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = CenterlineConfig.corrected()
    overlap = {(r["PatientID"][-4:], r["role"]): r
               for r in csv.DictReader(open("results/tables/hqcolon_overlap.csv", encoding="utf-8"))
               if r["in_hqcolon"] == "True"}
    cases = sorted(overlap) if args.all else [tuple(c.split(":")) for c in args.cases]
    rows = []
    for pid4, role in cases:
        rec = overlap[(pid4, role)]
        pid = rec["PatientID"]
        ct = sitk.ReadImage(f"data/work/{pid}/{role}/ct_lps_iso.nii.gz", sitk.sitkFloat32)
        with tempfile.TemporaryDirectory() as td:
            zname, folder = ev.ARCHIVES["gas_fluid"]
            ref = ev.to_our_grid(ev.extract_one(Path("data/hqcolon") / zname, folder,
                                                rec["hqcolon_label_file"], Path(td)), ct)
        croot = args.centerline_root / pid / role
        lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(croot / "lumen_mask.nii.gz"))) > 0
        lab = cc3d.connected_components(lumen.astype(np.uint8), connectivity=6)
        lum = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
        cur = np.array([(int(r["k"]), int(r["j"]), int(r["i"])) for r in csv.DictReader(
            open(croot / "centerline_points.csv"))])
        cands = candidates(lum, ct, cur, cfg)
        scored = {}
        for name, p in cands.items():
            d = distance_to(p, ct, ref.shape)
            scored[name] = {
                "length_mm": compute_path_length_mm(p, ct.GetSpacing()),
                "self_near": float(np.mean(d[lum] <= NEAR_MM)),
                "ref_near": float(np.mean(d[ref] <= NEAR_MM)),
                "ref_far": float(np.mean(d[ref] > FAR_MM)),
                "inside": float(ref[p[:, 0], p[:, 1], p[:, 2]].mean()),
            }
        best = max(scored, key=lambda k: scored[k]["self_near"])
        scored["select"] = dict(scored[best], chosen=best)
        row = {"series": f"{pid4}-{1 if role == 'primary' else 2}"}
        for name, sc in scored.items():
            for k, v in sc.items():
                row[f"{name}_{k}"] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
        print(row["series"], " | ".join(
            f"{n}: {s['length_mm']:.0f} mm far {s['ref_far']:.2f} inside {s['inside']:.2f}"
            for n, s in scored.items()), f"-> {best}", flush=True)
    if args.out and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "series", k))
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
