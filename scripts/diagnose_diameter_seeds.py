"""Private feasibility test: uniform-speed geodesic diameter seeds.

On the same lumen the pipeline already used, and on the reference mask, choose
seeds as the two ends of the longest geodesic (uniform speed: arrival time =
path length inside the mask), then run the usual wall-distance-weighted path
between them. Report how much reference colon lies > 60 mm from each path.
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import eval_hqcolon as ev  # noqa: E402

from ctc_core.centerline import (  # noqa: E402
    CenterlineConfig,
    backtrack_centerline,
    compute_arrival_time,
    compute_path_length_mm,
)

cfg = CenterlineConfig.corrected()


def uniform_arrival(mask: np.ndarray, grid: sitk.Image, seed_kji):
    speed = mask.astype(np.float32)  # 1 inside, 0 outside: arrival = geodesic length
    img = sitk.GetImageFromArray(speed)
    img.CopyInformation(grid)
    fm = sitk.FastMarchingImageFilter()
    fm.SetTrialPoints([(int(seed_kji[2]), int(seed_kji[1]), int(seed_kji[0]))])
    fm.SetStoppingValue(1e8)
    t = sitk.GetArrayFromImage(fm.Execute(img)).astype(np.float64)
    t[~mask] = -1
    t[t > 1e7] = -1
    return t


def geodesic_diameter(mask, grid):
    pts = np.argwhere(mask)
    p0 = tuple(pts[len(pts) // 2])
    t = uniform_arrival(mask, grid, p0)
    a = np.unravel_index(np.argmax(t), t.shape)
    t = uniform_arrival(mask, grid, a)
    b = np.unravel_index(np.argmax(t), t.shape)
    return tuple(int(v) for v in a), tuple(int(v) for v in b), float(t[b])


def path_between(mask, grid, a, b):
    lumen = sitk.GetImageFromArray(mask.astype(np.uint8))
    lumen.CopyInformation(grid)
    arr, lum = compute_arrival_time(lumen, a, cfg)
    p = np.array(backtrack_centerline(arr, lum, a, b), dtype=int)
    return p, compute_path_length_mm(p, grid.GetSpacing())


def far_frac(path, grid, ref):
    shape = ref.shape
    m = np.zeros(shape, np.uint8)
    m[path[:, 0], path[:, 1], path[:, 2]] = 1
    img = sitk.GetImageFromArray(m)
    img.CopyInformation(grid)
    d = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))[ref]
    return float(np.mean(d <= 30)), float(np.mean(d > 60))


overlap = {(r["PatientID"][-4:], r["role"]): r
           for r in csv.DictReader(open("results/tables/hqcolon_overlap.csv", encoding="utf-8"))
           if r["in_hqcolon"] == "True"}
cases = sys.argv[1:] or [
    "0007:primary", "0003:secondary", "0001:primary", "0004:secondary", "0030:primary",
]
for c in cases:
    pid4, role = c.split(":")
    rec = overlap[(pid4, role)]
    pid = rec["PatientID"]
    ct = sitk.ReadImage(f"data/work/{pid}/{role}/ct_lps_iso.nii.gz", sitk.sitkFloat32)
    with tempfile.TemporaryDirectory() as td:
        zname, folder = ev.ARCHIVES["gas_fluid"]
        ref = ev.to_our_grid(ev.extract_one(Path("data/hqcolon") / zname, folder,
                                            rec["hqcolon_label_file"], Path(td)), ct)
        zname, folder = ev.ARCHIVES["gas"]
        refgas = ev.to_our_grid(ev.extract_one(Path("data/hqcolon") / zname, folder,
                                               rec["hqcolon_label_file"], Path(td)), ct)
    lumen = sitk.GetArrayFromImage(sitk.ReadImage(
        f"data/centerline/corrected/{pid}/{role}/lumen_mask.nii.gz")) > 0
    out = [f"{pid4}-{1 if role == 'primary' else 2}"]
    # reference: old rule vs geodesic diameter (on the gas-only mask, largest component)
    import cc3d
    lab = cc3d.connected_components(refgas.astype(np.uint8), connectivity=6)
    big = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    rp_old, rl_old = ev.reference_centerline(refgas, ct, cfg)
    a, b, g = geodesic_diameter(big, ct)
    rp_new, rl_new = path_between(big, ct, a, b)
    # ours: same lumen, new seeds within its largest component
    lab2 = cc3d.connected_components(lumen.astype(np.uint8), connectivity=6)
    big2 = lab2 == np.argmax(np.bincount(lab2.ravel())[1:]) + 1
    a2, b2, g2 = geodesic_diameter(big2, ct)
    op_new, ol_new = path_between(big2, ct, a2, b2)
    ours_old = np.array([(int(r["k"]), int(r["j"]), int(r["i"])) for r in csv.DictReader(
        open(f"data/centerline/corrected/{pid}/{role}/centerline_points.csv"))])
    for name, p, L in (("ref old", rp_old, rl_old), ("ref diam", rp_new, rl_new),
                       ("our old", ours_old, compute_path_length_mm(ours_old, ct.GetSpacing())),
                       ("our diam", op_new, ol_new)):
        w, f = far_frac(p, ct, ref)
        out.append(f"{name}: {L:.0f} mm, <=30 {w:.2f}, >60 {f:.2f}")
    print(" | ".join(out), flush=True)
