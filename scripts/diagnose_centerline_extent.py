"""Private diagnostic: does the seed rule truncate both centerlines?

For a few reference series: distance of every reference-colon voxel (gas and
fluid) to (a) the reference centerline and (b) our centerline. Voxels more than
60 mm from a path cannot be explained by lumen width: they are segments the path
never visits. Printed only. (Section 8 of docs/EVALUATION_HQCOLON.md; run before
the whole-colon seed rule, it showed both paths missing about half the colon.)

    python scripts/diagnose_centerline_extent.py 0007:primary 0001:primary
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_hqcolon as ev  # noqa: E402

from ctc_core.centerline import CenterlineConfig  # noqa: E402

DEFAULT_CASES = ["0007:primary", "0003:secondary", "0001:primary", "0004:secondary",
                 "0030:primary"]


def dist_to_path(path_kji: np.ndarray, grid: sitk.Image) -> np.ndarray:
    shape = sitk.GetArrayFromImage(grid).shape
    m = np.zeros(shape, np.uint8)
    m[path_kji[:, 0], path_kji[:, 1], path_kji[:, 2]] = 1
    img = sitk.GetImageFromArray(m)
    img.CopyInformation(grid)
    return sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))


def main() -> None:
    cfg = CenterlineConfig.corrected()
    overlap = {(r["PatientID"][-4:], r["role"]): r
               for r in csv.DictReader(open("results/tables/hqcolon_overlap.csv", encoding="utf-8"))
               if r["in_hqcolon"] == "True"}
    for c in sys.argv[1:] or DEFAULT_CASES:
        pid4, role = c.split(":")
        rec = overlap[(pid4, role)]
        pid = rec["PatientID"]
        ct = sitk.ReadImage(f"data/work/{pid}/{role}/ct_lps_iso.nii.gz", sitk.sitkFloat32)
        with tempfile.TemporaryDirectory() as td:
            zname, folder = ev.ARCHIVES["gas_fluid"]
            mp = ev.extract_one(Path("data/hqcolon") / zname, folder, rec["hqcolon_label_file"],
                                Path(td))
            ref = ev.to_our_grid(mp, ct)
        rpath, rlen = ev.reference_centerline(ref, ct, cfg)
        ours = np.array([(int(r["k"]), int(r["j"]), int(r["i"])) for r in csv.DictReader(
            open(f"data/centerline/corrected/{pid}/{role}/centerline_points.csv"))])
        vox_ml = float(np.prod(ct.GetSpacing())) / 1000
        print(f"{pid4}-{1 if role == 'primary' else 2}: colon (gas+fluid) "
              f"{ref.sum() * vox_ml:.0f} mL; ref path {rlen:.0f} mm, our path {len(ours)} pts")
        for name, path in (("ref path", rpath), ("our path", ours)):
            d = dist_to_path(path, ct)[ref]
            mid = np.mean((d > 30) & (d <= 60))
            print(f"   {name}: within 30 mm {np.mean(d <= 30):.2f}; 30-60 mm {mid:.2f}; "
                  f"> 60 mm {np.mean(d > 60):.2f} (max {d.max():.0f} mm)")


if __name__ == "__main__":
    main()
