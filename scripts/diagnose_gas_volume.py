"""Private diagnostic: which gas-volume definition matches the reference colon gas?

  component  gas (<= -700 HU) in the lumen components the path passes through
             (the current definition in quality.gas_volume_ml)
  near30     lumen gas within 30 mm of the path

compared with the HQColon gas-filled volume on the 26 reference series.
Printed only.

    python scripts/diagnose_gas_volume.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctc_core.quality import QualityConfig, gas_volume_ml  # noqa: E402


def main() -> None:
    ev = {(r["PatientID"], r["role"]): r for r in csv.DictReader(
        open("results/tables/eval_hqcolon_auto.csv", encoding="utf-8"))}
    out = []
    for (pid, role), r in sorted(ev.items()):
        ct = sitk.ReadImage(f"data/work/{pid}/{role}/ct_lps_iso.nii.gz", sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(ct)
        root = Path(f"data/centerline/corrected/{pid}/{role}")
        lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(root / "lumen_mask.nii.gz"))) > 0
        ijk = np.array([(int(p["i"]), int(p["j"]), int(p["k"])) for p in csv.DictReader(
            open(root / "centerline_points.csv"))])
        sp = ct.GetSpacing()
        comp, gas = gas_volume_ml(arr, lumen, (sp[2], sp[1], sp[0]), ijk, QualityConfig())
        m = np.zeros(arr.shape, np.uint8)
        m[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = 1
        img = sitk.GetImageFromArray(m)
        img.CopyInformation(ct)
        d = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
            img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
        vox = float(np.prod(sp)) / 1000
        near = float((lumen & (arr <= -700) & (d <= 30)).sum()) * vox
        ref = float(r["ref_gas_ml"])
        out.append((ref, comp, near))
        print(f"{pid[-4:]}-{1 if role == 'primary' else 2}: reference {ref:6.0f}  component "
              f"{comp:6.0f}  near30 {near:6.0f}", flush=True)
    a = np.array(out)
    for j, name in ((1, "component"), (2, "near30")):
        rel = (a[:, j] - a[:, 0]) / a[:, 0]
        q1, q2, q3 = np.percentile(rel, [25, 50, 75])
        print(f"{name:10s} r = {np.corrcoef(a[:, 0], a[:, j])[0, 1]:.3f}; relative error median "
              f"{q2:+.3f} [{q1:+.3f}, {q3:+.3f}]")


if __name__ == "__main__":
    main()
