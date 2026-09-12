"""Private diagnostic: which gas components hold the colon the path misses?

For each series: the HQColon colon mask (gas and fluid), every retained gas
component of our air mask, the components our lumen used, and the path. Writes
coronal projections to ``data/diag/`` (gitignored: the images show HQColon
masks, which must not be published) and prints, per component, its volume and
how much of it overlaps the reference colon.

    python scripts/diagnose_colon_components.py 0007:primary 0001:primary
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
from ctc_core.masks import build_body_air_masks, label_air_components  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = CenterlineConfig.corrected()
    overlap = {(r["PatientID"][-4:], r["role"]): r
               for r in csv.DictReader(open("results/tables/hqcolon_overlap.csv", encoding="utf-8"))
               if r["in_hqcolon"] == "True"}
    out = Path("data/diag")
    out.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[1:]:
        pid4, role = c.split(":")
        rec = overlap[(pid4, role)]
        pid = rec["PatientID"]
        ct = sitk.ReadImage(f"data/work/{pid}/{role}/ct_lps_iso.nii.gz", sitk.sitkFloat32)
        vox = float(np.prod(ct.GetSpacing())) / 1000
        with tempfile.TemporaryDirectory() as td:
            zname, folder = ev.ARCHIVES["gas_fluid"]
            ref = ev.to_our_grid(ev.extract_one(Path("data/hqcolon") / zname, folder,
                                                rec["hqcolon_label_file"], Path(td)), ct)
        _b, air, diag = build_body_air_masks(
            ct, body_threshold=cfg.body_threshold, air_threshold=cfg.air_threshold,
            body_closing_radius=cfg.body_closing_radius,
            air_closing_radius=cfg.air_closing_radius, fill_holes=cfg.fill_holes,
            connectivity=cfg.connectivity)
        labels, sizes = label_air_components(air, connectivity=cfg.connectivity,
                                             dust_threshold_voxels=cfg.dust_threshold_voxels)
        lumen = sitk.GetArrayFromImage(sitk.ReadImage(
            f"data/centerline/corrected/{pid}/{role}/lumen_mask.nii.gz")) > 0
        path = np.array([(int(r["k"]), int(r["j"]), int(r["i"])) for r in csv.DictReader(
            open(f"data/centerline/corrected/{pid}/{role}/centerline_points.csv"))])
        print(f"{pid4}-{1 if role == 'primary' else 2}: reference colon {ref.sum() * vox:.0f} mL, "
              f"fill {diag.get('fill_holes_used')}")
        ids = [int(i) for i in np.unique(labels) if i > 0]
        rows = []
        for i in ids:
            m = labels == i
            n = int(m.sum())
            if n * vox < 5:
                continue
            in_ref = float((m & ref).sum()) / n
            used = float((m & lumen).sum()) / n
            zc = np.argwhere(m)[:, 0].mean()
            rows.append((n * vox, in_ref, used, zc, i))
        rows.sort(reverse=True)
        for v, in_ref, used, zc, i in rows[:14]:
            print(f"   cc {i:4d}: {v:7.0f} mL  in colon {in_ref:4.2f}  in our lumen {used:4.2f}  "
                  f"z-centre {zc:5.0f}")
        missed = ref & ~lumen
        print(f"   reference colon not in our lumen: {missed.sum() * vox:.0f} mL; of that, in our "
              f"gas mask {float((missed & (labels > 0)).sum()) * vox:.0f} mL, not gas "
              f"(fluid/collapsed/other) {float((missed & ~(labels > 0)).sum()) * vox:.0f} mL")

        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        ax.imshow(ref.sum(axis=1) > 0, cmap="Greys", alpha=0.35, origin="lower")
        colonic = np.isin(labels, [r[4] for r in rows if r[1] > 0.5])
        ax.contour((colonic.sum(axis=1) > 0).astype(float), levels=[0.5], colors="green",
                   linewidths=0.6, origin="lower")
        ax.contour((lumen.sum(axis=1) > 0).astype(float), levels=[0.5], colors="blue",
                   linewidths=0.8, origin="lower")
        ax.plot(path[:, 2], path[:, 0], color="red", linewidth=0.8)
        ax.set_title(f"{pid4}-{role}: grey HQColon, green colonic gas cc, blue our lumen, red path",
                     fontsize=7)
        fig.savefig(out / f"{pid4}_{role}_components.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
