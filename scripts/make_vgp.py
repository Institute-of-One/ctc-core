"""Render the VGP unfold of one series at the prototype's Ultra-HQ quality.

For the figure only. As in the prototype, the single case is first resampled to
0.5 mm isotropic from its DICOM series, then rendered with the Ultra-HQ preset
(720 angles, 0.25 mm ray sampling, opacity ramp 100 HU, tagged residue clamped
at 300 HU) along the whole-colon centerline smoothed with sigma 2 mm and
resampled every 1 mm. Resampling every series this finely is unnecessary: the
measurements run on the 1.0 mm grid.

The centerline is smoothed here, with the function the ring fat map uses, and
the renderer's own smoothing is switched off. The unfold then shares its arc
length with the fat map (Figure 2c-e); smoothing inside the renderer would move
the points but keep the arc length of the voxel path, about 7 % longer.

The case is the one Figure 2 shows (``make_case_figures.choose_pipeline_case``).
No electronic cleansing is applied, so tagged fluid appears as surface.
Needs VTK (``pip install -e .[vgp]``) and an OpenGL-capable GPU.

    python scripts/make_vgp.py --raw-root .
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctc_core.fatmap import FatConfig, resample_centerline, smooth_centerline  # noqa: E402
from ctc_core.io import get_or_make_resampled_ct  # noqa: E402
from ctc_core.vgp import (  # noqa: E402
    ULTRA_HQ,
    ULTRA_HQ_CT_SPACING_MM,
    ULTRA_HQ_STEP_MM,
    compute_vgp_unfold_gpu_cube,
)


def main() -> None:
    from make_case_figures import choose_pipeline_case

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=Path("."))
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--out", type=Path, default=Path("data/vgp"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pid, role = choose_pipeline_case(args.tables)
    rec = next(r for r in csv.DictReader(args.manifest.open(encoding="utf-8"))
               if r["PatientID"] == pid and r["role"] == role)
    t0 = time.perf_counter()
    ct = get_or_make_resampled_ct(
        dicom_dir=args.raw_root / rec["download_dir"],
        out_ct_path=args.work_dir / pid / role / f"ct_lps_iso_{ULTRA_HQ_CT_SPACING_MM}mm.nii.gz",
        iso_spacing=ULTRA_HQ_CT_SPACING_MM,
    )
    t_resample = time.perf_counter() - t0

    pts = np.array([(float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])) for r in
                    csv.DictReader((args.centerline_root / pid / role /
                                    "centerline_points.csv").open(encoding="utf-8"))])
    s_mm, xyz = resample_centerline(
        smooth_centerline(pts, FatConfig().frame_smooth_sigma_mm), ULTRA_HQ_STEP_MM)

    out_dir = args.out / pid / role
    t1 = time.perf_counter()
    rgb, diag = compute_vgp_unfold_gpu_cube(ct, xyz, s_mm, out_dir=out_dir,
                                            centerline_smooth_sigma_mm=0.0, **ULTRA_HQ)
    t_render = time.perf_counter() - t1
    diag.update({"PatientID": pid, "role": role, "resample_sec": round(t_resample, 1),
                 "render_sec": round(t_render, 1), "centerline_length_mm": float(s_mm[-1])})
    (out_dir / "vgp_diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"{pid[-4:]}-{1 if role == 'primary' else 2}: unfold {rgb.shape} "
          f"(angles x s), resample {t_resample:.0f} s, render {t_render:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
