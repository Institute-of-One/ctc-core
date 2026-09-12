"""Polyp measurement accuracy against exact geometric ground truth.

Evaluation pillar 2, as re-scoped in docs/EVALUATION_POLYPS.md: the collection's
annotations carry no 3-D coordinate and our cohort holds three annotated
lesions, so in-vivo accuracy cannot be assessed. What can be assessed exactly is
measurement accuracy on phantoms whose true diameter, volume and sphericity are
known in closed form.

Each phantom is a sphere or ellipsoid of soft tissue in air. The primary set
has a one-voxel partial-volume ramp at its edge, which is what a real CT edge
looks like; a second set of spheres has an ideal step edge, which carries no
partial-volume information and shows the limit of any surface-based method.
Both diameter routes are reported -- the binary-mask route the reference
implementation used, and the grayscale iso-surface route this port uses -- so
the correction is shown rather than asserted.

    python scripts/eval_polyp_phantoms.py --out results/tables/polyp_phantom_accuracy.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctc_core.polyp import PolypConfig, measure_polyp  # noqa: E402

AIR_HU = -1000.0
SOFT_HU = 60.0

# (label, semi-axes in mm as (z, y, x), edge). Sizes span the 6 and 10 mm
# reporting thresholds, plus an elongated lesion that a sphere-only check would
# miss, plus step-edge spheres for the no-partial-volume limit.
PHANTOMS = [
    ("sphere d=6", (3.0, 3.0, 3.0), "ramp"),
    ("sphere d=8", (4.0, 4.0, 4.0), "ramp"),
    ("sphere d=10", (5.0, 5.0, 5.0), "ramp"),
    ("sphere d=12", (6.0, 6.0, 6.0), "ramp"),
    ("sphere d=16", (8.0, 8.0, 8.0), "ramp"),
    ("ellipsoid 8x8x20", (4.0, 4.0, 10.0), "ramp"),
    ("sphere d=6", (3.0, 3.0, 3.0), "step"),
    ("sphere d=10", (5.0, 5.0, 5.0), "step"),
    ("sphere d=16", (8.0, 8.0, 8.0), "step"),
]


def ellipsoid_pv(
    semi: tuple[float, float, float], n: int, spacing: float, edge: str = "ramp"
) -> np.ndarray:
    """Ellipsoid with a one-voxel linear partial-volume ramp at its edge.

    ``edge="step"`` gives a hard edge instead: each voxel is lesion if its centre
    lies inside the surface, air otherwise.
    """
    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
    zz, yy, xx = (zz - c) * spacing, (yy - c) * spacing, (xx - c) * spacing
    # Signed distance to the surface, approximated by scaling the normalised
    # radius back to mm with the local semi-axis; exact for a sphere.
    r_norm = np.sqrt((zz / semi[0]) ** 2 + (yy / semi[1]) ** 2 + (xx / semi[2]) ** 2)
    scale = min(semi)
    dist = (r_norm - 1.0) * scale
    if edge == "step":
        frac = (r_norm <= 1.0).astype(float)
    else:
        frac = np.clip(0.5 - dist / spacing, 0.0, 1.0)
    return (AIR_HU + (SOFT_HU - AIR_HU) * frac).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=1.0, help="isotropic voxel size, mm")
    args = ap.parse_args()

    sp = args.spacing
    rows = []
    for label, semi, edge in PHANTOMS:
        n = int(np.ceil(2 * max(semi) / sp)) + 24
        ct = ellipsoid_pv(semi, n, sp, edge)
        centre = (n / 2.0, n / 2.0, n / 2.0)

        true_diam = 2.0 * max(semi)
        true_vol = (4.0 / 3.0) * np.pi * semi[0] * semi[1] * semi[2]

        gray = measure_polyp(ct, centre, (sp, sp, sp), PolypConfig(diameter_from_grayscale=True))
        binr = measure_polyp(ct, centre, (sp, sp, sp), PolypConfig(diameter_from_grayscale=False))
        if not (gray.ok and binr.ok):
            raise RuntimeError(f"{label}: measurement failed")
        g, b = gray.measurements, binr.measurements

        rows.append({
            "phantom": label,
            "edge": edge,
            "voxel_mm": sp,
            "true_max_diameter_mm": round(true_diam, 3),
            "binary_max_diameter_mm": b["max_diameter_mm"],
            "binary_diameter_error_mm": round(b["max_diameter_mm"] - true_diam, 3),
            "grayscale_max_diameter_mm": g["max_diameter_mm"],
            "grayscale_diameter_error_mm": round(g["max_diameter_mm"] - true_diam, 3),
            "true_volume_mm3": round(true_vol, 2),
            "voxel_count_volume_mm3": g["volume_voxel_count_mm3"],
            "voxel_count_volume_error_pct": round(
                100.0 * (g["volume_voxel_count_mm3"] - true_vol) / true_vol, 2
            ),
            "mesh_volume_mm3": g["volume_mm3"],
            "mesh_volume_error_pct": round(100.0 * (g["volume_mm3"] - true_vol) / true_vol, 2),
            "sphericity": g["sphericity"],
            "shape_index_median": g["shape_index_median"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'phantom':18s} {'true':>6s} {'binary':>14s} {'grayscale':>14s} "
          f"{'vol: voxel':>12s} {'vol: mesh':>11s} {'spher':>6s}")
    for r in rows:
        print(f"{r['phantom'] + ' ' + r['edge']:18s} {r['true_max_diameter_mm']:6.1f} "
              f"{r['binary_max_diameter_mm']:7.3f} ({r['binary_diameter_error_mm']:+.2f}) "
              f"{r['grayscale_max_diameter_mm']:7.3f} ({r['grayscale_diameter_error_mm']:+.2f}) "
              f"{r['voxel_count_volume_error_pct']:+10.1f}% {r['mesh_volume_error_pct']:+9.1f}% "
              f"{r['sphericity']:6.3f}")
    spheres = [r for r in rows if r["phantom"].startswith("sphere") and r["edge"] == "ramp"]
    ge = [abs(r["grayscale_diameter_error_mm"]) for r in spheres]
    be = [abs(r["binary_diameter_error_mm"]) for r in spheres]
    print(f"\nspheres, ramp edge, |diameter error|: grayscale max {max(ge):.3f} mm, "
          f"binary max {max(be):.3f} mm")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
