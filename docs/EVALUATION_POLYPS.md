# Polyp measurement — what the TCIA annotations can and cannot support

Evaluation pillar 2 as originally planned was "for
annotated polyps present in our cohort: VOI max diameter vs reported size class
(>=10 mm, 6-9 mm), segment location agreement". Having downloaded the
annotations (`scripts/download_tcia_annotations.py`), **that comparison is not
achievable**, for two independent reasons.

## 1. The cohort contains almost no annotated polyps

The collection publishes three sheets: 35 patients with polyps >= 10 mm, 69 with
polyps 6-9 mm, and 243 with no polyp found. Against our 30 patients:

| Sheet | Patients in sheet | In our cohort |
|---|---:|---:|
| polyps >= 10 mm | 35 | **0** |
| polyps 6-9 mm | 69 | **3** (0007, 0011, 0019) |

Three lesions, all in one size band. That cannot support a size-agreement
analysis, and with nothing in the >= 10 mm class it cannot test behaviour at the
threshold that matters most.

This is a consequence of the cohort selection rule, which took the first 30
patients by PatientID with no regard to polyp status
(`scripts/download_tcia.py`). The rule was reproducible, which was
its purpose, but it was blind to this pillar.

## 2. The annotations carry no 3-D location

Each lesion row gives a size, a segment code and, for some rows, a slice number
in the supine and/or prone series. There is **no in-plane coordinate**. A
spherical VOI needs a point inside the lesion, so the annotation cannot place
one; a reader must identify the lesion within the named slice.

Slice numbers are not always present either:

| Sheet | Rows | Supine slice given | Prone slice given |
|---|---:|---:|---:|
| polyps >= 10 mm | 35 | 35 | 26 |
| polyps 6-9 mm | 69 | 35 | 33 |

Of our three, one (0007) has no slice number at all, one (0019) has supine only,
and one (0011) has both.

So even on an extended cohort the comparison would be reader-dependent: roughly
100 lesions, each requiring a human to place a point. That conflicts with the
design decision that makes this study reproducible -- no human readers -- and
would make the polyp results the one part of the paper that nobody else can
regenerate from the manifest.

## 3. What is reported instead

The measurement itself is validated against **exact** ground truth on geometric
phantoms, which is stronger than a reader's categorical size class:

- Spheres of known diameter, 6 / 10 / 16 mm at 1 mm isotropic.
- Ellipsoids with a known major axis.
- Canonical shape-index surfaces (cap +1, ridge +0.5, cup -1).

That validation found and corrected two systematic biases in the reference
implementation, both material at the 6 and 10 mm reporting thresholds:

| Quantity | Before | After | Cause |
|---|---|---|---|
| max diameter | -0.45 to +0.05 mm on a partial-volume edge, the value depending on where the threshold falls on the ramp; +1.00 mm on an ideal binary mask (7.000 / 11.000 / 17.000 against 6 / 10 / 16) | -0.11 to +0.11 mm on a partial-volume edge | marching cubes at level 0.5 on a *binary* mask puts the surface half a voxel outside the outermost included voxel centre, at each end |
| volume | -28 / -17 / -14 % for radii 3 / 5 / 8 mm | -8 / -3 / -1 % | a thresholded voxel count is biased by where the threshold sits on the partial-volume ramp; -50 HU is about 90 % up it |
| sphericity (sphere) | 0.84-0.91 | 0.99+ | follows from taking volume, area and diameter from one iso-surface |

The honest framing for the manuscript: we report **measurement accuracy against
exact geometric truth**, and we state that the collection's annotations support
neither an automatic nor an adequately powered in-vivo comparison. No claim
about in-vivo polyp measurement accuracy, detection sensitivity or specificity
is made.

## 4. If pillar 2 is wanted in vivo

It requires all three of:

1. re-selecting the cohort to include annotated-polyp patients (104 available,
   which at roughly 330 MB per series is about 68 GB of raw data against a 60 GB
   budget, so it needs `--purge-ct-after-case`);
2. a reader placing a point in each lesion, since no coordinate is published;
3. accepting that the resulting numbers are reader-dependent and not
   regenerable from the manifest alone.

The first is mechanical. The second and third change the character of the study.

## 5. Measured accuracy (2026-09-11)

```
python scripts/eval_polyp_phantoms.py --out results/tables/polyp_phantom_accuracy.csv
```

Soft tissue in air, 1 mm isotropic. Rows marked *ramp* have a one-voxel
partial-volume ramp at the edge; rows marked *step* an ideal hard edge.

| Phantom | True diameter | Binary route | Grayscale route | Volume, voxel count | Volume, mesh | Sphericity |
|---|---:|---:|---:|---:|---:|---:|
| sphere | 6 mm | 5.745 (-0.26) | **6.029 (+0.03)** | -28.4 % | **-8.0 %** | 0.991 |
| sphere | 8 mm | 7.550 (-0.45) | **7.886 (-0.11)** | -33.2 % | **-10.4 %** | 0.993 |
| sphere | 10 mm | 10.050 (+0.05) | **10.074 (+0.07)** | -16.5 % | **-3.2 %** | 0.995 |
| sphere | 12 mm | 11.874 (-0.13) | **12.073 (+0.07)** | -18.3 % | **-2.5 %** | 0.996 |
| sphere | 16 mm | 16.031 (+0.03) | **16.107 (+0.11)** | -14.2 % | **-1.1 %** | 0.997 |
| ellipsoid | 20 mm | 19.000 (-1.00) | 19.578 (-0.42) | -27.3 % | -10.1 % | 0.878 |
| sphere, step | 6 mm | 7.000 (+1.00) | 7.000 (+1.00) | +8.8 % | +2.4 % | 0.869 |
| sphere, step | 10 mm | 11.000 (+1.00) | 11.000 (+1.00) | -1.6 % | -3.6 % | 0.902 |
| sphere, step | 16 mm | 17.000 (+1.00) | 17.000 (+1.00) | -1.7 % | -2.4 % | 0.923 |

All rows above the step rows have the partial-volume ramp.

On spheres with a partial-volume edge the grayscale diameter is within 0.11 mm
at every size (-0.11 to +0.11); the binary diameter ranges from -0.45 to +0.05.
The grayscale error is not of one sign -- it is negative at 8 mm -- so the
correct claim is a smaller error and a smaller spread across size (0.22 against
0.50 mm), not a constant offset. On an ideal step edge both routes read +1.00 mm
everywhere: with no partial-volume information the surface cannot be placed
better than half a voxel at each end. Real CT always carries the ramp.

**Correction (2026-09-11, later the same day).** An earlier version of this file
and of `docs/PARAMETERS.md` described the grayscale error as "same-signed across
size" and the binary error as "+1.000 mm at every size". The first is false (the
8 mm sphere reads -0.114 mm); the second holds only for an ideal binary mask, not
for the binary route on a realistic edge. Both statements are corrected above.

**Caveat on the ellipsoid row.** Its partial-volume ramp is built from an
approximate distance to the surface -- the normalised radius scaled by the
smallest semi-axis -- which is exact for a sphere but compresses the ramp along
the long axis. Part of its -0.42 mm is therefore the phantom's, not the
measurement's, and the sphere rows are the primary evidence. The ellipsoid row is
kept to show that sphericity falls correctly (0.878) for an elongated lesion.
