# Pre-specified plan: centerline perturbation experiment

Written 2026-09-12, **before any result of this experiment was computed**. The
plan was not followed in every respect: section 8 records each departure, when
it was found and when it was corrected. The hypothesis, the conditions and the
outcomes are as specified; the selection of subjects was not, and was repaired
in the analysis rather than by re-running.

This is a **control experiment on the images already analysed**, not an
independent validation: the same series, masks and measurement parameters are
reused and only the centerline is changed.

## 1. Hypothesis

Under the same segment and the same end points, local wobble of the centerline
**increases the traced path length** and **changes the centerline-based
pericolonic fat indices**.

- Length: directional hypothesis (increase).
- Fat indices: magnitude of change is assessed; no direction is predicted.
- No assumption is made that left-right asymmetry necessarily increases, or that
  a more symmetric map is more accurate.

Local wobble (which lengthens the path) and premature termination or shortcuts
(which shorten the traced extent) are different problems and are kept separate:
this experiment addresses only the first.

## 2. Where the perturbation enters the pipeline

Order in the existing code, unchanged by this experiment:

| Quantity | Path from the stored centerline |
|---|---|
| Traced centerline length | `quality.smooth_path` (Gaussian, sigma 1.5 points, ends fixed) -> sum of segment lengths |
| Ring fat indices | `fatmap.smooth_centerline` (Gaussian, sigma 2 mm, ends fixed) -> `resample_centerline` (5 mm) -> `fat_map_polar` (180 rays, wall = first sample outside the segmented gas mask, ring 5-15 mm beyond it) -> `summarise_fat_map`, `polar_asymmetry` |

The perturbation is added **once**, to the stored centerline points, i.e. before
those steps. What is measured is therefore the sensitivity of the reported
indices to wobble **in the input path, after the smoothing that the pipeline
already applies** -- not the sensitivity to residual wobble that survives
smoothing. The Methods will say this in those terms. No existing smoothing is
bypassed and none is applied twice.

## 3. Subjects

Series that met the centerline success criterion in the current results
(extraction without error and a traced path of at least 600 mm), taken from
`results/tables/centerline_auto.csv` at run time. The count is whatever that
file gives; no historical figure is copied. Series that failed remain failures
in the original evaluation and are not perturbed.

Images, masks, patient position and all measurement parameters are fixed; only
the centerline changes.

## 4. Perturbation conditions

Displacement of the stored path, by arc length `s`:

    d(s) = A * taper(s) * sin(2*pi*s / L) * u

- amplitude `A` in {1, 2} mm (maximum lateral displacement)
- wavelength `L` in {10, 20} mm along the arc
- direction `u` in {n, b}: the two orthogonal directions of the local plane
  perpendicular to the centerline. They are **not** anatomical left/right; the
  frame is the anterior-anchored frame of `fatmap.centerline_frames` computed on
  the sigma-2-mm smoothed control path, which is smooth along the path, so no
  artificial kink is introduced by frame flips.
- `taper(s)` rises from 0 to 1 over the first 15 mm and falls back to 0 over the
  last 15 mm (raised cosine), so both end points stay exactly where they were.

That is 2 x 2 x 2 = **8 perturbed conditions plus the unperturbed control**.
Conditions are not added, dropped or re-tuned after seeing the results, and a
perturbed centerline is never pushed back inside the lumen.

The amplitudes and wavelengths are a controlled probe. They are **not** claimed
to represent the size of real centerline error.

## 5. Outcomes

Primary:

- relative change in traced centerline length against the control, per series.

Secondary, all as changes against the control of the same series:

- change in ring fat attenuation (HU) and its absolute value
- change in ring fat fraction and its absolute value
- change in the left-right asymmetry index and its absolute value
- share of centerline points lying outside the lumen mask after perturbation
- share of stations at which the ring fat measurement remains valid

Diagnostic, recorded but not an outcome: the root-mean-square displacement that
survives the sigma-2-mm smoothing, which says how much of the injected wobble
the existing smoothing removes.

Fat indices are aggregated per series exactly as in the manuscript. Because
changing the path also changes the number of stations, the result is the
sensitivity of the **whole pipeline**, sampling included; this is stated rather
than controlled for.

## 6. Analysis

- Paired per series: each perturbed condition against the control of the same
  series.
- Summaries are medians with interquartile ranges over series, per condition.
- Where an interval is given, it is a bootstrap percentile interval resampling
  **patients** (not series and not series-condition pairs), so that the two
  positions of a patient and the eight conditions of a series are never counted
  as independent observations.
- Effect sizes and spread are reported. No family of significance tests is added.

The geometric fact that wobble lengthens a path is not presented as the finding.
The reportable information is how much wobble changes which index, and how far
the existing smoothing limits that.

No claim is made that sensitivity to an artificial perturbation detects real
centerline failures. A link between real path roughness and failure would be
exploratory only, and is not part of this plan.

## 7. Feasibility and the fallback that is fixed in advance

A first run on 2 series checks correctness and timing. If the full run is too
slow, the experiment switches -- **before any outcome is examined** -- to a
subset fixed by patient identifier: the first 10 patients of the manifest order
that have at least one eligible series, all of their eligible series included.
The switch and its reason are recorded in section 8.

## 8. Deviations from this plan

- 2026-09-12: none at the time of writing. Entries are appended here with the
  reason, before the results are read where that is possible.
- 2026-09-12: the eligible set is taken from `results/tables/indices.csv`
  (`ok == True`) intersected with the success criterion in
  `results/tables/centerline_auto.csv`, because the perturbation needs the
  per-series work files that the indices stage also requires. Reason: the two
  files must agree on the series actually measurable; no outcome was inspected.
- 2026-09-12, **found** while the first summary of the results was being read
  (21:33 local time, immediately after the run finished at 21:31) and **fixed**
  in the analysis the same hour, before any number entered the manuscript: the
  eligibility filter had no effect on the length criterion,
  because `centerline_auto.csv` records the traced length rather than a
  "failed" column, and the first version of the filter looked for the latter.
  The experiment therefore ran on all 60 series with a centerline. The four
  series below 600 mm (0014-2, 0015-2, 0021-1, 0024-2) are excluded in
  `scripts/analyse_perturbation.py`, which now applies the criterion; their
  rows stay in the released CSV, unused, so the discrepancy is visible rather
  than hidden. The analysed set is the 56 series the plan specifies, and the
  filter in `scripts/perturb_centerline.py` is corrected for any re-run.
