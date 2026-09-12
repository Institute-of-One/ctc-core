# Parameter and threshold registry — IORN-012

Every heuristic threshold that a reviewer could challenge is recorded here with
its value, where it is used, and its justification. The rule was set for
`quality.py` and is applied to every module. A threshold with
justification "uncalibrated" must be described as such in the manuscript.

Status legend: **A** = anatomically/physically justified, **E** = empirically
chosen from the observed cohort distribution, **U** = uncalibrated, inherited
from the reference workbench and not yet examined.

---

## Outcome classification (`scripts/make_tables.py`)

The 2026-08 failure analysis (`scripts/analyze_batch_metrics.py`) used 1200 mm;
the manuscript uses the recalibrated values below.

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| `COMPLETE_LENGTH_MM` | 1200 mm -> 900 mm -> **withdrawn** | -- | **Withdrawn 2026-09-11**: the 900 mm value was calibrated against reference centerlines that were truncated by the same seed rule as ours (`docs/EVALUATION_HQCOLON.md` section 8); whole-colon reference centerlines measure 1.7-2.1 m. Completeness is now measured directly as whole-colon coverage where a reference exists, and no length threshold for completeness is used. Earlier text: **Recalibrated 2026-09-10.** The original 1200 mm came from anatomical colon length (150-200 cm), which is the wrong quantity: a lumen geodesic cuts haustral corners and is systematically shorter. HQColon reference centerlines measure 1186 mm median (IQR 988-1304), so 14 of 26 reference standards scored below the old threshold. Measured coverage saturates above ~900 mm (0.982 median in the 900-1100 mm band against 0.419 in 600-900). Still a proxy: Spearman r with measured coverage is only 0.664, and the recalibration used the same 26 series, so report measured coverage where a reference exists. See `docs/EVALUATION_HQCOLON.md` section 3. |
| `PARTIAL_LENGTH_MM` | 600 mm | A | Roughly one third of a colon; below this the trace cannot span more than a couple of segments and is not usable for segment-wise analysis. |
| `POOR_INSUFFLATION_VOXELS` | 150 000 vox = 150 mL at 1.0 mm iso | A+E | An adequately insufflated colon holds 1-2 L of gas, and `distention_quality_score` already anchors its gas term at 500 mL. 150 mL is well below any evaluable colon. Empirically the 2026-08 distribution has an empty gap from 93 mL to 241 mL, so no observed series lies near the threshold. |
| `SEG_FAILURE_DICE` | 0.5 | E | A lumen segmentation counts as failed against HQColon below this Dice with the gas-filled reference. Observed values are bimodal with an empty gap from 0.002 to 0.721 (3-D fill only), so any cut inside the gap gives the same count. |
| `MIN_INSUFFLATION_VOXELS` | 10 000 vox | U | Inherited: `auto_detect_seed_endpoints` aborts below this ("probably no insufflation"). Not independently justified; superseded in practice by the 150 mL criterion above. |

## Masks and seeds (`ctc_core/masks.py`, `centerline.py`)

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| body threshold | > -300 HU | U | Inherited. Separates soft tissue/fat from air and the outside; standard practice but not tuned here. |
| body closing radius | 2 vox | U | Inherited (batch). |
| body hole filling | **`auto`**: 3-D (`bbox`) fill, falling back to slice-wise 2-D when the result holds no plausible colon | A+E | **Corrected twice.** 2026-09-09: the batch used `sitk.BinaryFillhole(..., fullyConnected=True)`, a 26-connected background, letting a cavity escape through one diagonal wall gap; on 0022/primary that cost a factor of 221 in recovered gas. 2026-09-10: HQColon showed that *any* 3-D fill loses the colon when its gas reaches a volume face or merges with exterior air (5 of 26 reference series, 1.6-3.3 L each). Slice-wise 2-D filling seals those but degrades 11 of the 21 working series (precision 0.786 -> 0.670), so the choice is made adaptively. See `docs/EVALUATION_HQCOLON.md` section 6 and `tests/test_masks.py`. |
| `MIN_COLON_GAS_VOXELS` | 150 000 vox = 150 mL at 1.0 mm iso | A+E | Threshold at which a 3-D fill is judged to have found no colon, triggering the slice-wise fall-back. Same value and justification as `POOR_INSUFFLATION_VOXELS`. The two groups separate by more than an order of magnitude -- 12-53 mL for the failing series against 1.5-3 L for the rest -- so nothing sits near the threshold. |
| air threshold | < -700 HU | U | Inherited. Between insufflated colonic gas (~ -1000 HU) and fat (-190 to -30 HU). |
| air closing radius | 1 vox | E | The batch was internally inconsistent: the centerline stage used 0 and the fat-map stage 1. A radius of 0 leaves the colon fragmented at haustral folds and fluid menisci into components the 500-voxel dust filter then deletes. 1 voxel at 1.0 mm iso bridges a single-voxel partial-volume gap without merging anatomically separate structures; larger radii are not used because they risk fusing colon to small bowel. |
| `dust_threshold_voxels` | 500 vox | U | Inherited. |
| CC connectivity | 6 (was 26) | A | **Corrected 2026-09-09.** `sitk.FastMarchingImageFilter` propagates on the 6-connected stencil, so labelling at 26 declares connected what the solver cannot traverse. Confirmed as the cause of all 5 `End seed is unreachable` failures, with no false positives among 9 control series (`docs/FAILURE_ANALYSIS.md` section 2.3). Labelling at 6 is conservative: it can only split components the solver could not cross, and the split is then handled by the CC-bridge graph. The backtrack's 26-neighbour stepping must be reconciled with this. |
| min largest air CC for seeding | 10 000 vox | U | See above. |
| `seed_rule` | `whole_colon` | E | **Added 2026-09-11.** The provisional seeds below build the lumen; the final path is the one of three candidates (provisional pair; rectal end to the voxel farthest along the lumen; rectal end to the farthest voxel of the largest wide-core component) that brings most of the lumen within `selection_near_mm` of the path. The rectal end is the lowest lumen voxel, where CT colonography is insufflated. Needs no reference; the choice among candidates is recorded per series. `skeleton_extremes` (the previous behaviour) left about half of the colon untraversed in every series checked (`docs/EVALUATION_HQCOLON.md` section 8). |
| `core_radius_mm` | 3 mm | E | Wall distance defining the wide core within which the far end is sought for the `rectal_core` candidate. Thin contacts between touching loops and narrow small-bowel links fall below it; a distended colon does not. Chosen on five series where 3 mm fixed three that 5 and 7 mm broke by fragmenting the core. |
| `selection_near_mm` | 30 mm | A | Radius for the reference-free selection score. A little over the radius of a distended colon, so lumen within it is served by the path; the same radius as the whole-colon coverage metric. |
| provisional seeds | max-distance pair of degree-1 endpoints of the Lee skeleton over **all retained** air CCs (`seed_span_components=True`) | E | **Corrected 2026-09-10.** The batch skeletonized only the largest component, so `start_cc == end_cc` held identically and the CC-bridge graph was unreachable dead code in 55/55 series (`docs/FAILURE_ANALYSIS.md` section 2.1). Spanning the retained set lets the bridge planner run; on 0001/primary the seeds first land in different components and the centerline grows 936 -> 1080 mm. Still a geometric heuristic, not an anatomical anchor: the pair is the most distant skeleton endpoints, not rectum and caecum. |

## Centerline (`ctc_core/centerline.py`)

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| speed exponent `k` | 2.0 | U | `F(p) = clip(d_wall(p)^k, 0.1, 1000)` inside the lumen, with `d_wall` the Euclidean distance to the lumen boundary in mm (`compute_arrival_time`); larger `k` pushes the geodesic toward the lumen axis. Inherited. (An earlier version of this row wrote `(1 + d_wall)^k`, which is not what the code computes.) |
| speed clip | [0.1, 1000] | U | Inherited; prevents zero speed inside the lumen. |
| FMM stopping value | 1e8 | A | Inherited, and verified not to be the cause of the unreachable-seed failures: those are a connectivity mismatch, not early termination (`docs/FAILURE_ANALYSIS.md` section 2.3). Effectively unbounded. |
| resample step | 1.5-2.0 mm | U | Inherited. |
| `max_bridge_mm` | 150 mm | U | Inherited. |
| `bridge_alignment_weight_mm` (alpha) | 40 mm | U | Inherited. |
| `bridge_wall_penalty_per_mm` (beta) | 6 /mm | U | Inherited. |
| `bridge_tube_radius_vox` | 2 vox | U | Inherited. |
| kimimaro teasar | `scale=2.0, const=50, pdrf_scale=100000, pdrf_exponent=4` | U | Inherited; used to obtain component endpoints and tangents only, never as the centerline itself. These are the values of the prototype's actual call (`_plan_cc_chain_bridges`), which differ from `scale=4, const=4` recorded in its notes. |

## Fat space map (`ctc_core/fatmap.py`)

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| ray step `Delta` | 0.5 mm | A | Half the 1.0 mm voxel — Nyquist-adequate sampling along the ray. |
| `max_ray_mm` | 60 mm | E | **Changed 2026-09-12** from the inherited 35 mm. The ring ends 15 mm beyond the wall, so a 35 mm ray truncates it wherever the wall lies beyond 20 mm, which in a distended colon is a quarter of the rays. 60 mm covers the wall plus the full ring at every lumen calibre observed. |
| `skin_skip_mm` | 1 mm | U | Inherited; skips the first sample so the wall is not detected at the ray origin. |
| fat HU window | [-190, -30] HU | A | Conventional adipose tissue window, widely used in body-composition CT. |
| `step_mm` | 5 mm | U | Inherited; along-centerline station interval, for both the ring and the sphere. |
| phase0 batch: `r_mm` | 15 mm | U | Inherited; spherical ROI radius. **Since 2026-09-12 the sphere is the previous definition, reported for comparison only**: centred on the centerline, it reaches tissue beyond the wall only where the lumen is narrower than 15 mm, so it was valid at a median of 61 % of stations (lowest 22 %) against 100 % for the ring. |
| phase0 batch: `min_fat_vox` | 50 | U | Inherited; minimum voxels for a valid sphere station. |

## Quality indices (`ctc_core/quality.py`)

All thresholds in this section are **uncalibrated** and must be stated as such
in the manuscript. They define the indices; they do not yet have validated
cut-points.

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| gas mask | lumen AND CT <= -700 HU | U | Inherited. |
| centerline smoothing | 1-D Gaussian, sigma = 1.5 points, endpoints fixed | U | Inherited; suppresses voxel-stepping noise before length integration. |
| CTI bands | Low <= 12, Moderate 12-18, High > 18 | U | Inherited, and **almost certainly wrong for this definition**. With `CTI = L/D` and both in the same unit, 0001/primary gives L = 104.4 cm over D = 45.6 cm, i.e. CTI = 2.29. A colon cannot plausibly reach 12, so these bands would never fire and must belong to some other quantity or unit convention. Do not use them until re-derived from the observed distribution. |
| `PLRI` reference length | 160 cm | A | Mid-point of the 150-200 cm nominal colon length. PLRI divides *traced* centerline length by this *anatomical* prediction, so its numerator and denominator are not the same quantity; report it as traced extent against a nominal colon, not as redundancy. |
| height-normalised length index | not computed | -- | The collection's clinical table provides no height, weight, age or BMI (`scripts/make_cohort_table.py`), so the index is left null rather than imputed. |
| collapse radius | < 4 mm Maurer radius | U | Inherited. |
| collapse band for score | 40 % | U | Inherited; `s_collapse = clip((1 - collapse_ratio/40) * 100)`. |
| LAG score anchors | 2 to 8 mL/cm | U | Inherited. |
| gas score anchors | 500 to 1500 mL | U | Inherited. |
| distension score weights | 0.45 collapse / 0.40 LAG / 0.15 gas | U | Inherited; **no empirical basis**. The composite shows prone/supine ICC(2,1) 0.19, so it must be presented as a descriptor rather than a validated score (`docs/EVALUATION_PRONE_SUPINE.md`). |
| distension grades | Excellent >= 80, Good 60-79, Fair 40-59, Poor < 40 | U | Inherited; arbitrary. |

## Fat map (`ctc_core/fatmap.py`), additions 2026-09-11

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| `ring_inner_mm` / `ring_outer_mm` | 5 / 15 mm beyond the wall | U | Inherited from `vgp_fat_heatmap.py`. Anchoring the ring to the detected wall rather than to the centerline is what makes stations of differing calibre comparable. **Primary pericolonic measurement since 2026-09-12** (`ring_fat_mean_hu_median`, `ring_fat_fraction_mean`). The wall is the first sample outside the gas mask; without cleansing, a ray that meets tagged fluid takes its surface as the wall. |
| `frame_smooth_sigma_mm` | 2 mm | E | **Added 2026-09-12.** Gaussian smoothing of the centerline before the ring frames, endpoints fixed, as the prototype's VGP unfold does. On the voxel path the tangent flips between neighbouring stations and the ray fan rotates with it; 2 mm removes the kinks (and, in the unfold, the black dots) without moving the path off the lumen axis. The spherical profile keeps the unsmoothed path of the 2026-08 batch. |
| `min_ring_fat_samples` | 50 | U | **Added 2026-09-12.** A station counts when its 180 pooled rays hold at least 50 fat samples; the same number as the sphere's voxel minimum, not tuned. Rays are pooled per station (sum of fat HU over sum of fat samples) rather than averaged per ray, so rays that meet little tissue do not weigh as much as full ones. |
| `n_theta` | 180 | E | 2-degree angular sampling. Fine enough that the left/right split is insensitive to it, coarse enough to stay cheap. |
| frame reference direction | anterior `(0, -1, 0)`, projected perpendicular to the tangent | A | Puts `theta = 0` anteriorly in every series and position. Falls back to `(0, 0, 1)` where the tangent is within ~8 degrees of anterior. **Corrected 2026-09-11**: this anchors `theta` but not its sense of rotation, which follows the tangent, so the halves of `theta` are not left and right (see the next row). |
| `lateral_cone_deg` | 45 degrees | A | **Added 2026-09-11.** `fat_asymmetry` counts a ray as left or right when it points within 45 degrees of the patient's left-right axis (+x / -x in LPS), so the sign means more fat on the patient's left and does not depend on the direction of travel. The original specification split at `theta = pi`; that index flips sign when the centerline is traversed the other way (the seed order is arbitrary) and between ascending and descending segments, so it measures side-of-travel, not left/right. It is kept as `fat_asymmetry_theta_halves` for comparison (`tests/test_fatmap.py` pins both behaviours). |
| air-mask structuring element | `sitk.BinaryMorphologicalClosing`, radius 1 | E | Differs from the reference workbench's strict `r <= 1` ball on 47 273 of 2.8 M air voxels (1.6 %), all at boundaries, which moves `fat_fraction_mean` by ~1 %. The SimpleITK form is kept for consistency with every other stage; `tests/test_fatmap.py` pins the exact reproduction under the reference element so the difference stays attributed to the mask. |

## Agreement statistics (`ctc_core/agreement.py`)

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| ICC model | ICC(2,1), two-way random, absolute agreement, single measurement | A | The two positions are not interchangeable replicates, so a systematic offset between them must count against agreement. ICC(3,1) treats raters as fixed and would ignore exactly that. Validated against Shrout & Fleiss (1979). |
| ICC confidence interval | McGraw & Wong (1996) case 2A, F-based, Satterthwaite degrees of freedom from `MS_cols` and `MS_err` | A | **Corrected 2026-09-12.** The earlier code used the rows mean square (`MS_rows`) where the formula needs the columns (position) mean square, so every ICC confidence interval reported before this date was wrong; point estimates were unaffected. Most intervals moved by 0.01 or less, a few by up to 0.2 at the lower bound (sphere fat attenuation over 19 pairs: 0.403 -> 0.599). `tests/test_agreement.py` now pins the interval of the Shrout-Fleiss example, [0.019, 0.761], as well as the coefficient. |
| limits of agreement | bias +/- 1.96 SD | A | Bland & Altman (1986). |
| sensitivity analysis exclusion | share of the reference colon beyond 60 mm > 0.05, and a reference is required | A | **Corrected 2026-09-12 (second).** The filter kept series with no reference, so the "whole colon reached" subset of 19 pairs was mostly pairs the reference had never scored: 11 had no reference at all and 10 only one. With the requirement, 7 of 28 pairs have a reference in both positions and 3 meet the criterion in both, too few to report, so no sensitivity analysis is published. `scripts/eval_prone_supine.py` drops unreferenced series (`select_covered`, pinned by `tests/test_agreement.py`) and refuses below `--min-pairs`. |

## Polyp VOI, shape index and measurement (`ctc_core/polyp.py`)

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| sphere VOI radius | 18 mm | U | Inherited; comfortably encloses a 10 mm polyp with margin. |
| polyp HU window | [-50, 180] HU | U | Inherited. |
| closing | 1.0 mm | U | Inherited. |
| `min_component_mm3` | 4.0 | U | Inherited. |
| shape-index `sigma_mm` | 2.0 | U | Inherited; curvature smoothing scale. |
| `diameter_from_grayscale` | True | A | **Corrected 2026-09-11.** Marching cubes at level 0.5 on a *binary* mask places the surface half a voxel outside the outermost included voxel centre at each end, so the maximum diameter is biased high by exactly one voxel: measured 7.000 / 11.000 / 17.000 mm for spheres of true diameter 6 / 10 / 16 at 1 mm isotropic. That is 17 % at the 6 mm reporting threshold. On a realistic partial-volume edge the binary route's error depends on where the segmentation threshold falls on the ramp (-0.45 to +0.05 mm over 6-16 mm spheres). Taking the iso-surface from the grayscale uses the partial-volume information and keeps the error within 0.11 mm (-0.11 to +0.11), with half the spread across size; it is not a constant offset. On an ideal step edge both routes read +1.00 mm, since no method can place the surface better than half a voxel at each end; real CT always carries the ramp (`scripts/eval_polyp_phantoms.py`, step rows). |
| diameter iso-level | per lesion: midpoint of median HU inside the mask and in a surrounding shell | A | Estimated per lesion rather than fixed, because a polyp may border lumen air (about -1000 HU) or tagged fluid (well above 0) and the boundary midpoint differs accordingly. |
| lesion volume | enclosed volume of the iso-surface mesh | A | **Corrected 2026-09-11.** A thresholded voxel count is biased by where the threshold sits on the partial-volume ramp: `soft_hu_min` of -50 HU is about 90 % up a lumen-to-soft-tissue ramp, so voxel counting *under*-reads volume by 28 / 17 / 14 % on spheres of radius 3 / 5 / 8 mm. The mesh volume reduces that to 8 / 3 / 1 % and refers to the same surface as the area and the diameter, which also makes sphericity self-consistent (0.99+ on a sphere against 0.84-0.91 before). The voxel-count volume is still reported as `volume_voxel_count_mm3` for comparison. |

## Unfolded view, VGP (`ctc_core/vgp.py`, `scripts/make_vgp.py`) -- visualisation only

No number depends on these; they set the look of Figure 2c. Ported unchanged
from the prototype's "VGP Cube (Ultra HQ, super-sampled)" preset.

| Parameter | Value | Status | Justification |
|---|---|:--:|---|
| CT grid for the figure | 0.5 mm isotropic, one series only | U | The prototype's highest-quality route: resample the one displayed series finely, render, and leave every other series on the 1.0 mm measurement grid. |
| `num_angles` | 720 | U | 0.5-degree columns. |
| `sample_distance_mm` | 0.25 mm | U | Ray-casting step, half the grid spacing. |
| `opacity_ramp_hu`, `clamp_high_hu` | 100 HU, 300 HU | U | Opacity rises over 100 HU from air to tissue; tagged residue is clamped at 300 HU so its gradient does not dominate the shading. |
| centerline | the ring map's (sigma 2 mm smoothing), resampled every 1 mm; the renderer's own smoothing off | E | Shares the ring map's arc length, so panels c-e of Figure 2 line up. Smoothing inside the renderer would move the points but keep the voxel path's arc length, about 7 % longer. |
| cube faces | 4 x 100 degrees, view-up = tangent, `tan` remap with overlap blending | U | Four renders per station instead of one per angle (the prototype's 47 min -> 24 s). |
