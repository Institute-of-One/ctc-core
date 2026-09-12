# Prone/supine agreement — 2026-09-11

Evaluation pillar 3. Reproduce with:

```
python scripts/run_indices.py --centerline-root data/centerline/corrected --polar --out results/tables/indices.csv
python scripts/eval_prone_supine.py --indices results/tables/indices.csv --out results/tables/prone_supine_agreement.csv
python scripts/eval_prone_supine.py --indices results/tables/indices.csv --min-coverage 0.6 --out results/tables/prone_supine_agreement_highcov.csv
```

Indices computed for **60 of 60 series**. Pairing is by **recovered acquisition
position**, never by the manifest's primary/secondary role, which carries no
position information (`docs/EVALUATION_HQCOLON.md`, and
`scripts/derive_patient_position.py`).

Statistics: ICC(2,1) — two-way random effects, absolute agreement, single
measurement — with a 95 % confidence interval, and Bland–Altman bias with 95 %
limits of agreement. ICC(2,1) rather than ICC(3,1) because the two positions are
not interchangeable replicates and a systematic offset between them *should*
count against agreement. The implementation is checked against the published
worked example in Shrout & Fleiss (1979); see `tests/test_agreement.py`.

**28 of 30 patients** paired. The two exclusions are informative:

- **0003** — both series were called supine. This is the single known error of
  the position rule (`0003/primary` is prone in HQColon but tagged `FFS`), and
  it costs one pair outright. A 96.2 % accurate position rule is not free.
- **0019** — only one series carries an identifiable position.

---

## 1. Results

Bias is prone minus supine. The right-hand column is the sensitivity analysis
restricted to series with HQColon coverage >= 0.6, which is available for the 26
series with a reference.

| Index | ICC(2,1) [95 % CI], n=28 | ICC, coverage >= 0.6, n=24 | Bias [95 % CI] | 95 % LoA |
|---|---|---|---|---|
| **fat_mean_hu_median** | **0.759 [0.484, 0.889]** | 0.745 [0.427, 0.889] | −0.39 HU [−1.63, 0.85] | ±6.2 HU |
| gas_volume_ml | 0.674 [0.386, 0.838] | 0.699 [0.377, 0.863] | −211 mL [−439, 18] | ±1155 mL |
| fat_asymmetry (patient frame, from 2026-09-11) | 0.575 [0.252, 0.780] | 0.449 [0.059, 0.719] | 0.002 [−0.028, 0.033] | ±0.155 |
| fat_asymmetry_theta_halves (original definition; was `fat_asymmetry`) | 0.394 [0.032, 0.665] | 0.626 [0.290, 0.822] | −0.009 [−0.025, 0.007] | ±0.075 |
| fat_fraction_mean | 0.520 [0.218, 0.739] | 0.554 [0.233, 0.773] | **−0.032 [−0.055, −0.008]** | ±0.118 |
| centerline_tortuosity_index | 0.398 [0.032, 0.669] | 0.492 [0.117, 0.744] | 0.026 [−0.219, 0.271] | ±1.24 |
| length_adjusted_gas_ml_per_cm | 0.357 [0.006, 0.635] | 0.490 [0.115, 0.743] | −2.49 [−5.72, 0.74] | ±16.3 |
| fat_volume_ml_sum | 0.328 [−0.036, 0.618] | 0.333 [−0.053, 0.639] | −36 mL [−101, 29] | ±326 mL |
| traced_centerline_length_cm | 0.318 [−0.061, 0.615] | 0.322 [−0.085, 0.637] | −1.7 cm [−13.9, 10.5] | ±61.8 cm |
| luminal_radius_mm_median | 0.205 [−0.185, 0.536] | 0.355 [−0.050, 0.659] | −0.03 mm [−1.10, 1.04] | ±5.4 mm |
| distension_quality_score | 0.188 [−0.194, 0.520] | 0.201 [−0.205, 0.551] | 2.85 [−4.31, 10.00] | ±36.2 |
| collapse_ratio_pct | 0.178 [−0.208, 0.513] | 0.211 [−0.189, 0.556] | −1.99 % [−8.16, 4.18] | ±31.2 % |

**Only one bias is distinguishable from zero**, and it is so in both analyses:
`fat_fraction_mean` is lower in prone by 0.032 (95 % CI 0.008–0.055).

---

## 2. Reading this correctly

A single "how reproducible is the pipeline" summary would be wrong here, because
these indices are not all supposed to agree. They fall into three groups, and
the group determines what a low ICC means.

### 2.1 Position-independent tissue properties — these *should* agree

**`fat_mean_hu_median`, ICC 0.759, bias −0.4 HU, limits ±6.2 HU.** Pericolonic
fat attenuation is reproduced across two separate acquisitions with the patient
physically turned over, with no detectable systematic offset and limits of
agreement of a few Hounsfield units. This is the strongest single result in the
project and the one that supports treating pericolonic fat attenuation as a
patient-level measurement.

~~**`fat_asymmetry`, ICC 0.626 once low-coverage series are removed** (0.394 over
all pairs). The direction-resolved index is reproducible when there is enough
colon to measure it on, which is what the `(s, theta)` representation is for.~~

**WITHDRAWN (2026-09-11, same day).** The index these numbers describe split the
map at `theta = pi`. `theta = 0` is anterior, but the sense of rotation follows
the centerline tangent, so the half facing the patient's left swaps with the
direction of travel -- which is set by an arbitrary seed order and reverses
between ascending and descending colon. It is a side-of-travel index, not a
left/right one (a phantom reads -1.000 forwards and +1.000 backwards). Kept as
`fat_asymmetry_theta_halves`.

`fat_asymmetry` is now defined in the patient frame (rays within 45 degrees of
the left-right axis). Over all 28 pairs its ICC is 0.575 [0.252, 0.780]; with
low-coverage series removed it is 0.449 [0.059, 0.719]. The two definitions move
in opposite directions under the same exclusion, so neither supports a claim
that the asymmetry index is reproducible "when there is enough colon"; the
confidence intervals are wide and overlap. The honest statement is moderate
agreement with an unstable estimate at n = 24-28.

### 2.2 Position-*dependent* state — these should **not** agree

`gas_volume_ml`, `collapse_ratio_pct`, `distension_quality_score`,
`luminal_radius_mm_median`. Gas redistributes and segments that collapse in one
position open in the other; that is the entire reason CT colonography is
acquired in two positions. Their low ICCs (0.18–0.70) are the expected physical
behaviour, not a measurement failure, and reporting them as reproducibility
failures would be a misreading.

The corollary is that these indices must be reported **per position**, never
pooled or averaged across the two, and any downstream analysis using them has to
carry the position with it.

### 2.3 Measurement-limited — a real limitation

**`traced_centerline_length_cm`, ICC 0.318, limits of agreement ±61.8 cm.** Anatomical colon
length is a patient property and should agree; ±62 cm on a ~100 cm measurement
does not. Unlike the indices in 2.1, it does **not** improve when low-coverage
series are removed (0.318 → 0.322), so this is not simply the four bad series.

The honest reading is that this is not anatomical colon length. It is the length
of the traceable gas-filled centerline, and how much of the colon is gas-filled
and traceable differs between the two acquisitions. The index has been renamed
accordingly; it was `colon_length_cm`, which overstated what is measured.

A consequence worth carrying: PLRI divides this traced length by a *predicted
anatomical* colon length of 160 cm, so its numerator and denominator are not the
same quantity. It is a descriptor of traced extent against a nominal colon, not
a redundancy measure, and must be reported as such.

### 2.4 The one systematic offset

`fat_fraction_mean` is lower in prone by 0.032 (CI 0.008–0.055), consistently in
both analyses, while `fat_mean_hu_median` shows no offset at all. The
dissociation is interpretable: the *composition* of pericolonic fat, its
attenuation, is a tissue property and does not care about position; the *amount*
of fat within a fixed-radius ROI depends on how the abdominal contents are packed
around the colon, which turning the patient over changes. Anyone using a fat
fraction as a patient-level measure has to account for position; a fat
attenuation can be used directly.

---

## 3. What this supports in the manuscript

Claimable:

- Indices computed for 60 of 60 series; 28 of 30 patients paired by recovered
  position.
- Pericolonic fat attenuation is position-independent: ICC(2,1) 0.759
  [0.484, 0.889], bias −0.4 HU, limits ±6.2 HU.
- The left-right asymmetry index, once defined in the patient frame, has ICC
  0.575 [0.252, 0.780] over 28 pairs and 0.449 [0.059, 0.719] over 24; the earlier
  0.626 belonged to a side-of-travel definition and is withdrawn.
- Fat fraction carries a small but real positional offset (−0.032,
  CI 0.008–0.055) that fat attenuation does not.
- The distension-related indices behave as position-dependent state, which is
  consistent with the rationale for two-position acquisition.

Not claimable, and to be stated as limitations:

- Traced centerline length is not reproducible between positions
  (limits ±61.8 cm) and is not anatomical colon length.
- `distension_quality_score` inherits arbitrary weights (0.45 / 0.40 / 0.15) and
  shows ICC 0.19; it should be presented as a descriptor, not a validated score.
- n = 28 pairs is small; every confidence interval here is wide, and four of the
  eleven lower bounds sit below zero.

## 4. Next

1. The cohort extension is now clearly worth doing, and should be chosen to
   maximise HQColon overlap: 113 HQColon subjects have both positions against
   our 8, which would sharpen every interval above.
2. ~~Rename the length index.~~ **DONE**: it is `traced_centerline_length_cm`
   throughout, so the name no longer implies an anatomical measurement.
3. Port `polyp.py` and `radiomics.py` for evaluation pillar 2.


---

## 2026-09-12: re-run on the whole-colon path -- figures above are superseded

Every figure above was computed on centerlines that missed about half the colon
(`docs/EVALUATION_HQCOLON.md` section 8). Re-run with the whole-colon seed rule
(`results/tables/prone_supine_agreement.csv`, 28 pairs; sensitivity analysis in
`prone_supine_agreement_reached.csv`, which now excludes series whose colon the
path did not reach -- share beyond 60 mm > 0.05 -- leaving 19 pairs):

| Index | ICC(2,1) [95 % CI], 28 pairs | colon reached, 19 pairs | Bias [95 % CI] |
|---|---|---|---|
| traced_centerline_length_cm | 0.325 [-0.047, 0.618] | 0.345 [-0.076, 0.675] | -9.372 [-33.44, 14.69] |
| centerline_tortuosity_index | 0.218 [-0.169, 0.544] | 0.302 [-0.160, 0.656] | -0.4895 [-2.773, 1.794] |
| gas_volume_ml | 0.622 [0.325, 0.806] | 0.470 [0.090, 0.746] | -228.3 [-461.3, 4.716] |
| length_adjusted_gas_ml_per_cm | 0.596 [0.286, 0.792] | 0.682 [0.288, 0.870] | -1.304 [-3.552, 0.9429] |
| collapse_ratio_pct | 0.136 [-0.249, 0.481] | 0.054 [-0.399, 0.486] | -1.504 [-5.937, 2.93] |
| luminal_radius_mm_median | 0.149 [-0.217, 0.484] | 0.293 [-0.187, 0.655] | -0.6607 [-1.673, 0.3513] |
| distension_quality_score | 0.169 [-0.217, 0.507] | 0.153 [-0.307, 0.557] | 2.046 [-4.52, 8.613] |
| fat_mean_hu_median | 0.724 [0.442, 0.868] | 0.822 [0.403, 0.938] | 0.114 [-1.157, 1.385] |
| fat_fraction_mean | 0.580 [0.267, 0.781] | 0.442 [0.040, 0.732] | -0.0146 [-0.0363, 0.0072] |
| fat_volume_ml_sum | -0.024 [-0.405, 0.354] | 0.105 [-0.305, 0.507] | -22.47 [-141.3, 96.4] |
| fat_asymmetry | 0.226 [-0.164, 0.550] | 0.533 [0.104, 0.792] | -0.0014 [-0.0331, 0.0303] |
| fat_asymmetry_theta_halves | -0.093 [-0.425, 0.273] | -0.266 [-0.581, 0.165] | 0.0173 [-0.007, 0.0416] |

What changes:

- **Fat attenuation still agrees**: ICC 0.724 [0.442, 0.868], bias +0.1 HU.
- **The fat-fraction offset is gone.** The earlier -0.032 [-0.055, -0.008] --
  the only significant bias, and the basis of the composition-versus-amount
  interpretation -- does not survive the whole-colon path (-0.015 [-0.036,
  0.007]). That interpretation is withdrawn.
- **Fat volume no longer agrees** (ICC -0.02): a sum over stations grows with
  path length, which varies between positions.
- **The asymmetry index** (patient frame) is 0.23 over 28 pairs and 0.53 where
  the colon was reached: unstable, no claim.
- **Distension indices** agree poorly to moderately (0.14-0.62), as before.
- **Traced centerline length** is now close to whole-colon length (median
  1566 mm) but still does not agree between positions (ICC 0.33).

---

## 2026-09-12 (later): wall-anchored ring primary; ICC intervals corrected

Two changes, both superseding the table in the previous section.

**ICC confidence intervals were wrong.** `ctc_core/agreement.py` computed the
Satterthwaite degrees of freedom with the rows mean square instead of the
columns (position) mean square. Point estimates were unaffected; every interval
reported before this date, in this file and in the earlier drafts, was not.
Most moved by 0.01 or less, a few by up to 0.2 at the lower bound. The test now
pins the Shrout-Fleiss interval [0.019, 0.761].

**The fat ring is primary.** Figure 2 showed gaps in the spherical profile: a
15 mm sphere centred on the centerline holds almost no tissue beyond the wall
where the lumen is wide (38 % of stations in 0007-1; cohort median 61 % valid,
lowest 22 %). The ring 5-15 mm beyond the wall, on the sigma-2-mm smoothed
centerline with 60 mm rays and stations pooled over 180 rays, is valid at every
station in 52 of 60 series and at >= 90.6 % in all. The decision was taken on
that sampling ground before the ring's agreement was computed. The sphere is
reported as the previous definition.

| Index | ICC(2,1) [95 % CI], 28 pairs | colon reached, 19 pairs | Bias [95 % CI] |
|---|---|---|---|
| ring_fat_mean_hu_median | 0.958 [0.913, 0.980] | 0.934 [0.757, 0.978] | -0.5315 [-1.563, 0.5] |
| ring_fat_fraction_mean | 0.962 [0.920, 0.982] | 0.948 [0.872, 0.980] | -0.0039 [-0.0204, 0.0125] |
| fat_asymmetry (ring, patient frame) | -0.070 [-0.437, 0.311] | 0.089 [-0.394, 0.520] | -0.0114 [-0.0447, 0.0218] |
| fat_asymmetry_theta_halves | -0.265 [-0.562, 0.104] | -0.254 [-0.588, 0.191] | 0.02 [-0.0057, 0.0458] |
| fat_mean_hu_median (sphere) | 0.724 [0.484, 0.862] | 0.822 [0.599, 0.927] | 0.114 [-1.157, 1.385] |
| fat_fraction_mean (sphere) | 0.580 [0.278, 0.779] | 0.442 [0.033, 0.734] | -0.0146 [-0.0363, 0.0072] |
| fat_volume_ml_sum (sphere) | -0.024 [-0.405, 0.354] | 0.105 [-0.302, 0.506] | -22.47 [-141.3, 96.4] |
| traced_centerline_length_cm | 0.325 [-0.049, 0.619] | 0.345 [-0.077, 0.675] | -9.372 [-33.44, 14.69] |
| centerline_tortuosity_index | 0.218 [-0.172, 0.546] | 0.302 [-0.164, 0.657] | -0.4895 [-2.773, 1.794] |
| gas_volume_ml | 0.622 [0.331, 0.805] | 0.470 [0.057, 0.752] | -228.3 [-461.3, 4.716] |
| length_adjusted_gas_ml_per_cm | 0.596 [0.299, 0.789] | 0.682 [0.337, 0.865] | -1.304 [-3.552, 0.9429] |
| collapse_ratio_pct | 0.136 [-0.250, 0.482] | 0.054 [-0.399, 0.486] | -1.504 [-5.937, 2.93] |
| luminal_radius_mm_median | 0.149 [-0.216, 0.484] | 0.293 [-0.196, 0.657] | -0.6607 [-1.673, 0.3513] |
| distension_quality_score | 0.169 [-0.219, 0.507] | 0.153 [-0.308, 0.558] | 2.046 [-4.52, 8.613] |

Reading it:

- **Ring fat attenuation and fraction agree closely** (ICC 0.96, both
  intervals above 0.9), no bias, limits of agreement -5.7 to 4.7 HU.
- **Part of the gain is spread, not precision.** Between-patient SD of fat
  attenuation is 8.8-9.6 HU in the ring against 3.7-4.9 HU in the sphere, which
  raises the ICC by itself; the limits of agreement narrow only from +/-6.4 to
  +/-5.2 HU. The manuscript says so.
- **Asymmetry does not agree** under either definition; the earlier 0.23/0.53
  came from 35 mm rays on the unsmoothed path. No claim.
- Distension indices are unchanged apart from their intervals.

---

## 2026-09-12 (third): the coverage sensitivity analysis is withdrawn

The author checked Table 4 against the code and the series list. The "whole
colon reached, 19 pairs" column was wrong: `eval_prone_supine.py` kept a series
when its coverage was **unknown** (`c is not None and c > max_unreached`), so
series without an HQColon reference passed the filter. Of the 28 pairs, 11 have
no reference in either position, 10 in one only, and 7 in both; of those 7, both
series meet the coverage criterion in 3.

| Pairs (n = 28) | count |
|---|---:|
| reference in both positions | 7 |
| coverage criterion met in both | 3 |
| no reference in either position | 11 |

Three pairs cannot support ICC, so **no coverage-restricted agreement is
reported**. Table 4 now has one ICC column (28 pairs). The filter requires a
reference and `--min-pairs` refuses small subsets; both are pinned by tests.

The earlier column also carried an obsolete header in the DOCX builder
("ICC(2,1), coverage >= 0.6"), from the withdrawn 0.6 coverage floor. That
inconsistency is what led to the check.
