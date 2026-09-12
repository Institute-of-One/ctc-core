# Centerline failure analysis — 2026-08 batch run

Input: the 60 `metrics.json` files of the mid-August 2026 headless batch run of
the in-house prototype (not distributed; read-only).
Reproduce with:

```
python scripts/analyze_batch_metrics.py --root <prototype batch output> --out results/tables/batch_metrics_2026-08.csv
```

Cohort: 30 TCIA CT COLONOGRAPHY patients x 2 positions = 60 series, resampled to
1.0 mm isotropic. All 60 completed `resample` and `auto_seed`; 5 raised in
`centerline`; `dc` and `phase0` ran on the remaining 55.

## 1. Outcome definition proposed for the manuscript

Judged on extracted centerline length, against a nominal total colon length of
150-200 cm:

| Outcome | Criterion | n | % |
|---|---|---:|---:|
| complete | length >= 1200 mm | 12 | 20.0 |
| partial | 600 <= length < 1200 mm | 29 | 48.3 |
| failed | length < 600 mm, or the stage raised | 19 | 31.7 |

Thresholds are registered in `docs/PARAMETERS.md`.

> **SUPERSEDED 2026-09-10 — the 1200 mm threshold is miscalibrated.** It was
> derived from anatomical colon length (150-200 cm), but a lumen geodesic cuts
> every haustral corner and is systematically shorter. The HQColon reference
> centerlines measure 1186 mm median (IQR 988-1304), so **14 of 26 reference
> standards would themselves score below "complete"**. Measured coverage
> saturates above roughly 900 mm. Every count in this table understates
> performance, and understates the corrected pipeline more than the baseline.
> See `docs/EVALUATION_HQCOLON.md` section 3.

## 2. Failure modes

| Mode | n | % of 60 |
|---|---:|---:|
| (none — complete) | 12 | 20.0 |
| single-component-traversal | 36 | 60.0 |
| insufficient-distension (2.2 — 2 mask artefact, 6 genuine) | 8 | 13.3 |
| fmm-unreachable-end-seed (2.3 — connectivity mismatch, confirmed) | 4 | 6.7 |

### 2.1 single-component-traversal — the dominant mode (36/60)

**This is an algorithm-configuration defect, not an anatomical limitation, and it
is the single highest-value thing to fix.**

`n_ccs_visited == 1` in **55 of 55** successful centerlines, while the air mask
was split into a median of 15 components (IQR 12-22, range 3-65).
`n_bridges == 0` and `used_kimimaro == false` in all 55: the CC-bridge graph —
the component credited in the prototype's notes with 2133 mm against 205 mm for TEASAR — never
executed once in the entire batch.

The cause is a structural interaction between the two stages, visible in the
reference sources:

- `run_cohort_batch.py::auto_detect_seed_endpoints` takes the **largest single
  air component**, skeletonizes only that component, and returns the pair of
  degree-1 skeleton endpoints at maximal physical distance
  (`seed_source == "skeleton_endpoints"` in 60/60 series).
- `auto_centerline.py::auto_extract_centerline` branches on
  `start_cc == end_cc`. When true it sets `augmented_mask = (labels == start_cc)`
  and skips bridging entirely.

Because both seeds are drawn from one component by construction,
`start_cc == end_cc` holds identically, the `else` branch is unreachable, and
the geodesic is confined to whichever single air component happens to be
largest. Median centerline over non-failed series is 988 mm (IQR 874-1224)
against an anatomical 1500-2000 mm — consistent with tracing roughly one
haustral segment chain and stopping at the first collapsed or fluid-filled
segment.

**Fix direction (task 4):** seed detection must span components. Either select
the seed pair across the retained component set (e.g. the two extreme endpoints
of the union skeleton, or an anatomically anchored rectum/caecum pair) so that
`start_cc != end_cc` engages the existing bridge graph, or call the bridge
planner unconditionally on the retained component set. The bridging code needs
no modification — only to be reached.

### 2.2 insufficient-distension (8/60)

Largest contiguous intra-abdominal gas component of 10-93 mL (10 352-93 453
voxels at 1 mm iso) against 1.2-2.5 M voxels in well-distended series. The
distribution is cleanly bimodal with an empty gap between 93 mL and 241 mL, so
the 150 mL threshold does not sit near any observed series.

Affects patients 0018/sec, 0022 (both), 0026 (both), 0027/sec, 0028/pri,
0030/pri. Where only one position is affected the other is often excellent
(0028/sec 1321 mm, 0030/sec 1275 mm), which is the expected prone/supine
complementarity rather than a pipeline defect.

**RESOLVED 2026-09-09 — a mask defect in 2 of the 8; genuine poor
distension in the other 6.**

> **THE SECOND HALF OF THAT VERDICT IS REFUTED (2026-09-10).** Five of the six
> series called "genuine poor distension" are in the HQColon overlap, and the
> reference finds **1.6 to 3.3 litres of colonic gas in every one of them**,
> against 9-53 mL for our mask, with Dice ~0 and recall 0.000 in four of the
> five. Those colons are fully insufflated. This is a **second, unfixed body/air
> mask failure mode**, distinct from the `fullyConnected` defect corrected
> below. **It has since been diagnosed and fixed** (`EVALUATION_HQCOLON.md`
> section 6): no 3-D hole fill can seal a cavity that reaches a volume face or
> merges with exterior air, and an adaptive fall-back to slice-wise filling
> repairs all six series, each recovering a 2-3.5 L colon. **None of the six was
> genuine poor distension.** The
> reasoning that led to the wrong call -- small gas volume, plausible body
> volume, opposite position often well distended -- is recorded here
> deliberately, because it was internally consistent and still wrong.
> See `docs/EVALUATION_HQCOLON.md` section 2.

The suspicion recorded here initially (that 10 mL of contiguous colonic gas is
not a plausible state for a CT colonography acquisition) was correct. The batch
logs already contained the contradiction: for 0022/primary the centerline stage
reported a largest air component of 10 352 voxels while the fat-map stage, on
the same series, reported 3 127 334 air-in-body voxels. The two stages each had
their own body-mask implementation.

Reproducing both on the resampled volume (`scripts/diagnose_mask_failures.py`,
experiment A):

| Body mask variant | Body | Air in body | Largest air CC |
|---|---:|---:|---:|
| `image26` — centerline stage | 18 910.8 mL | 29.0 mL | **10.4 mL** |
| `image6` | 22 300.3 mL | 3 120.6 mL | **2 300.0 mL** |
| `bbox` — fat-map stage | 22 300.3 mL | 3 120.6 mL | **2 300.0 mL** |

`image26` reproduces the batch's 10 352 voxels exactly. The corrected variants
recover a **2.3 L contiguous gas component — a factor of 221**. The colon was
fully insufflated throughout.

**Root cause: one argument.** `sitk.BinaryFillhole(body, fullyConnected=True)`
makes the *background* 26-connected. An enclosed cavity then escapes to the
exterior through any single diagonal gap in the wall surrounding it, and having
escaped it is no longer a hole, so it is never added to the body mask. Since
`air = (CT < -700) AND body`, the gas inside it is discarded.

The isolation is clean: `fullyConnected=False` on the same filter, over the same
full image, reproduces the fat-map result *to the decimal*. So the cause is that
argument alone — not the bounding-box cropping that the fat-map variant also
does, and not the image border (`recovered_gas_touches_border` is false; the
recovered gas never reaches a volume face). `tests/test_masks.py` pins the
mechanism on a synthetic phantom whose cavity is sealed except for a single
3-D diagonal step: `image26` leaves it unfilled, `image6` and `bbox` fill it.

**How far the artefact reaches.** Running experiment A over all eight affected
series plus two controls (`results/tables/mask_diagnosis.csv`) shows the defect
is series-specific, not systematic. Largest air component, mL:

| Series | `image26` (batch) | corrected, closing r=1 | Verdict |
|---|---:|---:|---|
| 0022/primary | 10.4 | **3 173.7** | mask artefact — colon was distended |
| 0028/primary | 93.5 | **2 841.9** | mask artefact — colon was distended |
| 0026/primary | 16.8 | 53.1 | genuine poor distension |
| 0027/secondary | 45.7 | 46.1 | genuine poor distension |
| 0022/secondary | 12.4 | 37.0 | genuine poor distension |
| 0026/secondary | 11.9 | 20.8 | genuine poor distension |
| 0018/secondary | 14.5 | 14.6 | genuine poor distension |
| 0030/primary | 12.1 | 12.2 | genuine poor distension |
| 0001/primary (control) | 1 878.1 | 2 861.8 | unaffected by the fill fix |
| 0008/secondary (control) | 2 154.2 | 2 906.7 | unaffected by the fill fix |

So **2 of 8 were mask artefacts and 6 of 8 are real**: those six retain
12-53 mL of fragmented gas under the corrected mask, against 1.5-3.2 L in
distended series, and in most of them the opposite position is well distended
(0018/pri 2 051 mL, 0030/sec 2 019 mL, 0027/pri 1 132 mL) — the expected
prone/supine complementarity. Body-mask volumes are anatomically plausible
throughout (16.9-52.6 L), so no case is explained by a body mask that grabbed
the wrong structure.

The correction is safe: on both controls `image26` and `image6` agree exactly
(ratio 1.00), so fixing the fill changes nothing on series that already worked.

**A second, independent contributor: the air-mask closing radius.** The
centerline stage used `closing_radius = 0` and the fat-map stage `1`. A radius
of 0 leaves the colon fragmented at haustral folds and fluid menisci into
components the 500-voxel dust filter then deletes. Under the corrected body
mask, going from 0 to 1 (experiment B):

| Series | components (r=0 -> r=1) | largest CC (r=0 -> r=1) |
|---|---|---|
| 0001/primary | 16 -> 8 | 1 878 -> 2 862 mL (**+52 %**) |
| 0021/secondary | 17 -> 11 | 1 629 -> 2 279 mL (+40 %) |
| 0022/primary | 9 -> 5 | 2 300 -> 3 174 mL (+38 %) |
| 0008/secondary | 13 -> 5 | 2 154 -> 2 907 mL (+35 %) |
| 0029/secondary | 18 -> 7 | 1 945 -> 2 403 mL (+24 %) |

On well-distended colons a single voxel of closing recovers 24-52 % more
contiguous lumen and roughly halves the fragment count. On genuinely collapsed
colons it changes almost nothing (1.01-1.17x), so it does not manufacture
distension where there is none.

Consequences:

- `ctc_core.masks.build_body_mask` defaults to the corrected fill and
  `build_air_mask` to `closing_radius = 1`. `image26` is retained only to
  reproduce the 2026-08 batch.
- One body/air mask definition must be shared by all stages. Carrying two
  divergent implementations is what let both defects run undetected for a whole
  cohort, and the contradiction was visible in the batch logs all along.
- The six genuinely poorly distended series are a legitimate, reportable
  limitation of automated CTC analysis and should be described as such, not as
  a pipeline failure.

### 2.3 fmm-unreachable-end-seed (4/60) — CONFIRMED

All five raising series fail identically at `centerline_from_seeds.py:593`,
`ValueError: End seed is unreachable in Fast Marching arrival map`. Four of them
(0021/sec, 0023 both, 0029/sec) had large, well-distended colons (1.5-1.9 L of
gas) and long seed separations (312-437 mm); the fifth (0026/sec) is counted
under section 2.2 by the gas-volume rule but fails by the same mechanism.

This should not be possible within one component: the Fast Marching stopping
value is 1e8 and speed is clipped to >= 0.1 everywhere inside the lumen, so
there is no interior barrier.

**Confirmed 2026-09-09 — connectivity mismatch.** Components are labelled with
`cc3d.connected_components(..., connectivity=26)` and the backtrack walks 26
neighbours, but `sitk.FastMarchingImageFilter` propagates on the 6-connected
(face) stencil. A colon joined only through a corner- or edge-touching voxel is
**one** component to cc3d and **two disjoint regions** to the solver, so the end
seed genuinely has no finite arrival time.

Experiment C reproduces the batch masks, re-derives the automatic seed pair, and
asks whether the two seeds share a component under each connectivity. On the 14
series tested:

| Series | same component at 26 | same component at 6 | centerline stage |
|---|:--:|:--:|---|
| 0021/secondary | yes | **no** | raised |
| 0023/primary | yes | **no** | raised |
| 0023/secondary | yes | **no** | raised |
| 0026/secondary | yes | **no** | raised |
| 0029/secondary | yes | **no** | raised |
| 9 other series | yes | yes | did not raise |

The set of series showing the mismatch is **exactly** the set whose centerline
stage raised — 5 of 5, with no false positives among the 9 others, including
both controls. The hypothesis is confirmed.

**Interaction with the section 2.2 fix.** Re-running experiment C under the
corrected body mask and `closing_radius = 1` removes the mismatch for four of
the five: a one-voxel closing turns the corner contact into a face contact, and
the seeds then share a 6-connected component. **0021/secondary still fails**, so
the closing is not a reliable repair.

**Fix:** make the labelling connectivity agree with the solver. Either label the
air mask with `connectivity=6`, or replace the Fast Marching solver with one
whose stencil matches the labelling. Labelling at 6 is the smaller change and is
the conservative direction — it can only split components that the solver could
not traverse anyway, and any resulting split is then handled by the CC-bridge
graph of section 2.1 rather than by a crash. Note that the backtrack's 26-step
neighbourhood must be reconciled at the same time.

This is a general hazard, not a quirk of this codebase: pairing a 26-connected
component labeller with a 6-connected PDE solver is a mismatch that stays silent
until a single diagonal voxel contact decides a case. It is worth stating
explicitly in the manuscript's Methods.

## 3. Consequence for the evaluation plan

Prone/supine agreement (evaluation pillar 3) needs both positions of the same
patient to be measurable:

| Requirement | Patients |
|---|---:|
| both positions complete | 3 / 30 |
| both positions complete or partial | 15 / 30 |
| both positions produced any centerline | 26 / 30 |

Three paired observations cannot support ICC or Bland-Altman.

**Superseded by section 5.2.** The table above is the 2026-08 batch. Under the
corrected pipeline, both positions are usable for **22 of 30** patients and all
30 produce a centerline in both positions, which is enough for the prone/supine
pillar at n=30. The cohort extension is therefore no longer blocked by this.

## 4. Runtime (for the Methods section)

Per stage, seconds, over all series that reached the stage:

| Stage | n | median | mean | max |
|---|---:|---:|---:|---:|
| resample | 60 | 17.9 | 26.4 | 159.2 |
| auto_seed | 60 | 14.5 | 14.8 | 24.4 |
| centerline | 60 | 18.0 | 18.3 | 31.9 |
| dc | 55 | 20.9 | 23.8 | 78.5 |
| phase0 | 55 | 67.8 | 72.8 | 122.0 |
| **total** | 55 | **148.7 (2.5 min)** | — | — |

Headless, single workstation, 1.0 mm isotropic. Note that these timings come
from the truncated centerlines of section 2.1; the fixed pipeline will trace
longer paths and `phase0` cost scales with centerline length, so they are a
lower bound and must be re-measured, not reused.

## 5. Applying the fixes (2026-09-10)

All three fixes were implemented in `ctc_core/centerline.py` as fields of
`CenterlineConfig`, and the whole cohort was run under both presets.
`CenterlineConfig.batch_2026_08()` reproduces the reference batch exactly
(0001/primary: 936.33 mm, 678 points, 16 components, 1 visited, 0 bridges,
largest air CC 1 878 117 voxels -- every field of the reference `metrics.json`),
and reaches the same 55/60 success count, so the port is sound and the
comparison below is attributable to the fixes alone.

```
python scripts/run_centerline.py --config batch     --raw-root .                --summary results/tables/centerline_batch.csv
python scripts/run_centerline.py --config corrected --raw-root .                --summary results/tables/centerline_gated.csv
python scripts/compare_centerline_configs.py --baseline results/tables/centerline_batch.csv --candidate results/tables/centerline_gated.csv --out results/tables/centerline_comparison.csv
```

### 5.1 An intermediate result that must not be reported

Before bridge admissibility was gated, the corrected pipeline looked
transformative: complete 12 -> 29 (48.3 %), failed 19 -> 5, median length
908.5 -> 1218.8 mm, bridges running on 53 of 60 series instead of none.

The bridges themselves invalidate it. Across 117 bridges the median wall
fraction was **0.639** -- the typical bridge spent more of its length inside
solid tissue than in lumen -- with 73 of 117 above 0.5 and a worst case of
0.978. Median alignment was 0.316 against a 1.0 ideal. Of the 29 series scoring
complete, **only 4 avoided a bridge above 0.5**, and all four traces exceeding
2000 mm (longer than an anatomical colon) depended on a bridge at 0.96-0.98.
Those paths are not colon centerlines.

The cause is structural, not a mis-set weight. The cost function penalises wall
fraction, but Dijkstra must connect the two seeds, so when the only chain runs
through tissue it takes it at whatever cost. A soft penalty inside a search that
is required to succeed cannot refuse.

`max_bridge_wall_fraction` (default 0.5) therefore removes such edges from the
graph outright, and `fall_back_to_largest_component` traces within the largest
retained component when no admissible chain exists, so a fragmented colon is
reported as partial coverage rather than stitched through solid organs.

### 5.2 What the fixes actually delivered

| | batch 2026-08 | corrected + gated |
|---|---:|---:|
| complete (>= 1200 mm) | 12 (20.0 %) | 10 (16.7 %) |
| partial (600-1200 mm) | 29 (48.3 %) | 40 (66.7 %) |
| failed | 19 (31.7 %) | **10 (16.7 %)** |
| series raising an exception | 5 | **0** |
| length median | 908.5 mm | 1015.1 mm |
| length max | 1614 mm | 1706 mm |
| paths > 2000 mm (implausible) | 0 | **0** |
| bridges drawn | 0 / 60 | 16 / 60 |
| worst bridge wall fraction | — | median 0.000, **0 series above 0.5** |
| components visited (median) | 1 | 1 |
| both positions usable | 15 / 30 | **22 / 30** |
| both positions produced a centerline | 26 / 30 | **30 / 30** |

**Robustness and validity improved; coverage did not.** Every series now
produces a centerline, hard failures are gone, failures by length are halved,
no path is anatomically implausible, and no bridge cuts through tissue. Paired
availability rises from 15 to 22 patients, which is what unblocks evaluation
pillar 3.

> **PARTLY SUPERSEDED 2026-09-10.** The "coverage did not improve" conclusion
> was an artefact of the miscalibrated 1200 mm threshold (section 1). Against a
> reference-calibrated 900 mm threshold the corrected pipeline goes from 28 to
> 38 of 60, where the baseline goes from 12 to 28 -- so coverage did improve,
> and the old threshold hid it. Measured coverage against HQColon is 0.940
> (IQR 0.782-0.982) on the 21 series where segmentation succeeds.
> The validity conclusions stand and are now externally confirmed: 97.1 % of
> centerline points fall inside the reference colon.
> See `docs/EVALUATION_HQCOLON.md`.

But `complete` fell from 12 to 10, the median length rose by only 107 mm
(+12.9 mm across the 55 paired series, 31 improved and 24 worsened), and the
median number of components visited is still **1**: only 16 of 60 series obtain
any admissible bridge, and 39 fall back to a single component. The three fixes
made the pipeline honest and robust. They did not make it cover the colon.

Bridging, where it is admissible, mostly rescues bad cases rather than extending
good ones: the largest gains are 0028/pri +548 mm, 0018/sec +547 mm,
0012/sec +417 mm and 0019/sec +318 mm, all from short baselines, while four
series lose ground (0013/sec -214 mm, 0028/sec -160 mm). Median length among
bridged series (983 mm) is no higher than among non-bridged (1015 mm).

### 5.3 Why coverage is still unsolved

Two design faults remain, and neither is a threshold that can be tuned.

**The traversal objective is wrong.** `plan_cc_chain_bridges` runs Dijkstra, so
it finds the *cheapest* chain from start to end. Colon centerline extraction
needs the path that *covers the colon*, which is close to the opposite
objective. This is why 39 series decline to bridge at all and why bridged series
are no longer than unbridged ones.

> **NOT SUPPORTED (2026-09-11).** This was inferred from component counts before
> any coverage measurement existed. Measured against HQColon, whose reference
> mask is a single connected component spanning the whole colon, the shortest-
> path configuration covers 93.7 % of the reference centerline with a median of
> one component visited: the largest gas component usually *is* most of the
> colon. A coverage-maximising traversal was built and measured, and it was
> worse on every quantity (coverage 0.575, path-to-reference 25.0 mm against
> 4.9 mm). See `EVALUATION_HQCOLON.md` section 7. The real limitation is
> narrower: coverage fails badly in 4 of 26 series, which are individual cases
> to diagnose, not a systematic fault.

**The seeds are not anatomically anchored.** They are the most distant pair of
skeleton endpoints, not rectum and caecum. The end-to-end straight distance
reaches 486 mm (0023/primary), well beyond the 25-30 cm that separates rectum
from caecum, which means at least one seed sits outside that span -- plausibly
in stomach or small bowel. Tortuosity gives the same signal from the other
direction: the median is 2.59 and the maximum 6.55, whereas a full colon
(L 150-200 cm over D 25-35 cm) should sit around 4-8. Our traces are too
straight for their span, which is what partial coverage looks like.

Four series with substantial gas are still traced far too short, and these are
pure algorithm failures rather than anatomy: 0015/pri (1 967 mL of gas, 524 mm),
0024/sec (1 431 mL, 398 mm), 0015/sec (649 mL, 238 mm), 0021/pri (449 mL,
244 mm). The other six of the ten remaining failures have 12-53 mL of gas and
are the genuinely poorly distended series of section 2.2.

### 5.4 Consequence for the manuscript

Length is a proxy that has now been shown to fail in both directions: it
rewarded tissue-cutting bridges with "complete" scores, and it cannot tell a
partially covered colon from a fully covered one. **No accuracy claim can rest
on it.** The HQColon comparison (evaluation pillar 1) is therefore not an
optional strengthening of the paper but a precondition for making any claim at
all, because it measures coverage against an external reference instead of
against a length threshold.

What the current results *can* support, and what the manuscript should be built
around, is the methodological content: two mask defects and a connectivity
mismatch found and fixed with controlled experiments, the demonstration that a
soft penalty inside a mandatory search fabricates anatomically invalid paths,
and a pipeline that produces a result for 60 of 60 series with no implausible
output. That is a genuine and reportable contribution; "48 % complete" is not.

## 6. Next actions

Sections 2.1 to 2.3 are closed: all three defects are diagnosed, fixed, and
their effect measured over the full cohort (section 5). What remains:

1. ~~**HQColon comparison (next step).**~~ **DONE 2026-09-10** --
   `docs/EVALUATION_HQCOLON.md`. It refuted two conclusions in this document
   (sections 1 and 2.2 above) and confirmed the bridge admissibility gate
   externally. The replacement top priority is the mask failure it exposed:
   5 of 26 reference series lose a 1.6-3.3 L colon completely.
2. **Replace the traversal objective** (section 5.3). Dijkstra minimises path
   cost; the task needs maximum colon coverage. Candidates: a coverage-maximising
   walk over the component graph, or a spanning traversal that visits every
   admissible component rather than the cheapest chain. This is a redesign, not
   a retune.
3. **Anchor the seeds anatomically** (section 5.3). Rectum and caecum rather
   than the most distant pair of skeleton endpoints, whose 486 mm span shows at
   least one seed leaving the colon.
4. Re-derive the `CTI` bands, which are inconsistent with their own definition
   (`docs/PARAMETERS.md`).
5. Only then re-decide the cohort extension to ~100 patients. Paired
   availability is now 22/30, which is enough for the prone/supine pillar at
   n=30, so extension is no longer blocking.

Deliberately **not** claimed: any accuracy or completeness figure. The
`complete` rate under the corrected pipeline (16.7 %) is a length proxy, not a
validated measure, and it is lower than the batch's for reasons section 5.2
explains.
