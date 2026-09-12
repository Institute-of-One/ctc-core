# External validation against HQColon — 2026-09-10

Evaluation pillar 1. First measurement of this pipeline against a reference
standard rather than against a length proxy. Reproduce with:

```
python scripts/download_hqcolon.py --out data/hqcolon
python scripts/hqcolon_overlap.py --out results/tables/hqcolon_overlap.csv
python scripts/eval_hqcolon.py --overlap results/tables/hqcolon_overlap.csv --out results/tables/eval_hqcolon.csv
```

Reference: HQColon (Finocchiaro et al., Sci Data 13:199, 2026,
doi:10.1038/s41597-025-06518-z; data doi:10.17605/OSF.IO/8TKPM), CC BY-NC-ND
4.0, evaluation use only. Its `meta-data.json` is JSON Lines keyed by
`InstanceUID`, which is the TCIA SeriesInstanceUID, so it joins to
`cohort/manifest.csv` directly. **26 of our 60 series** have a reference mask
(18 of 30 patients, 8 with both positions).

Masks are on the native DICOM grid but share our physical frame exactly
(identical origin, identity direction), so they are brought onto our 1.0 mm
isotropic grid by identity-transform nearest-neighbour resampling. The two
archives expand to 62 GB each, so each mask is extracted on demand and deleted;
peak extra disk is about 350 MB.

The reference centerline is built from the reference mask by **our own
procedure** — most distant pair of Lee-skeleton endpoints, then a Fast Marching
geodesic — so lengths and paths are compared like-for-like.

---

> **Reading order.** Sections 1 to 5 record the first measurement, before the
> segmentation failure was fixed. Section 6 diagnoses and fixes it and carries
> the current numbers; where the two disagree, section 6 is authoritative.

## 1. Headline: the pipeline separates cleanly into two failure classes

| Class | n / 26 | Signature |
|---|---:|---|
| Segmentation succeeded | 21 | Dice > 0.5 |
| **Segmentation failed** | **5** | Dice ~ 0, our mask misses the colon entirely |

On the 21 series where segmentation succeeded:

| Metric | Median | IQR |
|---|---:|---|
| Dice (our lumen vs reference gas) | **0.935** | 0.861–0.973 |
| Recall | **0.995** | 0.994–0.996 |
| Precision | 0.881 | 0.758–0.951 |
| Centerline points inside the reference colon | **0.971** | 0.950–0.982 |
| Our centerline length / reference length | **0.979** | 0.940–1.071 |
| Reference centerline covered within 10 mm | **0.940** | 0.782–0.982 |
| Mean symmetric surface distance | 2.50 mm | 0.51–4.48 |
| Mean distance, our path to reference path | 2.95 mm | 1.50–6.55 |

Two of these deserve emphasis. **Length ratio 0.979** means that where
segmentation works, our centerline reproduces the reference length to within
about 2 %. And **97.1 % of centerline points lie inside the reference colon**,
which is the direct external confirmation that the bridge admissibility gate of
`FAILURE_ANALYSIS.md` section 5.1 did its job: the ungated pipeline would have
placed points inside solid tissue, and no length criterion could have detected
that.

Precision (0.881) is consistently below recall (0.995). Our air mask contains
essentially all of the reference colon and then some — small bowel and stomach
gas. The error is over-inclusion, not omission, which is the more tractable
direction.

---

## 2. Correction: the "genuine poor distension" group was wrong

`FAILURE_ANALYSIS.md` section 2.2 concluded that 2 of 8 apparent distension
failures were mask artefacts and **6 were genuine poor distension**, reasoning
that 12–53 mL of contiguous gas under the corrected mask must reflect the
patient. Five of those six are in the HQColon overlap, and **the reference
refutes all five**:

| Series | HQColon gas | HQColon gas+fluid | Our lumen | Our air | Dice |
|---|---:|---:|---:|---:|---:|
| 0026/primary | **3 344.7 mL** | 3 485.6 | 53.1 | 157.4 | 0.000 |
| 0026/secondary | **3 223.9 mL** | 3 362.4 | 38.0 | 109.4 | 0.001 |
| 0018/secondary | **2 649.3 mL** | 2 902.5 | 9.0 | 60.5 | 0.002 |
| 0030/primary | **1 802.3 mL** | 1 952.1 | 12.2 | 34.8 | 0.000 |
| 0022/secondary | **1 614.6 mL** | 1 798.3 | 37.0 | 139.5 | 0.000 |

Every one of these colons holds 1.6 to 3.3 litres of gas. They are fully
insufflated. Recall is 0.000 in four of the five: our mask does not merely
undershoot, it has **no overlap at all** with the reference colon.

So the conclusion to record is the opposite of section 2.2's: this was a
**second body/air mask failure mode**, distinct from the `fullyConnected`
defect, affecting 5 of 26 reference series (19 %). It is diagnosed and fixed in
section 6 below.

The sixth case in that group, 0027/secondary, is not in the HQColon overlap, so
this note originally recorded it as unverified, with the prior that it was the
same defect. The fix confirms that prior: its gas goes from 46 mL to 2 378 mL.
**None of the six was genuine poor distension.**

This is the clearest possible vindication of refusing to publish a number before
the external reference existed. The internal evidence — small gas volume,
plausible body-mask volume, the opposite position often well distended — pointed
confidently at patient anatomy, and it was wrong. It was the highest-priority
defect, and section 6 resolves it.

---

## 3. Correction: the completeness threshold was miscalibrated

`FAILURE_ANALYSIS.md` section 1 set complete >= 1200 mm, justified by a nominal
anatomical colon length of 150–200 cm. The reference standard's own centerlines
measure:

| Reference | Median | IQR | Range |
|---|---:|---|---|
| gas-only | **1 186 mm** | 988–1 304 | 794–1 426 |
| gas+fluid | 1 188 mm | 968–1 320 | 745–1 426 |

**14 of the 26 reference centerlines would themselves score below "complete".**
The threshold is not slightly off; it is calibrated against the wrong quantity.
Anatomical colon length is measured along the bowel, including haustral
undulation, whereas a Fast Marching geodesic runs down the lumen axis and cuts
every corner, so it is systematically shorter. A rectum-to-caecum lumen geodesic
in this collection is about 1 200 mm, not 1 500–2 000 mm.

Coverage as a function of our length, on the segmentation-OK series, shows where
the real knee is:

| Our length | n | Median coverage |
|---|---:|---:|
| 600–900 mm | 4 | 0.419 |
| **900–1 100 mm** | 7 | **0.982** |
| 1 100–1 400 mm | 9 | 0.954 |
| > 1 400 mm | 1 | 0.993 |

Coverage saturates above roughly **900 mm**, not 1 200 mm. Nine series labelled
"partial" by the old threshold have coverage >= 0.92, including 0003/secondary at
coverage 1.000 with a 1 015 mm centerline.

Consequences for the cohort figures, both thresholds shown:

| Threshold | batch 2026-08 | corrected + gated |
|---|---:|---:|
| >= 1200 mm (old, anatomical) | 12 complete (20.0 %) | 10 complete (16.7 %) |
| >= 900 mm (reference-calibrated) | 28 complete (46.7 %) | **38 complete (63.3 %)** |

(After the section 6 mask fix this becomes **44 of 60 (73.3 %)**; the caveat
below applies to that figure equally.)

The old threshold understated both configurations and, worse, understated the
corrected pipeline more than the baseline, which is how section 5.2 came to
report that "coverage did not improve". With a threshold calibrated against the
reference, the corrected pipeline moves from 28 to 38 of 60.

**This does not license quoting 63.3 %.** Length remains a proxy, Spearman
correlation between length and measured coverage is only 0.664 (p = 0.0002), and
the recalibration was derived from the same 26 series it would be used to
describe. The manuscript should report **measured coverage on the 26** as the
result and use the length band only to describe the remaining 34, explicitly
labelled as a proxy.

---

## 4. The residual traversal failures

Four series have good segmentation but a path that does not traverse it — the
`FAILURE_ANALYSIS.md` section 5.3 objective-function fault, now measured
directly:

| Series | Dice | Length ratio | Coverage |
|---|---:|---:|---:|
| 0031/primary | 0.879 | 1.316 | **0.037** |
| 0028/primary | 0.973 | 0.548 | 0.154 |
| 0014/primary | 0.721 | 0.731 | 0.304 |
| 0009/primary | 0.807 | 0.570 | 0.533 |

0028/primary is the cleanest illustration: Dice 0.973 means the colon is
segmented almost perfectly, and the path still covers only 15 % of it. No
segmentation improvement can fix that; it is the traversal objective.

0031/primary needs a caveat of its own: coverage against the gas-only reference
is 0.037 but against gas+fluid it is 0.998. The gas-only reference mask is
fragmented there, so its own largest component — and therefore its reference
centerline — sits somewhere our path does not. Where the two references disagree
this sharply, the gas+fluid one is the more trustworthy, and single-reference
coverage figures for fragmented colons should be read with care.

---

## 5. What is now claimable, and what is next

Claimable, with the reference behind it:

- Dice 0.935 (IQR 0.861–0.973), recall 0.995, precision 0.881 on 21 of 26 series.
- Centerline length within 2 % of reference (ratio 0.979).
- 97.1 % of centerline points inside the reference colon, confirming the
  admissibility gate externally.
- Coverage 0.940 (IQR 0.782–0.982).
- A clean, quantified separation of segmentation failure (5/26) from traversal
  failure (4/26).

Next, in priority order:

1. ~~**The mask failure of section 2 above.**~~ **DONE** — section 6.
   All six affected series recover a 2–3.5 L colon; segmentation failures go
   from 5 of 26 to 0 of 26 with no series degraded.
2. Replace the traversal objective (section 4 here, section 5.3 there). This is
   now the only substantial defect left.
3. Recompute every completeness figure in `FAILURE_ANALYSIS.md`
   against the reference-calibrated threshold, with the proxy
   caveat attached.
4. Extend the cohort toward HQColon overlap: HQColon covers 315 subjects, 113
   with both positions, against our current 18 and 8.

---

## 6. Diagnosis and fix of the segmentation failure (same day)

The gas is unambiguously gas -- 99.4 % of the reference colon lies below -700 HU
-- yet only 0.1 % of it fell inside our body mask on 0026/primary and 0.0 % on
0030/primary, against 100 % on 0001/primary. Geometry was excluded first:
origins and extents match exactly, and every download directory holds one series
whose UID matches the manifest.

**A 3-D hole fill cannot help when the cavity is not a hole.** Labelling the
background before filling shows the component containing the colon:

| Series | Component holding the colon | Reaches a volume face |
|---|---:|:--:|
| 0026/primary | 52 510 mL -- merged with exterior air | **yes** |
| 0030/primary | 3 002 mL -- cut open by the field of view | **yes** |
| 0001/primary | 2 988 mL -- enclosed | no |

No choice of background connectivity or bounding box changes that, which is why
the earlier `fullyConnected` correction could not have caught it.

### 6.1 Slice-wise filling alone is a trade, not a fix

Within an axial slice the body wall encircles the colon, so 2-D filling seals
what 3-D filling cannot: it recovers 100 % of the reference colon on all five
failing series. But it also admits gas that is enclosed in-plane and is not
colon, and measured against the reference it **degraded 11 of the 21 series that
already worked**:

| Metric (median over 26) | 3-D `bbox` | 2-D `slicewise` |
|---|---:|---:|
| Dice | 0.878 | 0.801 |
| Precision | 0.786 | 0.670 |
| Centerline inside colon | 0.964 | 0.883 |
| Mean surface distance | 3.64 mm | 7.11 mm |

Swapping one for the other would have traded one defect for another.

### 6.2 The choice can be made adaptively, without a reference

Under 3-D filling the failing series retain 12-53 mL of gas while every
succeeding series retains 1.5-3 L: the groups separate by more than an order of
magnitude, so the failure is detectable from the image alone.
`build_body_air_masks` therefore tries the 3-D fill and falls back to slice-wise
only when the result holds no plausible colon, reusing the 150 mL threshold
already registered for that judgement. It records which fill was used and why.

Six of the 60 series take the fallback -- exactly the six of the original
"insufficient distension" group -- and every one recovers a 2-3.5 L colon.

| Metric (median over 26) | `bbox` | `slicewise` | **`auto`** |
|---|---:|---:|---:|
| Dice (lumen vs gas) | 0.878 | 0.801 | **0.882** |
| Precision | 0.786 | 0.670 | **0.793** |
| Recall | 0.995 | 0.995 | **0.995** |
| Centerline inside colon | 0.964 | 0.883 | **0.964** |
| Reference coverage | 0.919 | 0.936 | **0.937** |
| Length ratio | 0.973 | 1.022 | **1.001** |
| Mean surface distance | 3.64 mm | 7.11 mm | **3.31 mm** |
| **Segmentation failures** | **5 / 26** | 0 / 26 | **0 / 26** |

Per series: **5 fixed, 0 degraded, 21 unchanged within 0.02 Dice.** The adaptive
rule is at least as good as the better of the two on every metric.

The same change removes the structural cause of this family of bugs. The
workbench carried two divergent body/air implementations, one per stage, and
their disagreement is what let a whole cohort lose the colon unnoticed;
`build_body_air_masks` is now the single entry point for every stage.

### 6.3 Where the pipeline now stands

External validation over the 26 reference series:

| Metric | Median | IQR |
|---|---:|---|
| Dice (our lumen vs reference gas) | **0.882** | 0.810-0.962 |
| Recall / precision | 0.995 / 0.793 | |
| Centerline points inside reference colon | **0.964** | 0.907-0.981 |
| Our length / reference length | **1.001** | 0.944-1.108 |
| Reference centerline covered within 10 mm | **0.937** | 0.718-0.978 |
| Mean symmetric surface distance | 3.31 mm | 0.64-6.16 |
| Segmentation failures | **0 / 26** | |

Cohort of 60, against the 2026-08 batch:

| | batch | now |
|---|---:|---:|
| produced a centerline | 55 / 60 | **60 / 60** |
| failed (< 600 mm) | 19 (31.7 %) | **4 (6.7 %)** |
| complete, >= 900 mm reference-calibrated | 28 (46.7 %) | **44 (73.3 %)** |
| length median | 908.5 mm | 1 054.6 mm |
| paths > 2000 mm | 0 | 0 |
| bridges with wall fraction > 0.5 | -- | 0 |
| both positions usable | 15 / 30 | **27 / 30** |
| both positions produced a centerline | 26 / 30 | **30 / 30** |

**Still unsolved: the traversal objective.** The median number of components
visited is 1, 36 of 60 series fall back to the largest component, and the
coverage IQR still reaches down to 0.718. 0028/primary remains the clearest
case: Dice 0.973 against coverage 0.154 -- near-perfect segmentation that the
path does not traverse. That is `FAILURE_ANALYSIS.md` section 5.3, and no
segmentation work will fix it.

---

## 7. A traversal redesign, tested and rejected (2026-09-11)

Section 6.3 closed by naming the traversal objective as the last substantial
defect, on the reasoning that a median of one component visited must mean a
partially covered colon. **That reasoning was wrong, and the redesign built on
it made every measured quantity worse.** Both are recorded here because the
error is instructive and because the rejected code remains available.

### 7.1 The premise was false

The HQColon reference mask is a **single connected component covering 100 % of
the colon**, in both the gas-only and the gas+fluid variant (checked on eight
series: largest component 99.9-100 % of mask volume). The reference centerline
therefore spans the whole colon, and the coverage figure of 0.937 already
measured whole-colon coverage — achieved with a median of one component visited.

The largest gas component usually *is* most of the colon. Counting components is
not a measure of coverage, and where the two disagree the measurement wins.

### 7.2 What was built, and what it cost

`plan_coverage_traversal` maximises traversed lumen volume: drop components below
a volume floor, take the connected sub-graph of greatest total volume, reduce it
to its minimum spanning tree, return the node-weighted diameter. Two supporting
changes came with it, each forced by a phantom:

- the wall fraction measured over the **extra-luminal part** of a bridge rather
  than the whole segment, so a thick barrier cannot be diluted by lumen the
  segment runs along;
- an absolute **wall-thickness** gate, because the gap-only fraction is
  near-absolute — colon fragmentation is collapsed segments, so every real
  bridge crosses some tissue;
- terminal seeds from a **two-pass geodesic diameter** instead of from where the
  bridges attach.

Measured on the 26 reference series, against the shortest-path configuration:

| Metric (median) | shortest path | max coverage, gate 6 / 12 / 25 mm |
|---|---:|---:|
| Reference coverage | **0.937** | 0.575 / 0.575 / 0.575 |
| Centerline inside colon | **0.964** | 0.842 / 0.842 / 0.830 |
| Length ratio | **1.001** | 0.770 / 0.770 / 0.821 |
| Mean path-to-reference distance | **4.9 mm** | 25.0 / 25.0 / 25.0 mm |
| Series with any bridge | 16 / 60 | 2 / 26 |

### 7.3 Why each piece failed

**The gap-only wall fraction emptied the graph.** It is the principled measure,
but in practice almost no bridge survives it: `traversal_cluster_components` is 1
for 24 of 26 series, so the coverage traversal had nothing to traverse and
degenerated to the largest component. That the 6, 12 and 25 mm thickness gates
gave *identical* results was the tell — the thickness gate was never the binding
constraint.

**The double sweep picks the wrong endpoints for this metric.** Fast Marching
speed is `d_wall^k`, so arrival time is dominated by narrow passages and the
"farthest" voxel is whatever lies beyond the tightest constriction, not the
anatomically most distant point. The two-pass diameter construction is valid
where distance tracks geometric length; under a speed-weighted metric it is not.
On 0001/primary it gave 773 mm and coverage 0.319 where the max-distance
skeleton endpoint pair gave 1 045 mm and 0.984.

### 7.4 Disposition

Defaults are reverted to the configuration that measures best — shortest path,
whole-segment wall fraction gated at 0.5, adaptive masks — and the revert
reproduces the earlier numbers exactly, all eight summary medians unchanged to
three decimals with no series moving by more than 0.01.

The rejected machinery is kept, tested and selectable: `traversal="max_coverage"`,
`wall_frac_over_gap_only=True`, `max_bridge_wall_mm`. Each carries the measured
reason it is off by default, so this ground does not get re-covered.

### 7.5 What this changes in the manuscript

`FAILURE_ANALYSIS.md` section 5.3 claims coverage is limited by the traversal
objective. It is **not supported**: that claim was inferred from component counts
before any coverage measurement existed, and the measurement contradicts it. It
is corrected in place.

The real remaining limitation is narrower and should be stated as such: coverage
is high in most series (median 0.937) but fails badly in a few. Four of 26 fall
below 0.6 — 0031/primary 0.037, 0028/primary 0.154, 0014/primary 0.304,
0009/primary 0.533 — and these are individual failures to be diagnosed
case by case, not a systematic objective-function fault.


---

## 8. The seed rule truncates both centerlines -- coverage was not whole-colon (2026-09-11)

Found while drawing an illustration of case 0007-1: the path does not enter the
right colon, although that gas is part of the lumen. **Section 7.1's conclusion
("the reference centerline spans the whole colon") is REFUTED**, and with it the
reading of coverage 0.937 and length ratio 1.001 as whole-colon results.

Both centerlines take their seeds from the most distant pair of skeleton end
points by *Euclidean* distance. In a colon those are typically the rectum and a
flexure, not the rectum and the caecum, so the geodesic between them leaves a
large part of the colon as a side branch. The reference centerline is built by
the same rule, so the two agree while missing the same segments.

Direct check, fraction of the reference colon (gas and fluid) lying more than
60 mm from a path -- a distance no lumen width explains
(`scripts/diagnose_centerline_extent.py`, `scripts/diagnose_diameter_seeds.py`):

| Series | Reference, current rule | Reference, geodesic-diameter seeds | Ours, current rule |
|---|---|---|---|
| 0007-1 | 1222 mm, 0.48 | 1890 mm, 0.00 | 1193 mm, 0.48 |
| 0003-2 | 900 mm, 0.60 | 1666 mm, 0.00 | 1015 mm, 0.57 |
| 0001-1 | 933 mm, 0.52 | 1850 mm, 0.00 | 1045 mm, 0.52 |
| 0004-2 | 1177 mm, 0.39 | 1780 mm, 0.00 | 1488 mm, 0.23 |
| 0030-1 | 1151 mm, 0.61 | 2133 mm, 0.00 | 1214 mm, 0.61 |

Seeds at the two ends of the longest geodesic with **uniform** speed (arrival
time = path length inside the mask) make the reference centerline span the whole
colon in all five. Section 7's rejected attempt used the wall-distance-weighted
arrival time for the same search, which is dominated by narrow passages; the
uniform-speed version does not have that defect.

For our pipeline the new seeds help only where the colon lies in one gas
component (0003-2, 0030-1 fixed; 0007-1 and 0004-2 unchanged; 0001-1 worse),
because the lumen passed to the geodesic contains only the components chained
between the old seeds.

Consequences, to be worked through before any submission:

1. Pillar 1's centerline metrics must be recomputed against a whole-colon
   reference centerline (geodesic-diameter seeds), with a volume-based coverage
   (reference colon within 30 mm of the path; beyond 60 mm) as the primary
   measure. The 900 mm completeness threshold, calibrated on truncated
   reference paths, is void.
2. The pipeline's seeding must reach the whole colon across components:
   geodesic-diameter seeds over the bridged union of colonic components.
3. Every centerline-derived index (traced length, fat profile, collapse ratio,
   gas volume restricted to path components) is affected and must be re-run.


### 8.1 After the fix: the whole-colon seed rule (re-run 2026-09-11/12)

`seed_rule="whole_colon"` (`ctc_core/centerline.py`; registry entry in
`docs/PARAMETERS.md`): the lumen is built as before, and the final path is the
candidate -- provisional pair, rectal end to the farthest voxel along the lumen,
or rectal end to the farthest voxel of the largest wide-core component -- that
brings most of the lumen within 30 mm of the path. Evaluated first on all 26
reference series from the existing lumens (`scripts/diagnose_seed_rules.py`,
`results/tables/seed_rule_comparison.csv`), then by a full re-run whose values
match that evaluation exactly.

| Measure (26 reference series) | Before | After |
|---|---:|---:|
| Colon (gas and fluid) within 30 mm of the path | 0.426 | **0.860** [0.575, 0.961] |
| Colon beyond 60 mm (never visited), median | 0.427 | **0.005** [0.000, 0.212] |
| Whole colon reached (< 5 % beyond 60 mm) | 0 / 26 | **15 / 26** |
| Path points inside the colon | 0.964 | **0.980** |
| Length / whole-colon reference length | -- | 0.969 [0.725, 0.995] |
| Whole-colon reference centerline | (truncated) | 1907 mm [1792, 1976] |

Neither candidate alone suffices (10/26 and 9/26 reached); the reference-free
selection equals the best candidate in 25/26 and is never worse than the old
path. Eleven series remain partially covered, four with more than 30 % of the
colon beyond 60 mm; in the series inspected the cause is a shortcut where loops
touch or small bowel joins the colon gas, which turns the lumen into a loop
without anatomical ends.

The gas-volume index was validated against the reference gas-filled volume on
the same series: r = 0.841, median relative error +2.2%
(`scripts/diagnose_gas_volume.py`).

Slice-wise fall-back lumens contain 1-2 L of non-colonic gas, including lung
bases joined to the splenic flexure by the gas closing (precision 0.46-0.82 on
the five fall-back series with a reference; Figure 4). The path and the
gas-volume index, which uses unclosed gas, are not affected.
