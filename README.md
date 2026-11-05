# ctc-core

[![ci](https://github.com/Institute-of-One/ctc-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Institute-of-One/ctc-core/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22720528.svg)](https://doi.org/10.5281/zenodo.22720528)

Minimal, headless algorithm core for CT colonography (CTC) analysis, and the
reproducible evaluation behind the IORN-012 manuscript.

- **Masks**: body and gas masks with an adaptive hole fill (3-D, falling back to
  slice-wise when the 3-D fill retains no plausible colon).
- **Centerline**: a connected-component bridge graph with a wall-fraction
  admissibility gate, and a Fast Marching geodesic with a wall-distance speed
  between seeds chosen, without reference, to trace the whole colon.
- **Pericolonic fat map**: fat attenuation, fraction and volume along the
  centerline, and an anatomically anchored (arc length, angle) map with a
  left-right asymmetry index.
- **Distension and preparation descriptors**: gas volume, length-adjusted gas,
  collapse ratio, luminal radius, traced centerline length, tortuosity.
- **Polyp measurement**: spherical VOI, HU segmentation, maximum diameter and
  volume from a grayscale iso-surface, sphericity, shape index.
- **Agreement statistics**: ICC(2,1) and Bland-Altman limits of agreement.

No GUI, by design. Everything runs on one workstation.

## Data

Public data only. No image data, mask or per-case output is stored here;
`.gitignore` enforces that.

- **TCIA CT COLONOGRAPHY** (CC BY 3.0): the evaluation cohort, 30 patients /
  60 series, listed by Series Instance UID in `cohort/manifest.csv`.
  Smith K, Clark K, Bennett W, Nolan T, Kirby J, Wolfsberger M, Moulton J,
  Vendt B, Freymann J (2015) Data From CT COLONOGRAPHY. The Cancer Imaging
  Archive. https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1
- **HQColon** (CC BY-NC-ND 4.0): external reference segmentations, used for
  evaluation only. Never redistributed; no derived mask is published, only
  summary statistics. https://doi.org/10.17605/OSF.IO/8TKPM

## Install

Python 3.10-3.12. On Windows:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e ".[tcia,figures,dev]"
```

On Linux or macOS:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e ".[tcia,figures,dev]"
```

`requirements-lock.txt` pins the versions every committed result was produced
with. Continuous integration runs both blocks above, exactly as written, on a
machine with nothing installed, and then runs the test command below: a working
tree carries an editable install from months ago, so instructions that no
longer work stay green there indefinitely.

`scripts/run_all.py` checks the installed versions against the lock file before
the first stage and stops if they differ, because a minor release of an array
or plotting library can change a committed result without anything in this
repository changing:

```
python scripts/check_environment.py --all
```

## Reproduce

One entry point regenerates every table and figure of the manuscript, in
dependency order:

```
python scripts/run_all.py --raw-root . --download
```

`--download` fetches the 60 TCIA series (about 19 GB), the HQColon masks
(423 MB) and the TCIA polyp and clinical sheets; omit it when the data are
already on disk. The image stages take a few hours on one workstation (the
centerline stage about 1 min per series). `--stages` runs a subset, and
`--results <dir>` writes to another directory so that a re-run can be compared
with the committed results:

```
python scripts/run_all.py --stages phantoms tables figures   # no image data needed
python scripts/run_all.py --raw-root . --results out_check    # compare, don't overwrite
```

Run-time columns (`*_sec`) differ between runs by construction; every other
column is deterministic. One file is larger than a re-run would make it: the
committed `centerline_perturbation.csv` also carries the rows of four series
that the first run included by mistake and the analysis excludes (see
`docs/PLAN_CENTERLINE_PERTURBATION.md`, section 8); a re-run writes only the 56
eligible series, and Tables 5-6 are the same either way.

| Stage | Script(s) | Output in `results/tables/` |
|---|---|---|
| download | `download_tcia.py`, `download_hqcolon.py`, `download_tcia_annotations.py` | (data under `cohort/raw/`, `data/`) |
| centerline | `run_centerline.py` | `centerline_batch.csv` (prototype baseline), `centerline_auto.csv` (reported) |
| position | `hqcolon_overlap.py`, `derive_patient_position.py`, `summarise_acquisition.py` | `hqcolon_overlap.csv`, `patient_position.csv`, `acquisition.csv` |
| hqcolon | `eval_hqcolon.py` | `eval_hqcolon_auto.csv` (reported), `eval_hqcolon_batch.csv` (the Table 4 comparator) |
| ablation | `run_centerline.py`, `eval_hqcolon.py` | `centerline_fill_*.csv`, `eval_hqcolon_fill_*.csv`, `centerline_seed_extremes.csv`, `eval_hqcolon_seed_extremes.csv` |
| sweep | `sweep_thresholds.py` | `threshold_sweep.csv`, `threshold_sweep.json` |
| gas | `measure_gas_attenuation.py` | `gas_attenuation.csv`, `gas_attenuation.json` |
| coverage | `analyse_coverage_anatomy.py` | `coverage_anatomy.csv`, `coverage_anatomy.json` |
| indices | `run_indices.py` | `indices.csv` |
| agreement | `eval_prone_supine.py` | `prone_supine_agreement.csv` |
| perturbation | `perturb_centerline.py`, `analyse_perturbation.py` | `centerline_perturbation.csv`, `manuscript/table5_perturbation_path.csv`, `manuscript/table6_perturbation_fat.csv`, `manuscript/perturbation_numbers.json` |
| noise | `measure_image_noise.py` | `image_noise.csv`, `image_noise.json` |
| phantoms | `eval_polyp_phantoms.py`, `eval_polyp_realism.py` | `polyp_phantom_accuracy.csv`, `polyp_realism.csv`, `polyp_realism.json` |
| cohort | `make_cohort_table.py`, `report_cohort_lesions.py` | `cohort_characteristics.csv`, `cohort_lesions.csv`, `cohort_lesions.json` |
| clustered | `stats_clustered.py` | `clustered_statistics.csv`, `clustered_statistics.json` |
| strata | `stratify_acquisition.py` | `stratified_acquisition.csv`, `stratified_acquisition.json` |
| tables | `make_tables.py`, `pair_reference_status.py` | `manuscript/table1-12_*.csv`, `manuscript/table_s5-s8_*.csv`, `manuscript/numbers.json`, `pair_reference_status.csv` |
| figures | `make_figures.py` | `results/figures/figure_{1,3,5,6,7}.{pdf,png,tif}` |
| vgp | `make_vgp.py` | (unfold of the Figure 2 series under `data/vgp/`; needs `pip install -e .[vgp]` and a GPU) |
| case_figures | `make_case_figures.py` | `results/figures/figure_{2,4}.{pdf,png,tif}` (needs image data) |

`manuscript/numbers.json` holds every number quoted in the manuscript text,
keyed by name. `python scripts/run_all.py --list-outputs` prints what each
stage writes; that list is what the checks below hold the build to.

### Checking a re-run against the committed results

```
python scripts/check_rerun.py --stages phantoms strata tables figures
```

This copies the committed results, **deletes** the files those stages claim to
write, runs them, and requires every file back: tables compared value by value
(run-time `*_sec` columns excluded), figures by their pixels and by the
drawing inside the PDF rather than by the compressed bytes. Deleting first is
the point. Run the stages into a fresh directory and compare, and every file a
stage did not write is identical to the copy it came from, so the comparison
passes without having checked anything.

The stages above need no image data. The rest need the DICOM series and the
HQColon masks; for those, run them and compare directly:

```
python scripts/run_all.py --raw-root . --results out_check
python scripts/compare_tables.py out_check/tables results/tables
```

Other tables in `results/tables/` are the record of method development
(configuration sweeps, a rejected traversal redesign, mask diagnostics). They
are described, with the commands that produced them, in `docs/`.

## Tests

```
python -m pytest
```

Two regression tests need image data and are skipped without it:
`test_centerline.py` reproduces the prototype batch on case 0001 (set
`CTC_RAW_ROOT` to the directory holding `cohort/raw/`), and `test_fatmap.py`
compares against the prototype's own batch output, which is not distributed.

## Documentation

| File | Contents |
|---|---|
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | Every heuristic threshold, with its justification and calibration status. |
| [docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md) | Centerline failure modes of the prototype batch and the fixes. |
| [docs/EVALUATION_HQCOLON.md](docs/EVALUATION_HQCOLON.md) | External validation against HQColon, including the hole-fill ablation and a rejected redesign. |
| [docs/EVALUATION_PRONE_SUPINE.md](docs/EVALUATION_PRONE_SUPINE.md) | Prone/supine agreement of the fat and distension indices. |
| [docs/EVALUATION_POLYPS.md](docs/EVALUATION_POLYPS.md) | Why polyp measurement is validated on phantoms, and the result. |
| [docs/PLAN_CENTERLINE_PERTURBATION.md](docs/PLAN_CENTERLINE_PERTURBATION.md) | The centerline perturbation experiment as specified before any result was computed, with the record of where the run departed from it. |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | Installation and running instructions, submitted with the article. |

The documents are dated working records: where a later measurement overturned
an earlier conclusion, the earlier text is kept and marked as such.

## Author and licence

Shuji Yamamoto (Institute of One / LISIT). Code: MIT licence (`LICENSE`).
Please cite the data through the DOIs above, and the software through Zenodo.
The badge and `CITATION.cff` carry the concept DOI
([10.5281/zenodo.22720528](https://doi.org/10.5281/zenodo.22720528)), which
always resolves to the newest release; the article cites the DOI of the one
version its results were produced with, and `CITATION.cff` lists that version
under `identifiers`.
