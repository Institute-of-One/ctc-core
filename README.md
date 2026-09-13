# ctc-core

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

Python 3.10-3.12.

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e ".[tcia,figures,dev]"
```

`requirements-lock.txt` pins the versions every committed result was produced
with.

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
| hqcolon | `eval_hqcolon.py` | `eval_hqcolon_auto.csv` |
| ablation | `run_centerline.py`, `eval_hqcolon.py` | `centerline_fill_*.csv`, `eval_hqcolon_fill_*.csv`, `centerline_seed_extremes.csv`, `eval_hqcolon_seed_extremes.csv` |
| indices | `run_indices.py` | `indices.csv` |
| agreement | `eval_prone_supine.py` | `prone_supine_agreement.csv`, `prone_supine_agreement_reached.csv` |
| perturbation | `perturb_centerline.py`, `analyse_perturbation.py` | `centerline_perturbation.csv`, `manuscript/table5_perturbation_path.csv`, `manuscript/table6_perturbation_fat.csv`, `manuscript/perturbation_numbers.json` |
| phantoms | `eval_polyp_phantoms.py` | `polyp_phantom_accuracy.csv` |
| cohort | `make_cohort_table.py` | `cohort_characteristics.csv` |
| tables | `make_tables.py`, `pair_reference_status.py` | `manuscript/table1-4_*.csv`, `manuscript/table5_phantoms.csv`, `manuscript/numbers.json`, `pair_reference_status.csv` |
| figures | `make_figures.py` | `results/figures/figure_{1,3,5,6,7}.{pdf,png,tif}` |
| vgp | `make_vgp.py` | (unfold of the Figure 2 series under `data/vgp/`; needs `pip install -e .[vgp]` and a GPU) |
| case_figures | `make_case_figures.py` | `results/figures/figure_{2,4}.{pdf,png,tif}` (needs image data) |

`manuscript/numbers.json` holds every number quoted in the manuscript text,
keyed by name.

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
Please cite the software through its Zenodo DOI (`CITATION.cff`) and the data
through the DOIs above.
