# ctc-core: installation and running instructions

Submitted with the manuscript because the journal asks that software offered for
readers' use come with a manual for installation and use. It is written for a
reader who has not seen the project before, and every command below was run on
the machine that produced the reported results.

Version: 0.1.1, archived at https://doi.org/10.5281/zenodo.22731519.
Licence: MIT. No restriction on use by non-academics.

## 1. What the software does

`ctc-core` takes CT colonography series and produces, without a graphical
interface and without human input:

- a body mask, an intra-abdominal gas mask and a colon lumen mask;
- a colon centerline (Fast Marching minimal path, with a seed rule that aims at
  the whole colon and a bridge graph across gas components);
- pericolonic fat indices in a ring anchored to the segmented gas boundary;
- distension and preparation descriptors;
- polyp measurement on a spherical volume of interest;
- an unfolded ("virtual gross pathology") view of the colon surface, for
  illustration only;
- every table and figure of the article, from the cohort manifest.

## 2. Requirements

| | |
|---|---|
| Operating system | platform independent; tested on Windows 11 |
| Python | 3.10 to 3.12 (3.12.10 was used) |
| Memory | 16 GB is enough for one series at 1.0 mm isotropic; 32 GB was used |
| Disk | about 2 GB per series for the resampled volume and masks |
| GPU | not needed for any measurement; only the unfolded view uses one (VTK, OpenGL) |

Exact package versions are pinned in `requirements-lock.txt`.

## 3. Installation

```
git clone https://github.com/Institute-of-One/ctc-core
cd ctc-core
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements-lock.txt
pip install -e .
```

Optional extras, each independent of the others:

```
pip install -e .[tcia]      # download the cohort from the TCIA REST API
pip install -e .[figures]   # matplotlib, for the tables and figures
pip install -e .[vgp]       # vtk 9.6.2, for the unfolded view only
pip install -e .[dev]       # pytest and ruff
```

Check the installation:

```
python -m pytest            # 138 tests, about 20 s; 2 are skipped without the data
python -m ruff check .
```

## 4. Data

No patient data is included, and none may be added to the repository. The
images come from the Cancer Imaging Archive collection "CT COLONOGRAPHY"
(CC BY 3.0) and the reference segmentations from HQColon (CC BY-NC-ND 4.0, used
for evaluation only and not redistributed). `cohort/manifest.csv` holds the
Series Instance UIDs of the 60 series analysed, which is what makes the cohort
reproducible.

```
python scripts/download_tcia.py --raw-root .         # needs the tcia extra
python scripts/download_hqcolon.py --out data/hqcolon --extract
python scripts/download_tcia_annotations.py --out data/tcia
```

## 5. Running one series

```
python scripts/run_centerline.py --config corrected --raw-root . \
    --case 0007:primary --summary results/tables/one.csv
python scripts/run_indices.py --centerline-root data/centerline/corrected --polar \
    --out results/tables/one_indices.csv
```

The first command writes, under `data/centerline/corrected/<PatientID>/<role>/`,
the lumen mask, the centerline points (physical and voxel coordinates) and a
diagnostic JSON; the second writes one row of indices per series. About 70 s per
series for the centerline and 30 s for the indices on the machine used.

## 6. Reproducing the article

```
python scripts/run_all.py --raw-root .
```

Stages, each of which can be run alone with `--stages`:

| Stage | What it does |
|---|---|
| download | fetches the images, the reference masks and the annotations |
| centerline | the reported pipeline and the prototype configuration |
| position | recovers prone/supine from the DICOM header |
| hqcolon | compares lumen and centerline with the reference |
| ablation | the hole-fill and seed-rule ablations |
| indices | fat and distension indices for every series |
| agreement | prone/supine ICC(2,1) and Bland-Altman |
| phantoms | polyp measurement against exact geometry |
| perturbation | the centerline perturbation experiment (Tables 5-6, Figure 7) |
| cohort | cohort characteristics |
| tables | the article's tables and every in-text number |
| figures | Figures 1, 3, 5, 6 and 7 from the committed tables |
| vgp | the unfolded view of the illustrated series (needs the vgp extra) |
| case_figures | Figures 2 and 4, from the image data |

The perturbation experiment is a control experiment rather than part of the
pipeline, but it runs as the `perturbation` stage of the same entry point. Its
two steps can also be run on their own:

```
python scripts/perturb_centerline.py          # writes centerline_perturbation.csv
python scripts/analyse_perturbation.py        # writes tables 5 and 6 and their numbers
```

Run times measured on an Intel Core i7-10700K with 32 GB: about 70 s per series
for the centerline, 30 s for the indices, 30 s per series for the nine
perturbation conditions, and 25 s on an NVIDIA GeForce RTX 2080 Ti for the
unfolded view. The image stages over the whole cohort take a few hours.

`--results <dir>` writes to another directory, so that a re-run can be compared
with the committed results instead of overwriting them. Run-time columns differ
between runs by construction; every other column is deterministic.

## 7. Where the numbers in the article come from

| Article item | File |
|---|---|
| Tables 1-4 | `results/tables/manuscript/table[1-4]_*.csv` |
| Tables 5-6 | `results/tables/manuscript/table5_perturbation_path.csv`, `table6_perturbation_fat.csv` |
| Supplementary phantom accuracy | `results/tables/manuscript/table5_phantoms.csv` |
| Every number quoted in the text | `results/tables/manuscript/numbers.json` |
| Perturbation numbers | `results/tables/manuscript/perturbation_numbers.json` |
| Figures | `results/figures/figure_[1-7].{tif,pdf,png}` |
| Per-series results | `results/tables/*.csv` |

Thresholds and parameters, with their source and whether they were calibrated,
are listed in `docs/PARAMETERS.md`. The evaluation against the reference is
described in `docs/EVALUATION_HQCOLON.md`, the prone/supine analysis in
`docs/EVALUATION_PRONE_SUPINE.md`, and the perturbation experiment in
`docs/PLAN_CENTERLINE_PERTURBATION.md`.

## 8. Known limitations of the software

- Electronic cleansing is not applied, so fluid-filled colon is outside the gas
  mask and a ray that meets tagged fluid starts at the fluid surface.
- The slice-by-slice hole-fill fall-back can admit non-colonic gas, such as the
  lung bases, into the lumen mask.
- Most thresholds are inherited from the prototype and are not calibrated; they
  are marked as such in the parameter registry.
- The unfolded view is for illustration; no measurement uses it.
