"""Regenerate every table and figure of the manuscript, in dependency order.

This is the single entry point of the reproduction. Each stage is one script in
``scripts/`` and writes to ``results/`` (committed summaries) or ``data/``
(per-case outputs, gitignored). After a full run, ``git diff results/`` shows
whether anything moved.

Stages (``--stages`` selects a subset, in this order):

========== =============================================================
download   TCIA series (~19 GB), HQColon masks (423 MB), TCIA sheets.
           Only with ``--download``: the data licences are the user's
           to accept, and the cohort may already be on disk.
centerline 60 series, twice: the 2026-08 batch preset (baseline) and the
           corrected pipeline (the reported one). About 1 h each on one
           workstation.
position   HQColon subject overlap; acquisition position and parameters
           from the DICOM headers.
hqcolon    Pillar 1: lumen and centerline against the HQColon reference.
ablation   The hole-fill ablation behind the adaptive rule (3-D only,
           slice-wise only) and the seed-rule ablation (the previous
           skeleton-extremes rule), each run over the cohort and scored
           against HQColon. Three more centerline runs.
indices    Fat and quality indices for every series with a centerline.
agreement  Pillar 3: prone/supine ICC(2,1) and Bland-Altman. The coverage
           sensitivity analysis is not run: only 3 of the 28 pairs have a
           reference in both positions and adequate coverage in both
           (make_tables records the counts).
phantoms   Pillar 2: polyp measurement against exact geometric truth.
cohort     Cohort characteristics from the TCIA clinical table.
tables     Manuscript tables 1-5, every in-text number (numbers.json), and the
           per-patient reference/coverage audit behind Table 4.
figures    Figures 1, 3, 5 and 6, from the committed tables only.
vgp        The VGP unfold of the Figure 2 series: that one series is
           resampled to 0.5 mm and rendered on the GPU (needs the vgp extra,
           about 1 min). Visualisation only; no number depends on it.
case_figures  Figures 2 and 4: one series through the pipeline and one
           hole-fill failure, drawn from the image data.
========== =============================================================

    python scripts/run_all.py --raw-root . --download     # from nothing
    python scripts/run_all.py --raw-root .                 # data on disk
    python scripts/run_all.py --stages phantoms figures    # no image data needed
    python scripts/run_all.py --results out_check          # compare, don't overwrite

Run-time columns (``*_sec``) differ between runs by construction; every other
column is deterministic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = ("download", "centerline", "position", "hqcolon", "ablation", "indices",
          "agreement", "perturbation", "phantoms", "cohort", "tables", "figures", "vgp",
          "case_figures")


def commands(raw_root: str, results: str) -> dict[str, list[list[str]]]:
    T = f"{results}/tables"
    return {
        "download": [
            ["download_tcia.py", "--raw-root", raw_root],
            ["download_hqcolon.py", "--out", "data/hqcolon", "--extract"],
            ["download_tcia_annotations.py", "--out", "data/tcia"],
        ],
        "centerline": [
            ["run_centerline.py", "--config", "batch", "--raw-root", raw_root,
             "--summary", f"{T}/centerline_batch.csv"],
            ["run_centerline.py", "--config", "corrected", "--raw-root", raw_root,
             "--summary", f"{T}/centerline_auto.csv"],
        ],
        "position": [
            ["hqcolon_overlap.py", "--hqcolon", "data/hqcolon",
             "--out", f"{T}/hqcolon_overlap.csv"],
            ["derive_patient_position.py", "--raw-root", raw_root,
             "--overlap", f"{T}/hqcolon_overlap.csv", "--out", f"{T}/patient_position.csv"],
            ["summarise_acquisition.py", "--raw-root", raw_root, "--out", f"{T}/acquisition.csv"],
        ],
        "hqcolon": [
            ["eval_hqcolon.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--centerline-root", "data/centerline/corrected",
             "--position", f"{T}/patient_position.csv",
             "--out", f"{T}/eval_hqcolon_auto.csv"],
        ],
        "ablation": [
            cmd
            for fill in ("bbox", "slicewise")
            for cmd in (
                ["run_centerline.py", "--config", "corrected", "--fill-holes", fill,
                 "--tag", f"fill_{fill}", "--raw-root", raw_root,
                 "--summary", f"{T}/centerline_fill_{fill}.csv"],
                ["eval_hqcolon.py", "--overlap", f"{T}/hqcolon_overlap.csv",
                 "--centerline-root", f"data/centerline/fill_{fill}", "--fill-holes", fill,
                 "--position", f"{T}/patient_position.csv",
                 "--out", f"{T}/eval_hqcolon_fill_{fill}.csv"],
            )
        ] + [
            # Seed-rule ablation: the previous rule, everything else as reported.
            ["run_centerline.py", "--config", "corrected", "--seed-rule", "skeleton_extremes",
             "--tag", "seed_extremes", "--raw-root", raw_root,
             "--summary", f"{T}/centerline_seed_extremes.csv"],
            ["eval_hqcolon.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--centerline-root", "data/centerline/seed_extremes",
             "--position", f"{T}/patient_position.csv",
             "--out", f"{T}/eval_hqcolon_seed_extremes.csv"],
        ],
        "indices": [
            ["run_indices.py", "--centerline-root", "data/centerline/corrected", "--polar",
             "--position", f"{T}/patient_position.csv", "--out", f"{T}/indices.csv"],
        ],
        "agreement": [
            ["eval_prone_supine.py", "--indices", f"{T}/indices.csv",
             "--out", f"{T}/prone_supine_agreement.csv"],
        ],
        # The pre-specified centerline perturbation experiment (Tables 5-6,
        # Figure 7; docs/PLAN_CENTERLINE_PERTURBATION.md). It reuses the images,
        # centerlines and indices of the stages above and changes only the path.
        "perturbation": [
            ["perturb_centerline.py", "--tables", T,
             "--centerline-root", "data/centerline/corrected",
             "--out", f"{T}/centerline_perturbation.csv"],
            ["analyse_perturbation.py", "--perturbation", f"{T}/centerline_perturbation.csv",
             "--centerline", f"{T}/centerline_auto.csv", "--out", f"{T}/manuscript"],
        ],
        "phantoms": [
            ["eval_polyp_phantoms.py", "--out", f"{T}/polyp_phantom_accuracy.csv"],
        ],
        "cohort": [
            ["make_cohort_table.py", "--position", f"{T}/patient_position.csv",
             "--out", f"{T}/cohort_characteristics.csv"],
        ],
        "tables": [
            ["make_tables.py", "--tables", T],
            ["pair_reference_status.py", "--tables", T],
        ],
        "figures": [
            ["make_figures.py", "--tables", T, "--out", f"{results}/figures"],
        ],
        "vgp": [
            ["make_vgp.py", "--raw-root", raw_root, "--tables", T, "--out", "data/vgp"],
        ],
        "case_figures": [
            ["make_case_figures.py", "--tables", T, "--vgp-root", "data/vgp",
             "--out", f"{results}/figures"],
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=".",
                    help="directory the manifest's download_dir paths are relative to")
    ap.add_argument("--results", default="results",
                    help="where tables and figures go; point elsewhere to compare a "
                         "re-run against the committed results without overwriting them")
    ap.add_argument("--download", action="store_true", help="include the download stage")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    stages = args.stages or [s for s in STAGES if s != "download" or args.download]
    plan = commands(args.raw_root, args.results)
    t_all = time.perf_counter()
    for stage in STAGES:
        if stage not in stages:
            continue
        for cmd in plan[stage]:
            full = [sys.executable, str(ROOT / "scripts" / cmd[0]), *cmd[1:]]
            print(f"\n[{stage}] {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            t0 = time.perf_counter()
            rc = subprocess.run(full, cwd=ROOT).returncode
            if rc != 0:
                print(f"[{stage}] failed with exit code {rc}; stopping", file=sys.stderr)
                return rc
            print(f"[{stage}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nall requested stages done in {(time.perf_counter() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
