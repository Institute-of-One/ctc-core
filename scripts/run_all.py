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
sweep      Each mask and path threshold varied one at a time, masks and
           centerline rebuilt per setting (Table 10).
gas        The attenuation of the colonic gas inside the reference mask,
           and the share each candidate threshold admits (Table 10, and
           the bounded limitation in the Methods).
coverage   Coverage by position along the reference centerline, and the
           fluid or collapsed volume the gas mask excludes (Tables 8-9).
indices    Fat and quality indices for every series with a centerline.
agreement  Pillar 3: prone/supine ICC(2,1) and Bland-Altman. The coverage
           sensitivity analysis is not run: only 3 of the 28 pairs have a
           reference in both positions and adequate coverage in both
           (make_tables records the counts).
perturbation  The pre-specified centerline perturbation experiment: the
           path is wobbled, smoothly at two wavelengths and independently,
           and the indices are recomputed (Tables 5-6, Figure 7).
noise      The image noise of the cohort, measured from the images: the
           level the phantom sweep injects, so the sweep has no noise level
           of its own to drift.
phantoms   Pillar 2: polyp measurement against exact geometric truth, in the
           noise-free limit and then under noise, a fluid interface and
           adhesion to a wall or a fold.
cohort     Cohort characteristics from the TCIA clinical table, and every
           annotated lesion in the cohort decoded from the published key.
clustered  Patient-cluster intervals for the reported quantities, by
           bootstrap over patients rather than series. Runs before the
           tables stage, which prints its interval column.
strata     The results by manufacturer, kernel, slice thickness, tube
           voltage and tube current (Supplementary file 6). Reads the
           tables above; no image data.
tables     Manuscript tables 1-12 and the supplementary tables, every
           in-text number (numbers.json), and the per-patient
           reference/coverage audit behind Table 4.
figures    Figures 1, 3, 5, 6 and 7, from the committed tables only.
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

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

STAGES = ("download", "centerline", "position", "hqcolon", "ablation", "sweep",
          "gas", "coverage", "indices", "agreement", "perturbation", "noise",
          "phantoms", "cohort", "clustered", "strata", "tables", "figures",
          "vgp", "case_figures")


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
            # The prototype configuration against the same reference: Table 4's
            # comparator. It reads the masks and paths the "batch" centerline
            # run above left in data/centerline/batch.
            ["eval_hqcolon.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--centerline-root", "data/centerline/batch",
             "--position", f"{T}/patient_position.csv",
             "--out", f"{T}/eval_hqcolon_batch.csv"],
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
        # The threshold sweep, the gas attenuation and the coverage analysis.
        # All three are reported (Table 10, Tables 8-9, the Methods) and all
        # three were missing from this plan until 2026-11-05: the register in
        # check_tables.py named their scripts, nothing ran them, and the entry
        # point that calls itself the whole reproduction did not write them.
        # Each rebuilds masks from the images, so none of them is cheap.
        "sweep": [
            ["sweep_thresholds.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--hqcolon", "data/hqcolon", "--out", f"{T}/threshold_sweep.csv"],
        ],
        "gas": [
            ["measure_gas_attenuation.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--hqcolon", "data/hqcolon", "--acquisition", f"{T}/acquisition.csv",
             "--out", f"{T}/gas_attenuation.csv"],
        ],
        "coverage": [
            ["analyse_coverage_anatomy.py", "--overlap", f"{T}/hqcolon_overlap.csv",
             "--hqcolon", "data/hqcolon",
             "--centerline-root", "data/centerline/corrected",
             "--out", f"{T}/coverage_anatomy.csv"],
        ],
        "indices": [
            ["run_indices.py", "--centerline-root", "data/centerline/corrected", "--polar",
             "--position", f"{T}/patient_position.csv", "--out", f"{T}/indices.csv"],
        ],
        "agreement": [
            # --coverage is passed although the reported run does not use it
            # (it is read only with --max-unreached, the sensitivity analysis
            # the Methods says is not run): its default is the committed
            # table, so leaving it off would mean that enabling the analysis
            # in a re-run read the committed tree whatever --results said.
            ["eval_prone_supine.py", "--indices", f"{T}/indices.csv",
             "--coverage", f"{T}/eval_hqcolon_auto.csv",
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
        # Image noise, measured in the cohort rather than assumed, because the
        # phantom sweep below reads its level from this file and has none of
        # its own. It needs the images; the sweep does not, because the
        # summary it reads is a committed result.
        "noise": [
            ["measure_image_noise.py", "--out", f"{T}/image_noise.csv"],
        ],
        "phantoms": [
            ["eval_polyp_phantoms.py", "--out", f"{T}/polyp_phantom_accuracy.csv"],
            ["eval_polyp_realism.py", "--noise", f"{T}/image_noise.json",
             "--out", f"{T}/polyp_realism.csv"],
        ],
        "cohort": [
            ["make_cohort_table.py", "--position", f"{T}/patient_position.csv",
             "--out", f"{T}/cohort_characteristics.csv"],
            ["report_cohort_lesions.py", "--out", f"{T}/cohort_lesions.csv"],
        ],
        # Patient-cluster intervals. This runs before "tables" because
        # make_tables.py prints the interval column of the reference table from
        # clustered_statistics.csv; without this stage the table would be built
        # from whatever interval file happened to be left on disk.
        "clustered": [
            ["stats_clustered.py", "--tables", T,
             "--out", f"{T}/clustered_statistics.csv",
             "--hq", "eval_hqcolon_auto.csv", "--batch", "centerline_auto.csv"],
        ],
        # The acquisition strata (Supplementary file 6, and the collinearity
        # and Exposure evidence the Discussion quotes). It reads the tables the
        # stages above wrote, so it runs after them and before make_tables.py,
        # which prints its stratified table.
        "strata": [
            ["stratify_acquisition.py", "--tables", T,
             "--out", f"{T}/stratified_acquisition.csv"],
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


# What each stage writes under ``--results``, so that the shipped artefacts can
# be counted one by one against the steps that produce them.
#
# Nothing resolves this map at run time: a stage that quietly stops writing a
# file would otherwise be invisible, because the committed copy stays where it
# is and a comparison of the re-run against the committed tree compares that
# copy with itself. ``scripts/check_rerun.py`` deletes these paths before it
# runs the stage and requires each one back, and ``check_tables.py`` requires
# every registered result to appear here exactly once unless it is declared an
# orphan. Paths are relative to the results directory.
#
# The download stage writes only under ``cohort/raw`` and ``data/`` (gitignored
# image data), and the vgp stage only under ``data/vgp``; both are empty here.
OUTPUTS: dict[str, tuple[str, ...]] = {
    "download": (),
    "centerline": ("tables/centerline_batch.csv", "tables/centerline_auto.csv"),
    "position": ("tables/hqcolon_overlap.csv", "tables/patient_position.csv",
                 "tables/acquisition.csv"),
    "hqcolon": ("tables/eval_hqcolon_auto.csv", "tables/eval_hqcolon_batch.csv"),
    "ablation": ("tables/centerline_fill_bbox.csv",
                 "tables/eval_hqcolon_fill_bbox.csv",
                 "tables/centerline_fill_slicewise.csv",
                 "tables/eval_hqcolon_fill_slicewise.csv",
                 "tables/centerline_seed_extremes.csv",
                 "tables/eval_hqcolon_seed_extremes.csv"),
    "sweep": ("tables/threshold_sweep.csv", "tables/threshold_sweep.json"),
    "gas": ("tables/gas_attenuation.csv", "tables/gas_attenuation.json"),
    "coverage": ("tables/coverage_anatomy.csv", "tables/coverage_anatomy.json"),
    "indices": ("tables/indices.csv",),
    "agreement": ("tables/prone_supine_agreement.csv",),
    "perturbation": ("tables/centerline_perturbation.csv",
                     "tables/manuscript/table5_perturbation_path.csv",
                     "tables/manuscript/table6_perturbation_fat.csv",
                     "tables/manuscript/perturbation_numbers.json"),
    "noise": ("tables/image_noise.csv", "tables/image_noise.json"),
    "phantoms": ("tables/polyp_phantom_accuracy.csv",
                 "tables/polyp_realism.csv", "tables/polyp_realism.json"),
    "cohort": ("tables/cohort_characteristics.csv",
               "tables/cohort_lesions.csv", "tables/cohort_lesions.json"),
    "clustered": ("tables/clustered_statistics.csv",
                  "tables/clustered_statistics.json"),
    "strata": ("tables/stratified_acquisition.csv",
               "tables/stratified_acquisition.json"),
    "tables": ("tables/manuscript/table1_cohort.csv",
               "tables/manuscript/table2_reference.csv",
               "tables/manuscript/table3_cohort_outcome.csv",
               "tables/manuscript/table4_agreement.csv",
               "tables/manuscript/table4_comparators.csv",
               "tables/manuscript/table5_phantoms.csv",
               "tables/manuscript/table8_coverage_by_position.csv",
               "tables/manuscript/table9_excluded_volume.csv",
               "tables/manuscript/table10_threshold_sweep.csv",
               "tables/manuscript/table11_polyp_realism.csv",
               "tables/manuscript/table12_cohort_lesions.csv",
               "tables/manuscript/table_s5_reference_development_tuned.csv",
               "tables/manuscript/table_s6_acquisition.csv",
               "tables/manuscript/table_s7_exploratory_composites.csv",
               "tables/manuscript/table_s8_polyp_realism_rows.csv",
               "tables/manuscript/numbers.json",
               "tables/pair_reference_status.csv"),
    "figures": tuple(f"figures/figure_{n}.{ext}"
                     for n in (1, 3, 5, 6, 7) for ext in ("png", "pdf", "tif")),
    "vgp": (),
    "case_figures": tuple(f"figures/figure_{n}.{ext}"
                          for n in (2, 4) for ext in ("png", "pdf", "tif")),
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
    ap.add_argument("--list-outputs", action="store_true",
                    help="print what each requested stage writes, and stop")
    ap.add_argument("--allow-environment-drift", action="store_true",
                    help="run even when the installed packages differ from "
                         "requirements-lock.txt (the committed results were "
                         "produced with the locked versions)")
    args = ap.parse_args()

    stages = args.stages or [s for s in STAGES if s != "download" or args.download]

    if args.list_outputs:
        for stage in STAGES:
            if stage in stages:
                print(f"[{stage}] {' '.join(OUTPUTS[stage]) or '(no committed output)'}")
        return 0

    # The committed results were produced with the versions in
    # requirements-lock.txt, and a plotting or array library can change what it
    # writes between minor versions without changing this repository at all.
    # Checked before the first stage, not after a long run.
    if not args.dry_run:
        env = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_environment.py")],
                             cwd=ROOT).returncode
        if env != 0:
            if not args.allow_environment_drift:
                print("the environment does not match requirements-lock.txt; "
                      "install it, or pass --allow-environment-drift and say so "
                      "wherever the numbers are reported", file=sys.stderr)
                return env
            print("continuing with a drifting environment at the caller's request",
                  file=sys.stderr)
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
