"""Download the 60 cohort series listed in ``cohort/manifest.csv`` from TCIA.

The cohort is fixed by the committed manifest, not re-selected at download
time: the collection's metadata can change between releases, and the
manuscript's numbers belong to these exact Series Instance UIDs.

Each series is fetched as a ZIP through the NBIA ``getImage`` endpoint and
extracted to ``<raw-root>/<download_dir>`` (``cohort/raw/<PatientID>/<SeriesUID>``
by default; gitignored). A ``.downloaded`` marker makes re-runs skip completed
series. The DICOM files are checked after extraction: the count must equal the
manifest's ``ImageCount`` and every file must carry the manifest's
SeriesInstanceUID.

How the manifest was selected (``--verify-selection`` re-runs the rule against
live metadata and reports any difference; it never rewrites the manifest):

* collection "CT COLONOGRAPHY", modality CT, BodyPartExamined in
  {COLON, ABDOMEN, ABDOMENPELVIS, ABDOMEN PELVIS};
* 200 <= ImageCount <= 900 and FileSize <= 700 MiB;
* patients with at least two such series, sorted by PatientID; the first 30;
* per patient the two series with the most images (ties broken by UID).
  ``primary`` is the larger. It carries **no** position information -- the
  acquisition position is recovered separately
  (``scripts/derive_patient_position.py``).

Data: Smith K, et al. (2015) Data From CT COLONOGRAPHY. The Cancer Imaging
Archive. https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1 (CC BY 3.0). About 19 GB.

    python scripts/download_tcia.py --raw-root .
    python scripts/download_tcia.py --verify-selection   # metadata only
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOGGER = logging.getLogger("download_tcia")

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "CT COLONOGRAPHY"
BODY_PARTS = {"COLON", "ABDOMEN", "ABDOMENPELVIS", "ABDOMEN PELVIS"}
MIN_IMAGE_COUNT = 200
MAX_IMAGE_COUNT = 900
MAX_FILE_SIZE_BYTES = 700 * 1024 * 1024
N_PATIENTS = 30


# ---------------------------------------------------------------------------
# Selection rule (pure; unit-tested)
# ---------------------------------------------------------------------------


def is_candidate_series(s: dict) -> bool:
    if s.get("Modality") != "CT":
        return False
    if str(s.get("BodyPartExamined", "")).upper() not in BODY_PARTS:
        return False
    n = int(s.get("ImageCount", 0))
    if not MIN_IMAGE_COUNT <= n <= MAX_IMAGE_COUNT:
        return False
    return int(s.get("FileSize", 0)) <= MAX_FILE_SIZE_BYTES


def select_cohort(series: list[dict], n_patients: int = N_PATIENTS) -> list[tuple[str, str, str]]:
    """``(PatientID, role, SeriesInstanceUID)`` for the deterministic selection."""
    by_pid: dict[str, list[dict]] = {}
    for s in series:
        if is_candidate_series(s):
            by_pid.setdefault(str(s["PatientID"]), []).append(s)
    out: list[tuple[str, str, str]] = []
    n = 0
    for pid in sorted(by_pid):
        cand = by_pid[pid]
        if len(cand) < 2:
            continue
        pair = sorted(cand, key=lambda x: (-int(x["ImageCount"]), x["SeriesInstanceUID"]))[:2]
        out.append((pid, "primary", pair[0]["SeriesInstanceUID"]))
        out.append((pid, "secondary", pair[1]["SeriesInstanceUID"]))
        n += 1
        if n >= n_patients:
            break
    return out


def compare_selection(
    manifest: list[dict], selected: list[tuple[str, str, str]]
) -> list[str]:
    """Human-readable differences between the manifest and a re-selection."""
    have = {(r["PatientID"], r["role"], r["SeriesInstanceUID"]) for r in manifest}
    want = set(selected)
    diffs = [f"only in manifest:  {t}" for t in sorted(have - want)]
    diffs += [f"only in selection: {t}" for t in sorted(want - have)]
    return diffs


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _check_series(dest: Path, series_uid: str, expected: int) -> str | None:
    """None if ``dest`` holds exactly ``expected`` files of ``series_uid``."""
    import SimpleITK as sitk

    files = sorted(dest.glob("*.dcm"))
    if len(files) != expected:
        return f"{len(files)} files, manifest says {expected}"
    reader = sitk.ImageFileReader()
    for f in files:
        reader.SetFileName(str(f))
        reader.ReadImageInformation()  # header only
        uid = reader.GetMetaData("0020|000e").strip()
        if uid != series_uid:
            return f"{f.name} belongs to series {uid}"
    return None


def download_series(session, row: dict, raw_root: Path, max_retries: int = 3) -> tuple[bool, str]:
    uid = row["SeriesInstanceUID"]
    dest = raw_root / row["download_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".downloaded"
    expected = int(row["ImageCount"])
    if marker.exists() and len(list(dest.glob("*.dcm"))) == expected:
        return True, "already downloaded"

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(f"{NBIA_BASE}/getImage", params={"SeriesInstanceUID": uid},
                             stream=True, timeout=600) as resp:
                resp.raise_for_status()
                buf = io.BytesIO()
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    buf.write(chunk)
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".dcm")] or zf.namelist()
                for name in names:
                    with zf.open(name) as src, (dest / Path(name).name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            problem = _check_series(dest, uid, expected)
            if problem:
                raise RuntimeError(problem)
            marker.write_text(f"files={expected}\n", encoding="utf-8")
            return True, f"downloaded ({buf.getbuffer().nbytes / 1e6:.0f} MB)"
        except Exception as exc:  # network errors and failed checks alike
            LOGGER.warning("attempt %d/%d for %s: %s", attempt, max_retries, uid, exc)
            time.sleep(2 * attempt)
    return False, "max retries exceeded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("cohort/manifest.csv"))
    ap.add_argument("--raw-root", type=Path, default=Path("."),
                    help="directory the manifest's download_dir paths are relative to")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-selection", action="store_true",
                    help="re-run the selection rule on live metadata; download nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import requests

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))

    if args.verify_selection:
        r = requests.get(f"{NBIA_BASE}/getSeries", params={"Collection": COLLECTION}, timeout=120)
        r.raise_for_status()
        diffs = compare_selection(manifest, select_cohort(r.json()))
        for d in diffs:
            print(d)
        print("selection matches the manifest" if not diffs else f"{len(diffs)} differences")
        return 0 if not diffs else 1

    session = requests.Session()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_series, session, r, args.raw_root): r for r in manifest}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = futures[fut]
            ok, msg = fut.result()
            failed += not ok
            LOGGER.info("[%d/%d] %s %s: %s", i, len(manifest), row["PatientID"], row["role"], msg)
    print(f"{len(manifest) - failed} ok / {failed} failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
