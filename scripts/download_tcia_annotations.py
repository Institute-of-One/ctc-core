"""Download the TCIA CT COLONOGRAPHY supporting sheets.

The collection publishes its polyp findings as spreadsheets alongside the
images, plus a clinical-data table. All are small and are covered by the
collection's CC BY 3.0 licence, so unlike the image data they may be cited and
their derived summaries published -- but the files themselves still land in
``data/tcia/``, which is gitignored, and only aggregate results reach
``results/``.

Data citation: Smith K, Clark K, Bennett W, Nolan T, Kirby J, Wolfsberger M,
Moulton J, Vendt B, Freymann J. (2015). Data From CT COLONOGRAPHY. The Cancer
Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1

    python scripts/download_tcia_annotations.py --out data/tcia
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = "https://www.cancerimagingarchive.net/wp-content/uploads/"
USER_AGENT = "ctc-core/0.1 (research; https://github.com/Institute-of-One/ctc-core)"

FILES = {
    "TCIA-CTC-large-10-mm-polyps.xls": "polyps >= 10 mm",
    "TCIA-CTC-6-to-9-mm-polyps.xls": "polyps 6-9 mm",
    "TCIA-CTC-no-polyp-found.xls": "no polyp found",
    "CT-Colonography_clinical-data_v01_20260824.tsv": "clinical data",
    # The clinical columns are coded (a0e9, i1e3, ...); unusable without this.
    "CT-Colonography_machine_readable_data_dictionary_v01_20260824-1.tsv": "data dictionary",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/tcia"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for name, what in FILES.items():
        dest = args.out / name
        if not dest.exists():
            url = BASE + name
            print(f"  {name} ({what}) <- {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
        else:
            print(f"  {name}: already present")
        manifest[name] = {
            "description": what,
            "size_bytes": dest.stat().st_size,
            "sha256": sha256(dest),
        }
        print(f"      {manifest[name]['size_bytes']:,} bytes  "
              f"sha256 {manifest[name]['sha256'][:16]}...")

    (args.out / "download_manifest.json").write_text(
        json.dumps({"source": BASE, "files": manifest}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
