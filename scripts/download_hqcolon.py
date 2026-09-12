"""Download the HQColon reference segmentations from OSF.

HQColon (Finocchiaro et al., Sci Data 13:199, 2026, doi:10.1038/s41597-025-06518-z;
data doi:10.17605/OSF.IO/8TKPM) provides two masks for
each of 435 CT colonography scans taken from the TCIA "CT COLONOGRAPHY"
collection: gas-filled only, and gas plus fluid. It is used here as the external
reference standard for lumen segmentation and centerline coverage.

**Licence: CC BY-NC-ND 4.0 — evaluation use only.** The masks and anything
derived from them must never be redistributed or published. They land in
``data/hqcolon/``, which is gitignored; only summary statistics (Dice, surface
distance, coverage) reach ``results/``.

Only the three files needed for evaluation are fetched by default. The
TotalSegmentator masks and the RootPainter project (another 896 MB) are not
used by this work.

    python scripts/download_hqcolon.py --out data/hqcolon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

OSF_NODE = "8tkpm"
DOI = "10.17605/OSF.IO/8TKPM"
USER_AGENT = "ctc-core/0.1 (research; https://github.com/Institute-of-One/ctc-core)"

# name -> (OSF download id, expected size in bytes). Sizes are from the OSF API
# on 2026-09-10 and are checked after download.
WANTED = {
    "meta-data.json": ("8w6q7", 83_510),
    "gas-filled-colon-segmentation.zip": ("y3ad2", 208_576_256),
    "gas-and-fluid-filled-colon-segmentation.zip": ("d4sc3", 214_537_517),
}

# Present in the OSF node but not used here; listed so the omission is explicit.
NOT_USED = {
    "masks-totalsegmentator.zip": 489_957_133,
    "root_painter_colon_fluid_project.zip": 406_118_996,
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, file_id: str, expected: int, out_dir: Path) -> Path:
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size == expected:
        print(f"  {name}: already present ({expected:,} bytes)")
        return dest

    url = f"https://osf.io/download/{file_id}/"
    print(f"  {name}: fetching {expected:,} bytes from {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        done = 0
        while block := resp.read(1 << 20):
            fh.write(block)
            done += len(block)
            if done % (32 << 20) < (1 << 20):
                print(f"      {done:,} / {expected:,}", flush=True)

    got = tmp.stat().st_size
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: expected {expected:,} bytes, got {got:,}")
    tmp.replace(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/hqcolon"))
    ap.add_argument("--extract", action="store_true",
                    help="unzip the mask archives next to the zips")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"HQColon (doi:{DOI}) -> {args.out}")
    print("Licence CC BY-NC-ND 4.0: evaluation only, never redistribute.")
    print(f"Not fetched: {', '.join(NOT_USED)}")

    manifest = {"doi": DOI, "osf_node": OSF_NODE, "files": {}}
    for name, (file_id, size) in WANTED.items():
        path = fetch(name, file_id, size, args.out)
        manifest["files"][name] = {
            "osf_id": file_id,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        print(f"      sha256 {manifest['files'][name]['sha256'][:16]}...")

    if args.extract:
        for name in WANTED:
            if not name.endswith(".zip"):
                continue
            target = args.out / name[: -len(".zip")]
            if target.exists():
                print(f"  {name}: already extracted")
                continue
            print(f"  {name}: extracting", flush=True)
            with zipfile.ZipFile(args.out / name) as zf:
                zf.extractall(target)

    # Provenance record. Small and contains no mask data, so it is safe to keep
    # even though the directory itself is gitignored.
    (args.out / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
