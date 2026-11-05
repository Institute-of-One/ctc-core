"""Re-run chosen stages and require them to reproduce the committed results.

    python scripts/check_rerun.py --stages phantoms strata tables figures
    python scripts/check_rerun.py --stages phantoms --keep

Why this is not "run the stages into another directory and diff": the stages
read the tables earlier stages wrote, so a re-run directory has to start as a
copy of the committed one. Compare that copy with the original and every file
the stage never wrote comes out identical -- a comparison of a file with
itself, green and empty of information. The same mistake was shipped in a
sibling project, whose public regeneration job claimed to check every result
and in fact checked a third of them.

So the files the chosen stages claim in ``run_all.OUTPUTS`` are **deleted** from
the copy before the stages run, and each one is required back. A stage that
stops writing a result fails here, not silently.

Tables are compared with ``compare_tables.py`` (run-time ``*_sec`` columns
excluded by construction). JSON is compared value by value, excluding keys
whose name ends in ``_sec``. Figures are compared by what they draw: the
pixels of a PNG or a TIFF, and the inflated streams of a PDF. The file bytes
are not the figure -- the same drawing compressed by two zlib builds gives two
files, which is what the first run of this check in continuous integration
reported about five identical PNGs -- and anything else is compared as bytes.

What this does not establish: that a stage computed its result rather than
copying it. The committed tree stays readable while the stages run, so a
script that resolves an input or an output through a path of its own, instead
of through --results, would be fed the committed data and agree with it. That
channel is checked statically instead, by stage_inputs() in check_tables.py,
which refuses any stage script that builds a path from a module-level constant
pointing into results/ outside an argparse default, or that defaults an
argument there without the stage passing the flag. Short of running each stage
with the committed tree out of reach, which no portable filesystem trick gives
us, those two together are the guarantee: deletion proves something was
written, and the static rule proves it could not have come from the committed
copy.

Exit code 0 when every claimed file came back and matched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from compare_tables import compare as compare_csv
from run_all import OUTPUTS, STAGES

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Written into the re-run directory as it is created, and required before this
# script ever deletes one. Nothing else marks a directory as this tool's to
# destroy: the first version deleted whatever path it was handed, including
# one its own guard had just refused.
MARKER = ".check_rerun"


def claimed(stages: list[str]) -> list[str]:
    """What the chosen stages say they write, in stage order, without repeats."""
    out: list[str] = []
    for stage in STAGES:
        if stage in stages:
            out.extend(OUTPUTS[stage])
    return sorted(dict.fromkeys(out))


def json_diffs(a: Path, b: Path) -> list[str]:
    """Differences between two JSON files, ignoring run-time fields."""
    x, y = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    out: list[str] = []

    def walk(u, v, path: str) -> None:
        if isinstance(u, dict) and isinstance(v, dict):
            for k in sorted(set(u) | set(v)):
                if k.endswith("_sec"):
                    continue
                if k not in u or k not in v:
                    out.append("%s%s only in %s" % (path, k,
                               "re-run" if k in u else "committed"))
                else:
                    walk(u[k], v[k], "%s%s." % (path, k))
        elif isinstance(u, list) and isinstance(v, list):
            if len(u) != len(v):
                out.append("%s length %d vs %d" % (path, len(u), len(v)))
            for i, (ui, vi) in enumerate(zip(u, v, strict=False)):
                walk(ui, vi, "%s[%d]." % (path, i))
        elif isinstance(u, bool) or isinstance(v, bool) or u is None or v is None:
            if u is not v:
                out.append("%s %r vs %r" % (path, u, v))
        elif isinstance(u, (int, float)) and isinstance(v, (int, float)):
            # json.dump writes NaN, and math.isclose(nan, nan) is False, so a
            # file that reproduces exactly would be reported as differing --
            # and the CSV side treats two empty cells as equal, which would
            # have made the two rules disagree about the same quantity.
            both_nan = u != u and v != v
            if not (both_nan or math.isclose(u, v, rel_tol=1e-9, abs_tol=1e-12)):
                out.append("%s %r vs %r" % (path, u, v))
        elif u != v:
            out.append("%s %r vs %r" % (path, u, v))

    walk(x, y, "")
    return out


def raster_diffs(rerun: Path, committed: Path) -> list[str]:
    """Compare two images by their pixels rather than by their file bytes.

    A PNG or a TIFF is a compressed container, and the compressor is not the
    figure: the first run of this check in continuous integration regenerated
    every table identically on Linux and reported all five PNGs as different,
    because zlib there does not emit the same bytes as zlib on the machine the
    committed figures were drawn on. The pixels were identical. Comparing the
    pixels says what the check is for -- the figure was redrawn and it is the
    same figure -- and says it on any machine.
    """
    from PIL import Image, UnidentifiedImageError

    out: list[str] = []
    try:
        with Image.open(rerun) as a, Image.open(committed) as b:
            if (a.size, a.mode) != (b.size, b.mode):
                return ["%s %s vs %s %s" % (a.size, a.mode, b.size, b.mode)]
            pa, pb = np.asarray(a), np.asarray(b)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # Not a readable image: say so and fall back to the bytes rather than
        # letting the check die on a file it cannot open.
        if rerun.read_bytes() == committed.read_bytes():
            return []
        return ["not identical, and the pixels could not be read to compare "
                "them instead (%s)" % type(exc).__name__]
    if pa.shape != pb.shape:
        out.append("pixel array %s vs %s" % (pa.shape, pb.shape))
    elif not np.array_equal(pa, pb):
        moved = int((pa != pb).any(axis=-1).sum()) if pa.ndim == 3 else int((pa != pb).sum())
        out.append("%d pixel(s) differ" % moved)
    return out


def pdf_diffs(rerun: Path, committed: Path) -> list[str]:
    """Compare two PDFs by what they draw, not by how it was compressed.

    matplotlib writes its content in FlateDecode streams, so the same drawing
    compressed by two zlib builds gives two files. Every stream is inflated
    and the rest of the file is compared as it stands; a PDF this cannot take
    apart is compared byte for byte instead, and says so.
    """
    import zlib

    # The keyword, not the six letters: "endstream" ends with them too, and
    # looking for the substring walked straight into the one that closes the
    # stream it had just read.
    opener = re.compile(rb"(?<![A-Za-z])stream\r?\n")

    def parts(path: Path) -> list[bytes] | None:
        raw = path.read_bytes()
        out, at = [], 0
        while True:
            match = opener.search(raw, at)
            if match is None:
                out.append(raw[at:])
                return out
            end = raw.find(b"endstream", match.end())
            if end < 0:
                return None
            out.append(raw[at:match.end()])
            body = raw[match.end():end]
            try:
                out.append(zlib.decompress(body))
            except zlib.error:
                out.append(body)
            at = end

    # A coordinate written with one digit more or less is the same drawing.
    # The two PDFs that survived every other normalisation differed by one
    # character in a marker path -- 2.645751 against 2.6457513, the square
    # root of seven -- and by two in a page of coordinates, while the 300 dpi
    # rasters of both were pixel for pixel identical. Numbers are compared as
    # numbers, to four decimals of a PDF point: a ten-thousandth of a point is
    # three millionths of an inch, and anything that moves a mark by that much
    # moves it by less than a thousandth of a pixel.
    NUMBER = re.compile(rb"(-?\d+\.\d+)")

    def is_text(stream: bytes) -> bool:
        printable = sum(32 <= c < 127 or c in b"\n\r\t" for c in stream[:4096])
        return printable >= 0.9 * min(len(stream), 4096) and bool(stream)

    def same_drawing(x: bytes, y: bytes, tol: float = 1e-3) -> bool:
        """Equal text, and numbers equal within a thousandth of a point.

        Compared as numbers rather than rounded to a fixed number of decimals:
        rounding puts a boundary somewhere, and a value that sits on it --
        471.525949 against 471.525950 -- comes out different although it has
        moved by a millionth of a point.
        """
        if not (is_text(x) and is_text(y)):
            return x == y
        px, py = NUMBER.split(x), NUMBER.split(y)
        if len(px) != len(py):
            return False
        for i, (u, v) in enumerate(zip(px, py, strict=True)):
            if i % 2 == 0:
                if u != v:
                    return False
            elif abs(float(u) - float(v)) > tol:
                return False
        return True

    def plumbing(chunk: bytes) -> bytes:
        """A chunk between streams, with what compression decides taken out.

        The text around a stream carries its compressed length, and the table
        at the end carries byte offsets into the file. Compress the same
        drawing differently and both move, which is how the first version of
        this comparison still reported two figures as different after the
        pixels had been shown identical. Numbers inside a stream -- the
        coordinates, the widths, the colours -- are not touched.
        """
        chunk = re.sub(rb"/Length\s+\d+", b"/Length N", chunk)
        return re.sub(rb"\b\d{6,}\b", b"OFFSET", chunk)

    a, b = parts(rerun), parts(committed)
    if a is not None and b is not None:
        # Even indices are the text between streams; odd ones are inflated.
        a = [plumbing(c) if i % 2 == 0 else c for i, c in enumerate(a)]
        b = [plumbing(c) if i % 2 == 0 else c for i, c in enumerate(b)]
    if a is None or b is None:
        if rerun.read_bytes() == committed.read_bytes():
            return []
        return ["not identical, and the streams could not be read to compare "
                "the drawing instead"]
    if len(a) != len(b):
        return ["%d part(s) vs %d" % (len(a), len(b))]
    differing = [i for i, (x, y) in enumerate(zip(a, b, strict=True))
                 if not (x == y or (i % 2 == 1 and same_drawing(x, y)))]
    if not differing:
        return []

    # Say which parts and what they are. "Three parts differ" sent two
    # continuous-integration rounds guessing; a stream's kind is written in
    # the dictionary just before it.
    out = ["%d of %d part(s) of the drawing differ" % (len(differing), len(a))]
    for i in differing[:6]:
        kind = "the text between streams"
        if i % 2 == 1:
            # The dictionary this stream belongs to is the last one in the
            # text before it. Reading a fixed number of bytes back instead
            # picked up a neighbouring object and called a page of
            # coordinates an image.
            head = a[i - 1] if i else b""
            head = head[head.rfind(b"<<"):] if b"<<" in head else head[-200:]
            for key in (b"/FontFile", b"/Type1", b"/TrueType", b"/CIDFontType",
                        b"/Image", b"/ToUnicode", b"/Subtype /Form"):
                if key in head:
                    kind = "a stream declared %s" % key.decode()
                    break
            else:
                kind = "a content stream"
        out.append("    part %d (%s): %d bytes vs %d"
                   % (i, kind, len(a[i]), len(b[i])))
    return out


def diffs(rerun: Path, committed: Path) -> list[str]:
    if rerun.suffix == ".csv":
        return compare_csv(rerun, committed)
    if rerun.suffix == ".json":
        return json_diffs(rerun, committed)
    if rerun.suffix in (".png", ".tif", ".tiff"):
        return raster_diffs(rerun, committed)
    if rerun.suffix == ".pdf":
        return pdf_diffs(rerun, committed)
    if rerun.read_bytes() == committed.read_bytes():
        return []
    return ["%d bytes vs %d, not identical"
            % (rerun.stat().st_size, committed.stat().st_size)]


def refuse_work_directory(work: Path, results: Path,
                          root: Path | None = None) -> str | None:
    """Why this directory must not be the re-run's, or None if it may be.

    verify() deletes the re-run directory and writes it again, so the one
    thing it must never be handed is a directory that holds something else.
    The first version refused only the committed results directory itself,
    which let results/rerun, or the repository root, through -- and a tool
    that deletes takes the strictest rule it can, not the narrowest one that
    covers the mistake somebody already made.

    What is refused: the committed results, anything inside them, anything
    that contains them, the repository itself, anything that contains the
    repository, and any directory that already holds something without this
    script's marker file. What is allowed, deliberately, is a directory
    inside the repository: the default --work is ROOT/out_rerun, and
    .gitignore carries out_*/.

    *root* is resolved when the function runs, not when the module is
    imported, so that a caller -- a test, above all -- can point the whole
    rule at a tree of its own and have it mean something.
    """
    root = ROOT if root is None else root
    work, results, root = (p.resolve() for p in (work, results, root))
    if work == results or results.is_relative_to(work):
        return "it is the committed results directory, or contains it"
    if work.is_relative_to(results):
        return "it is inside the committed results directory"
    if work == root or root.is_relative_to(work):
        return "it is the repository, or contains it"
    if work.exists() and not work.is_dir():
        return "it exists and is not a directory"
    if work.exists() and work.is_dir() and any(work.iterdir()):
        # Only a directory this tool made is replaced without asking, and that
        # is decided by the marker it leaves, not by the shape of the tree: a
        # directory holding "tables" and "figures" is also what another
        # project's committed results look like.
        # is_file, not exists: a directory called .check_rerun would otherwise
        # have been taken as this tool's own mark.
        if not (work / MARKER).is_file():
            return ("it is not empty and carries no %s marker, so it was not "
                    "left by an earlier re-run" % MARKER)
    return None


def may_delete(work: Path) -> bool:
    """Only a directory this script created and marked may be removed."""
    return work.is_dir() and (work / MARKER).is_file()


def verify(files: list[str], results: Path, work: Path, runner, echo=print) -> list[str]:
    """Delete the claimed files from a copy of *results*, re-run, require them back.

    *runner* takes the re-run directory and returns a process exit code. It is a
    parameter so that the test suite can plant a stage that writes nothing, or
    writes something else, and prove this function notices: a checker nobody has
    seen fail is not evidence of anything.
    """
    missing = ["claimed by the build but not committed: %s" % f
               for f in files if not (results / f).exists()]
    if missing:
        return missing

    refusal = refuse_work_directory(work, results)
    if refusal:
        return ["refusing to re-run into %s: %s" % (work, refusal)]

    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(results, work)
    (work / MARKER).write_text(
        "Written by scripts/check_rerun.py. This directory is a copy of the "
        "committed results with the chosen stages' outputs removed, and the "
        "script will delete it again; nothing else belongs here.\n",
        encoding="utf-8")
    for f in files:
        (work / f).unlink()
    echo("%d claimed file(s) deleted from the copy" % len(files))

    rc = runner(work)
    if rc != 0:
        return ["the re-run itself failed with exit code %d" % rc]

    bad: list[str] = []
    for f in files:
        rerun = work / f
        if not rerun.exists():
            echo("%s: NOT WRITTEN" % f)
            bad.append("%s: the re-run did not write it" % f)
            continue
        found = diffs(rerun, results / f)
        echo("%s: %s" % (f, "reproduced" if not found else "DIFFERS"))
        for d in found:
            echo("    %s" % d)
        bad += ["%s: %s" % (f, d) for d in found]
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    ap.add_argument("--work", type=Path, default=ROOT / "out_rerun",
                    help="where the re-run writes (replaced on every run)")
    ap.add_argument("--raw-root", default=".")
    ap.add_argument("--keep", action="store_true",
                    help="leave the re-run directory in place for inspection")
    args = ap.parse_args()

    files = claimed(args.stages)
    if not files:
        print("those stages claim no committed output; nothing to check")
        return 1

    work = args.work.resolve()
    # The refusal is read here, before anything is created, and the clean-up
    # below is reached only if the directory was acceptable. Deciding twice --
    # verify refusing and main deleting on its own test -- is what made the
    # first version destroy a refused path, and a marker on a refused path
    # would have been enough to do it again.
    refusal = refuse_work_directory(work, RESULTS)
    if refusal:
        print("refusing to re-run into %s: %s" % (work, refusal))
        print("\nnothing was compared")
        return 1

    def runner(target: Path) -> int:
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"),
                               "--stages", *args.stages, "--results", str(target),
                               "--raw-root", args.raw_root], cwd=ROOT).returncode

    bad = verify(files, RESULTS, work, runner)
    # Everything the check found is printed. The earlier version hid any line
    # holding ": ", which is every refusal and every missing-file reason, and
    # then counted files by splitting on that same colon, so a run that
    # compared nothing at all could still announce that all but one file had
    # reproduced.
    claimed_set = set(files)
    failed = {b.split(":")[0] for b in bad if b.split(":")[0] in claimed_set}
    blocked = [b for b in bad if b.split(":")[0] not in claimed_set]
    for b in blocked:
        print(b)
    # Only a directory that passed the refusal above and carries this
    # script's marker is deleted.
    if not args.keep and may_delete(work):
        shutil.rmtree(work)
    if blocked:
        print("\nnothing was compared")
        return 1
    print("\n%d of %d claimed file(s) reproduced"
          % (len(files) - len(failed), len(files)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
