"""Check the installed packages against requirements-lock.txt before a run.

The lock file records the versions every committed result was produced with.
Nothing enforced it: ``run_all.py`` launched the stages with ``sys.executable``,
so whichever interpreter happened to be active rebuilt the tables and figures,
and a minor version of a plotting or array library is free to change what it
writes without anything in this repository changing at all. That is not a
hypothetical: in a sibling project a matplotlib upgrade from 3.9 to 3.10
silently replaced five figures that a human had already reviewed.

Only the packages that can move a committed number or byte are checked by
default -- the numerical and image libraries, pandas, and matplotlib. A
difference in, say, ``click`` does not change a result, and failing on it would
teach the reader to pass the override. ``--all`` checks every pinned package,
and ``--json`` prints the comparison for a log.

    python scripts/check_environment.py
    python scripts/check_environment.py --all

Exit code 0 when the checked packages match the lock file, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"

# The packages whose version can change a committed result. Names as pip
# reports them; comparison is on the normalised name, because the lock file
# carries "simpleitk" and the distribution calls itself "SimpleITK".
DECIDES_RESULTS = (
    "numpy", "scipy", "pandas", "SimpleITK", "scikit-image", "networkx",
    "connected-components-3d", "kimimaro", "matplotlib",
    # The figures are compared by scripts/check_rerun.py, and matplotlib does
    # not draw them alone: Pillow decodes the PNG and the TIFF whose pixels
    # are compared, and fontTools subsets the fonts whose outlines end up in
    # the PDF. Either one can move a pixel with matplotlib itself unchanged.
    "pillow", "fonttools", "tifffile",
    # kimimaro is pinned above, but it does none of this arithmetic itself:
    # the distance transform, the path search and the hole filling behind the
    # centerline length and the coverage numbers are these. A change in any
    # of them moves a reported result while kimimaro's own version stands
    # still.
    "edt", "dijkstra3d", "fill_voids", "fastremap", "xs3d", "osteoid",
    "crackle-codec",
)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def locked(path: Path = LOCK) -> tuple[dict[str, str], list[str]]:
    """The pinned versions, and the lines this parser could not read.

    The unreadable lines are returned rather than skipped. A lock file
    regenerated with hashes, extras or environment markers would otherwise
    lose packages quietly, and --all would report success over a shrinking
    set without saying that it had shrunk.
    """
    out: dict[str, str] = {}
    unread: list[str] = []
    continuation = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if continuation:
            continuation = line.endswith("\\")
            if line.startswith("--hash"):
                continue
        if not line:
            continue
        if line.startswith(("-r", "-c")):
            # A lock that includes another file pins packages somewhere this
            # check never reads; skipping the line silently would let --all
            # report agreement over a set it does not know the size of.
            unread.append(raw.strip())
            continue
        if line.startswith("--"):
            unread.append(raw.strip())
            continue
        continuation = line.endswith("\\")
        spec = line.rstrip("\\").strip()
        if ";" in spec:
            # An environment marker means the pin applies to some interpreters
            # and not others. Taking the text before it would record a pin
            # that may not be this interpreter's, and the last such line would
            # quietly win over the first.
            unread.append(raw.strip())
            continue
        spec = re.sub(r"\[[^\]]*\]", "", spec)              # drop extras
        if "==" not in spec:
            unread.append(raw.strip())
            continue
        name, version = spec.split("==", 1)
        # Trailing tokens are not dropped. "numpy==2.5.3 --hash=sha256:..."
        # is a pin this parser does not understand, and taking the first word
        # would record it as understood.
        parts = version.split()
        version = parts[0] if parts else ""
        key = normalise(name)
        if not name.strip() or not version or len(parts) > 1 or name.strip().split() != [name.strip()]:
            unread.append(raw.strip())
            continue
        if key in out and out[key] != version:
            unread.append("%s (pinned twice, as %s and %s)"
                          % (raw.strip(), out[key], version))
            continue
        out[key] = version
    return out, unread


def installed(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def compare(want: dict[str, str], scope: list[str], lock_name: str = LOCK.name) -> list[str]:
    """Report every checked package that is absent or at another version."""
    bad = []
    have = installed(scope)
    for name in scope:
        key = normalise(name)
        if key not in want:
            bad.append("%s is checked but not pinned in %s" % (name, lock_name))
            continue
        if have[name] is None:
            bad.append("%s is not installed; %s pins %s"
                       % (name, lock_name, want[key]))
        elif normalise(have[name]) != normalise(want[key]):
            bad.append("%s is %s; %s pins %s"
                       % (name, have[name], lock_name, want[key]))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="check every package pinned in the lock file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lock", type=Path, default=LOCK)
    args = ap.parse_args()

    want, unread = locked(args.lock)
    scope = sorted(want) if args.all else list(DECIDES_RESULTS)
    bad = compare(want, scope, args.lock.name)
    bad += ["%s has a line this check cannot read, so the package in it is "
            "not being checked: %r" % (args.lock.name, line) for line in unread]

    if args.json:
        print(json.dumps({
            "python": sys.version.split()[0],
            "lock": args.lock.name,
            "checked": scope,
            "installed": installed(scope),
            "mismatches": bad,
        }, indent=2))
    else:
        print("python %s, %d package(s) checked against %s"
              % (sys.version.split()[0], len(scope), args.lock.name))
        for b in bad:
            print("  %s" % b)
        if not bad:
            print("  every checked package is at the locked version")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
