"""Compare a re-run's tables with the committed ones, ignoring run-time columns.

    python scripts/compare_tables.py out_check/tables results/tables [name ...]

It lives in ``scripts/`` rather than in the private ``tools/`` directory
because the release has to be able to do this: the README tells a reader to
re-run into another directory and compare, and a comparison they cannot run is
not an instruction. ``scripts/check_rerun.py`` drives it over the files a
stage claims to write.

Reports, per table, the columns whose values differ (numeric tolerance 1e-9
relative) and how many rows are affected. Run-time columns (``*_sec``) differ by
construction and are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["PatientID", "role", "index", "phantom", "edge", "characteristic", "value", "metric"]


def compare(a_path: Path, b_path: Path) -> list[str]:
    # Read as text, with pandas' own missing-value list switched off. Left to
    # itself it turns the strings NA, N/A, None, null and nan into the same
    # empty cell, so a result that changed from "None" to "" would compare
    # equal, and it infers numbers, so the identifier 007 would equal 7.
    a, b = (pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[])
            for p in (a_path, b_path))
    out = []
    if len(a) != len(b):
        out.append(f"row count {len(a)} vs {len(b)}")
    common = [c for c in a.columns if c in b.columns]
    for c in sorted(set(a.columns) ^ set(b.columns)):
        out.append(f"column only in {'re-run' if c in a.columns else 'committed'}: {c}")
    n = min(len(a), len(b))
    for c in common:
        if c.endswith("_sec"):
            continue
        x, y = a[c].iloc[:n], b[c].iloc[:n]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        # Two cells agree when they are the same text, or both empty, or both
        # numeric and close. The earlier version took "both coerce to NaN" as
        # agreement, which in a column holding numbers and words made every
        # pair of different words equal: REFUSED and ACCEPTED compared the
        # same, and a changed verdict was reported as reproduced. Emptiness is
        # read from the cells, not from the failure to parse them.
        same_text = x == y
        # Numeric agreement is for a number written differently -- 1.50 and
        # 1.5 -- and not for text that happens to parse. A leading zero makes
        # an identifier, not a number: reading the file as text was not enough
        # on its own, because to_numeric then made PatientID 0001 equal to 1.
        looks_like_an_id = (x.str.match(r"^[+-]?0\d") | y.str.match(r"^[+-]?0\d"))
        both_numeric = xn.notna() & yn.notna() & ~looks_like_an_id
        close = both_numeric & np.isclose(xn, yn, rtol=1e-9, atol=1e-12)
        diff = ~(same_text | close)
        if diff.any():
            out.append(f"{c}: {int(diff.sum())} rows differ")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rerun", type=Path, help="the re-run's tables directory")
    ap.add_argument("committed", type=Path, help="the committed tables directory")
    ap.add_argument("names", nargs="*",
                    help="table names to compare; default is every CSV in the re-run")
    args = ap.parse_args()
    a_dir, b_dir = args.rerun, args.committed
    names = args.names or sorted(p.name for p in a_dir.glob("*.csv"))
    # Nothing to compare is a failure, not a pass: pointed at an empty or
    # misspelled directory, this used to print nothing and exit 0, and a
    # caller reading only the exit code would have taken that for agreement.
    if not names:
        print(f"no CSV to compare in {a_dir}")
        return 1
    bad = 0
    compared = 0
    for name in names:
        name = name if name.endswith(".csv") else name + ".csv"
        if not (a_dir / name).exists():
            print(f"{name}: not in the re-run")
            bad += 1
            continue
        if not (b_dir / name).exists():
            # Asked for by name, or found in the re-run, and absent from the
            # committed tree: either way nothing was checked about it.
            print(f"{name}: not committed")
            bad += 1
            continue
        diffs = compare(a_dir / name, b_dir / name)
        bad += bool(diffs)
        compared += 1
        print(f"{name}: {'identical (excluding *_sec)' if not diffs else ''}")
        for d in diffs:
            print(f"    {d}")
    print(f"{compared} of {len(names)} table(s) compared")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
