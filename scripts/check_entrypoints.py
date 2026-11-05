"""Every entry point must survive a console that is not UTF-8, and stand alone.

Two defects this catches, both of which have been shipped by this programme
before:

* **the console.** GitHub's runners are UTF-8, so a script that prints an em
  dash passes in CI for ever and dies on the author's Japanese Windows console
  with ``UnicodeEncodeError: 'cp932'``. In a sibling project two of the twelve
  scripts the public job runs died that way on the first character of their
  first message, and one of them was the script the manuscript cited as the
  evidence that a design was frozen. The fix is one line per entry point, and
  this check is what keeps the line there.
* **the standalone tree.** The published release does not contain ``tools/``
  or ``manuscript/``. A script that imports from either is a script the reader
  cannot run, and byte-comparing the export against the public repository
  cannot see it: a module missing from both sides matches.

    python scripts/check_entrypoints.py
    python scripts/check_entrypoints.py --run-help   # argparse under cp932
    python scripts/check_entrypoints.py --selftest

``--run-help`` runs every entry point with ``--help`` and ``PYTHONIOENCODING``
set to cp932, which is the author's console and the one encoding that has
actually broken a release. argparse prints the module docstring, so this
exercises the text each script leads with, not just the statement above it.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Directories that the public release does not carry (tools/export_public.py).
NOT_PUBLISHED = {"tools", "manuscript"}


def entry_points(folder: Path = SCRIPTS) -> list[Path]:
    return sorted(p for p in folder.glob("*.py")
                  if "__main__" in p.read_text(encoding="utf-8"))


# What may stand between the top of a module and the line that fixes its
# console. None of these writes to stdout; anything else might, and would then
# write before the fix (a bare print, a loop, a with-block, a call that logs).
QUIET_PROLOGUE = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                  ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

# Calls that configure and do not print. Named one by one rather than allowing
# calls in general: the point of the ordering rule is that an arbitrary call
# may write, and the twenty-four occurrences in this tree are all of these two.
QUIET_CALLS = frozenset({
    "sys.path.insert", "sys.path.append", "matplotlib.use",
    "warnings.filterwarnings", "warnings.simplefilter", "logging.basicConfig",
})


def is_print_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = ast.unparse(node.func)
    return name == "print" or name.endswith((".write", ".print"))


def writes(node: ast.stmt) -> bool:
    """Does this statement print, anywhere inside it?

    An assignment, a class body and a decorator all run at import, so the
    statement kinds in QUIET_PROLOGUE are not silent by nature; this looks
    through them for the one call that certainly writes. A call that prints
    indirectly, through a function defined elsewhere, is still not caught --
    stated here rather than implied.
    """
    def runs_now(n: ast.AST) -> list[ast.AST]:
        """The parts of a definition that execute where it is written."""
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return (list(n.decorator_list)
                    + [d for d in n.args.defaults if d is not None]
                    + [d for d in n.args.kw_defaults if d is not None])
        return [n]

    def scan(n: ast.AST) -> bool:
        # The node itself first: a decorator or a default argument can be the
        # printing call, not merely contain one.
        if is_print_call(n):
            return True
        for child in ast.iter_child_nodes(n):
            # A function body does not run where it is defined. A class body
            # does: it is executed to build the class, and that includes the
            # decorators and default arguments of the methods inside it, so
            # skipping a nested function means skipping its body only.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(scan(part) for part in runs_now(child)):
                    return True
                continue
            if scan(child):
                return True
        return False

    # A class's decorators, bases and body all run at the point of definition.
    return any(scan(part) for part in runs_now(node))


def is_reconfigure(node: ast.stmt) -> tuple[str, str] | None:
    """(stream, encoding) if this statement is a stdout/stderr reconfigure."""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "reconfigure"):
        return None
    encoding = next((kw.value.value for kw in node.value.keywords
                     if kw.arg == "encoding" and isinstance(kw.value, ast.Constant)),
                    None)
    return (ast.unparse(node.value.func.value), str(encoding or ""))


def imported_roots(tree: ast.AST):
    """Every top-level package name imported anywhere in the module.

    Every name in the statement, not the first: ``import json, tools`` hid the
    second one from the earlier version of this check.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.level:
                yield (node.module or "").split(".")[0]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]


def inspect(path: Path, entry_point: bool = True) -> list[str]:
    """Findings for one file: the console line, and imports it cannot have."""
    bad: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))

    if entry_point:
        reconfigured: set[str] = set()
        at, noisy_at, noisy_kind = None, None, None
        for i, node in enumerate(tree.body):
            found = is_reconfigure(node)
            if found:
                stream, encoding = found
                if encoding.lower().replace("-", "") == "utf8":
                    reconfigured.add(stream)
                    if stream == "sys.stdout" and at is None:
                        at = i
                continue
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                continue  # the module docstring
            if writes(node):
                if noisy_at is None:
                    noisy_at, noisy_kind = i, "printing"
                continue
            if isinstance(node, QUIET_PROLOGUE):
                continue
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) in QUIET_CALLS):
                continue
            if noisy_at is None:
                noisy_at = i
                noisy_kind = (ast.unparse(node.value.func)
                              if isinstance(node, ast.Expr)
                              and isinstance(node.value, ast.Call)
                              else type(node).__name__)
        preceded_by_output = (noisy_kind if at is not None and noisy_at is not None
                              and noisy_at < at else None)

        if "sys.stdout" not in reconfigured:
            bad.append("%s does not set its stdout to UTF-8, so it dies on a "
                       "cp932 console at its first non-ASCII character" % path.name)
        elif preceded_by_output:
            # Setting the encoding after something has already run is not the
            # same as setting it first, and the AST rule used to accept the
            # call anywhere in the module body.
            bad.append("%s sets its stdout to UTF-8 only after a %s statement "
                       "has run, so anything that statement prints is written "
                       "with the console's own encoding"
                       % (path.name, preceded_by_output))

    for module in imported_roots(tree):
        if module in NOT_PUBLISHED:
            bad.append("%s imports %s, which the published release does not "
                       "contain" % (path.name, module))

    # An import by name is not the only way in. Putting a directory on
    # sys.path and importing a module out of it by its bare name reads the
    # same as any other import, and sys.path.insert is on the list of calls
    # that may precede the console line.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and ast.unparse(node.func) in ("sys.path.insert", "sys.path.append",
                                               "sys.path.extend",
                                               "site.addsitedir")):
            # Every string in the call, by path segment: matching the quoted
            # folder name alone missed "../tools" and "x/tools". A directory
            # held in a variable still cannot be followed from here.
            for child in ast.walk(node):
                if not (isinstance(child, ast.Constant)
                        and isinstance(child.value, str)):
                    continue
                parts = [p for p in child.value.replace("\\", "/").split("/") if p]
                for folder in sorted(NOT_PUBLISHED & set(parts)):
                    bad.append("%s puts %s on sys.path, and the published "
                               "release does not contain it"
                               % (path.name, folder))
    return bad


def check(folder: Path = SCRIPTS) -> list[str]:
    """Entry points in full; every other module for the import rule.

    A helper in scripts/ that imports tools/ is not an entry point, so the
    earlier version never read it -- and the entry point that imports the
    helper inherits the problem.
    """
    entries = set(entry_points(folder))
    bad = [b for p in sorted(entries) for b in inspect(p)]
    for path in sorted(folder.glob("*.py")):
        if path not in entries:
            bad += inspect(path, entry_point=False)
    return bad


def run_help(folder: Path = SCRIPTS, encoding: str = "cp932") -> list[str]:
    env = dict(os.environ, PYTHONIOENCODING=encoding)
    bad = []
    for path in entry_points(folder):
        proc = subprocess.run([sys.executable, str(path), "--help"],
                              cwd=ROOT, env=env, capture_output=True)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            bad.append("%s --help fails with PYTHONIOENCODING=%s: %s"
                       % (path.name, encoding, tail[-1] if tail else "no output"))
        print("  %-36s %s" % (path.name, "ok" if proc.returncode == 0 else "FAILS"))
    return bad


def selftest() -> int:
    """Plant each defect in a directory of its own and require it to fire."""
    ok = True
    good = ('"""Doc."""\nimport sys\n\n'
            'sys.stdout.reconfigure(encoding="utf-8")\n\n'
            'if __name__ == "__main__":\n    pass\n')
    cases = [
        ("an entry point without the console line",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n', ""),
         "dies on a cp932 console"),
        ("the console line set to the wrong encoding",
         good.replace('encoding="utf-8"', 'encoding="latin-1"'),
         "dies on a cp932 console"),
        ("an entry point that imports the private tools directory",
         good.replace("import sys\n", "import sys\nfrom tools import thing\n"),
         "the published release does not contain"),
        # The console line is set, but after something has already printed.
        ("the console line set after a statement that writes",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'print("starting \\u2014 now")\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         "only after a"),
        # The print is hidden in an assignment, which the prologue allows.
        ("something prints inside a statement the prologue allows",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'BANNER = print("starting \\u2014 now")\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         "only after a"),
        ("the console line set inside main() rather than at import",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n\n',
                      "")
             .replace('if __name__ == "__main__":\n    pass\n',
                      'def main():\n    sys.stdout.reconfigure(encoding="utf-8")\n\n'
                      'if __name__ == "__main__":\n    main()\n'),
         "dies on a cp932 console"),
        # The private import is the second name in the statement.
        ("a private import hidden behind another name in the same statement",
         good.replace("import sys\n", "import sys\nimport json, tools\n"),
         "the published release does not contain"),
        # A class body runs when the class is defined, unlike a function body.
        ("something prints in a class body before the console line",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'class Banner:\n    print("starting \\u2014 now")\n\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         "only after a"),
        # The printing call is the default argument itself, not inside one.
        ("a method's default argument prints while the class is defined",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'class Banner:\n'
                      '    def show(self, line=print("starting \\u2014 now")):\n'
                      '        pass\n\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         "only after a"),
        ("a print inside a function defined before the console line is not a "
         "finding, because defining it runs nothing",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'def later():\n    print("only when called")\n\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         None),
        ("a private directory reached through sys.path rather than by name",
         good.replace("import sys\n",
                      'import sys\nsys.path.insert(0, "tools")\n'),
         "on sys.path"),
        ("the same directory spelled relatively",
         good.replace("import sys\n",
                      'import sys\nsys.path.insert(0, "../tools")\n'),
         "on sys.path"),
        ("a quiet configuration call before the console line is not a finding",
         good.replace('sys.stdout.reconfigure(encoding="utf-8")\n',
                      'sys.path.insert(0, ".")\n'
                      'sys.stdout.reconfigure(encoding="utf-8")\n'),
         None),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean"
        clean.mkdir()
        (clean / "fine.py").write_text(good, encoding="utf-8")
        if check(clean):
            print("  NOISY    a correct entry point was reported")
            ok = False
        else:
            print("  SILENT   a correct entry point (negative control)")

        for name, text, expect in cases:
            folder = Path(tmp) / name.replace(" ", "_")[:40]
            folder.mkdir()
            (folder / "planted.py").write_text(text, encoding="utf-8")
            if text == good:
                print("  NO-OP    %s (the injection changed nothing)" % name)
                ok = False
                continue
            found = check(folder)
            if expect is None:                     # must stay silent
                print("  %-8s %s" % ("SILENT" if not found else "NOISY", name))
                if found:
                    ok = False
                    print("           reported %s" % found)
                continue
            hit = [f for f in found if expect in f]
            print("  %-8s %s" % ("DETECTED" if hit else "MISSED", name))
            if not hit:
                ok = False
                print("           expected %r, got %s" % (expect, found or "nothing"))
    print("selftest %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-help", action="store_true",
                    help="also run every entry point's --help under cp932")
    ap.add_argument("--encoding", default="cp932")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    bad = check()
    if args.run_help:
        bad += run_help(encoding=args.encoding)
    for b in bad:
        print("FAIL %s" % b)
    if not bad:
        print("OK   %d entry point(s): UTF-8 stdout, no private imports%s"
              % (len(entry_points()),
                 ", and --help survives %s" % args.encoding if args.run_help else ""))
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
