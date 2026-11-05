"""Run each gate with a defect planted in it, on a tree built here.

Every check in scripts/ carries a --selftest that plants the defects it exists
for. Those live behind a command-line flag, so a suite that never calls them
proves nothing about the gates: an independent review of 2026-11-05 pointed
out that the only executed tests covered the comparison rules and nothing
else. These cases call the gates directly.

Everything they need is written here, in a temporary directory. The first
version of this file copied scripts/ and asserted that the live repository was
clean, which made it fail wherever the whole tree is not present -- including
the review harness, which copies only the files it was given. Those assertions
live in tests/test_repository.py instead; these must pass anywhere.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_entrypoints  # noqa: E402
import check_environment  # noqa: E402
import check_tables  # noqa: E402
from run_all import OUTPUTS, STAGES  # noqa: E402

GOOD_ENTRY_POINT = ('"""Doc."""\nimport sys\n\n'
                    'sys.stdout.reconfigure(encoding="utf-8")\n\n'
                    'if __name__ == "__main__":\n    pass\n')

# A stage script as the gate expects one: a committed constant, an argparse
# default on a flag the stage passes, and a helper that takes the directory.
STAGE_SCRIPT = '''"""A stage script."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

sys.stdout.reconfigure(encoding="utf-8")


def read(name: str, tables: Path = TABLES) -> str:
    return (tables / name).read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=TABLES)
    args = ap.parse_args()
    print(read("indices.csv", args.tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class StageMap(unittest.TestCase):
    def test_every_stage_has_an_output_claim(self):
        self.assertEqual(sorted(OUTPUTS), sorted(STAGES))

    def test_no_claim_belongs_to_two_stages(self):
        seen: dict[str, str] = {}
        for stage, files in OUTPUTS.items():
            for f in files:
                self.assertNotIn(f, seen, "%s claimed by %s and %s"
                                 % (f, seen.get(f), stage))
                seen[f] = stage


class BuildCoverageFires(unittest.TestCase):
    """The register against the stage map; neither file tree is needed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # An empty results directory: the shipped-file walk then has nothing
        # to say, and the register rules are what these cases exercise.
        self.results = Path(self._tmp.name) / "results"
        self.results.mkdir(parents=True)
        self.baseline = check_tables.build_coverage(results=self.results)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def fresh(self) -> dict:
        return {k: tuple(v) for k, v in OUTPUTS.items()}

    def found(self, outputs) -> list[str]:
        return [f for f in check_tables.build_coverage(outputs=outputs,
                                                       results=self.results)
                if f not in self.baseline]

    def test_a_result_with_no_stage_to_write_it(self):
        maimed = self.fresh()
        maimed["sweep"] = ()
        self.assertTrue([f for f in self.found(maimed)
                         if "threshold_sweep.csv" in f])

    def test_a_hand_run_claimed_as_a_build_product(self):
        claimed = self.fresh()
        claimed["tables"] = claimed["tables"] + ("tables/eval_hqcolon.csv",)
        self.assertTrue([f for f in self.found(claimed)
                         if "registered as an orphan" in f])

    def test_a_claim_no_command_is_pointed_at(self):
        elsewhere = self.fresh()
        elsewhere["strata"] = ("figures/stratified.png",)
        self.assertTrue([f for f in self.found(elsewhere)
                         if "none of its commands is pointed at" in f])


class StageInputsFires(unittest.TestCase):
    """Each plant is one script written here, under a stage script's name.

    stage_inputs also reports the stage scripts that are not in the directory;
    those findings are expected and ignored, which is why each case looks for
    its own sentence rather than counting.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "scripts").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def plant(self, name: str, text: str) -> list[str]:
        (self.root / "scripts" / name).write_text(text, encoding="utf-8")
        return check_tables.stage_inputs(self.root)

    def test_a_correct_stage_script_is_not_reported(self):
        found = self.plant("stats_clustered.py", STAGE_SCRIPT)
        self.assertEqual([f for f in found if "stats_clustered" in f], [])

    def test_a_helper_call_that_stops_passing_the_directory(self):
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                'read("indices.csv")'))
        self.assertTrue([f for f in found if "without tables" in f])

    def test_a_committed_path_spelled_out(self):
        found = self.plant("make_cohort_table.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                'open("results/tables/x.csv").read()'))
        self.assertTrue([f for f in found if "is the committed tree" in f])

    def test_a_constant_used_outside_an_argparse_default(self):
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                '(TABLES / "indices.csv").read_text()'))
        self.assertTrue([f for f in found if "outside an argparse default" in f])

    def test_a_default_that_says_results_on_its_own(self):
        # The committed tree reached through a bare segment, on a flag no
        # stage passes: the default-override rule had its own, narrower idea
        # of what a committed path looks like until 2026-11-05.
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace(
                               'ap.add_argument("--tables", type=Path, default=TABLES)',
                               'ap.add_argument("--extra", type=Path,\n'
                               '                    default=Path("results") / "tables")\n'
                               '    ap.add_argument("--tables", type=Path, default=TABLES)'))
        self.assertTrue([f for f in found if "no stage passes that flag" in f])

    def test_a_helper_name_that_belongs_to_another_module(self):
        # csv.read() is not this script's read(): following every attribute
        # call by name alone reported an honest script.
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                'csv.read("indices.csv")'))
        self.assertEqual([f for f in found if "without tables" in f], [])

    def test_a_default_no_stage_overrides(self):
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace('ap.add_argument("--tables", type=Path, default=TABLES)',
                                                'ap.add_argument("--extra", type=Path, default=TABLES)\n'
                                                '    ap.add_argument("--tables", type=Path, default=TABLES)'))
        self.assertTrue([f for f in found if "no stage passes that flag" in f])

    def test_a_committed_path_spelled_relatively(self):
        found = self.plant("make_cohort_table.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                'open("../results/tables/x.csv").read()'))
        self.assertTrue([f for f in found if "is the committed tree" in f])

    def test_a_committed_path_built_from_pieces(self):
        found = self.plant("make_cohort_table.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                '(ROOT / "results" / "tables").iterdir()'))
        self.assertTrue([f for f in found if "is the committed tree" in f])

    def test_a_helper_reached_as_an_attribute(self):
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace('read("indices.csv", args.tables)',
                                                'self.read("indices.csv")'))
        self.assertTrue([f for f in found if "without tables" in f])

    def test_a_helper_passed_around_rather_than_called(self):
        found = self.plant("stats_clustered.py",
                           STAGE_SCRIPT.replace("    return 0",
                                                "    loader = read\n    return 0"))
        self.assertTrue([f for f in found if "passes read around" in f])


class EntryPointsFire(unittest.TestCase):
    def plant(self, text: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "planted.py").write_text(text, encoding="utf-8")
            return check_entrypoints.check(folder)

    def test_a_correct_entry_point_is_silent(self):
        self.assertEqual(self.plant(GOOD_ENTRY_POINT), [])

    def test_a_missing_console_line(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            'sys.stdout.reconfigure(encoding="utf-8")\n', ""))
        self.assertTrue([f for f in found if "cp932" in f])

    def test_a_print_before_the_console_line(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            'sys.stdout.reconfigure(encoding="utf-8")\n',
            'print("x")\nsys.stdout.reconfigure(encoding="utf-8")\n'))
        self.assertTrue([f for f in found if "only after a" in f])

    def test_a_print_in_a_class_body_before_the_console_line(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            'sys.stdout.reconfigure(encoding="utf-8")\n',
            'class B:\n    print("x")\n\nsys.stdout.reconfigure(encoding="utf-8")\n'))
        self.assertTrue([f for f in found if "only after a" in f])

    def test_a_function_defined_before_it_may_print_when_called(self):
        self.assertEqual(self.plant(GOOD_ENTRY_POINT.replace(
            'sys.stdout.reconfigure(encoding="utf-8")\n',
            'def later():\n    print("x")\n\n'
            'sys.stdout.reconfigure(encoding="utf-8")\n')), [])

    def test_a_private_import(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            "import sys\n", "import sys\nimport json, tools\n"))
        self.assertTrue([f for f in found if "does not contain" in f])

    def test_a_private_directory_on_sys_path(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            "import sys\n", 'import sys\nsys.path.insert(0, "tools")\n'))
        self.assertTrue([f for f in found if "on sys.path" in f])

    def test_a_private_directory_spelled_relatively(self):
        found = self.plant(GOOD_ENTRY_POINT.replace(
            "import sys\n", 'import sys\nsys.path.insert(0, "../tools")\n'))
        self.assertTrue([f for f in found if "on sys.path" in f])


class EnvironmentFires(unittest.TestCase):
    LOCK = "numpy==2.5.3\nmatplotlib==3.11.1\npillow==12.3.0\n"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def lock(self, text: str) -> Path:
        path = self.dir / "lock.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_plain_lock_reads_cleanly(self):
        want, unread = check_environment.locked(self.lock(self.LOCK))
        self.assertEqual(unread, [])
        self.assertEqual(want["numpy"], "2.5.3")

    def test_a_version_that_moved(self):
        # Neither the package nor its installation is what is under test, so
        # neither is relied on: compare() is given an installed version of our
        # own. Two earlier versions of this case asserted only that some
        # finding appeared, and then that it named the pinned version -- which
        # the "not installed" message names as well, so on a machine without
        # matplotlib both passed without comparing anything.
        want, _ = check_environment.locked(
            self.lock(self.LOCK.replace("3.11.1", "3.9.4")))
        with mock.patch.object(check_environment, "installed",
                               return_value={"matplotlib": "3.11.1"}):
            found = check_environment.compare(want, ["matplotlib"])
        self.assertEqual(len(found), 1, found)
        self.assertIn("matplotlib is 3.11.1", found[0])
        self.assertIn("3.9.4", found[0])

    def test_a_package_that_is_not_installed_at_all(self):
        want, _ = check_environment.locked(self.lock(self.LOCK))
        with mock.patch.object(check_environment, "installed",
                               return_value={"matplotlib": None}):
            found = check_environment.compare(want, ["matplotlib"])
        self.assertEqual(len(found), 1, found)
        self.assertIn("not installed", found[0])

    def test_a_version_that_matches_is_silent(self):
        want, _ = check_environment.locked(self.lock(self.LOCK))
        with mock.patch.object(check_environment, "installed",
                               return_value={"matplotlib": "3.11.1"}):
            self.assertEqual(
                check_environment.compare(want, ["matplotlib"]), [])

    def test_a_line_with_a_trailing_token(self):
        _, unread = check_environment.locked(
            self.lock(self.LOCK.replace("numpy==2.5.3",
                                        "numpy==2.5.3 --hash=sha256:dead")))
        self.assertTrue(unread)

    def test_a_line_with_an_environment_marker(self):
        _, unread = check_environment.locked(
            self.lock(self.LOCK.replace(
                "numpy==2.5.3", 'numpy==1.26.4 ; python_version < "3.12"')))
        self.assertTrue(unread)

    def test_a_package_pinned_twice(self):
        _, unread = check_environment.locked(
            self.lock(self.LOCK + "numpy==1.26.4\n"))
        self.assertTrue([u for u in unread if "pinned twice" in u])

    def test_an_include_of_another_file(self):
        _, unread = check_environment.locked(
            self.lock("-r other.txt\n" + self.LOCK))
        self.assertTrue(unread)


if __name__ == "__main__":
    unittest.main()
