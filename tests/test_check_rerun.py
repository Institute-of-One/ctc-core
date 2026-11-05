"""Prove the reproduction check fails when a stage stops reproducing.

A gate that has never been seen to fail is not evidence. The three ways a
re-run can be wrong are planted here, each on a tiny results tree with a stage
of our own: the stage writes nothing, it writes a different number, and it
writes different bytes. The fourth case -- it writes exactly what is committed
-- has to stay silent, or the gate would fail every honest run.

The deletion is the part under test. Without it the committed copy would still
be sitting in the re-run directory and the first case, the one where a stage
quietly stops writing a result, would pass.

Written as unittest.TestCase rather than with pytest fixtures so that one file
serves both runners: pytest runs it with the rest of the suite, and
``unittest discover`` runs it where that is the only accepted command.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_rerun  # noqa: E402
import compare_tables  # noqa: E402

from check_rerun import (  # noqa: E402
    MARKER,
    claimed,
    json_diffs,
    may_delete,
    refuse_work_directory,
    verify,
)
from compare_tables import compare as compare_csv  # noqa: E402

FILES = ["tables/thing.csv", "tables/thing.json", "figures/thing.png"]


def quiet(*args) -> None:
    """The echo the checker prints with; silenced for the tests."""


class ReRunCheck(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.results, self.work = root / "results", root / "work"
        (self.results / "tables").mkdir(parents=True)
        (self.results / "figures").mkdir(parents=True)
        (self.results / "tables" / "thing.csv").write_text(
            "PatientID,value,elapsed_sec\n0001,1.5,12.0\n", encoding="utf-8")
        (self.results / "tables" / "thing.json").write_text(
            json.dumps({"median": 1.5, "n": 3, "elapsed_sec": 12.0}), encoding="utf-8")
        (self.results / "figures" / "thing.png").write_bytes(
            b"\x89PNG\r\n\x1a\n-pixels-")
        # A committed file no stage in scope claims: it must not be deleted,
        # and it must not be reported either.
        (self.results / "tables" / "other.csv").write_text("x\n1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def copy_back(self):
        """A stage that reproduces the committed files exactly."""
        def runner(work: Path) -> int:
            for name in FILES:
                (work / name).write_bytes((self.results / name).read_bytes())
            return 0
        return runner

    def test_an_honest_rerun_is_silent(self):
        self.assertEqual(
            verify(FILES, self.results, self.work, self.copy_back(), echo=quiet), [])
        self.assertTrue((self.work / "tables" / "other.csv").exists())

    def test_a_stage_that_writes_nothing_is_caught(self):
        bad = verify(FILES, self.results, self.work, lambda work: 0, echo=quiet)
        self.assertEqual(len(bad), 3)
        self.assertTrue(all("did not write it" in b for b in bad))

    def test_a_changed_number_is_caught(self):
        # The run-time column is left alone, so this cannot pass because the
        # *_sec rule is broken: only the value moves.
        def runner(work: Path) -> int:
            self.copy_back()(work)
            (work / "tables" / "thing.csv").write_text(
                "PatientID,value,elapsed_sec\n0001,1.6,12.0\n", encoding="utf-8")
            (work / "tables" / "thing.json").write_text(
                json.dumps({"median": 1.6, "n": 3, "elapsed_sec": 12.0}),
                encoding="utf-8")
            return 0

        bad = verify(FILES, self.results, self.work, runner, echo=quiet)
        self.assertTrue([b for b in bad if b.startswith("tables/thing.csv")])
        self.assertTrue([b for b in bad if b.startswith("tables/thing.json")])
        self.assertFalse([b for b in bad if b.startswith("figures/")])

    def test_a_changed_figure_is_caught(self):
        def runner(work: Path) -> int:
            self.copy_back()(work)
            (work / "figures" / "thing.png").write_bytes(b"\x89PNG\r\n\x1a\n-other-")
            return 0

        bad = verify(FILES, self.results, self.work, runner, echo=quiet)
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("figures/thing.png"))

    def test_a_run_time_difference_alone_is_not_reported(self):
        def runner(work: Path) -> int:
            self.copy_back()(work)
            (work / "tables" / "thing.csv").write_text(
                "PatientID,value,elapsed_sec\n0001,1.5,99.0\n", encoding="utf-8")
            (work / "tables" / "thing.json").write_text(
                json.dumps({"median": 1.5, "n": 3, "elapsed_sec": 99.0}),
                encoding="utf-8")
            return 0

        self.assertEqual(verify(FILES, self.results, self.work, runner, echo=quiet), [])

    def test_a_failed_rerun_is_reported_rather_than_compared(self):
        bad = verify(FILES, self.results, self.work, lambda work: 3, echo=quiet)
        self.assertEqual(bad, ["the re-run itself failed with exit code 3"])

    def test_a_claimed_file_that_is_not_committed_stops_the_check(self):
        bad = verify(FILES + ["tables/absent.csv"], self.results, self.work,
                     self.copy_back(), echo=quiet)
        self.assertEqual(len(bad), 1)
        self.assertIn("not committed", bad[0])


class Comparators(unittest.TestCase):
    """What the two comparison rules do with the cases that are easy to get wrong.

    Every case here was named by an independent review on 2026-11-05 as one
    the suite did not plant. Two of them were real defects at the time: a
    changed word in a column that also holds numbers compared equal, and two
    NaNs compared unequal.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def csv(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def js(self, name: str, obj) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def test_a_changed_word_in_a_mixed_column_is_caught(self):
        a = self.csv("a.csv", "metric,value\nmean,1.5\nverdict,REFUSED\n")
        b = self.csv("b.csv", "metric,value\nmean,1.5\nverdict,ACCEPTED\n")
        self.assertTrue(compare_csv(a, b))

    def test_two_empty_cells_in_a_mixed_column_are_equal(self):
        a = self.csv("a.csv", "metric,value\nmean,1.5\nverdict,\n")
        b = self.csv("b.csv", "metric,value\nmean,1.5\nverdict,\n")
        self.assertEqual(compare_csv(a, b), [])

    def test_a_changed_number_in_a_mixed_column_is_caught(self):
        a = self.csv("a.csv", "metric,value\nmean,1.5\nverdict,REFUSED\n")
        b = self.csv("b.csv", "metric,value\nmean,1.6\nverdict,REFUSED\n")
        self.assertTrue(compare_csv(a, b))

    def test_a_number_that_only_looks_different_is_not_reported(self):
        a = self.csv("a.csv", "metric,value\nmean,1.50\n")
        b = self.csv("b.csv", "metric,value\nmean,1.5\n")
        self.assertEqual(compare_csv(a, b), [])

    def test_an_identifier_that_loses_its_leading_zeros_is_caught(self):
        a = self.csv("a.csv", "PatientID,value\n0001,1.50\n")
        b = self.csv("b.csv", "PatientID,value\n1,1.5\n")
        self.assertTrue(any("PatientID" in d for d in compare_csv(a, b)))

    def test_a_decimal_without_its_leading_zero_is_not_reported(self):
        a = self.csv("a.csv", "metric,value\nx,0.5\n")
        b = self.csv("b.csv", "metric,value\nx,.5\n")
        self.assertEqual(compare_csv(a, b), [])

    def test_the_words_pandas_reads_as_missing_are_kept_apart(self):
        # NA, N/A, None, null and nan are pandas' default missing values.
        a = self.csv("a.csv", "metric,value\nmean,1.5\nnote,None\n")
        b = self.csv("b.csv", "metric,value\nmean,1.5\nnote,\n")
        self.assertTrue(compare_csv(a, b))

    def test_extra_rows_and_columns_are_reported(self):
        a = self.csv("a.csv", "metric,value\na,1\n")
        b = self.csv("b.csv", "metric,value\na,1\nb,2\n")
        self.assertTrue(any("row count" in d for d in compare_csv(a, b)))
        c = self.csv("c.csv", "metric,value,extra\na,1,x\n")
        self.assertTrue(any("column only" in d for d in compare_csv(c, a)))

    def test_two_nan_values_in_json_are_equal(self):
        nan = float("nan")
        self.assertEqual(json_diffs(self.js("a.json", {"m": nan}),
                                    self.js("b.json", {"m": nan})), [])

    def test_a_nan_against_a_number_is_reported(self):
        self.assertTrue(json_diffs(self.js("a.json", {"m": float("nan")}),
                                   self.js("b.json", {"m": 1.0})))

    def test_a_runtime_key_nested_in_a_list_is_ignored(self):
        self.assertEqual(
            json_diffs(self.js("a.json", {"rows": [{"n": "a", "elapsed_sec": 1.0}]}),
                       self.js("b.json", {"rows": [{"n": "a", "elapsed_sec": 9.0}]})),
            [])

    def test_a_value_beside_a_runtime_key_is_still_compared(self):
        self.assertTrue(
            json_diffs(self.js("a.json", {"rows": [{"n": "a", "elapsed_sec": 1.0}]}),
                       self.js("b.json", {"rows": [{"n": "b", "elapsed_sec": 1.0}]})))

    def test_true_is_not_one(self):
        self.assertTrue(json_diffs(self.js("a.json", {"flag": True}),
                                   self.js("b.json", {"flag": 1})))


class TheComparisonCommand(unittest.TestCase):
    """compare_tables as a reader runs it: the exit code has to mean something.

    It reported success when the committed directory did not hold the table
    it was asked for, and when its own discovery found no CSV at all -- so a
    caller reading the exit code, which is what the README tells a reader to
    do, could take "nothing was compared" for "everything agrees".
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.rerun = Path(self._tmp.name) / "rerun"
        self.committed = Path(self._tmp.name) / "committed"
        self.rerun.mkdir()
        self.committed.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_cli(self, *names: str) -> int:
        argv = ["compare_tables.py", str(self.rerun), str(self.committed), *names]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("builtins.print", lambda *a, **k: None):
            return compare_tables.main()

    def write(self, where: Path, name: str, text: str) -> None:
        (where / name).write_text(text, encoding="utf-8")

    def test_two_matching_tables_succeed(self):
        for where in (self.rerun, self.committed):
            self.write(where, "t.csv", "a,b\n1,2\n")
        self.assertEqual(self.run_cli(), 0)

    def test_a_table_missing_from_the_committed_tree_fails(self):
        self.write(self.rerun, "t.csv", "a,b\n1,2\n")
        self.assertEqual(self.run_cli(), 1)

    def test_a_table_asked_for_and_absent_fails(self):
        for where in (self.rerun, self.committed):
            self.write(where, "t.csv", "a,b\n1,2\n")
        self.assertEqual(self.run_cli("absent.csv"), 1)

    def test_nothing_to_compare_fails(self):
        self.assertEqual(self.run_cli(), 1)


class FigureComparison(unittest.TestCase):
    """The same drawing, compressed differently, is the same figure.

    The first continuous-integration run of check_rerun regenerated every
    table identically on Linux and called all five PNGs different: zlib there
    does not emit the bytes zlib emits on the machine the committed figures
    were drawn on. These cases hold both directions of that -- a different
    compressor is silent, different pixels are not -- without needing two
    machines to show it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def png(self, name: str, value: int, compress_level: int) -> Path:
        from PIL import Image
        path = self.dir / name
        Image.new("RGB", (8, 8), (value, 10, 20)).save(
            path, format="PNG", compress_level=compress_level)
        return path

    def pdf(self, name: str, content: bytes, level: int) -> Path:
        import zlib
        path = self.dir / name
        body = zlib.compress(content, level)
        head = ("%PDF-1.4\n1 0 obj\n<< /Length " + str(len(body))
                + " >>\nstream\n").encode("ascii")
        path.write_bytes(head + body + b"\nendstream\nendobj\n%%EOF\n")
        return path

    def test_the_same_pixels_compressed_differently_agree(self):
        a = self.png("a.png", 100, 1)
        b = self.png("b.png", 100, 9)
        self.assertNotEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(check_rerun.diffs(a, b), [])

    def test_different_pixels_are_reported(self):
        a = self.png("a.png", 100, 9)
        b = self.png("b.png", 101, 9)
        found = check_rerun.diffs(a, b)
        self.assertTrue(found)
        self.assertIn("pixel", found[0])

    def test_an_image_of_another_size_is_reported(self):
        from PIL import Image
        a = self.png("a.png", 100, 9)
        b = self.dir / "b.png"
        Image.new("RGB", (9, 8), (100, 10, 20)).save(b, format="PNG")
        self.assertTrue(check_rerun.diffs(a, b))

    def test_the_same_drawing_compressed_differently_agrees_in_a_pdf(self):
        # Long enough that the two compression levels give different lengths,
        # so the /Length beside the stream differs too: that number moved on
        # the Linux runner and made this comparison report two figures as
        # changed after the pixels had been shown identical.
        drawing = b"BT /F1 12 Tf (figure) Tj ET\n" + b"0 0 m 1 1 l S\n" * 200
        a = self.pdf("a.pdf", drawing, 1)
        b = self.pdf("b.pdf", drawing, 9)
        self.assertNotEqual(a.stat().st_size, b.stat().st_size)
        self.assertEqual(check_rerun.diffs(a, b), [])

    def test_a_coordinate_that_moves_in_the_seventh_decimal_agrees(self):
        # The two PDFs that survived every other normalisation differed by one
        # character in a number: 2.645751 against 2.6457513. A ten-thousandth
        # of a point is three millionths of an inch.
        a = self.pdf("a.pdf", b"0 -2.645751 m 2.645751 0 l h B", 6)
        b = self.pdf("b.pdf", b"0 -2.6457513 m 2.6457513 0 l h B", 6)
        self.assertEqual(check_rerun.diffs(a, b), [])

    def test_a_coordinate_that_really_moves_is_reported(self):
        a = self.pdf("a.pdf", b"0 -2.645751 m 2.645751 0 l h B", 6)
        b = self.pdf("b.pdf", b"0 -2.645751 m 2.655751 0 l h B", 6)
        self.assertTrue(check_rerun.diffs(a, b))

    def test_a_coordinate_on_a_rounding_boundary_agrees(self):
        # Rounding to four decimals would put these two on opposite sides.
        a = self.pdf("a.pdf", b"471.525949 0 l", 6)
        b = self.pdf("b.pdf", b"471.525950 0 l", 6)
        self.assertEqual(check_rerun.diffs(a, b), [])

    def test_negative_zero_is_zero(self):
        # The difference between the two machines, exactly: one wrote
        # "0 0 m" where the other wrote "0 -0 m".
        a = self.pdf("a.pdf", b"0 0 m 471.525949 0 l h", 6)
        b = self.pdf("b.pdf", b"0 -0 m 471.525949 -0 l h", 6)
        self.assertEqual(check_rerun.diffs(a, b), [])

    def test_an_integer_that_changes_is_still_reported(self):
        a = self.pdf("a.pdf", b"0 0 m 471 0 l h", 6)
        b = self.pdf("b.pdf", b"0 0 m 472 0 l h", 6)
        self.assertTrue(check_rerun.diffs(a, b))

    def test_a_changed_operator_is_reported_even_with_equal_numbers(self):
        a = self.pdf("a.pdf", b"0 -2.645751 m 2.645751 0 l h B", 6)
        b = self.pdf("b.pdf", b"0 -2.645751 m 2.645751 0 l h f", 6)
        self.assertTrue(check_rerun.diffs(a, b))

    def test_a_changed_drawing_is_reported_in_a_pdf(self):
        a = self.pdf("a.pdf", b"BT /F1 12 Tf (figure) Tj ET", 9)
        b = self.pdf("b.pdf", b"BT /F1 12 Tf (other) Tj ET", 9)
        found = check_rerun.diffs(a, b)
        self.assertTrue(found)
        self.assertIn("drawing", found[0])


class WorkDirectory(unittest.TestCase):
    """The re-run directory is deleted and written again, so it is checked first."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        (self.results / "tables").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def refusal(self, work: Path):
        # root is passed, not patched: it is an argument of the rule, and a
        # test that patches the module instead reaches the default and can
        # pass for a reason it did not mean to test.
        return refuse_work_directory(work, self.results, self.root)

    def test_a_directory_beside_the_results_is_allowed(self):
        """The default --work lives inside the repository, so that is allowed."""
        self.assertIsNone(self.refusal(self.root / "out_rerun"))

    def test_the_repository_is_refused_on_its_own_account(self):
        """With the results elsewhere, only the repository rule can refuse it.

        Two earlier versions of this case kept the committed results inside
        the repository, so "this directory contains the results" fired first
        and the branch the test is named after never ran. An independent
        review caught that twice. Here the results sit beside the repository,
        not inside it.
        """
        far = self.root / "elsewhere" / "results"
        (far / "tables").mkdir(parents=True)
        repo = self.root / "repo"
        repo.mkdir()
        reason = refuse_work_directory(repo, far, repo)
        self.assertIsNotNone(reason)
        self.assertIn("repository", reason)

    def test_a_directory_above_the_repository_is_refused(self):
        far = self.root / "elsewhere" / "results"
        (far / "tables").mkdir(parents=True)
        repo = self.root / "deep" / "repo"
        repo.mkdir(parents=True)
        reason = refuse_work_directory(repo.parent, far, repo)
        self.assertIsNotNone(reason)
        self.assertIn("repository", reason)

    def test_the_committed_results_directory_is_refused(self):
        self.assertIsNotNone(self.refusal(self.results))

    def test_a_directory_inside_the_results_tree_is_refused(self):
        self.assertIsNotNone(self.refusal(self.results / "rerun"))

    def test_a_directory_containing_the_results_tree_is_refused(self):
        self.assertIsNotNone(self.refusal(self.root))

    def test_a_directory_holding_something_else_is_refused(self):
        other = self.root / "mine"
        other.mkdir()
        (other / "notes.txt").write_text("keep me", encoding="utf-8")
        self.assertIsNotNone(self.refusal(other))

    def test_a_previous_rerun_and_a_fresh_name_are_allowed(self):
        again = self.root / "out_rerun"
        (again / "tables").mkdir(parents=True)
        (again / MARKER).write_text("left by an earlier run", encoding="utf-8")
        self.assertIsNone(self.refusal(again))
        self.assertIsNone(self.refusal(self.root / "somewhere_new"))

    def test_a_directory_shaped_like_results_but_not_ours_is_refused(self):
        """Another project's results/ holds tables and figures too."""
        theirs = self.root / "their_results"
        (theirs / "tables").mkdir(parents=True)
        (theirs / "figures").mkdir(parents=True)
        self.assertIsNotNone(self.refusal(theirs))

    def test_a_marked_directory_in_a_refused_place_is_still_refused(self):
        """The marker says this tool made it; it does not say it may be used.

        main() reads the refusal before anything is created and returns, so
        the clean-up at the end is never reached for a refused path. Were the
        two decided separately again, a marker inside the committed results
        would be enough to delete them.
        """
        inside = self.results / "rerun"
        inside.mkdir()
        (inside / MARKER).write_text("ours", encoding="utf-8")
        self.assertIsNotNone(self.refusal(inside))

    def test_only_a_marked_directory_may_be_deleted(self):
        theirs = self.root / "their_results"
        (theirs / "tables").mkdir(parents=True)
        self.assertFalse(may_delete(theirs))
        self.assertFalse(may_delete(self.root))
        (theirs / MARKER).write_text("ours now", encoding="utf-8")
        self.assertTrue(may_delete(theirs))

    def test_verify_marks_the_directory_it_creates(self):
        """The marker is what makes the clean-up safe, so verify must write it."""
        results = self.root / "committed"
        (results / "tables").mkdir(parents=True)
        (results / "tables" / "t.csv").write_text("a\n1\n", encoding="utf-8")
        work = self.root / "fresh"
        verify(["tables/t.csv"], results, work,
               lambda w: (w / "tables" / "t.csv").write_text("a\n1\n",
                                                             encoding="utf-8") and 0 or 0,
               echo=lambda *a: None)
        self.assertTrue((work / MARKER).is_file())
        self.assertTrue(may_delete(work))


class TheCommandLine(unittest.TestCase):
    """main() itself: the round-one critical defect lived here, not in verify.

    It deleted whatever --work named, including a path its own guard had
    refused. The refusal is read once now, before anything is created, and
    this drives the whole entry point to prove that the directory survives.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        (self.results / "tables").mkdir(parents=True)
        # Everything the chosen stage claims is committed here, so the
        # "claimed by the build but not committed" return cannot be what
        # stops the run. Without this, these cases returned 1 either way and
        # would have passed with the refusal removed.
        for name in claimed(["strata"]):
            path = self.results / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("a\n1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_main(self, work: Path) -> tuple[int, str]:
        argv = ["check_rerun.py", "--stages", "strata", "--work", str(work)]
        said: list[str] = []
        with mock.patch.object(check_rerun, "RESULTS", self.results), \
                mock.patch.object(check_rerun, "ROOT", self.root), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch("builtins.print",
                           lambda *a, **k: said.append(" ".join(str(x) for x in a))):
            return check_rerun.main(), "\n".join(said)

    def test_a_refused_directory_is_left_alone(self):
        inside = self.results / "rerun"
        inside.mkdir()
        (inside / "keep.txt").write_text("mine", encoding="utf-8")
        code, said = self.run_main(inside)
        self.assertEqual(code, 1)
        self.assertIn("refusing to re-run into", said)
        self.assertTrue((inside / "keep.txt").exists())

    def test_a_directory_holding_the_committed_results_is_left_alone(self):
        """Named for what it tests: this tree holds the results, so the
        containment rule refuses it. The repository rule has its own cases in
        WorkDirectory, where the results sit outside the repository."""
        code, said = self.run_main(self.root)
        self.assertEqual(code, 1)
        self.assertIn("refusing to re-run into", said)
        self.assertTrue((self.results / "tables").is_dir())

    def test_a_refused_directory_that_carries_the_marker_survives(self):
        """The one case that separates main's refusal from verify's.

        verify refuses the same paths, so with main's early return removed
        the run would still report a refusal -- and then main's clean-up,
        which asks only for the marker, would delete the directory. That was
        the round-one critical defect. A marked directory in a refused place
        is therefore what tells the two layers apart, and an independent
        review had to point that out twice before this case existed.
        """
        marked = self.results / "rerun"
        marked.mkdir()
        (marked / check_rerun.MARKER).write_text("ours", encoding="utf-8")
        (marked / "keep.txt").write_text("mine", encoding="utf-8")
        code, said = self.run_main(marked)
        self.assertEqual(code, 1)
        self.assertIn("refusing to re-run into", said)
        self.assertTrue((marked / "keep.txt").exists(),
                        "a refused directory was deleted because it was marked")

    def test_someone_elses_directory_is_left_alone(self):
        theirs = self.root / "theirs"
        (theirs / "tables").mkdir(parents=True)
        (theirs / "tables" / "notes.csv").write_text("a\n1\n", encoding="utf-8")
        code, said = self.run_main(theirs)
        self.assertEqual(code, 1)
        self.assertIn("refusing to re-run into", said)
        self.assertTrue((theirs / "tables" / "notes.csv").exists())


class StageClaims(unittest.TestCase):
    def test_claimed_resolves_the_stage_order_without_repeats(self):
        files = claimed(["tables", "figures"])
        self.assertEqual(files, sorted(set(files)))
        self.assertIn("tables/manuscript/numbers.json", files)
        self.assertTrue(any(f.endswith("figure_7.tif") for f in files))


if __name__ == "__main__":
    unittest.main()
