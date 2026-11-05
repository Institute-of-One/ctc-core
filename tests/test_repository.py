"""The gates, run against this repository as it stands.

Separate from tests/test_gates.py, which plants defects in trees it builds
itself and must pass anywhere. These need the whole working tree -- the
scripts, the committed results, the two documents -- so they belong where that
tree exists: a developer's checkout and continuous integration.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_entrypoints  # noqa: E402
import check_environment  # noqa: E402
import check_rerun  # noqa: E402
import check_tables  # noqa: E402


class TheTreeIsClean(unittest.TestCase):
    def test_the_build_covers_every_shipped_result(self):
        self.assertEqual(check_tables.build_coverage(), [])

    def test_no_stage_reads_the_committed_results(self):
        self.assertEqual(check_tables.stage_inputs(), [])

    def test_every_entry_point_fixes_its_console(self):
        self.assertEqual(check_entrypoints.check(), [])

    def test_the_environment_matches_the_lock(self):
        want, unread = check_environment.locked()
        self.assertEqual(unread, [])
        self.assertEqual(
            check_environment.compare(want, list(check_environment.DECIDES_RESULTS)),
            [])


class TheCommittedFiguresSurviveAnotherCompressor(unittest.TestCase):
    """A reader's machine may compress the PDF differently. It is the same figure.

    Continuous integration on Linux reported two of the committed PDFs as
    changed when every table had reproduced exactly: zlib there does not emit
    what zlib here emits, and the length beside each stream and the offsets at
    the end of the file move with it. This rebuilds each committed figure the
    way another writer would and requires the comparison to stay silent.
    """

    def recompress(self, raw: bytes, level: int) -> bytes:
        import re
        import zlib
        opener = re.compile(rb"(?<![A-Za-z])stream\r?\n")
        pieces, at = [], 0
        while True:
            match = opener.search(raw, at)
            if match is None:
                pieces.append(raw[at:])
                return b"".join(pieces)
            end = raw.find(b"endstream", match.end())
            if end < 0:
                pieces.append(raw[at:])
                return b"".join(pieces)
            head, body = raw[at:match.end()], raw[match.end():end]
            try:
                again = zlib.compress(zlib.decompress(body), level)
            except zlib.error:
                again = body
            head = re.sub(rb"/Length\s+\d+",
                          b"/Length " + str(len(again)).encode(), head)
            pieces += [head, again]
            at = end

    def test_every_committed_pdf_reads_as_the_same_drawing(self):
        figures = sorted((ROOT / "results" / "figures").glob("*.pdf"))
        self.assertTrue(figures, "no committed figures to check")
        with tempfile.TemporaryDirectory() as tmp:
            for pdf in figures:
                for level in (1, 9):
                    other = Path(tmp) / ("%s.%d.pdf" % (pdf.stem, level))
                    other.write_bytes(self.recompress(pdf.read_bytes(), level))
                    self.assertEqual(check_rerun.diffs(other, pdf), [],
                                     "%s at level %d" % (pdf.name, level))


if __name__ == "__main__":
    unittest.main()
