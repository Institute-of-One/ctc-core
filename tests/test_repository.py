"""The gates, run against this repository as it stands.

Separate from tests/test_gates.py, which plants defects in trees it builds
itself and must pass anywhere. These need the whole working tree -- the
scripts, the committed results, the two documents -- so they belong where that
tree exists: a developer's checkout and continuous integration.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_entrypoints  # noqa: E402
import check_environment  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
