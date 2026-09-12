"""The deterministic cohort-selection rule of scripts/download_tcia.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_tcia import compare_selection, is_candidate_series, select_cohort  # noqa: E402


def series(pid, uid, n=500, size=300 << 20, body="COLON", modality="CT"):
    return {"PatientID": pid, "SeriesInstanceUID": uid, "ImageCount": n, "FileSize": size,
            "BodyPartExamined": body, "Modality": modality}


def test_candidate_filters():
    assert is_candidate_series(series("p", "u"))
    assert not is_candidate_series(series("p", "u", modality="MR"))
    assert not is_candidate_series(series("p", "u", body="CHEST"))
    assert not is_candidate_series(series("p", "u", n=150))
    assert not is_candidate_series(series("p", "u", n=950))
    assert not is_candidate_series(series("p", "u", size=800 << 20))
    assert is_candidate_series(series("p", "u", body="abdomen pelvis"))


def test_selection_is_sorted_and_takes_two_largest():
    meta = [
        series("B", "b1", n=400), series("B", "b2", n=600), series("B", "b3", n=500),
        series("A", "a1", n=300), series("A", "a2", n=300),
        series("C", "c1"),                      # only one candidate: skipped
        series("D", "d1"), series("D", "d2", body="CHEST"),  # one candidate left
    ]
    sel = select_cohort(meta, n_patients=5)
    assert sel == [("A", "primary", "a1"), ("A", "secondary", "a2"),   # tie broken by UID
                   ("B", "primary", "b2"), ("B", "secondary", "b3")]


def test_selection_stops_at_n_patients_and_reports_differences():
    meta = [series(p, f"{p}{i}") for p in "ABC" for i in (1, 2)]
    sel = select_cohort(meta, n_patients=2)
    assert {s[0] for s in sel} == {"A", "B"}
    manifest = [{"PatientID": p, "role": r, "SeriesInstanceUID": u} for p, r, u in sel]
    assert compare_selection(manifest, sel) == []
    manifest[0]["SeriesInstanceUID"] = "other"
    assert len(compare_selection(manifest, sel)) == 2
