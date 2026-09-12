"""Unit tests for the outcome/failure-mode classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_batch_metrics import classify  # noqa: E402


def row(**kw):
    base = {
        "centerline_ok": True,
        "largest_air_cc_voxels": 1_500_000,
        "n_ccs_total": 15,
        "n_ccs_visited": 1,
        "centerline_length_mm": 1300.0,
    }
    base.update(kw)
    return base


def test_complete_has_no_failure_mode():
    assert classify(row()) == ("complete", "")


def test_partial_attributed_to_single_component_traversal():
    assert classify(row(centerline_length_mm=900.0)) == (
        "partial",
        "single-component-traversal",
    )


def test_short_centerline_counts_as_failed():
    outcome, _ = classify(row(centerline_length_mm=200.0))
    assert outcome == "failed"


def test_small_gas_component_dominates_attribution():
    assert classify(row(centerline_length_mm=48.0, largest_air_cc_voxels=16_825)) == (
        "failed",
        "insufficient-distension",
    )


def test_raised_stage_with_large_colon_is_unreachable_seed():
    assert classify(row(centerline_ok=False, centerline_length_mm=None)) == (
        "failed",
        "fmm-unreachable-end-seed",
    )


def test_raised_stage_with_tiny_colon_is_distension():
    assert classify(
        row(centerline_ok=False, centerline_length_mm=None, largest_air_cc_voxels=11_908)
    ) == ("failed", "insufficient-distension")


def test_multi_component_traversal_is_not_flagged_when_complete():
    # A complete centerline that visited several components is a success even
    # though n_ccs_visited > 1.
    assert classify(row(n_ccs_visited=4)) == ("complete", "")
