"""Tests for the realistic-condition polyp sweep.

The noise generator is the part of that script that can be wrong without
anything downstream looking wrong: a field whose standard deviation or
correlation does not match what the paper says was injected would still produce
a plausible table. So it is pinned against the two quantities it claims --
the standard deviation and the in-plane lag-1 autocorrelation -- measured back
out of the generated field rather than assumed from the construction.

The expected values here were not written before they were measured: the
tolerances come from the sampling error of an estimate over a field of this
size, and the autocorrelation of white noise through a known kernel has a
closed form, which is what ``kernel_rho1`` computes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_polyp_realism import (  # noqa: E402
    WALL_THICKNESS_MM,
    build,
    gauss_kernel,
    kernel_rho1,
    noise_field,
    sigma_for_rho1,
    size_class,
)

SHAPE = (24, 96, 96)


def _rho1_of(field: np.ndarray) -> float:
    r = field - field.mean()
    num = float((r[:, :, :-1] * r[:, :, 1:]).sum())
    den = float((r * r).sum())
    return num / den


def test_white_field_has_the_requested_standard_deviation():
    rng = np.random.default_rng(1)
    f = noise_field(SHAPE, 15.0, 0.0, rng)
    assert f.std() == pytest.approx(15.0, rel=0.02)
    # White means no in-plane correlation, to within the sampling error of a
    # field of this size (1/sqrt(n) is about 0.002 here).
    assert abs(_rho1_of(f)) < 0.02


def test_correlated_field_matches_both_targets():
    rng = np.random.default_rng(2)
    f = noise_field(SHAPE, 15.0, 0.195, rng)
    assert f.std() == pytest.approx(15.0, rel=0.03)
    assert _rho1_of(f) == pytest.approx(0.195, abs=0.02)


def test_the_smoothing_scale_is_solved_not_guessed():
    for rho in (0.05, 0.195, 0.35):
        assert kernel_rho1(sigma_for_rho1(rho)) == pytest.approx(rho, abs=1e-3)
    # A larger scale correlates more, which is the monotonicity the solver uses.
    assert kernel_rho1(0.5) < kernel_rho1(1.0) < kernel_rho1(2.0)
    assert gauss_kernel(1.0).sum() == pytest.approx(1.0)


def test_scaling_is_analytic_so_the_spread_of_the_estimate_survives():
    # Rescaling each realization by its own standard deviation would make every
    # field have exactly the target, which would understate the variation the
    # sweep is meant to show. Different seeds must give different standard
    # deviations.
    sds = [noise_field((8, 32, 32), 15.0, 0.195,
                       np.random.default_rng(s)).std() for s in range(6)]
    assert len(set(round(s, 6) for s in sds)) == len(sds)
    assert np.std(sds) > 0.0


def test_fluid_background_lowers_the_contrast_and_air_does_not():
    semi = (5.0, 5.0, 5.0)
    air = build(semi, 38, 1.0, "air")
    fluid = build(semi, 38, 1.0, "fluid")
    # Centre is lesion in both; the corner is background.
    assert air[19, 19, 19] == pytest.approx(60.0, abs=1.0)
    assert fluid[19, 19, 19] == pytest.approx(60.0, abs=1.0)
    assert air[1, 1, 1] == pytest.approx(-1000.0, abs=1.0)
    assert fluid[1, 1, 1] == pytest.approx(20.0, abs=1.0)


def test_wall_touches_the_lesion_and_the_fold_touches_it_twice():
    semi = (5.0, 5.0, 5.0)
    n = 38
    c = n // 2
    air = build(semi, n, 1.0, "air")
    wall = build(semi, n, 1.0, "wall_tangent")
    fold = build(semi, n, 1.0, "fold")
    # Just beyond the lesion's far pole there is air in the plain phantom and
    # tissue once a wall is there. The probe sits inside the wall's thickness,
    # not at its far edge, where the partial-volume ramp has begun.
    beyond = (c + 6, c, c)
    assert air[beyond] < -500.0
    assert wall[beyond] > 0.0
    # Beside the lesion: air with one wall, tissue with a fold.
    beside = (c, c, c + 6)
    assert wall[beside] < -500.0
    assert fold[beside] > 0.0


def test_the_sessile_wall_cuts_the_lesion_at_its_equator():
    semi = (5.0, 5.0, 5.0)
    n = 38
    c = n // 2
    tangent = build(semi, n, 1.0, "wall_tangent")
    sessile = build(semi, n, 1.0, "wall_sessile")
    # Just past the centre and outside the lesion itself (radius 6.1 mm
    # against the lesion's 5 mm), on the side away from the lumen: air with a
    # tangent wall, wall tissue once the wall passes through the centre.
    inside_half = (c + 1, c + 6, c)
    assert tangent[inside_half] < -500.0
    assert sessile[inside_half] > 0.0
    # The protruding side is lesion in both, so the two are comparable.
    protruding = (c - 3, c, c)
    assert tangent[protruding] > 0.0
    assert sessile[protruding] > 0.0
    # A sessile lesion is continuous with its wall: through the thickness of
    # the wall the tissue is one region, with no gap of air between them.
    thickness = int(WALL_THICKNESS_MM)
    assert (sessile[c:c + thickness, c, c] > 0.0).all()


def test_the_wall_is_backed_by_fat_that_stops_a_growing_region():
    # A wall that never ends would fail the measurement for a reason no
    # patient has. Beyond the wall is fat, below the segmentation window.
    semi = (5.0, 5.0, 5.0)
    n = 38
    c = n // 2
    sessile = build(semi, n, 1.0, "wall_sessile")
    # Beyond the wall and outside the lesion, so what is read is the backing
    # and not the lesion itself.
    beyond = (c + int(WALL_THICKNESS_MM) + 2, c + 8, c)
    assert sessile[beyond] == pytest.approx(-100.0, abs=1.0)
    assert sessile[beyond] < -50.0
    # A thicker wall pushes the fat further out, and that is the only change.
    thick = build(semi, n, 1.0, "wall_sessile", 10.0)
    assert thick[beyond] > 0.0
    assert thick[c + 12, c + 8, c] == pytest.approx(-100.0, abs=1.0)


def test_size_class_turns_on_the_reporting_thresholds():
    assert size_class(5.99) == "<6"
    assert size_class(6.0) == "6-9"
    assert size_class(9.99) == "6-9"
    assert size_class(10.0) == ">=10"
