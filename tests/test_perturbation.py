"""Geometry of the centerline perturbation experiment.

The experiment's claim rests on the displacement being what the plan says it
is: end points fixed, amplitude as specified, in the plane perpendicular to the
path, and lengthening a straight path by a predictable amount.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from perturb_centerline import arc_length, perturb, taper  # noqa: E402


def straight_path(length_mm: float = 200.0, step: float = 1.0) -> np.ndarray:
    n = int(length_mm / step) + 1
    return np.column_stack([np.arange(n) * step, np.zeros(n), np.zeros(n)])


def test_taper_is_zero_at_the_ends_and_one_in_the_middle():
    s = np.linspace(0, 200, 201)
    t = taper(s, 15.0)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(0.0)
    assert t[100] == pytest.approx(1.0, abs=1e-6)
    assert (t >= 0).all() and (t <= 1 + 1e-12).all()


def test_end_points_do_not_move():
    xyz = straight_path()
    out = perturb(xyz, 2.0, 10.0, "n", 2.0)
    np.testing.assert_allclose(out[0], xyz[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], xyz[-1], atol=1e-9)


def test_displacement_is_perpendicular_and_within_the_amplitude():
    xyz = straight_path()
    for direction in ("n", "b"):
        out = perturb(xyz, 1.0, 20.0, direction, 2.0)
        d = out - xyz
        # The path runs along x, so a perpendicular displacement has no x part.
        assert np.abs(d[:, 0]).max() < 1e-9
        assert np.linalg.norm(d, axis=1).max() <= 1.0 + 1e-9


def test_the_two_directions_are_orthogonal():
    xyz = straight_path()
    dn = perturb(xyz, 1.0, 20.0, "n", 2.0) - xyz
    db = perturb(xyz, 1.0, 20.0, "b", 2.0) - xyz
    middle = slice(50, 150)  # away from the taper
    cos = (dn[middle] * db[middle]).sum(axis=1) / (
        np.linalg.norm(dn[middle], axis=1) * np.linalg.norm(db[middle], axis=1))
    assert np.abs(cos).max() < 1e-9


def test_wobble_lengthens_a_straight_path_as_the_geometry_predicts():
    """A sinusoid of amplitude A and wavelength L lengthens a straight path by
    about (1/4) (2 pi A / L)^2 for small slopes; the test checks the order and
    that a shorter wavelength lengthens more."""
    xyz = straight_path()
    lengths = {}
    for amp, wav in ((1.0, 20.0), (1.0, 10.0), (2.0, 10.0)):
        out = perturb(xyz, amp, wav, "n", 2.0)
        lengths[(amp, wav)] = arc_length(out)[-1] / arc_length(xyz)[-1] - 1.0
    assert all(v > 0 for v in lengths.values())
    assert lengths[(1.0, 10.0)] > lengths[(1.0, 20.0)]
    assert lengths[(2.0, 10.0)] > lengths[(1.0, 10.0)]
    predicted = 0.25 * (2 * np.pi * 1.0 / 20.0) ** 2
    assert lengths[(1.0, 20.0)] == pytest.approx(predicted, rel=0.35)


def test_control_and_perturbed_differ_only_by_the_displacement():
    xyz = straight_path()
    out = perturb(xyz, 2.0, 10.0, "n", 2.0)
    assert not np.allclose(out, xyz)
    # No point is dropped, added or reordered: the experiment changes positions
    # only, so the paired comparison is between the same stations.
    assert out.shape == xyz.shape
