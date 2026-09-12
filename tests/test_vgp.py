"""VGP unfold: the parts that run without a GPU.

Rendering needs VTK and an OpenGL context and is exercised by
``scripts/make_vgp.py`` only; these tests pin the geometry around it.
"""

import numpy as np
import pytest

from ctc_core.fatmap import resample_centerline, smooth_centerline
from ctc_core.vgp import _build_remap_table


def test_remap_blend_weights_sum_to_one():
    pf, _, _, wp, sf, _, _, ws = _build_remap_table(720, 256, face_fov_deg=100.0)
    np.testing.assert_allclose(wp + ws, 1.0)
    assert (wp >= 0.5).all()  # the primary face always dominates
    assert (pf != sf).all()


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
def test_remap_face_centres(quarter):
    # Column k * 90 degrees looks straight down face k: primary, full weight.
    pf, x_low, w_high, wp, *_ = _build_remap_table(720, 256, face_fov_deg=100.0)
    col = quarter * 180
    assert pf[col] == quarter
    assert wp[col] == pytest.approx(1.0)
    assert x_low[col] + w_high[col] == pytest.approx(256 / 2 - 0.5, abs=1.0)


def test_rejects_faces_without_overlap():
    with pytest.raises(ValueError):
        _build_remap_table(720, 256, face_fov_deg=90.0)


def _tangent_turn_deg(xyz):
    t = np.diff(xyz, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    cos = np.clip((t[:-1] * t[1:]).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_smoothing_removes_voxel_kinks():
    # A diagonal voxel staircase: the raw tangent turns 90 degrees at every
    # step, which is what rotates the camera frame and leaves black dots.
    steps = np.array([[1.0, 0, 0], [0, 1.0, 0]] * 60)
    stair = np.vstack([[0.0, 0, 0], np.cumsum(steps, axis=0)])
    assert _tangent_turn_deg(stair).max() == pytest.approx(90.0)
    _, xyz = resample_centerline(smooth_centerline(stair, 2.0), 1.0)
    inner = _tangent_turn_deg(xyz)[5:-5]  # away from the fixed end points
    assert inner.max() < 5.0


def test_smoothed_path_shares_arc_length_with_ring_map():
    # make_vgp.py and the ring map resample the same smoothed path, so the
    # unfold's s axis ends where the fat map's does (to within one step).
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.normal(0, 1, (400, 3)) + [1.0, 0, 0], axis=0)
    sm = smooth_centerline(path, 2.0)
    s_vgp, _ = resample_centerline(sm, 1.0)
    s_ring, _ = resample_centerline(sm, 5.0)
    assert s_vgp[-1] == pytest.approx(s_ring[-1])
    s_raw, _ = resample_centerline(path, 1.0)
    assert s_raw[-1] > s_vgp[-1]  # the voxel path is longer
