"""Tests for ctc_core.quality.

Phantoms are chosen so the right answer is known in closed form, because none of
these indices has a reference implementation output to regress against.
"""

import numpy as np
import pytest

from ctc_core.quality import (
    QualityConfig,
    collapse_ratio_pct,
    compute_quality,
    distension_score,
    gas_volume_ml,
    luminal_radius_mm,
    path_length_mm,
    smooth_path,
    straight_line_mm,
)

AIR_HU = -1000.0
TISSUE_HU = 40.0


# ---------------------------------------------------------------------------
# Index / axis conventions -- the easiest thing to get wrong
# ---------------------------------------------------------------------------


def test_sampling_uses_ijk_as_xyz_not_zyx():
    """centerline_ijk is (i, j, k) = (x, y, z); arrays are (z, y, x)."""
    arr = np.zeros((7, 8, 9), dtype=np.float64)
    arr[1, 2, 3] = 5.0  # (z=1, y=2, x=3)
    from ctc_core.quality import _sample_at_centerline

    # The point at x=3, y=2, z=1 must find it.
    vals, inside = _sample_at_centerline(arr, np.array([[3, 2, 1]]))
    assert inside[0] and vals[0] == 5.0

    # The transposed reading must not.
    vals2, _ = _sample_at_centerline(arr, np.array([[1, 2, 3]]))
    assert vals2[0] == 0.0


def test_out_of_bounds_points_are_ignored_not_wrapped():
    arr = np.ones((5, 5, 5))
    from ctc_core.quality import _sample_at_centerline

    vals, inside = _sample_at_centerline(arr, np.array([[0, 0, 0], [99, 0, 0], [-1, 0, 0]]))
    assert list(inside) == [True, False, False]
    assert vals[0] == 1.0


# ---------------------------------------------------------------------------
# Length and tortuosity
# ---------------------------------------------------------------------------


def test_length_of_a_known_polyline():
    xyz = np.array([[0.0, 0, 0], [3, 4, 0], [3, 4, 12]])
    assert path_length_mm(xyz) == pytest.approx(5.0 + 12.0)
    assert straight_line_mm(xyz) == pytest.approx(13.0)


def test_degenerate_paths_give_zero():
    assert path_length_mm(np.zeros((1, 3))) == 0.0
    assert straight_line_mm(np.zeros((0, 3))) == 0.0


def test_smoothing_shortens_a_staircase_but_keeps_the_endpoints():
    # A staircase along x/y that a smooth line would cut across.
    pts = []
    for n in range(20):
        pts.append([n, 0, 0])
        pts.append([n, 1, 0])
    xyz = np.array(pts, dtype=float)
    sm = smooth_path(xyz, 1.5)

    assert path_length_mm(sm) < path_length_mm(xyz)
    assert np.allclose(sm[0], xyz[0])
    assert np.allclose(sm[-1], xyz[-1])


def test_smoothing_is_a_noop_on_short_or_disabled_input():
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert np.allclose(smooth_path(xyz, 1.5), xyz)  # fewer than 5 points
    long = np.random.default_rng(0).normal(size=(30, 3))
    assert np.allclose(smooth_path(long, 0.0), long)


# ---------------------------------------------------------------------------
# Gas volume
# ---------------------------------------------------------------------------


def test_gas_volume_counts_only_lumen_below_the_air_threshold():
    ct = np.full((10, 10, 10), TISSUE_HU, dtype=np.float64)
    lumen = np.zeros((10, 10, 10), dtype=bool)
    lumen[2:6, 2:6, 2:6] = True  # 64 voxels of lumen
    ct[2:6, 2:6, 2:4] = AIR_HU  # half of it is gas: 32 voxels

    cfg = QualityConfig(restrict_gas_to_centerline_components=False)
    vol, gas = gas_volume_ml(ct, lumen, (1.0, 1.0, 1.0), None, cfg)
    assert int(gas.sum()) == 32
    assert vol == pytest.approx(32 / 1000.0)


def test_gas_volume_scales_with_voxel_size():
    ct = np.full((6, 6, 6), AIR_HU)
    lumen = np.ones((6, 6, 6), dtype=bool)
    cfg = QualityConfig(restrict_gas_to_centerline_components=False)
    v1, _ = gas_volume_ml(ct, lumen, (1.0, 1.0, 1.0), None, cfg)
    v2, _ = gas_volume_ml(ct, lumen, (2.0, 1.0, 1.0), None, cfg)
    assert v2 == pytest.approx(2 * v1)


def test_gas_restriction_drops_air_the_centerline_never_visits():
    """Lung or stomach gas must not inflate a colonic gas volume."""
    ct = np.full((20, 20, 20), TISSUE_HU)
    lumen = np.zeros((20, 20, 20), dtype=bool)
    lumen[2:8, 2:8, 2:8] = True  # colon, visited
    lumen[12:18, 12:18, 12:18] = True  # separate pocket, not visited
    ct[lumen] = AIR_HU

    # centerline through the first block only, as (i, j, k) = (x, y, z)
    path = np.array([[x, 5, 5] for x in range(3, 8)])

    unrestricted = QualityConfig(restrict_gas_to_centerline_components=False)
    restricted = QualityConfig(restrict_gas_to_centerline_components=True)
    v_all, _ = gas_volume_ml(ct, lumen, (1.0, 1.0, 1.0), path, unrestricted)
    v_colon, gas = gas_volume_ml(ct, lumen, (1.0, 1.0, 1.0), path, restricted)

    assert v_all == pytest.approx(2 * 6**3 / 1000.0)
    assert v_colon == pytest.approx(6**3 / 1000.0)
    assert not gas[15, 15, 15]


# ---------------------------------------------------------------------------
# Luminal radius and collapse
# ---------------------------------------------------------------------------


def test_luminal_radius_recovers_a_known_cylinder_radius():
    r = 6
    n = 40
    lumen = np.zeros((20, n, n), dtype=bool)
    c = n // 2
    _, yy, xx = np.mgrid[0:20, 0:n, 0:n]
    lumen[((yy - c) ** 2 + (xx - c) ** 2) <= r**2] = True

    path = np.array([[c, c, z] for z in range(4, 16)])  # (i, j, k) = (x, y, z)
    radii = luminal_radius_mm(lumen, path, (1.0, 1.0, 1.0))
    assert np.median(radii) == pytest.approx(r, abs=1.0)


def stepped_tube(nz=40, n=40, r_wide=8, r_narrow=2, step_z=20):
    lumen = np.zeros((nz, n, n), dtype=bool)
    c = n // 2
    _, yy, xx = np.mgrid[0:nz, 0:n, 0:n]
    disc2 = (yy - c) ** 2 + (xx - c) ** 2
    lumen[:step_z][disc2[:step_z] <= r_wide**2] = True
    lumen[step_z:][disc2[step_z:] <= r_narrow**2] = True
    return lumen, c


def test_collapse_ratio_finds_the_narrow_half():
    """Half the tube is below the threshold, so the ratio must sit near half.

    Not exactly half: the inscribed radius legitimately shrinks either side of a
    sharp calibre step, because the nearest wall is then the step itself. A real
    colon tapers, so this over-reads more than practice would.
    """
    lumen, c = stepped_tube()
    zs = list(range(2, 38))
    path = np.array([[c, c, z] for z in zs])
    xyz = np.array([[float(c), float(c), float(z)] for z in zs])

    pct, radii = collapse_ratio_pct(lumen, path, xyz, (1.0, 1.0, 1.0),
                                    QualityConfig(collapse_radius_mm=4.0))
    assert 50.0 <= pct <= 70.0
    assert radii[:15].mean() > radii[-15:].mean()


def test_collapse_ratio_is_length_weighted_not_point_weighted():
    """Sampling the narrow half more densely must not change the answer.

    This is the property the length weighting exists for: a point-weighted ratio
    would be dragged toward whichever stretch happens to carry more samples.
    """
    lumen, c = stepped_tube()

    uniform_z = [float(z) for z in range(2, 38)]
    # Same geometry, same extent, but the narrow half sampled four times denser.
    dense_z = [float(z) for z in range(2, 20)] + list(np.arange(20.0, 38.0, 0.25))

    out = {}
    for name, zs in (("uniform", uniform_z), ("dense_narrow", dense_z)):
        xyz = np.array([[float(c), float(c), z] for z in zs])
        path = np.rint(xyz).astype(int)[:, [0, 1, 2]]
        pct, _ = collapse_ratio_pct(lumen, path, xyz, (1.0, 1.0, 1.0),
                                    QualityConfig(collapse_radius_mm=4.0))
        out[name] = pct

    assert out["uniform"] == pytest.approx(out["dense_narrow"], abs=3.0)

    # A point-weighted ratio would have moved a long way on the same input.
    n_narrow_uniform = sum(1 for z in uniform_z if z >= 20)
    n_narrow_dense = sum(1 for z in dense_z if z >= 20)
    point_uniform = 100.0 * n_narrow_uniform / len(uniform_z)
    point_dense = 100.0 * n_narrow_dense / len(dense_z)
    assert abs(point_dense - point_uniform) > 20.0


def test_collapse_ratio_of_a_uniformly_wide_lumen_is_zero():
    n, r = 40, 8
    lumen = np.zeros((30, n, n), dtype=bool)
    c = n // 2
    _, yy, xx = np.mgrid[0:30, 0:n, 0:n]
    lumen[((yy - c) ** 2 + (xx - c) ** 2) <= r**2] = True
    zs = list(range(3, 27))
    path = np.array([[c, c, z] for z in zs])
    xyz = np.array([[float(c), float(c), float(z)] for z in zs])
    pct, _ = collapse_ratio_pct(lumen, path, xyz, (1.0, 1.0, 1.0), QualityConfig())
    assert pct == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_distension_score_endpoints():
    cfg = QualityConfig()
    worst, grade_w = distension_score(0.0, 0.0, 100.0, cfg)
    best, grade_b = distension_score(2000.0, 10.0, 0.0, cfg)
    assert worst == pytest.approx(0.0)
    assert grade_w == "Poor"
    assert best == pytest.approx(100.0)
    assert grade_b == "Excellent"


def test_distension_score_weights_sum_over_the_three_terms():
    cfg = QualityConfig()
    # collapse perfect, the other two at zero -> exactly the collapse weight.
    score, _ = distension_score(0.0, 0.0, 0.0, cfg)
    assert score == pytest.approx(100.0 * cfg.w_collapse)


def test_distension_grades_are_monotone():
    cfg = QualityConfig()
    seen = [distension_score(g, lag, col, cfg)[0]
            for g, lag, col in ((0, 0, 100), (750, 4, 20), (2000, 10, 0))]
    assert seen == sorted(seen)


# ---------------------------------------------------------------------------
# The inherited CTI bands
# ---------------------------------------------------------------------------


def test_cti_band_is_absent_unless_calibrated():
    """The inherited bands could never fire; emitting a label would mislead."""
    assert QualityConfig().cti_bands == ()
    ct = np.full((12, 12, 12), TISSUE_HU)
    lumen = np.ones((12, 12, 12), dtype=bool)
    path = np.array([[6, 6, z] for z in range(2, 10)])
    xyz = np.array([[6.0, 6.0, float(z)] for z in range(2, 10)])
    res = compute_quality(ct, lumen, path, xyz, (1.0, 1.0, 1.0))
    assert res.ok
    assert res.indices["centerline_tortuosity_index"] is not None
    assert res.indices["cti_band"] is None


def test_cti_band_is_emitted_when_bands_are_supplied():
    cfg = QualityConfig(cti_bands=((2.0, "Low"), (4.0, "Moderate")))
    ct = np.full((12, 12, 12), TISSUE_HU)
    lumen = np.ones((12, 12, 12), dtype=bool)
    path = np.array([[6, 6, z] for z in range(2, 10)])
    xyz = np.array([[6.0, 6.0, float(z)] for z in range(2, 10)])
    res = compute_quality(ct, lumen, path, xyz, (1.0, 1.0, 1.0), cfg)
    assert res.indices["cti_band"] == "Low"  # a straight line has CTI 1.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def straight_gas_tube():
    n, r, nz = 40, 8, 40
    ct = np.full((nz, n, n), TISSUE_HU, dtype=np.float64)
    lumen = np.zeros((nz, n, n), dtype=bool)
    c = n // 2
    _, yy, xx = np.mgrid[0:nz, 0:n, 0:n]
    lumen[((yy - c) ** 2 + (xx - c) ** 2) <= r**2] = True
    ct[lumen] = AIR_HU
    zs = list(range(3, 37))
    path = np.array([[c, c, z] for z in zs])
    xyz = np.array([[float(c), float(c), float(z)] for z in zs])
    return ct, lumen, path, xyz


def test_compute_quality_on_a_known_tube():
    ct, lumen, path, xyz = straight_gas_tube()
    res = compute_quality(ct, lumen, path, xyz, (1.0, 1.0, 1.0))
    assert res.ok, res.message

    ind = res.indices
    assert ind["traced_centerline_length_cm"] == pytest.approx(3.3, abs=0.2)
    assert ind["centerline_tortuosity_index"] == pytest.approx(1.0, abs=0.01)
    assert ind["collapse_ratio_pct"] == pytest.approx(0.0, abs=1e-6)
    assert ind["predicted_length_redundancy_index"] == pytest.approx(3.3 / 160.0, abs=0.01)
    # gas volume equals the tube volume
    assert ind["gas_volume_ml"] == pytest.approx(int(lumen.sum()) / 1000.0, abs=0.06)


def test_compute_quality_reports_failure_without_raising():
    res = compute_quality(
        np.zeros((4, 4, 4)), np.zeros((4, 4, 4), dtype=bool),
        np.zeros((0, 3), dtype=int), np.zeros((0, 3)), (1.0, 1.0, 1.0),
    )
    # An empty centerline is degenerate but must not raise.
    assert isinstance(res.ok, bool)


def test_height_index_only_when_height_is_known():
    ct, lumen, path, xyz = straight_gas_tube()
    without = compute_quality(ct, lumen, path, xyz, (1.0, 1.0, 1.0))
    assert without.indices["height_normalized_colon_length_index"] is None

    with_h = compute_quality(ct, lumen, path, xyz, (1.0, 1.0, 1.0), patient_height_cm=170.0)
    assert with_h.indices["height_normalized_colon_length_index"] == pytest.approx(
        with_h.indices["traced_centerline_length_cm"] / 170.0, abs=1e-3
    )
