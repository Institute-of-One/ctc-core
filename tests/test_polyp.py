"""Tests for ctc_core.polyp.

Geometric phantoms with closed-form answers: a sphere's largest diameter is
twice its radius and its sphericity is 1, an ellipsoid's is twice its semi-major
axis. The shape index has canonical values for standard surfaces (cap +1,
ridge +0.5, cup -1), which pins both the formula and the sign convention.

The size thresholds CT colonography reporting turns on are 6 and 10 mm, so the
diameter is held to sub-millimetre accuracy where that is achievable, and where
it is not the tests say exactly how much bias remains and why.
"""

import numpy as np
import pytest

from ctc_core.polyp import (
    PolypConfig,
    boundary_iso_level,
    extract_sphere_voi,
    max_diameter_mm,
    measure_polyp,
    segment_polyp,
    shape_index_curvedness,
)

AIR_HU = -1000.0
SOFT_HU = 60.0
BONE_HU = 600.0


def ball_mask(shape, centre, radii, spacing=(1.0, 1.0, 1.0)):
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    dz = (zz - centre[0]) * spacing[0] / radii[0]
    dy = (yy - centre[1]) * spacing[1] / radii[1]
    dx = (xx - centre[2]) * spacing[2] / radii[2]
    return (dz * dz + dy * dy + dx * dx) <= 1.0


def sphere_phantom(radius_mm=5.0, n=48, spacing=(1.0, 1.0, 1.0), hu=SOFT_HU):
    """Hard-edged sphere: every voxel is fully lesion or fully air."""
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    c = (n / 2.0, n / 2.0, n / 2.0)
    m = ball_mask((n, n, n), c, (radius_mm,) * 3, spacing)
    ct[m] = hu
    return ct, c


def sphere_phantom_pv(radius_mm=5.0, n=64, hu=SOFT_HU):
    """Sphere with a one-voxel partial-volume ramp at the edge.

    This is what a real CT of a polyp looks like, and it is the information the
    grayscale diameter uses. A hard-edged phantom carries none of it, so no
    method can localise its surface better than half a voxel.
    """
    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
    rad = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    frac = np.clip(0.5 - (rad - radius_mm), 0.0, 1.0)
    return (AIR_HU + (hu - AIR_HU) * frac).astype(np.float32), (c, c, c)


# ---------------------------------------------------------------------------
# VOI extraction
# ---------------------------------------------------------------------------


def test_sphere_voi_is_centred_and_the_right_physical_size():
    ct, c = sphere_phantom()
    crop, sphere, origin, local = extract_sphere_voi(ct, c, (1.0, 1.0, 1.0), 10.0)

    assert crop.shape == sphere.shape
    # The sphere mask volume matches 4/3 pi r^3 to a few percent.
    assert sphere.sum() == pytest.approx((4 / 3) * np.pi * 10.0**3, rel=0.05)
    # The supplied centre maps back onto itself.
    assert np.allclose(np.array(origin) + np.array(local), np.array(c))


def test_sphere_voi_respects_anisotropic_spacing():
    ct, c = sphere_phantom(n=60)
    spacing = (2.0, 1.0, 1.0)
    _, sphere, _, _ = extract_sphere_voi(ct, c, spacing, 10.0)
    vox_mm3 = float(np.prod(spacing))
    assert sphere.sum() * vox_mm3 == pytest.approx((4 / 3) * np.pi * 10.0**3, rel=0.06)


def test_sphere_voi_outside_the_volume_raises():
    ct, _ = sphere_phantom()
    with pytest.raises(ValueError):
        extract_sphere_voi(ct, (-50.0, -50.0, -50.0), (1.0, 1.0, 1.0), 5.0)


# ---------------------------------------------------------------------------
# Diameter -- the measurement the 6 and 10 mm thresholds turn on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [3.0, 5.0, 8.0])
def test_binary_mask_diameter_is_biased_high_by_one_voxel(radius):
    """The reference implementation's behaviour, pinned as a known bias.

    Marching cubes at level 0.5 places the surface half a voxel outside the
    outermost included voxel centre, at each end, so a binary mask overestimates
    the diameter by exactly one voxel. At 1 mm isotropic that is +1 mm, which is
    17 % at the 6 mm reporting threshold.
    """
    n = 64
    mask = ball_mask((n, n, n), (n / 2,) * 3, (radius,) * 3)
    d, p1, p2, area, _vol = max_diameter_mm(mask, (1.0, 1.0, 1.0))

    assert d == pytest.approx(2 * radius + 1.0, abs=0.25)
    assert area == pytest.approx(4 * np.pi * radius**2, rel=0.25)
    assert p1 is not None and p2 is not None


@pytest.mark.parametrize("radius", [3.0, 5.0, 8.0])
def test_grayscale_diameter_is_unbiased_given_partial_volume(radius):
    """With partial-volume information the bias essentially disappears."""
    ct, c = sphere_phantom_pv(radius_mm=radius)
    res = measure_polyp(ct, c, (1.0, 1.0, 1.0), PolypConfig(diameter_from_grayscale=True))
    assert res.ok
    assert res.measurements["max_diameter_mm"] == pytest.approx(2 * radius, abs=0.25)


def test_grayscale_diameter_is_consistent_where_the_binary_one_is_not():
    """The grayscale route's value is consistency, not per-case superiority.

    The binary error depends on both the lesion size and where the segmentation
    threshold happens to fall on the edge: on a realistic ramp it runs from
    -0.26 mm at radius 3 to +0.03 mm at radius 8, and on an ideal step edge it is
    +1.00 mm everywhere. Those can coincidentally cancel, and at radius 8 the
    binary value is the closer of the two. What matters for a measurement used
    against fixed 6 and 10 mm thresholds is that the error does not move around
    with lesion size, and that is what the grayscale route provides.
    """
    radii = [3.0, 5.0, 8.0]
    gray_err, bin_err = [], []
    for r in radii:
        ct, c = sphere_phantom_pv(radius_mm=r)
        g = measure_polyp(ct, c, (1.0, 1.0, 1.0), PolypConfig(diameter_from_grayscale=True))
        b = measure_polyp(ct, c, (1.0, 1.0, 1.0), PolypConfig(diameter_from_grayscale=False))
        gray_err.append(g.measurements["max_diameter_mm"] - 2 * r)
        bin_err.append(b.measurements["max_diameter_mm"] - 2 * r)

    # Every grayscale error is small. (They happen to share a sign at these three
    # sizes; at 8 mm the grayscale error is -0.11, so no constant offset is claimed.)
    assert all(abs(e) < 0.25 for e in gray_err), gray_err
    # The binary error changes sign across the size range.
    assert min(bin_err) < 0 < max(bin_err), bin_err
    # And its spread is the larger.
    assert (max(bin_err) - min(bin_err)) > (max(gray_err) - min(gray_err))


def test_grayscale_diameter_cannot_help_on_an_ideal_step_edge():
    """A limit of the method, stated rather than hidden.

    A hard-edged phantom carries no partial-volume information, so the surface
    cannot be located better than half a voxel and the grayscale route inherits
    the same one-voxel bias as the binary one. Real CT always has the ramp; a
    synthetic step does not, and a test that ignored this would overstate the
    correction.
    """
    ct, c = sphere_phantom(radius_mm=5.0, n=64)
    res = measure_polyp(ct, c, (1.0, 1.0, 1.0), PolypConfig(diameter_from_grayscale=True))
    assert res.measurements["max_diameter_mm"] == pytest.approx(11.0, abs=0.3)


def test_boundary_level_adapts_to_what_surrounds_the_lesion():
    """A polyp against tagged fluid has a different boundary level than one in air."""
    n = 64
    levels = {}
    for background in (AIR_HU, 250.0):  # lumen air, then tagged fluid
        c = n / 2.0
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
        rad = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
        frac = np.clip(0.5 - (rad - 6.0), 0.0, 1.0)
        ct = (background + (SOFT_HU - background) * frac).astype(np.float32)

        crop, sphere, _, local = extract_sphere_voi(ct, (c, c, c), (1.0, 1.0, 1.0), 18.0)
        mask = segment_polyp(crop, sphere, local, (1.0, 1.0, 1.0), PolypConfig())
        levels[background] = boundary_iso_level(crop, mask)

    assert levels[AIR_HU] < 0 < levels[250.0]
    # Each sits about midway between the lesion and its surroundings.
    assert levels[AIR_HU] == pytest.approx((SOFT_HU + AIR_HU) / 2, abs=80)
    assert levels[250.0] == pytest.approx((SOFT_HU + 250.0) / 2, abs=80)


def test_max_diameter_of_an_ellipsoid_follows_the_major_axis():
    n = 80
    mask = ball_mask((n, n, n), (n / 2,) * 3, (4.0, 4.0, 12.0))
    d, _, _, _, _ = max_diameter_mm(mask, (1.0, 1.0, 1.0))
    assert d == pytest.approx(24.0, abs=1.0)


def test_max_diameter_uses_physical_spacing_not_voxel_counts():
    n = 48
    # Same voxel geometry, different spacing along z.
    mask = ball_mask((n, n, n), (n / 2,) * 3, (5.0, 5.0, 5.0))
    d_iso, _, _, _, _ = max_diameter_mm(mask, (1.0, 1.0, 1.0))
    d_aniso, _, _, _, _ = max_diameter_mm(mask, (3.0, 1.0, 1.0))
    assert d_aniso > d_iso * 1.5


def test_max_diameter_of_a_degenerate_mask_is_zero():
    d, p1, p2, area, vol = max_diameter_mm(np.zeros((8, 8, 8), dtype=bool), (1.0, 1.0, 1.0))
    assert d == 0.0 and p1 is None and p2 is None and area == 0.0 and vol == 0.0


# ---------------------------------------------------------------------------
# Shape index -- canonical values pin formula and sign convention
# ---------------------------------------------------------------------------


def test_shape_index_of_a_bright_sphere_is_a_cap():
    """A polyp is a bright blob; the convention must make that read +1."""
    ct, c = sphere_phantom(radius_mm=8.0, n=48)
    si, cv, gmag = shape_index_curvedness(ct, (1.0, 1.0, 1.0), sigma_mm=2.0)

    # Evaluate on the iso-surface, where the gradient is strongest.
    shell = gmag > np.percentile(gmag, 99)
    assert np.median(si[shell]) == pytest.approx(1.0, abs=0.15)
    assert np.median(cv[shell]) > 0


def test_shape_index_of_a_bright_cylinder_is_a_ridge():
    n = 48
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    _, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
    ct[((yy - n / 2) ** 2 + (xx - n / 2) ** 2) <= 8.0**2] = SOFT_HU

    si, _, gmag = shape_index_curvedness(ct, (1.0, 1.0, 1.0), sigma_mm=2.0)
    interior = np.zeros_like(si, dtype=bool)
    interior[8:-8] = True  # away from the ends
    shell = interior & (gmag > np.percentile(gmag[interior], 99))
    assert np.median(si[shell]) == pytest.approx(0.5, abs=0.15)


def test_shape_index_of_a_cavity_is_a_cup():
    """Inverting the contrast must flip the sign."""
    n = 48
    ct = np.full((n, n, n), SOFT_HU, dtype=np.float32)
    ct[ball_mask((n, n, n), (n / 2,) * 3, (8.0,) * 3)] = AIR_HU

    si, _, gmag = shape_index_curvedness(ct, (1.0, 1.0, 1.0), sigma_mm=2.0)
    shell = gmag > np.percentile(gmag, 99)
    assert np.median(si[shell]) == pytest.approx(-1.0, abs=0.15)


def test_shape_index_is_zeroed_where_the_gradient_vanishes():
    ct = np.full((24, 24, 24), SOFT_HU, dtype=np.float32)
    si, cv, _ = shape_index_curvedness(ct, (1.0, 1.0, 1.0))
    assert np.allclose(si, 0.0)
    assert np.allclose(cv, 0.0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_segmentation_excludes_lumen_air_and_tagged_material():
    n = 48
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    ct[ball_mask((n, n, n), (n / 2,) * 3, (6.0,) * 3)] = SOFT_HU
    # Tagged fluid abutting the lesion, above the soft-tissue window.
    ct[n // 2 + 6:, :, :] = BONE_HU

    crop, sphere, _, local = extract_sphere_voi(
        ct, (n / 2, n / 2, n / 2), (1.0, 1.0, 1.0), 18.0
    )
    mask = segment_polyp(crop, sphere, local, (1.0, 1.0, 1.0), PolypConfig())

    values = crop[mask]
    assert values.min() >= PolypConfig().soft_hu_min
    assert values.max() <= PolypConfig().soft_hu_max


def test_segmentation_keeps_only_the_component_at_the_given_point():
    n = 56
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    ct[ball_mask((n, n, n), (n / 2, n / 2, n / 2), (5.0,) * 3)] = SOFT_HU
    ct[ball_mask((n, n, n), (n / 2, n / 2, n / 2 + 16), (5.0,) * 3)] = SOFT_HU

    crop, sphere, _, local = extract_sphere_voi(
        ct, (n / 2, n / 2, n / 2), (1.0, 1.0, 1.0), 18.0
    )
    mask = segment_polyp(crop, sphere, local, (1.0, 1.0, 1.0), PolypConfig())

    from scipy import ndimage

    _, n_comp = ndimage.label(mask)
    assert n_comp == 1
    assert mask.sum() == pytest.approx((4 / 3) * np.pi * 5.0**3, rel=0.2)


def test_segmentation_recovers_when_the_point_lands_off_tissue():
    """A reader clicks near the lesion, not necessarily inside it."""
    n = 56
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    ct[ball_mask((n, n, n), (n / 2, n / 2, n / 2 + 8), (5.0,) * 3)] = SOFT_HU

    crop, sphere, _, local = extract_sphere_voi(
        ct, (n / 2, n / 2, n / 2), (1.0, 1.0, 1.0), 18.0
    )
    mask = segment_polyp(crop, sphere, local, (1.0, 1.0, 1.0), PolypConfig())
    assert mask.any()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [3.0, 5.0])
def test_measure_polyp_recovers_a_known_sphere(radius):
    ct, c = sphere_phantom_pv(radius_mm=radius)
    res = measure_polyp(ct, c, (1.0, 1.0, 1.0))
    assert res.ok, res.message

    m = res.measurements
    assert m["max_diameter_mm"] == pytest.approx(2 * radius, abs=0.3)
    assert m["volume_mm3"] == pytest.approx((4 / 3) * np.pi * radius**3, rel=0.12)
    assert m["equivalent_diameter_mm"] == pytest.approx(2 * radius, abs=0.5)
    assert m["sphericity"] == pytest.approx(1.0, abs=0.05)
    assert m["shape_index_median"] > 0.5  # a cap

    # The iso-surface volume must beat the thresholded voxel count, which is
    # biased low because -50 HU sits high on the partial-volume ramp.
    truth = (4 / 3) * np.pi * radius**3
    assert abs(m["volume_mm3"] - truth) < abs(m["volume_voxel_count_mm3"] - truth)


def test_measure_polyp_distinguishes_a_6mm_from_a_10mm_lesion():
    """The reporting thresholds; the measurement has to separate them cleanly."""
    ct_small, c_small = sphere_phantom_pv(radius_mm=3.0)
    ct_large, c_large = sphere_phantom_pv(radius_mm=5.0)

    small = measure_polyp(ct_small, c_small, (1.0, 1.0, 1.0))
    large = measure_polyp(ct_large, c_large, (1.0, 1.0, 1.0))

    assert small.ok and large.ok
    assert small.measurements["max_diameter_mm"] < 8.0
    assert large.measurements["max_diameter_mm"] > 8.0


def test_measure_polyp_reports_failure_without_raising():
    ct = np.full((32, 32, 32), AIR_HU, dtype=np.float32)
    res = measure_polyp(ct, (16.0, 16.0, 16.0), (1.0, 1.0, 1.0))
    assert res.ok is False
    assert "no soft tissue" in res.message


def test_measure_polyp_sphericity_is_lower_for_an_elongated_lesion():
    n = 80
    ct = np.full((n, n, n), AIR_HU, dtype=np.float32)
    ct[ball_mask((n, n, n), (n / 2,) * 3, (4.0, 4.0, 11.0))] = SOFT_HU
    res = measure_polyp(ct, (n / 2, n / 2, n / 2), (1.0, 1.0, 1.0))
    assert res.ok
    assert res.measurements["sphericity"] < 0.95
    assert res.measurements["max_diameter_mm"] == pytest.approx(22.0, abs=1.5)
