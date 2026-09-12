"""Tests for ctc_core.fatmap.

The spherical profile has a real regression target: the 2026-08 batch recorded
its fat summary per series, so the port is checked against it on 0001/primary.
The polar map has none, so it is tested on phantoms with known answers.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ctc_core.fatmap import (
    FatConfig,
    centerline_frames,
    fat_map_polar,
    fat_profile_spherical,
    polar_asymmetry,
    resample_centerline,
    summarise_fat_profile,
)

FAT_HU = -100.0
MUSCLE_HU = 50.0
AIR_HU = -1000.0


# ---------------------------------------------------------------------------
# Arc-length resampling
# ---------------------------------------------------------------------------


def test_resample_is_uniform_in_arc_length():
    # A polyline whose points are unevenly spaced along a straight line.
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [1.5, 0, 0], [10, 0, 0]])
    s, pts = resample_centerline(xyz, 2.0)
    assert s[0] == 0.0
    assert s[-1] == pytest.approx(10.0)
    steps = np.diff(s)[:-1]  # the last step is the remainder to the end
    assert np.allclose(steps, 2.0)
    # Positions must follow arc length, not point index.
    assert pts[1] == pytest.approx([2.0, 0.0, 0.0])


def test_resample_always_includes_the_far_end():
    xyz = np.array([[0.0, 0, 0], [7.0, 0, 0]])
    s, _ = resample_centerline(xyz, 5.0)
    assert s[-1] == pytest.approx(7.0)


def test_resample_rejects_degenerate_input():
    with pytest.raises(ValueError):
        resample_centerline(np.zeros((1, 3)), 1.0)
    with pytest.raises(ValueError):
        resample_centerline(np.zeros((4, 3)), 1.0)  # zero total length


# ---------------------------------------------------------------------------
# Centerline frames
# ---------------------------------------------------------------------------


def test_frames_are_orthonormal():
    t = np.linspace(0, 4 * np.pi, 60)
    xyz = np.column_stack([20 * np.cos(t), 20 * np.sin(t), np.linspace(0, 60, 60)])
    t_hat, n_hat, b_hat = centerline_frames(xyz)

    assert np.allclose(np.linalg.norm(t_hat, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(n_hat, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(b_hat, axis=1), 1.0, atol=1e-6)
    assert np.allclose((t_hat * n_hat).sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose((t_hat * b_hat).sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose((n_hat * b_hat).sum(axis=1), 0.0, atol=1e-6)


def test_frame_normal_points_anteriorly_for_a_craniocaudal_path():
    """theta must be anatomically anchored or asymmetry means nothing."""
    xyz = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 100, 20)])
    _, n_hat, _ = centerline_frames(xyz)
    # Anterior is -y in LPS.
    assert np.allclose(n_hat, np.array([0.0, -1.0, 0.0]), atol=1e-6)


def test_frame_falls_back_when_the_tangent_is_anterior():
    """A path running straight anteriorly would make the reference degenerate."""
    xyz = np.column_stack([np.zeros(20), np.linspace(0, -100, 20), np.zeros(20)])
    t_hat, n_hat, _ = centerline_frames(xyz)
    assert np.allclose(np.linalg.norm(n_hat, axis=1), 1.0, atol=1e-6)
    assert np.allclose((t_hat * n_hat).sum(axis=1), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------


def fat_sleeve_phantom(
    fat_sides: str = "all",
    n: int = 80,
    nz: int = 60,
    r_lumen: int = 6,
    r_fat_out: int = 24,
):
    """Air lumen along z, a thin wall, then a fat sleeve, in muscle.

    ``fat_sides`` selects "all", "left" (x < centre) or "right".
    """
    arr = np.full((nz, n, n), MUSCLE_HU, dtype=np.float32)
    c = n // 2
    _, yy, xx = np.mgrid[0:nz, 0:n, 0:n]
    rad2 = (yy - c) ** 2 + (xx - c) ** 2

    sleeve = (rad2 > (r_lumen + 2) ** 2) & (rad2 <= r_fat_out**2)
    if fat_sides == "left":
        sleeve &= xx < c
    elif fat_sides == "right":
        sleeve &= xx >= c
    arr[sleeve] = FAT_HU
    arr[rad2 <= r_lumen**2] = AIR_HU

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    ct_arr = sitk.GetArrayFromImage(img)
    air = ct_arr < -700.0
    body = np.ones_like(air, dtype=bool)
    xyz = np.column_stack(
        [np.full(nz - 10, float(c)), np.full(nz - 10, float(c)), np.arange(5, nz - 5, dtype=float)]
    )
    return img, ct_arr, body, air, xyz


def test_spherical_profile_finds_a_uniform_fat_sleeve():
    img, ct_arr, body, air, xyz = fat_sleeve_phantom()
    s, pts = resample_centerline(xyz, 5.0)
    rows = fat_profile_spherical(img, ct_arr, body, air, s, pts, FatConfig())
    summary = summarise_fat_profile(rows)

    assert summary["n_valid"] == summary["n_points"]
    assert summary["fat_mean_hu_median"] == pytest.approx(FAT_HU, abs=1.0)
    assert summary["fat_fraction_mean"] > 0.5


def test_spherical_profile_excludes_the_lumen_from_the_roi():
    """Colon gas must never be counted as pericolonic tissue."""
    img, ct_arr, body, air, xyz = fat_sleeve_phantom()
    s, pts = resample_centerline(xyz, 5.0)
    rows = fat_profile_spherical(img, ct_arr, body, air, s, pts, FatConfig())

    # ROI voxels must be well below the full sphere, since the lumen is excised.
    sphere_voxels = (4 / 3) * np.pi * 15.0**3
    assert all(r["roi_voxels"] < sphere_voxels for r in rows)
    assert all(r["roi_voxels"] > 0 for r in rows)


def test_spherical_profile_reports_nan_with_a_reason_when_there_is_no_fat():
    img, ct_arr, body, air, xyz = fat_sleeve_phantom()
    ct_arr = np.where(ct_arr == FAT_HU, MUSCLE_HU, ct_arr)  # remove all fat
    s, pts = resample_centerline(xyz, 5.0)
    rows = fat_profile_spherical(img, ct_arr, body, air, s, pts, FatConfig())
    summary = summarise_fat_profile(rows)

    assert summary["n_valid"] == 0
    assert summary["fat_mean_hu_median"] is None
    assert summary["n_zero_fat"] == summary["n_points"]


# ---------------------------------------------------------------------------
# Polar map
# ---------------------------------------------------------------------------


def test_polar_map_has_the_expected_shape_and_finds_the_wall():
    img, ct_arr, body, air, xyz = fat_sleeve_phantom()
    s, pts = resample_centerline(xyz, 5.0)
    cfg = FatConfig(n_theta=36)
    fmap = fat_map_polar(img, ct_arr, body, air, s, pts, cfg)

    assert fmap["fat_fraction"].shape == (len(s), 36)
    assert fmap["theta"].shape == (36,)
    # The wall sits just outside the 6 mm lumen radius.
    wall = fmap["wall_mm"]
    assert np.nanmedian(wall) == pytest.approx(6.5, abs=2.0)


def test_polar_map_is_symmetric_for_a_uniform_sleeve():
    img, ct_arr, body, air, xyz = fat_sleeve_phantom("all")
    s, pts = resample_centerline(xyz, 5.0)
    fmap = fat_map_polar(img, ct_arr, body, air, s, pts, FatConfig(n_theta=36))
    asym = polar_asymmetry(fmap)

    assert asym["fat_asymmetry"] == pytest.approx(0.0, abs=0.05)
    assert asym["fat_fraction_left"] > 0.5


def test_asymmetry_is_in_the_patient_frame_whatever_the_direction_of_travel():
    """Fat on the patient's left must read positive, forwards and backwards.

    The phantom's image x axis is the LPS +x axis, i.e. the patient's left, so
    ``fat_sleeve_phantom("right")`` (fat at x >= centre) is fat on the patient's
    left. Defining the sides by the halves of theta instead flips the sign when
    the path is traversed the other way (the seed order is arbitrary) and between
    ascending and descending segments.
    """
    for side, sign in (("right", +1.0), ("left", -1.0)):
        img, ct_arr, body, air, xyz = fat_sleeve_phantom(side)
        for path in (xyz, xyz[::-1]):
            s, pts = resample_centerline(path, 5.0)
            fmap = fat_map_polar(img, ct_arr, body, air, s, pts, FatConfig(n_theta=36))
            asym = polar_asymmetry(fmap)["fat_asymmetry"]
            assert np.sign(asym) == sign
            assert abs(asym) > 0.5


def test_theta_halves_index_flips_with_the_direction_of_travel():
    """Why the original theta-half index is not a left/right measure: pinned."""
    img, ct_arr, body, air, xyz = fat_sleeve_phantom("right")
    signs = []
    for path in (xyz, xyz[::-1]):
        s, pts = resample_centerline(path, 5.0)
        fmap = fat_map_polar(img, ct_arr, body, air, s, pts, FatConfig(n_theta=36))
        signs.append(np.sign(polar_asymmetry(fmap)["fat_asymmetry_theta_halves"]))
    assert signs[0] == -signs[1]


def test_asymmetry_ignores_rays_that_are_not_lateral():
    """Fat only anteriorly and posteriorly has no left-right asymmetry."""
    img, ct_arr, body, air, xyz = fat_sleeve_phantom("all")
    s, pts = resample_centerline(xyz, 5.0)
    fmap = fat_map_polar(img, ct_arr, body, air, s, pts, FatConfig(n_theta=36))
    lateral = np.abs(fmap["lateral"]) >= np.cos(np.deg2rad(45.0))
    fmap["fat_fraction"] = np.where(lateral, 0.0, 1.0)  # fat only on non-lateral rays
    asym = polar_asymmetry(fmap)
    assert asym["fat_fraction_left"] == 0.0 and asym["fat_fraction_right"] == 0.0
    assert asym["fat_asymmetry"] is None


def test_polar_map_detects_one_sided_fat_with_opposite_signs():
    """The asymmetry index must change sign when the fat moves to the other side."""
    out = {}
    for side in ("left", "right"):
        img, ct_arr, body, air, xyz = fat_sleeve_phantom(side)
        s, pts = resample_centerline(xyz, 5.0)
        fmap = fat_map_polar(img, ct_arr, body, air, s, pts, FatConfig(n_theta=36))
        out[side] = polar_asymmetry(fmap)["fat_asymmetry"]

    assert abs(out["left"]) > 0.5
    assert abs(out["right"]) > 0.5
    assert np.sign(out["left"]) == -np.sign(out["right"])


def test_polar_ring_is_anchored_to_the_wall_not_the_centerline():
    """Stations of different calibre must still sample pericolonic tissue.

    A ring measured from the centerline would fall inside the lumen of a wide
    segment and outside the fat of a narrow one.
    """
    wide, _, _, _, _ = fat_sleeve_phantom(r_lumen=6)
    narrow_img, narrow_arr, body, air, xyz = fat_sleeve_phantom(r_lumen=3)
    s, pts = resample_centerline(xyz, 5.0)
    fmap = fat_map_polar(narrow_img, narrow_arr, body, air, s, pts, FatConfig(n_theta=36))
    # Fat is still found despite the lumen being half the radius.
    assert np.nanmean(fmap["fat_fraction"]) > 0.4


# ---------------------------------------------------------------------------
# Regression against the 2026-08 batch
# ---------------------------------------------------------------------------

# The prototype's own batch output (not distributed): the test runs only where it exists.
REFERENCE_ROOT = Path(os.environ.get("CTC_REFERENCE_ROOT", "__not_available__"))
CASE = REFERENCE_ROOT / "out_cohort/1.3.6.1.4.1.9328.50.4.0001/primary"
WORK = Path("data/work/1.3.6.1.4.1.9328.50.4.0001/primary/ct_lps_iso.nii.gz")


@pytest.mark.skipif(
    not (CASE / "metrics.json").exists() or not WORK.exists(),
    reason="reference batch output or resampled volume not available",
)
def test_spherical_profile_reproduces_the_2026_08_batch():
    """Exact against the batch, given the batch's own structuring element.

    The port matches the recorded values to floating-point rounding when the air
    mask is closed with the reference's ball of offsets. The pipeline itself uses
    sitk.BinaryMorphologicalClosing, which differs on about 1.6 % of air voxels
    at boundaries and moves fat_fraction_mean by ~1 %; that difference is in the
    mask, not in the fat computation, which is what this test pins.
    """
    from ctc_core.masks import build_body_mask

    def ball(r):
        return [
            (dz, dy, dx)
            for dz in range(-r, r + 1)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dz * dz + dy * dy + dx * dx <= r * r
        ]

    def shift_or(m, offs, op):
        nz, ny, nx = m.shape
        out = np.zeros_like(m) if op == "dilate" else np.ones_like(m)
        for dz, dy, dx in offs:
            sh = np.zeros_like(m)
            zs = slice(max(0, dz), nz + min(0, dz))
            zd = slice(max(0, -dz), nz + min(0, -dz))
            ys = slice(max(0, dy), ny + min(0, dy))
            yd = slice(max(0, -dy), ny + min(0, -dy))
            xs = slice(max(0, dx), nx + min(0, dx))
            xd = slice(max(0, -dx), nx + min(0, -dx))
            sh[zd, yd, xd] = m[zs, ys, xs]
            out = (out | sh) if op == "dilate" else (out & sh)
        return out

    ref = json.loads((CASE / "metrics.json").read_text())["stages"]["phase0"]
    ct = sitk.ReadImage(str(WORK), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(ct)
    body = build_body_mask(ct, -300.0, 2, "bbox")
    offs = ball(1)
    air = shift_or(shift_or((arr < -700.0) & body, offs, "dilate"), offs, "erode")

    pts = []
    with (CASE / "centerline_points.csv").open() as fh:
        for row in csv.DictReader(fh):
            pts.append((float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])))

    s, xyz = resample_centerline(np.array(pts), 5.0)
    got = summarise_fat_profile(fat_profile_spherical(ct, arr, body, air, s, xyz, FatConfig()))

    assert got["n_points"] == ref["n_points"]
    assert got["n_valid"] == ref["n_valid"]
    assert got["fat_mean_hu_median"] == pytest.approx(ref["fat_mean_hu_median"], abs=1e-4)
    assert got["fat_fraction_mean"] == pytest.approx(ref["fat_fraction_mean"], abs=1e-5)
    assert got["fat_volume_ml_sum"] == pytest.approx(ref["fat_volume_ml_sum"], abs=0.01)


# ---------------------------------------------------------------------------
# Ring indices (primary since 2026-09-12)
# ---------------------------------------------------------------------------


def test_ring_indices_measure_a_wide_lumen_that_the_sphere_cannot():
    """A distended colon: lumen radius 16 mm, fat sleeve beyond a 2 mm wall.

    The 15 mm sphere centred on the centerline lies wholly inside the gas, so
    the spherical profile has nothing to measure; the ring, anchored to the
    detected wall, still finds the fat.
    """
    from ctc_core.fatmap import summarise_fat_map

    img, ct_arr, body, air, xyz = fat_sleeve_phantom(n=100, r_lumen=16, r_fat_out=40)
    s, pts = resample_centerline(xyz, 5.0)
    sphere = summarise_fat_profile(fat_profile_spherical(img, ct_arr, body, air, s, pts,
                                                         FatConfig()))
    assert sphere["n_valid"] == 0

    ring = summarise_fat_map(fat_map_polar(img, ct_arr, body, air, s, pts,
                                           FatConfig(n_theta=36)), FatConfig())
    assert ring["ring_valid_fraction"] == 1.0
    assert ring["ring_fat_mean_hu_median"] == pytest.approx(FAT_HU, abs=1.0)
    assert ring["ring_fat_fraction_mean"] > 0.7


def test_smooth_centerline_keeps_ends_and_removes_voxel_kinks():
    from ctc_core.fatmap import smooth_centerline

    z = np.arange(0, 60, dtype=float)
    x = np.where(np.arange(60) % 2 == 0, 0.0, 1.0)  # 1-voxel zigzag
    p = np.column_stack([x, np.zeros(60), z])
    q = smooth_centerline(p, 2.0)
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1])
    assert np.ptp(q[5:-5, 0]) < 0.2  # zigzag flattened
