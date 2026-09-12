"""Tests for ctc_core.masks.

The fill-hole tests encode the root cause found on 2026-09-09: passing
``fullyConnected=True`` to ``sitk.BinaryFillhole`` makes the *background*
26-connected, so an enclosed cavity escapes through a single diagonal gap in the
surrounding wall and is not filled. On patient 0022/primary that discarded
2.3 L of colonic gas. See docs/FAILURE_ANALYSIS.md section 2.2.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from ctc_core.masks import (
    build_air_mask,
    build_body_air_masks,
    build_body_mask,
    label_air_components,
    touches_image_border,
)

BODY_HU = 0.0
AIR_HU = -1000.0


def phantom(diagonal_gap: bool) -> sitk.Image:
    """A hollow box of tissue with an air cavity inside.

    The wall is two voxels thick. When ``diagonal_gap`` is set, two wall voxels
    are removed at a full 3-D diagonal offset, so the cavity connects to the
    outside via a 26-connected background path but not via a 6-connected one.
    """
    arr = np.full((13, 13, 13), AIR_HU, dtype=np.float32)
    arr[1:12, 1:12, 1:12] = BODY_HU  # solid block
    arr[3:10, 3:10, 3:10] = AIR_HU  # hollow it out -> wall thickness 2

    if diagonal_gap:
        arr[1, 6, 6] = AIR_HU
        arr[2, 7, 7] = AIR_HU  # reachable from (1,6,6) only diagonally

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


CAVITY_VOXELS = 7**3


def test_sealed_cavity_is_filled_by_every_variant():
    ct = phantom(diagonal_gap=False)
    for variant in ("image26", "image6", "bbox"):
        body = build_body_mask(ct, closing_radius=0, fill_holes=variant)
        assert body[6, 6, 6], f"{variant} failed to fill a fully sealed cavity"


def test_diagonal_gap_defeats_fullyconnected_fill():
    """The 2026-08 batch's setting loses the cavity; the other two keep it."""
    ct = phantom(diagonal_gap=True)

    leaky = build_body_mask(ct, closing_radius=0, fill_holes="image26")
    assert not leaky[6, 6, 6], "image26 unexpectedly sealed the diagonal gap"

    for variant in ("image6", "bbox"):
        body = build_body_mask(ct, closing_radius=0, fill_holes=variant)
        assert body[6, 6, 6], f"{variant} failed to fill a diagonally-gapped cavity"


def test_diagonal_gap_costs_the_whole_air_cavity():
    """The consequence downstream: the air mask loses the cavity entirely."""
    ct = phantom(diagonal_gap=True)
    ct_arr = sitk.GetArrayFromImage(ct)

    def air_voxels(variant: str) -> int:
        body = build_body_mask(ct, closing_radius=0, fill_holes=variant)
        return int(build_air_mask(ct_arr, body, closing_radius=0).sum())

    assert air_voxels("image26") == 0
    assert air_voxels("image6") >= CAVITY_VOXELS
    assert air_voxels("image6") == air_voxels("bbox")


def test_image6_and_bbox_agree():
    for gap in (False, True):
        ct = phantom(diagonal_gap=gap)
        a = build_body_mask(ct, closing_radius=0, fill_holes="image6")
        b = build_body_mask(ct, closing_radius=0, fill_holes="bbox")
        assert np.array_equal(a, b)


def test_fill_holes_none_leaves_cavity_open():
    ct = phantom(diagonal_gap=False)
    body = build_body_mask(ct, closing_radius=0, fill_holes="none")
    assert not body[6, 6, 6]


# --------------------------------------------------------------------------
# Labelling connectivity vs. the Fast Marching stencil (section 2.3)
# --------------------------------------------------------------------------


def corner_touching_blocks() -> np.ndarray:
    """Two cubes sharing exactly one corner: one 26-component, two 6-components."""
    air = np.zeros((12, 12, 12), dtype=bool)
    air[1:5, 1:5, 1:5] = True
    air[5:9, 5:9, 5:9] = True  # touches the first only at (4,4,4)/(5,5,5)
    return air


def test_corner_contact_is_one_component_at_26_and_two_at_6():
    air = corner_touching_blocks()
    labels26, _ = label_air_components(air, connectivity=26, dust_threshold_voxels=0)
    labels6, _ = label_air_components(air, connectivity=6, dust_threshold_voxels=0)

    a, b = (2, 2, 2), (6, 6, 6)
    assert labels26[a] == labels26[b]
    assert labels6[a] != labels6[b]


def test_dust_threshold_removes_small_components():
    air = np.zeros((12, 12, 12), dtype=bool)
    air[1:6, 1:6, 1:6] = True  # 125 voxels
    air[9, 9, 9] = True  # dust
    labels, _ = label_air_components(air, dust_threshold_voxels=100)
    assert labels[3, 3, 3] != 0
    assert labels[9, 9, 9] == 0


def test_dust_threshold_rejecting_everything_raises():
    air = np.zeros((12, 12, 12), dtype=bool)
    air[1, 1, 1] = True
    with pytest.raises(RuntimeError, match="dust threshold"):
        label_air_components(air, dust_threshold_voxels=100)


def test_touches_image_border():
    m = np.zeros((5, 5, 5), dtype=bool)
    assert not touches_image_border(m)
    m[2, 2, 2] = True
    assert not touches_image_border(m)
    m[0, 2, 2] = True
    assert touches_image_border(m)


def test_air_closing_bridges_a_one_voxel_gap():
    """Closing radius 1 rejoins fragments split by a partial-volume gap."""
    ct_arr = np.zeros((16, 16, 16), dtype=np.float32)
    body = np.ones((16, 16, 16), dtype=bool)
    ct_arr[4:12, 4:12, 4:8] = AIR_HU
    ct_arr[4:12, 4:12, 9:13] = AIR_HU  # separated by the plane z index 8

    open0 = build_air_mask(ct_arr, body, closing_radius=0)
    _, sizes0 = label_air_components(open0, dust_threshold_voxels=0)
    assert (sizes0 > 0).sum() == 2

    closed1 = build_air_mask(ct_arr, body, closing_radius=1)
    _, sizes1 = label_air_components(closed1, dust_threshold_voxels=0)
    assert (sizes1 > 0).sum() == 1


# ---------------------------------------------------------------------------
# Slice-wise filling: the fix for a cavity open in z (2026-09-10)
# ---------------------------------------------------------------------------


def phantom_cavity_open_in_z() -> sitk.Image:
    """A cavity sealed within every axial slice but running out of the volume in z.

    This is the situation HQColon exposed on 5 of 26 reference series: the gas
    is enclosed by the body wall in-plane, but its component reaches a volume
    face, so it is not a hole in 3-D and no 3-D fill can recover it.
    """
    arr = np.full((14, 14, 14), AIR_HU, dtype=np.float32)
    arr[:, 2:12, 2:12] = BODY_HU  # a tissue tube running the full z extent
    arr[:, 5:9, 5:9] = AIR_HU  # gas channel inside it, open at both z faces
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_three_d_fills_lose_a_cavity_that_is_open_in_z():
    ct = phantom_cavity_open_in_z()
    for variant in ("image26", "image6", "bbox"):
        body = build_body_mask(ct, closing_radius=0, fill_holes=variant)
        assert not body[7, 6, 6], f"{variant} unexpectedly sealed a z-open cavity"


def test_slicewise_fill_recovers_a_cavity_open_in_z():
    ct = phantom_cavity_open_in_z()
    body = build_body_mask(ct, closing_radius=0, fill_holes="slicewise")
    assert body[7, 6, 6]
    # And the gas is then available to the air mask, which is the point.
    ct_arr = sitk.GetArrayFromImage(ct)
    air = build_air_mask(ct_arr, body, closing_radius=0)
    assert air[7, 6, 6]
    assert int(air.sum()) >= 14 * 4 * 4 * 0.9


def test_build_body_mask_rejects_auto():
    """The adaptive choice needs the air mask, so it lives one level up."""
    ct = phantom_cavity_open_in_z()
    with pytest.raises(ValueError, match="build_body_air_masks"):
        build_body_mask(ct, closing_radius=0, fill_holes="auto")


def test_auto_falls_back_to_slicewise_when_3d_finds_no_colon():
    ct = phantom_cavity_open_in_z()
    body, air, diag = build_body_air_masks(
        ct, body_closing_radius=0, air_closing_radius=0
    )
    assert diag["fill_holes_used"] == "slicewise"
    assert diag["fallback_reason"]
    assert body[7, 6, 6] and air[7, 6, 6]


def test_auto_keeps_the_3d_fill_when_it_already_works():
    """A sealed cavity: slice-wise gains nothing, so the 3-D fill is kept.

    This is what protects the 21 reference series that the slice-wise fill
    degraded (precision 0.786 -> 0.670 across the 26).
    """
    ct = phantom(diagonal_gap=False)
    body, air, diag = build_body_air_masks(
        ct, body_closing_radius=0, air_closing_radius=0
    )
    assert diag["fill_holes_used"] == "bbox"
    assert body[6, 6, 6] and air[6, 6, 6]


def test_explicit_fill_holes_bypasses_the_adaptive_logic():
    ct = phantom_cavity_open_in_z()
    _, air, diag = build_body_air_masks(
        ct, body_closing_radius=0, air_closing_radius=0, fill_holes="bbox"
    )
    assert diag["fill_holes_used"] == "bbox"
    assert int(air.sum()) == 0  # the z-open cavity is lost, as it must be
    assert "fallback_reason" not in diag


def test_slicewise_still_fills_a_sealed_cavity():
    ct = phantom(diagonal_gap=False)
    body = build_body_mask(ct, closing_radius=0, fill_holes="slicewise")
    assert body[6, 6, 6]


def test_slicewise_survives_the_diagonal_gap_case():
    """The section 2.2 phantom: slice-wise filling must also seal it."""
    ct = phantom(diagonal_gap=True)
    body = build_body_mask(ct, closing_radius=0, fill_holes="slicewise")
    assert body[6, 6, 6]
