"""Tests for ctc_core.centerline.

Most tests run on synthetic phantoms. The regression test at the end pins the
2026-08 batch result for 0001/primary and is skipped when the reference DICOM
is not mounted.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ctc_core.centerline import (
    CenterlineConfig,
    backtrack_centerline,
    compute_arrival_time,
    compute_path_length_mm,
    detect_seed_endpoints,
    extract_centerline,
)

AIR_HU = -1000.0
TISSUE_HU = 0.0


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------


def test_batch_preset_reproduces_the_old_defects():
    cfg = CenterlineConfig.batch_2026_08()
    assert cfg.fill_holes == "image26"
    assert cfg.air_closing_radius == 0
    assert cfg.connectivity == 26
    assert cfg.seed_span_components is False


def test_corrected_preset_matches_the_defaults():
    assert CenterlineConfig.corrected() == CenterlineConfig()
    cfg = CenterlineConfig.corrected()
    # The labelling connectivity must match the Fast Marching stencil.
    assert cfg.connectivity == 6
    assert cfg.seed_span_components is True


# ---------------------------------------------------------------------------
# Path length
# ---------------------------------------------------------------------------


def test_path_length_of_axis_aligned_steps():
    path = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    assert compute_path_length_mm(path, (1.0, 1.0, 1.0)) == pytest.approx(2.0)


def test_path_length_uses_anisotropic_spacing():
    # spacing is (x, y, z); a step along k is a step of z.
    path = [(0, 0, 0), (1, 0, 0)]
    assert compute_path_length_mm(path, (0.5, 0.5, 3.0)) == pytest.approx(3.0)


def test_path_length_of_a_diagonal_step():
    path = [(0, 0, 0), (1, 1, 1)]
    assert compute_path_length_mm(path, (1.0, 1.0, 1.0)) == pytest.approx(np.sqrt(3))


def test_path_length_of_degenerate_path():
    assert compute_path_length_mm([(0, 0, 0)], (1.0, 1.0, 1.0)) == 0.0
    assert compute_path_length_mm([], (1.0, 1.0, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# Geodesic on a straight tube
# ---------------------------------------------------------------------------


def straight_tube(length: int = 40, radius: int = 4, cap: int = 3) -> sitk.Image:
    """A cylindrical air lumen along z, embedded in tissue.

    ``cap`` voxels of tissue close each end. Without them the lumen would reach
    the volume face, where hole filling correctly refuses to seal it and the
    body mask then excludes the whole tube.
    """
    n = 2 * radius + 9
    arr = np.full((length, n, n), TISSUE_HU, dtype=np.float32)
    c = n // 2
    _, yy, xx = np.mgrid[0:length, 0:n, 0:n]
    arr[((yy - c) ** 2 + (xx - c) ** 2) <= radius**2] = AIR_HU
    if cap > 0:
        arr[:cap] = TISSUE_HU
        arr[-cap:] = TISSUE_HU
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_geodesic_traverses_a_straight_tube():
    ct = straight_tube(length=40)
    lumen_arr = (sitk.GetArrayFromImage(ct) < -700).astype(np.uint8)
    lumen = sitk.GetImageFromArray(lumen_arr)
    lumen.CopyInformation(ct)

    c = lumen_arr.shape[1] // 2
    start, end = (4, c, c), (35, c, c)
    arrival, mask = compute_arrival_time(lumen, start, CenterlineConfig())
    path = backtrack_centerline(arrival, mask, start, end)

    assert path[0] == start and path[-1] == end
    # A straight tube should give a near-straight path.
    assert compute_path_length_mm(path, (1.0, 1.0, 1.0)) == pytest.approx(31.0, abs=3.0)


def test_unreachable_end_seed_raises():
    """Two tube halves separated by a tissue plug: the far seed has no arrival time."""
    ct = straight_tube(length=40)
    arr = sitk.GetArrayFromImage(ct)
    arr[20:23] = TISSUE_HU  # plug
    lumen_arr = (arr < -700).astype(np.uint8)
    lumen = sitk.GetImageFromArray(lumen_arr)
    lumen.CopyInformation(ct)

    c = lumen_arr.shape[1] // 2
    arrival, mask = compute_arrival_time(lumen, (4, c, c), CenterlineConfig())
    with pytest.raises(ValueError, match="unreachable"):
        backtrack_centerline(arrival, mask, (4, c, c), (35, c, c))


# ---------------------------------------------------------------------------
# Seed selection: the section 2.1 defect and its fix
# ---------------------------------------------------------------------------


def two_component_air() -> np.ndarray:
    """Two collinear tube segments separated by a gap.

    Tubular rather than blocky so the Lee skeleton has unambiguous degree-1
    endpoints, as a colon does; a solid box skeletonizes to a sheet with none.
    """
    n, length, radius = 25, 80, 4
    air = np.zeros((length, n, n), dtype=bool)
    c = n // 2
    _, yy, xx = np.mgrid[0:length, 0:n, 0:n]
    tube = ((yy - c) ** 2 + (xx - c) ** 2) <= radius**2
    air[tube] = True
    air[36:44] = False  # gap splitting it in two
    air[:3] = False
    air[-3:] = False
    return air


SEED_CFG = {"min_air_cc_voxels": 500, "dust_threshold_voxels": 200}


def test_batch_seeding_keeps_both_seeds_in_one_component():
    """Reproduces the defect: bridging can never be reached."""
    cfg = CenterlineConfig(
        seed_span_components=False, connectivity=26, **SEED_CFG
    )
    _, _, diag = detect_seed_endpoints(two_component_air(), (1.0, 1.0, 1.0), cfg)
    assert diag["seeds_cross_components"] is False
    assert diag["start_cc"] == diag["end_cc"]


def test_corrected_seeding_spans_components():
    """The fix: the extreme endpoint pair may lie in different components."""
    cfg = CenterlineConfig(seed_span_components=True, **SEED_CFG)
    _, _, diag = detect_seed_endpoints(two_component_air(), (1.0, 1.0, 1.0), cfg)
    assert diag["seeds_cross_components"] is True
    assert diag["start_cc"] != diag["end_cc"]


def test_seed_detection_rejects_an_uninsufflated_study():
    air = np.zeros((40, 20, 20), dtype=bool)
    air[5:10, 5:10, 5:10] = True  # 125 voxels
    cfg = CenterlineConfig(min_air_cc_voxels=10_000, dust_threshold_voxels=10)
    with pytest.raises(RuntimeError, match="no insufflation"):
        detect_seed_endpoints(air, (1.0, 1.0, 1.0), cfg)


# ---------------------------------------------------------------------------
# End to end on a phantom with a gap the bridge planner must cross
# ---------------------------------------------------------------------------


def test_bridging_joins_two_tube_segments():
    """A tube split by a short plug: the corrected config bridges it, batch cannot."""
    ct = straight_tube(length=60, radius=4)
    arr = sitk.GetArrayFromImage(ct)
    arr[28:33] = TISSUE_HU  # 5 mm plug, well inside max_bridge_mm
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    c = arr.shape[1] // 2
    cfg = CenterlineConfig(
        min_air_cc_voxels=100, dust_threshold_voxels=50, air_closing_radius=0
    )
    res = extract_centerline(img, (4, c, c), (55, c, c), cfg)

    assert res.diag["n_bridges"] == 1
    assert res.diag["used_kimimaro"] is True
    assert res.diag["n_ccs_visited"] == 2
    assert res.length_mm > 50.0


# ---------------------------------------------------------------------------
# Regression against the 2026-08 batch
# ---------------------------------------------------------------------------

# Directory the manifest's download_dir paths are relative to (scripts/download_tcia.py).
REFERENCE_ROOT = Path(os.environ.get("CTC_RAW_ROOT", "."))
BASELINE_DIR = REFERENCE_ROOT / "cohort/raw/1.3.6.1.4.1.9328.50.4.0001/1.3.6.1.4.1.9328.50.4.563"


@pytest.mark.skipif(not BASELINE_DIR.is_dir(), reason="reference DICOM not available")
def test_batch_config_reproduces_0001_primary():
    """The pinned regression baseline: 936.33 mm, 678 points, 16 CCs, 0 bridges.

    These are the exact values in the reference batch's metrics.json for
    patient 0001, primary series. Any change to the batch preset must keep them.
    """
    from ctc_core.io import get_or_make_resampled_ct
    from ctc_core.masks import build_air_mask, build_body_mask

    cfg = CenterlineConfig.batch_2026_08()
    ct = get_or_make_resampled_ct(
        dicom_dir=BASELINE_DIR,
        out_ct_path=Path("data/work/1.3.6.1.4.1.9328.50.4.0001/primary/ct_lps_iso.nii.gz"),
        iso_spacing=1.0,
    )
    ct_arr = sitk.GetArrayFromImage(ct)
    body = build_body_mask(ct, cfg.body_threshold, cfg.body_closing_radius, cfg.fill_holes)
    air = build_air_mask(ct_arr, body, cfg.air_threshold, cfg.air_closing_radius, reference=ct)
    start, end, sdiag = detect_seed_endpoints(air, (1.0, 1.0, 1.0), cfg)

    assert sdiag["largest_air_cc_voxels"] == 1_878_117
    assert sdiag["seed_source"] == "skeleton_endpoints"

    res = extract_centerline(ct, start, end, cfg)
    assert res.length_mm == pytest.approx(936.33, abs=0.01)
    assert len(res.path_kji) == 678
    assert res.diag["n_ccs_total"] == 16
    assert res.diag["n_ccs_visited"] == 1
    assert res.diag["n_bridges"] == 0
    assert res.diag["used_kimimaro"] is False


# ---------------------------------------------------------------------------
# Bridge admissibility gate and fallback (2026-09-10)
# ---------------------------------------------------------------------------


def parallel_tubes() -> sitk.Image:
    """Two parallel air tubes separated laterally by solid tissue.

    Every candidate bridge between their skeleton endpoints crosses that tissue,
    so all edges exceed the wall-fraction gate. Contrast with a tube split by a
    plug, where a long bridge spanning a short barrier has a *low* wall fraction
    because the metric is a fraction of bridge length, not an absolute
    thickness -- see test_wall_fraction_is_diluted_by_bridge_length.

    The two tubes have deliberately different lengths so that a fallback to the
    largest component is distinguishable from a fallback to whichever component
    the start seed is in.
    """
    n, length, radius = 44, 60, 3
    arr = np.full((length, n, n), TISSUE_HU, dtype=np.float32)
    _, yy, xx = np.mgrid[0:length, 0:n, 0:n]
    short = ((yy - 11) ** 2 + (xx - 11) ** 2) <= radius**2
    long_ = ((yy - 33) ** 2 + (xx - 33) ** 2) <= radius**2
    arr[short] = AIR_HU
    arr[long_] = AIR_HU
    arr[20:] = np.where(short[20:], TISSUE_HU, arr[20:])  # short tube ends at z=20
    arr[:4] = TISSUE_HU
    arr[-4:] = TISSUE_HU
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


GATE_CFG = {
    "min_air_cc_voxels": 100,
    "dust_threshold_voxels": 50,
    "air_closing_radius": 0,
}

# The gate and fallback tests below describe the shortest-path formulation, in
# which a connection between two given seeds is mandatory and can therefore
# fail. max_coverage never demands a connection, so it has no equivalent
# failure; its behaviour is covered separately at the end of the file.
SP_GATE_CFG = {**GATE_CFG, "traversal": "shortest_path"}

# max_coverage is available but not the default: measured against HQColon it was
# worse than shortest path (coverage 0.575 vs 0.937). Its tests pin it
# explicitly, together with the gap-only wall measurement and thickness gate it
# was designed against. See docs/EVALUATION_HQCOLON.md section 7.
MC_CFG = {
    **GATE_CFG,
    "traversal": "max_coverage",
    "wall_frac_over_gap_only": True,
    "max_bridge_wall_fraction": 1.0,
    "max_bridge_wall_mm": 12.0,
}


def test_ungated_planner_bridges_straight_through_tissue():
    """The pre-gate behaviour: a bridge is drawn however much tissue it crosses."""
    img = parallel_tubes()
    cfg = CenterlineConfig(
        max_bridge_wall_fraction=1.0,
        max_bridge_wall_mm=float("inf"),
        fall_back_to_largest_component=False,
        **SP_GATE_CFG,
    )
    res = extract_centerline(img, (6, 11, 11), (50, 33, 33), cfg)
    assert res.diag["n_bridges"] == 1
    assert res.diag["max_bridge_wall_frac"] > 0.5


def test_gate_refuses_a_tissue_bridge_and_falls_back():
    """With the gate on, the series is reported as fragmented, not stitched."""
    img = parallel_tubes()
    cfg = CenterlineConfig(
        max_bridge_wall_mm=1.0, fall_back_to_largest_component=True, **SP_GATE_CFG
    )
    # Start seed deliberately in the SHORT tube.
    res = extract_centerline(img, (6, 11, 11), (50, 33, 33), cfg)

    assert res.diag["n_bridges"] == 0
    assert res.diag["bridge_fallback"]  # non-empty: records why
    assert res.diag["n_ccs_visited"] == 1
    # The fallback must trace the LARGEST component (the long tube, ~48 mm),
    # not the start seed's component (the short tube, ~14 mm).
    assert res.length_mm > 40.0


def test_gate_without_fallback_raises():
    img = parallel_tubes()
    cfg = CenterlineConfig(
        max_bridge_wall_mm=1.0, fall_back_to_largest_component=False, **SP_GATE_CFG
    )
    with pytest.raises(RuntimeError, match="No bridge chain"):
        extract_centerline(img, (6, 11, 11), (50, 33, 33), cfg)


def test_gate_still_allows_a_thin_barrier():
    """A thin plug must remain bridgeable under the thickness gate."""
    ct = straight_tube(length=60, radius=4)
    arr = sitk.GetArrayFromImage(ct)
    arr[28:31] = TISSUE_HU
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    c = arr.shape[1] // 2
    cfg = CenterlineConfig(max_bridge_wall_mm=12.0, **SP_GATE_CFG)
    res = extract_centerline(img, (4, c, c), (55, c, c), cfg)
    assert res.diag["n_bridges"] == 1
    # The barrier is thin in absolute terms even though, measured over the gap
    # alone, essentially all of it is tissue.
    assert res.diag["max_bridge_wall_mm"] <= 12.0


def test_batch_preset_keeps_the_gate_open():
    """The regression baseline must not be altered by the new gate."""
    cfg = CenterlineConfig.batch_2026_08()
    assert cfg.max_bridge_wall_fraction == 1.0
    assert cfg.fall_back_to_largest_component is False


def test_wall_thickness_gates_by_barrier_not_by_ratio():
    """A thick barrier is refused however long the bridge that spans it.

    This is the property the fraction could not provide. Measured over the whole
    segment, a long bridge dilutes a thick barrier and passes; measured over the
    gap alone, every bridge crossing any tissue fails. Thickness distinguishes
    the collapsed segment we want to bridge from the tunnel we do not.
    """
    results = {}
    for plug in (2, 5, 20, 40):
        ct = straight_tube(length=90, radius=4)
        arr = sitk.GetArrayFromImage(ct)
        arr[40 : 40 + plug] = TISSUE_HU
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        cfg = CenterlineConfig(**MC_CFG)
        res = extract_centerline(img, (4, 8, 8), (85, 8, 8), cfg)
        results[plug] = res.diag["n_bridges"]

    assert results[2] == 1, "a 2 mm collapsed segment must be bridgeable"
    assert results[5] == 1, "a 5 mm collapsed segment must be bridgeable"
    assert results[20] == 0, "a 20 mm barrier must not be bridged"
    assert results[40] == 0, "a 40 mm barrier must not be bridged"


def test_wall_fraction_is_measured_over_the_gap_only():
    """Lumen the segment runs along must not count toward the wall fraction."""
    import numpy as np_

    from ctc_core.centerline import _wall_hit_fraction

    ct_arr = np_.full((40, 10, 10), TISSUE_HU, dtype=np_.float32)
    lumen = np_.zeros((40, 10, 10), dtype=bool)
    # Lumen along z except a 5 mm tissue plug in the middle.
    lumen[:18] = True
    lumen[23:] = True
    ct_arr[lumen] = AIR_HU

    p0 = np_.array([1.0, 5.0, 5.0])
    p1 = np_.array([38.0, 5.0, 5.0])
    frac, mm = _wall_hit_fraction(ct_arr, p0, p1, (1.0, 1.0, 1.0), -300.0, lumen=lumen)
    # Only the plug is considered, and it is all tissue.
    assert frac == pytest.approx(1.0)
    assert 3.0 <= mm <= 8.0

    # Without the lumen argument the same barrier is diluted by the lumen run.
    frac_all, _ = _wall_hit_fraction(ct_arr, p0, p1, (1.0, 1.0, 1.0), -300.0)
    assert frac_all < 0.25


def test_coverage_traversal_needs_no_mandatory_connection():
    """Where shortest path must fail, max_coverage simply covers the best cluster."""
    img = parallel_tubes()
    cfg = CenterlineConfig(**{**MC_CFG, "max_bridge_wall_mm": 1.0})
    res = extract_centerline(img, (6, 11, 11), (50, 33, 33), cfg)

    assert res.diag["n_bridges"] == 0
    assert not res.diag["bridge_fallback"]  # nothing failed; nothing to fall back from
    assert res.diag["traversal_n_clusters"] == 2
    assert res.diag["n_ccs_visited"] == 1
    assert res.length_mm > 20.0


def test_coverage_traversal_covers_the_whole_chain():
    """Three collinear segments: the traversal must span all three, not two."""
    ct = straight_tube(length=120, radius=4)
    arr = sitk.GetArrayFromImage(ct)
    arr[40:43] = TISSUE_HU
    arr[80:83] = TISSUE_HU
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    cfg = CenterlineConfig(**MC_CFG)
    res = extract_centerline(img, (4, 8, 8), (115, 8, 8), cfg)

    assert res.diag["n_ccs_visited"] == 3
    assert res.diag["n_bridges"] == 2
    assert res.diag["traversal_path_volume_frac_of_cluster"] == pytest.approx(1.0)
    assert res.length_mm > 90.0


# ---------------------------------------------------------------------------
# Whole-colon seed selection (docs/EVALUATION_HQCOLON.md section 8)
# ---------------------------------------------------------------------------


def _hook_colon():
    """An inverted-U colon with a short bump at the top, 1 mm voxels.

    Ascending from the "rectum" (lowest point, x = 20) to z = 100, across to
    x = 100, and down to the "caecum" at z = 20. The bump rises from the middle
    of the transverse limb. By Euclidean distance the most distant end points
    are the rectum and the bump, so a path between them skips the descending
    limb; along the lumen the caecum is far the more distant.
    """
    shape = (125, 40, 125)  # z, y, x
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    r = 5.0

    def seg(p, q):
        p, q = np.array(p, float), np.array(q, float)
        v = q - p
        w = np.stack([zz - p[0], yy - p[1], xx - p[2]], axis=-1)
        t = np.clip((w @ v) / (v @ v), 0, 1)
        d = w - t[..., None] * v
        return (d ** 2).sum(-1) <= r * r

    y = 20
    m = (seg((5, y, 20), (100, y, 20)) | seg((100, y, 20), (100, y, 100))
         | seg((100, y, 100), (20, y, 100)) | seg((100, y, 60), (118, y, 60)))
    ref = sitk.GetImageFromArray(m.astype(np.uint8))
    ref.SetSpacing((1.0, 1.0, 1.0))
    return m, ref


def test_whole_colon_selection_reaches_the_far_end_the_euclidean_rule_misses():
    from ctc_core.centerline import CenterlineConfig, select_whole_colon_path

    m, ref = _hook_colon()
    provisional = ((1, 20, 20), (122, 20, 60))  # rectum and bump tip, as the old rule picks
    path, diag = select_whole_colon_path(m, ref, provisional, CenterlineConfig())

    assert diag["seed_rule_chosen"] in ("rectal", "rectal_core")
    ends = [np.array(path[0]), np.array(path[-1])]
    end = min(ends, key=lambda e: abs(e[2] - 100))
    assert abs(end[2] - 100) <= 6 and end[0] <= 30  # at the caecum end
    assert diag[f"cand_{diag['seed_rule_chosen']}_lumen_coverage"] > \
        diag["cand_provisional_lumen_coverage"]
    assert diag[f"cand_{diag['seed_rule_chosen']}_length_mm"] > \
        diag["cand_provisional_length_mm"] + 50


def test_whole_colon_selection_keeps_a_path_that_already_spans_the_tube():
    """With no bump the provisional seeds are the true ends and must not lose."""
    from ctc_core.centerline import CenterlineConfig, select_whole_colon_path

    m, ref = _hook_colon()
    m[106:, :, :] = False  # remove the bump
    provisional = ((5, 20, 20), (20, 20, 100))
    path, diag = select_whole_colon_path(m, ref, provisional, CenterlineConfig())
    best = max(v for k, v in diag.items() if k.endswith("_lumen_coverage"))
    assert diag[f"cand_{diag['seed_rule_chosen']}_lumen_coverage"] == best
    assert diag["cand_provisional_lumen_coverage"] >= best - 0.01
