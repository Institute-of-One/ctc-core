"""Automatic colon centerline: seed detection, Fast Marching geodesic, CC bridging.

Ported from ``auto_centerline.py`` / ``centerline_from_seeds.py`` /
``run_cohort_batch.py::auto_detect_seed_endpoints`` in the in-house prototype (not distributed).

Three defects found in the 2026-08 batch (``docs/FAILURE_ANALYSIS.md``) are
fixed here, and each is a field of :class:`CenterlineConfig` so that the old
behaviour stays reproducible:

1. ``seed_span_components`` (section 2.1). The batch drew both seeds from the
   skeleton of the *largest single* air component, so ``start_cc == end_cc``
   held identically and the CC-bridge graph was unreachable dead code in all 55
   successful series. Seeding across the retained component set lets the bridge
   planner run.
2. ``connectivity`` (section 2.3). The batch labelled 26-connected while
   ``sitk.FastMarchingImageFilter`` propagates on the 6-connected stencil, so
   components joined only by a corner contact were declared traversable and the
   solver then could not reach the end seed. Confirmed as the cause of all five
   ``End seed is unreachable`` failures.
3. ``fill_holes`` / ``air_closing_radius`` (section 2.2), delegated to
   :mod:`ctc_core.masks`.

:meth:`CenterlineConfig.batch_2026_08` reproduces the original batch exactly and
is used as the regression baseline; :meth:`CenterlineConfig.corrected` is the
pipeline going forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk

from ctc_core.masks import (
    AIR_THRESHOLD_HU,
    BODY_THRESHOLD_HU,
    FillHoles,
    build_body_air_masks,
    label_air_components,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CenterlineConfig",
    "CenterlineResult",
    "detect_seed_endpoints",
    "extract_centerline",
    "compute_path_length_mm",
    "path_to_physical_mm",
    "geodesic_diameter_seeds",
    "plan_coverage_traversal",
]

OFFSETS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]

# Fast Marching leaves unreached voxels at a very large value rather than inf.
UNREACHED = 1e30


@dataclass(frozen=True)
class CenterlineConfig:
    """Every knob of the centerline stage. Values are registered in docs/PARAMETERS.md."""

    # -- masks --------------------------------------------------------------
    body_threshold: float = BODY_THRESHOLD_HU
    air_threshold: float = AIR_THRESHOLD_HU
    body_closing_radius: int = 2
    air_closing_radius: int = 1
    fill_holes: FillHoles = "auto"
    connectivity: int = 6
    dust_threshold_voxels: int = 500

    # -- seeds / traversal --------------------------------------------------
    seed_span_components: bool = True
    min_air_cc_voxels: int = 10_000
    # "shortest_path" connects the two automatically chosen seeds by the cheapest
    # admissible chain -- the in-house prototype's formulation. "max_coverage"
    # instead searches the admissible component graph for the traversal that
    # covers the most lumen and derives its own seeds from the result.
    #
    # "max_coverage" is kept, tested and available but not the default; it
    # made every measured quantity worse (docs/EVALUATION_HQCOLON.md section 7).
    # The reasoning recorded there -- that the reference centerline spans the
    # whole colon -- was itself refuted later (section 8): both centerlines
    # were truncated by the same seed rule, which ``seed_rule`` below fixes.
    traversal: Literal["max_coverage", "shortest_path"] = "shortest_path"
    # Components below this share of the largest one are ignored when choosing
    # the traversal, so a scrap of small bowel cannot outvote colon.
    traversal_min_volume_frac: float = 0.02

    # -- final seeds ---------------------------------------------------------
    # "skeleton_extremes" runs the path between the provisional seeds: the most
    # distant skeleton end points by *Euclidean* distance. In a colon those are
    # typically the rectum and a flexure, so the geodesic leaves much of the
    # colon as a side branch (docs/EVALUATION_HQCOLON.md section 8).
    # "whole_colon" keeps the lumen built from the provisional seeds but chooses
    # the final path among three candidates, all inside that lumen:
    #   provisional   the path between the provisional seeds;
    #   rectal        from the lowest lumen voxel (the rectal end, where CT
    #                 colonography is insufflated) to the voxel farthest from it
    #                 along the lumen (uniform speed: arrival time = length);
    #   rectal_core   the same start, with the far end sought only within the
    #                 largest component of the lumen's wide core (wall distance
    #                 >= core_radius_mm), so thin contacts between touching loops
    #                 and narrow small-bowel links cannot redirect the search;
    # and keeps the one that brings most of the lumen within selection_near_mm
    # of the path. The choice needs no reference.
    seed_rule: Literal["skeleton_extremes", "whole_colon"] = "whole_colon"
    core_radius_mm: float = 3.0
    selection_near_mm: float = 30.0

    # -- geodesic -----------------------------------------------------------
    speed_power: float = 2.0
    min_speed: float = 0.1
    max_speed: float = 1000.0

    # -- bridging -----------------------------------------------------------
    max_bridge_mm: float = 150.0
    bridge_alignment_weight_mm: float = 40.0
    bridge_wall_penalty_per_mm: float = 6.0
    bridge_tube_radius_vox: int = 2
    # Hard admissibility gate. A bridge spending more than this fraction of its
    # length inside body tissue is not a colon lumen and the edge is simply not
    # offered to the planner. The soft penalty alone cannot refuse: Dijkstra
    # must connect the two seeds, so when the only chain runs through tissue it
    # takes it. Measured on the 2026-09-10 run, the median bridge had
    # wall_frac 0.64 and 25 of 29 "complete" traces leaned on one above 0.5.
    # Set to 1.0 to restore the ungated behaviour.
    max_bridge_wall_fraction: float = 0.5
    # Hard gate on the absolute thickness of tissue a bridge crosses, outside
    # existing lumen. This replaces the fraction as the operative constraint:
    # measured over the gap only, essentially every real bridge crosses *some*
    # tissue, so a fractional gate is either vacuous (measured over the whole
    # segment, where a long bridge dilutes a thick barrier) or absolute
    # (measured over the gap, where it refuses every bridge). Thickness
    # separates the two cases the fraction cannot: a collapsed colon segment is
    # a few millimetres of apposed wall, a tunnel through a solid organ is tens.
    # Disabled by default: with the fraction measured over the whole segment
    # (see wall_frac_over_gap_only) the fraction gate is the effective one, and
    # a sweep of 6, 12 and 25 mm changed nothing measurable. See
    # docs/EVALUATION_HQCOLON.md section 7.
    max_bridge_wall_mm: float = float("inf")
    # Measure the wall fraction over the extra-luminal part of the bridge only.
    # Principled -- it stops a thick barrier being diluted by lumen the segment
    # runs along -- but in practice near-absolute, because colon fragmentation
    # is collapsed segments and every real bridge crosses some tissue. With it
    # on, only 2 of 26 series obtained any bridge at all. Off by default.
    wall_frac_over_gap_only: bool = False
    # When no admissible chain reaches the end seed, trace within the *largest*
    # retained component instead of raising -- i.e. degrade to the pre-bridging
    # behaviour, which is substantial, rather than to whichever component the
    # start seed happened to land in, which is often tiny. A fragmented colon is
    # then reported as partial coverage rather than stitched through solid
    # organs.
    fall_back_to_largest_component: bool = True
    kimimaro_teasar: dict[str, Any] = field(
        default_factory=lambda: {
            "scale": 2.0,
            "const": 50,
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
        }
    )

    @classmethod
    def batch_2026_08(cls) -> CenterlineConfig:
        """Exactly the 2026-08 batch. Regression baseline only -- do not use for new runs."""
        return cls(
            air_closing_radius=0,
            fill_holes="image26",
            connectivity=26,
            seed_span_components=False,
            max_bridge_wall_fraction=1.0,
            max_bridge_wall_mm=float("inf"),
            fall_back_to_largest_component=False,
            traversal="shortest_path",
            seed_rule="skeleton_extremes",
        )

    @classmethod
    def corrected(cls) -> CenterlineConfig:
        """The three fixes of docs/FAILURE_ANALYSIS.md applied. Same as the defaults."""
        return cls()


@dataclass
class CenterlineResult:
    path_kji: list[tuple[int, int, int]]
    lumen: sitk.Image
    length_mm: float
    diag: dict[str, Any]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _skeleton_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(degree-1 voxels, skeleton)`` of the Lee skeleton of ``mask``."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    skel = skeletonize(mask, method="lee")
    if not skel.any():
        return np.zeros((0, 3), dtype=np.int64), skel

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return np.argwhere(skel & (degree == 1)), skel



def _reseed_within_component(
    component: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Most distant pair of skeleton endpoints inside a single component.

    Used by the bridge fallback and equivalent to the pre-2026-09 seeding, but
    applied to the largest component explicitly rather than by construction.
    """
    endpoints, _ = _skeleton_endpoints(component)
    sz, sy, sx = spacing_zyx
    if len(endpoints) >= 2:
        phys = endpoints.astype(np.float64) * np.array([sz, sy, sx])
        d2 = ((phys[:, None, :] - phys[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        return (
            tuple(int(v) for v in endpoints[i]),
            tuple(int(v) for v in endpoints[j]),
        )
    coords = np.argwhere(component)
    if coords.size == 0:
        raise RuntimeError("Fallback component is empty.")
    return (
        tuple(int(v) for v in coords[coords[:, 0].argmin()]),
        tuple(int(v) for v in coords[coords[:, 0].argmax()]),
    )


def detect_seed_endpoints(
    air: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: CenterlineConfig,
) -> tuple[tuple[int, int, int], tuple[int, int, int], dict]:
    """Two seed voxels far apart on the colonic air skeleton.

    With ``cfg.seed_span_components`` the skeleton is taken over the union of all
    retained components, so the extreme endpoint pair may lie in *different*
    components and the bridge planner is reached. With it off, only the largest
    component is skeletonized -- the 2026-08 behaviour, which guarantees both
    seeds share a component and makes bridging unreachable.
    """
    labels, sizes = label_air_components(
        air, connectivity=cfg.connectivity, dust_threshold_voxels=cfg.dust_threshold_voxels
    )
    n_largest = int(sizes.max())
    diag: dict[str, Any] = {
        "air_cc_count": int((sizes > 0).sum()),
        "largest_air_cc_voxels": n_largest,
        "n_ccs_retained": int(np.unique(labels[labels > 0]).size),
        "seed_span_components": cfg.seed_span_components,
    }
    if n_largest < cfg.min_air_cc_voxels:
        raise RuntimeError(
            f"Largest air CC is only {n_largest} voxels "
            f"(< {cfg.min_air_cc_voxels}); probably no insufflation in this study."
        )

    search = labels > 0 if cfg.seed_span_components else labels == int(np.argmax(sizes))
    endpoints, skel = _skeleton_endpoints(search)
    diag["skeleton_voxels"] = int(skel.sum())

    sz, sy, sx = spacing_zyx
    if len(endpoints) >= 2:
        phys = endpoints.astype(np.float64) * np.array([sz, sy, sx])
        d2 = ((phys[:, None, :] - phys[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        start = tuple(int(v) for v in endpoints[i])
        end = tuple(int(v) for v in endpoints[j])
        diag["seed_source"] = "skeleton_endpoints"
        diag["seed_distance_mm"] = float(np.sqrt(d2[i, j]))
    else:
        coords = np.argwhere(search)
        if coords.size == 0:
            raise RuntimeError("No retained air component to seed from.")
        start = tuple(int(v) for v in coords[coords[:, 0].argmin()])
        end = tuple(int(v) for v in coords[coords[:, 0].argmax()])
        diag["seed_source"] = "cc_z_extremes"
        diag["seed_distance_mm"] = float(abs(end[0] - start[0]) * sz)

    diag["start_cc"] = int(labels[start])
    diag["end_cc"] = int(labels[end])
    diag["seeds_cross_components"] = diag["start_cc"] != diag["end_cc"]
    return start, end, diag


# ---------------------------------------------------------------------------
# Geodesic
# ---------------------------------------------------------------------------


def _nearest_foreground(seed_kji: tuple[int, int, int], mask: np.ndarray) -> tuple[int, int, int]:
    if all(0 <= s < n for s, n in zip(seed_kji, mask.shape, strict=True)) and mask[seed_kji]:
        return seed_kji
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty while snapping a seed.")
    d2 = ((coords - np.asarray(seed_kji, dtype=np.float64)) ** 2).sum(axis=1)
    best = coords[int(np.argmin(d2))]
    return int(best[0]), int(best[1]), int(best[2])


def compute_arrival_time(
    lumen: sitk.Image,
    start_kji: tuple[int, int, int],
    cfg: CenterlineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast Marching arrival times with speed ``(d_wall)^k`` clipped into the lumen.

    Note that the solver's stencil is 6-connected. ``cfg.connectivity`` must
    match it or components will be declared traversable that are not; see
    docs/FAILURE_ANALYSIS.md section 2.3.
    """
    distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(lumen > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    lumen_arr = sitk.GetArrayFromImage(lumen) > 0
    dist_arr = sitk.GetArrayFromImage(distance).astype(np.float32)

    speed_arr = np.zeros_like(dist_arr, dtype=np.float32)
    speed_arr[lumen_arr] = np.clip(
        np.maximum(dist_arr[lumen_arr], 0.0) ** float(cfg.speed_power),
        cfg.min_speed,
        cfg.max_speed,
    )
    speed_img = sitk.GetImageFromArray(speed_arr)
    speed_img.CopyInformation(lumen)

    fm = sitk.FastMarchingImageFilter()
    fm.SetTrialPoints([(int(start_kji[2]), int(start_kji[1]), int(start_kji[0]))])  # x, y, z
    fm.SetStoppingValue(1e8)
    arrival = fm.Execute(speed_img)

    return sitk.GetArrayFromImage(arrival).astype(np.float64), lumen_arr


def backtrack_centerline(
    arrival_arr: np.ndarray,
    lumen_arr: np.ndarray,
    start_kji: tuple[int, int, int],
    end_kji: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Steepest descent on the arrival map from ``end`` back to ``start``."""
    shape = arrival_arr.shape
    for name, seed in (("Start", start_kji), ("End", end_kji)):
        if not all(0 <= s < n for s, n in zip(seed, shape, strict=True)):
            raise ValueError(f"{name} seed index out of bounds for the arrival volume.")

    if not np.isfinite(arrival_arr[end_kji]) or arrival_arr[end_kji] > UNREACHED:
        raise ValueError("End seed is unreachable in Fast Marching arrival map.")

    current = np.array(end_kji, dtype=int)
    start = np.array(start_kji, dtype=int)
    path = [tuple(current.tolist())]
    visited = {tuple(current.tolist())}
    max_steps = max(10_000, int(lumen_arr.sum()) * 4)

    for _ in range(max_steps):
        if np.array_equal(current, start):
            break

        curr_t = float(arrival_arr[tuple(current)])
        best, best_t, fallback = None, curr_t, []
        for dz, dy, dx in OFFSETS_26:
            cand = current + np.array([dz, dy, dx], dtype=int)
            ck, cj, ci = int(cand[0]), int(cand[1]), int(cand[2])
            if not (0 <= ck < shape[0] and 0 <= cj < shape[1] and 0 <= ci < shape[2]):
                continue
            if not lumen_arr[ck, cj, ci]:
                continue
            tval = float(arrival_arr[ck, cj, ci])
            if not np.isfinite(tval) or tval > UNREACHED:
                continue
            if tval < best_t - 1e-6:
                best_t = tval
                best = np.array([ck, cj, ci], dtype=int)
            key = (ck, cj, ci)
            if key not in visited and tval <= curr_t + 1e-6:
                fallback.append((tval, float(np.linalg.norm(cand - start)), key))

        if best is None:
            if not fallback:
                raise ValueError("Backtracking got stuck before reaching start.")
            fallback.sort(key=lambda x: (x[0], x[1]))
            best = np.array(fallback[0][2], dtype=int)

        current = best
        key = tuple(current.tolist())
        if key in visited:
            raise ValueError("Backtracking loop detected.")
        visited.add(key)
        path.append(key)

    if not np.array_equal(current, start):
        raise ValueError("Backtracking exceeded max steps without reaching start.")

    path.reverse()
    return path


# ---------------------------------------------------------------------------
# CC bridging
# ---------------------------------------------------------------------------


def _wall_hit_fraction(
    ct_arr: np.ndarray,
    p0_mm: np.ndarray,
    p1_mm: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    body_threshold: float,
    lumen: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fraction of the bridge's *extra-luminal* part that lies in body tissue.

    Samples the segment at ~1 mm and returns the share of tissue samples among
    those that are not already inside ``lumen``. Restricting the denominator to
    the extra-luminal part is what makes the figure mean "is the gap I am about
    to cross a lumen path or solid tissue".

    Measuring over the whole segment instead -- as this did until 2026-09-10 --
    dilutes a thick barrier with any lumen the segment happens to run along, so
    a long bridge scores better than a short honest one. On a phantom with a
    5 mm tissue plug the direct tip-to-tip bridge scored 0.83 and was refused,
    while a bridge from the same tip to the *far* tip of the other tube crossed
    the identical plug at 0.17 and was accepted -- and its tube then
    short-circuited that tube lengthwise, cutting coverage. With ``lumen``
    supplied both score ~1.0 and both are refused, which is the right answer for
    a solid plug.

    Returns ``(fraction, thickness_mm)``. The fraction is kept for reporting
    and for the cost term; the *thickness* is what the hard gate uses, because
    what separates a bridgeable gap from a tunnel is how much tissue is crossed,
    not what share of the segment it occupies. A collapsed colon segment is a
    few millimetres of apposed wall; crossing the liver is tens.

    0.0 means the gap is clear; 1.0 means it is entirely tissue.
    """
    sz, sy, sx = spacing_zyx
    diff = p1_mm - p0_mm
    n = max(2, int(float(np.linalg.norm(diff))) + 1)
    nz, ny, nx = ct_arr.shape
    hits = 0
    considered = 0
    for s_i in range(n):
        p = p0_mm + diff * (s_i / float(n - 1))
        kz, jy, ix = int(round(p[0] / sz)), int(round(p[1] / sy)), int(round(p[2] / sx))
        if not (0 <= kz < nz and 0 <= jy < ny and 0 <= ix < nx):
            continue
        if lumen is not None and lumen[kz, jy, ix]:
            continue  # already lumen; says nothing about the gap
        considered += 1
        if ct_arr[kz, jy, ix] > body_threshold:
            hits += 1
    step_mm = float(np.linalg.norm(diff)) / float(n - 1) if n > 1 else 0.0
    thickness_mm = hits * step_mm
    if considered == 0:
        # The entire segment lies inside lumen, so there is no gap to cross.
        return 0.0, 0.0
    return hits / float(considered), thickness_mm


def _draw_bridge_tube(
    mask: np.ndarray,
    p0_kji: tuple[int, int, int],
    p1_kji: tuple[int, int, int],
    tube_radius_vox: int = 2,
) -> None:
    """In place, add a thick voxel line from ``p0_kji`` to ``p1_kji``."""
    p0 = np.asarray(p0_kji, dtype=np.float64)
    p1 = np.asarray(p1_kji, dtype=np.float64)
    diff = p1 - p0
    n_steps = int(max(np.max(np.abs(diff)), 1)) * 2 + 1
    nz, ny, nx = mask.shape
    r = int(tube_radius_vox)
    r2 = r * r
    for s in range(n_steps + 1):
        p = p0 + diff * (s / float(n_steps))
        kc, jc, ic = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        for dk in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if dk * dk + dj * dj + di * di > r2:
                        continue
                    k, j, i = kc + dk, jc + dj, ic + di
                    if 0 <= k < nz and 0 <= j < ny and 0 <= i < nx:
                        mask[k, j, i] = True


def _cc_endpoints(
    labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: CenterlineConfig,
) -> dict[int, list[dict]]:
    """Skeleton endpoints and outgoing tangents per component, via kimimaro."""
    import kimimaro

    sz, sy, sx = spacing_zyx
    skels = kimimaro.skeletonize(
        labels,
        teasar_params=dict(cfg.kimimaro_teasar),
        anisotropy=spacing_zyx,
        dust_threshold=cfg.dust_threshold_voxels,
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
        progress=False,
        parallel=1,
    )
    if not skels:
        raise RuntimeError("kimimaro produced no skeletons.")

    out: dict[int, list[dict]] = {}
    for cc_id, skel in skels.items():
        verts_mm = np.asarray(skel.vertices, dtype=np.float64)  # (z_mm, y_mm, x_mm)
        edges = np.asarray(skel.edges, dtype=np.int64)
        n = verts_mm.shape[0]
        if n == 0:
            continue

        deg = np.zeros(n, dtype=np.int64)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for a, b in edges:
            ai, bi = int(a), int(b)
            deg[ai] += 1
            deg[bi] += 1
            adj[ai].append(bi)
            adj[bi].append(ai)

        endpoints = np.where(deg == 1)[0].tolist()
        if not endpoints:
            endpoints = [0] if n == 1 else [0, n - 1]

        eps: list[dict] = []
        for ep in endpoints:
            # Tangent: walk back up to 4 hops and take the unit displacement.
            cur, prev = ep, -1
            for _ in range(4):
                nbrs = [x for x in adj[cur] if x != prev]
                if not nbrs:
                    break
                prev, cur = cur, nbrs[0]
            tv = verts_mm[ep] - verts_mm[cur] if cur != ep else np.zeros(3)
            tnorm = float(np.linalg.norm(tv))
            mm = verts_mm[ep]
            eps.append(
                {
                    "mm": mm,
                    "kji": (
                        int(np.clip(round(mm[0] / sz), 0, labels.shape[0] - 1)),
                        int(np.clip(round(mm[1] / sy), 0, labels.shape[1] - 1)),
                        int(np.clip(round(mm[2] / sx), 0, labels.shape[2] - 1)),
                    ),
                    "tangent": tv / tnorm if tnorm > 1e-9 else np.zeros(3),
                }
            )
        out[int(cc_id)] = eps
    return out


def _build_bridge_graph(
    labels: np.ndarray,
    ct_arr: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: CenterlineConfig,
) -> tuple[Any, dict[tuple[int, int], dict], dict[int, list[dict]]]:
    """Admissible bridge graph over the retained air components.

    An edge exists between two components when some pair of their skeleton
    endpoints is within ``max_bridge_mm`` and spends no more than
    ``max_bridge_wall_fraction`` of the gap inside tissue. Edge weight is
    ``gap + alpha (1 - alignment) + beta * wall_fraction * gap``.

    Returns ``(graph, best_bridge_by_pair, endpoints_by_component)``.
    """
    import networkx as nx

    cc_endpoints = _cc_endpoints(labels, spacing_zyx, cfg)
    lumen = (labels > 0) if cfg.wall_frac_over_gap_only else None

    graph = nx.Graph()
    graph.add_nodes_from(int(c) for c in cc_endpoints)

    best_bridge: dict[tuple[int, int], dict] = {}
    cc_ids = list(cc_endpoints)
    for ai in range(len(cc_ids)):
        for bi in range(ai + 1, len(cc_ids)):
            a_cc, b_cc = cc_ids[ai], cc_ids[bi]
            best = None
            for ea in cc_endpoints[a_cc]:
                for eb in cc_endpoints[b_cc]:
                    pa, pb = ea["mm"], eb["mm"]
                    gap = pb - pa
                    d = float(np.linalg.norm(gap))
                    if d > cfg.max_bridge_mm:
                        continue
                    if d < 1e-6:
                        cost, align, wall_frac, wall_mm = 0.0, 1.0, 0.0, 0.0
                    else:
                        gap_dir = gap / d
                        align = 0.5 * (
                            max(0.0, float(np.dot(ea["tangent"], gap_dir)))
                            + max(0.0, float(np.dot(eb["tangent"], -gap_dir)))
                        )
                        wall_frac, wall_mm = _wall_hit_fraction(
                            ct_arr, pa, pb, spacing_zyx, cfg.body_threshold,
                            lumen=lumen,
                        )
                        if wall_mm > cfg.max_bridge_wall_mm:
                            continue  # a tunnel, not a gap: refuse outright
                        if wall_frac > cfg.max_bridge_wall_fraction:
                            continue
                        cost = (
                            d
                            + cfg.bridge_alignment_weight_mm * (1.0 - align)
                            + cfg.bridge_wall_penalty_per_mm * wall_mm
                        )
                    if best is None or cost < best["cost"]:
                        best = {
                            "cost": cost,
                            "p0_kji": ea["kji"],
                            "p1_kji": eb["kji"],
                            "from_cc": int(a_cc),
                            "to_cc": int(b_cc),
                            "length_mm": d,
                            "alignment": align,
                            "wall_frac": wall_frac,
                            "wall_mm": wall_mm,
                        }
            if best is not None:
                graph.add_edge(int(a_cc), int(b_cc), weight=best["cost"])
                best_bridge[(int(a_cc), int(b_cc))] = best
                best_bridge[(int(b_cc), int(a_cc))] = best

    return graph, best_bridge, cc_endpoints


def plan_cc_chain_bridges(
    labels: np.ndarray,
    ct_arr: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    start_cc: int,
    end_cc: int,
    cfg: CenterlineConfig,
) -> tuple[list[int], list[dict]]:
    """Cheapest admissible chain of components from ``start_cc`` to ``end_cc``.

    The in-house prototype's formulation, kept for ``traversal="shortest_path"``
    and for the batch reproduction. See :func:`plan_coverage_traversal` for why
    it is the wrong objective.
    """
    import networkx as nx

    graph, best_bridge, _ = _build_bridge_graph(labels, ct_arr, spacing_zyx, cfg)

    if start_cc not in graph.nodes or end_cc not in graph.nodes:
        raise RuntimeError(
            f"Start CC ({start_cc}) or End CC ({end_cc}) has no skeleton "
            "(too small for kimimaro)."
        )
    try:
        cc_path = nx.shortest_path(graph, source=start_cc, target=end_cc, weight="weight")
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            f"No bridge chain from CC {start_cc} to CC {end_cc} within "
            f"max_bridge_mm={cfg.max_bridge_mm}."
        ) from exc

    bridges = [
        best_bridge[(int(u), int(v))]
        for u, v in zip(cc_path[:-1], cc_path[1:], strict=True)
    ]
    return [int(c) for c in cc_path], bridges


def plan_coverage_traversal(
    labels: np.ndarray,
    sizes: np.ndarray,
    ct_arr: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: CenterlineConfig,
) -> tuple[list[int], list[dict], dict]:
    """Traversal that covers the most lumen, rather than the cheapest one.

    Shortest path answers "how do I get from this seed to that seed most
    cheaply", which is not the question. The colon is a tube and the goal is to
    run down as much of it as possible, so the objective here is the total lumen
    volume of the components traversed.

    Maximum-weight path is NP-hard on a general graph, but the admissible
    component graph of a colon is nearly a chain. The algorithm therefore:

    1. keeps components at least ``traversal_min_volume_frac`` of the largest,
       so a scrap of small bowel cannot outvote colon;
    2. takes the connected sub-graph of greatest total volume -- the colon
       cluster -- since separate clusters cannot be joined admissibly anyway;
    3. reduces that cluster to its minimum spanning tree by bridge cost, which
       keeps the cheapest way of realising each connection and makes the path
       between any two nodes unique;
    4. returns the node-weighted diameter of that tree, i.e. the pair of nodes
       whose connecting path carries the greatest total lumen volume.

    Returns ``(cc_path, bridges, diag)``.
    """
    import networkx as nx

    graph, best_bridge, _ = _build_bridge_graph(labels, ct_arr, spacing_zyx, cfg)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("No air component has a skeleton for traversal planning.")

    volume = {int(n): float(sizes[int(n)]) if int(n) < sizes.size else 0.0
              for n in graph.nodes}
    largest = max(volume.values()) if volume else 0.0
    keep = {n for n, v in volume.items()
            if v >= cfg.traversal_min_volume_frac * largest}
    if not keep:
        raise RuntimeError("Every component fell below the traversal volume floor.")
    graph = graph.subgraph(keep).copy()

    clusters = list(nx.connected_components(graph))
    cluster = max(clusters, key=lambda c: sum(volume[int(n)] for n in c))
    sub = graph.subgraph(cluster).copy()

    diag = {
        "traversal": "max_coverage",
        "n_components_considered": int(len(keep)),
        "n_clusters": int(len(clusters)),
        "cluster_components": int(len(cluster)),
        "cluster_volume_voxels": int(sum(volume[int(n)] for n in cluster)),
    }

    if sub.number_of_nodes() == 1:
        only = int(next(iter(cluster)))
        diag["path_volume_voxels"] = int(volume[only])
        return [only], [], diag

    tree = nx.minimum_spanning_tree(sub, weight="weight")
    nodes = list(tree.nodes)
    paths = dict(nx.all_pairs_shortest_path(tree))

    best_path, best_weight = None, -1.0
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            path = paths[a][b]
            w = sum(volume[int(n)] for n in path)
            if w > best_weight:
                best_weight, best_path = w, path
    assert best_path is not None

    cc_path = [int(n) for n in best_path]
    bridges = [
        best_bridge[(int(u), int(v))]
        for u, v in zip(cc_path[:-1], cc_path[1:], strict=True)
    ]
    diag["path_volume_voxels"] = int(best_weight)
    diag["path_volume_frac_of_cluster"] = round(
        best_weight / max(diag["cluster_volume_voxels"], 1), 4
    )
    return cc_path, bridges, diag


def _farthest_reachable(
    lumen: sitk.Image,
    seed: tuple[int, int, int],
    cfg: CenterlineConfig,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray]:
    """Voxel of greatest Fast Marching arrival time from ``seed``."""
    arrival, mask = compute_arrival_time(lumen, seed, cfg)
    reachable = mask & np.isfinite(arrival) & (arrival <= UNREACHED)
    if not reachable.any():
        raise RuntimeError("No reachable voxel from the seed.")
    masked = np.where(reachable, arrival, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return tuple(int(v) for v in idx), arrival, mask


def geodesic_diameter_seeds(
    lumen: sitk.Image,
    cfg: CenterlineConfig,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Endpoints of the longest path through ``lumen``, by a double sweep.

    Sweep once from an arbitrary lumen voxel to find the farthest reachable one,
    then again from there: the standard two-pass diameter construction. This
    replaces deriving terminal seeds from where the bridges happen to attach,
    which was fragile -- when the cheapest bridge entered a component at its far
    tip, the derived seed landed mid-component and the traversal clipped most of
    it. Asking directly for the longest traversable path needs no such
    assumption, and it is the same question the traversal is trying to answer.
    """
    arr = sitk.GetArrayFromImage(lumen) > 0
    coords = np.argwhere(arr)
    if coords.size == 0:
        raise RuntimeError("Lumen mask is empty.")
    anywhere = tuple(int(v) for v in coords[0])
    first, _, _ = _farthest_reachable(lumen, anywhere, cfg)
    second, _, _ = _farthest_reachable(lumen, first, cfg)
    return first, second


# ---------------------------------------------------------------------------
# Whole-colon seed selection
# ---------------------------------------------------------------------------


def _image_like(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    return img


def _largest_component(mask: np.ndarray, connectivity: int = 6) -> np.ndarray:
    import cc3d

    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def uniform_arrival(
    mask: np.ndarray, ref: sitk.Image, seed_kji: tuple[int, int, int]
) -> np.ndarray:
    """Geodesic distance (mm) from ``seed_kji`` inside ``mask``; -1 where unreached.

    Unit speed makes the arrival time a path length, unlike the wall-weighted
    speed of the centerline, whose arrival time is dominated by narrow passages
    and is therefore no measure of how far along the colon a voxel lies.
    """
    speed = sitk.GetImageFromArray(mask.astype(np.float32))
    speed.CopyInformation(ref)
    fm = sitk.FastMarchingImageFilter()
    fm.SetTrialPoints([(int(seed_kji[2]), int(seed_kji[1]), int(seed_kji[0]))])
    fm.SetStoppingValue(1e8)
    t = sitk.GetArrayFromImage(fm.Execute(speed)).astype(np.float64)
    t[~mask] = -1.0
    t[t > 1e7] = -1.0
    return t


def _farthest_along(mask: np.ndarray, ref: sitk.Image, seed: tuple[int, int, int]):
    t = uniform_arrival(mask, ref, seed)
    idx = np.unravel_index(int(np.argmax(t)), t.shape)
    return tuple(int(v) for v in idx), float(t[idx])


def _lowest_voxel(mask: np.ndarray) -> tuple[int, int, int]:
    """A voxel of the most inferior slice of ``mask`` (LPS: lowest k is inferior)."""
    pts = np.argwhere(mask)
    bottom = pts[pts[:, 0] == pts[:, 0].min()]
    return tuple(int(v) for v in bottom[len(bottom) // 2])


def lumen_coverage(path_kji: np.ndarray, mask: np.ndarray, ref: sitk.Image,
                   near_mm: float) -> float:
    """Share of ``mask`` voxels within ``near_mm`` of the path."""
    m = np.zeros(mask.shape, dtype=np.uint8)
    p = np.asarray(path_kji, dtype=int)
    m[p[:, 0], p[:, 1], p[:, 2]] = 1
    d = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        _image_like(m, ref), insideIsPositive=False, squaredDistance=False,
        useImageSpacing=True))
    return float(np.mean(d[mask] <= near_mm)) if mask.any() else 0.0


def select_whole_colon_path(
    lumen_arr: np.ndarray,
    ref: sitk.Image,
    provisional: tuple[tuple[int, int, int], tuple[int, int, int]],
    cfg: CenterlineConfig,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """The candidate path (see ``CenterlineConfig.seed_rule``) covering most lumen."""
    lum = _largest_component(lumen_arr, connectivity=6)
    lum_img = _image_like(lum, ref)
    cands: dict[str, list[tuple[int, int, int]]] = {}

    a = _nearest_foreground(provisional[0], lum)
    b = _nearest_foreground(provisional[1], lum)
    arr_a, mask_a = compute_arrival_time(lum_img, a, cfg)
    cands["provisional"] = backtrack_centerline(arr_a, mask_a, a, b)

    r0 = _lowest_voxel(lum)
    arr_r, mask_r = compute_arrival_time(lum_img, r0, cfg)  # shared by both rectal candidates
    far, _ = _farthest_along(lum, ref, r0)
    cands["rectal"] = backtrack_centerline(arr_r, mask_r, r0, far)

    dwall = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        lum_img, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))
    core = _largest_component(lum & (dwall >= cfg.core_radius_mm), connectivity=6)
    if core.any():
        pts = np.argwhere(core)
        near = tuple(int(v) for v in pts[np.argmin(((pts - np.array(r0)) ** 2).sum(axis=1))])
        far_c, _ = _farthest_along(core, ref, near)
        cands["rectal_core"] = backtrack_centerline(arr_r, mask_r, r0, far_c)

    spacing_xyz = ref.GetSpacing()
    scores = {
        name: {
            "lumen_coverage": round(lumen_coverage(np.array(p), lum, ref,
                                                   cfg.selection_near_mm), 4),
            "length_mm": round(compute_path_length_mm(p, spacing_xyz), 1),
        }
        for name, p in cands.items()
    }
    chosen = max(scores, key=lambda k: scores[k]["lumen_coverage"])  # first wins ties
    diag = {"seed_rule_chosen": chosen,
            **{f"cand_{k}_{m}": v for k, sc in scores.items() for m, v in sc.items()}}
    return cands[chosen], diag


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def extract_centerline(
    ct: sitk.Image,
    start_kji: tuple[int, int, int],
    end_kji: tuple[int, int, int],
    cfg: CenterlineConfig | None = None,
) -> CenterlineResult:
    """Centerline from ``start_kji`` to ``end_kji``, bridging components as needed."""
    cfg = cfg or CenterlineConfig()
    spacing_xyz = ct.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    ct_arr = sitk.GetArrayFromImage(ct)

    _body, air, mask_diag = build_body_air_masks(
        ct,
        body_threshold=cfg.body_threshold,
        air_threshold=cfg.air_threshold,
        body_closing_radius=cfg.body_closing_radius,
        air_closing_radius=cfg.air_closing_radius,
        fill_holes=cfg.fill_holes,
        connectivity=cfg.connectivity,
    )

    start_kji = _nearest_foreground(start_kji, air)
    end_kji = _nearest_foreground(end_kji, air)

    labels, sizes = label_air_components(
        air, connectivity=cfg.connectivity, dust_threshold_voxels=cfg.dust_threshold_voxels
    )
    kept_air = labels > 0
    if not kept_air[start_kji]:
        start_kji = _nearest_foreground(start_kji, kept_air)
    if not kept_air[end_kji]:
        end_kji = _nearest_foreground(end_kji, kept_air)

    start_cc, end_cc = int(labels[start_kji]), int(labels[end_kji])
    n_ccs_total = int(np.unique(labels[labels > 0]).size)
    LOGGER.info("Air CCs retained: %d; start CC=%d end CC=%d", n_ccs_total, start_cc, end_cc)

    bridges: list[dict] = []
    used_kimimaro = False
    bridge_fallback = ""
    traversal_diag: dict[str, Any] = {}

    if cfg.traversal == "max_coverage":
        # The seeds handed in are only a fallback here: the traversal chooses
        # which components to cover and then derives its own terminal seeds.
        try:
            cc_path, bridges, traversal_diag = plan_coverage_traversal(
                labels, sizes, ct_arr, spacing_zyx, cfg
            )
            used_kimimaro = True
        except RuntimeError as exc:
            if not cfg.fall_back_to_largest_component:
                raise
            LOGGER.warning("coverage traversal failed (%s); using largest CC", exc)
            bridge_fallback = str(exc)
            cc_path, bridges = [int(np.argmax(sizes))], []

        augmented = np.isin(labels, cc_path)
        for br in bridges:
            _draw_bridge_tube(augmented, br["p0_kji"], br["p1_kji"], cfg.bridge_tube_radius_vox)

        # Seeds come from the mask the geodesic will actually run on, so the
        # bridges are already in place and the sweep can use them.
        _probe = sitk.GetImageFromArray(augmented.astype(np.uint8))
        _probe.CopyInformation(ct)
        start_kji, end_kji = geodesic_diameter_seeds(_probe, cfg)
    elif start_cc == end_cc:
        cc_path = [start_cc]
        augmented = labels == start_cc
    else:
        try:
            cc_path, bridges = plan_cc_chain_bridges(
                labels, ct_arr, spacing_zyx, start_cc, end_cc, cfg
            )
            used_kimimaro = True
        except RuntimeError as exc:
            if not cfg.fall_back_to_largest_component:
                raise
            # No chain of admissible bridges reaches the end seed. Report the
            # colon as fragmented rather than forcing a path through solid
            # tissue: fall back to the largest retained component and re-derive
            # its own extreme endpoint pair inside it. Falling back to the
            # *start* seed's component instead would be far worse -- with seeds
            # spanning components the start seed often sits in a small one, and
            # doing that gave a median trace of 317 mm against 1014 mm for the
            # planned cases (2026-09-10 run).
            LOGGER.warning("bridge planning failed (%s); falling back to largest CC", exc)
            bridge_fallback = str(exc)
            largest_cc = int(np.argmax(sizes))
            cc_path, bridges = [largest_cc], []
            start_kji, end_kji = _reseed_within_component(
                labels == largest_cc, spacing_zyx
            )
        augmented = np.isin(labels, cc_path)
        for br in bridges:
            _draw_bridge_tube(augmented, br["p0_kji"], br["p1_kji"], cfg.bridge_tube_radius_vox)

    lumen = sitk.GetImageFromArray(augmented.astype(np.uint8))
    lumen.CopyInformation(ct)

    start_adj = _nearest_foreground(start_kji, augmented)
    end_adj = _nearest_foreground(end_kji, augmented)

    seed_diag: dict[str, Any] = {"seed_rule": cfg.seed_rule}
    if cfg.seed_rule == "whole_colon" and cfg.traversal == "shortest_path":
        path_kji, extra = select_whole_colon_path(augmented, lumen, (start_adj, end_adj), cfg)
        seed_diag.update(extra)
    else:
        arrival_arr, lumen_arr = compute_arrival_time(lumen, start_adj, cfg)
        path_kji = backtrack_centerline(arrival_arr, lumen_arr, start_adj, end_adj)

    diag = {
        "n_ccs_total": n_ccs_total,
        "n_ccs_visited": len(cc_path),
        "cc_path": cc_path,
        "n_bridges": len(bridges),
        "bridges": [
            {
                "from_cc": b["from_cc"],
                "to_cc": b["to_cc"],
                "length_mm": float(b["length_mm"]),
                "alignment": float(b["alignment"]),
                "wall_frac": float(b["wall_frac"]),
                "wall_mm": float(b["wall_mm"]),
            }
            for b in bridges
        ],
        "used_kimimaro": used_kimimaro,
        "bridge_fallback": bridge_fallback,
        "max_bridge_wall_frac": max([b["wall_frac"] for b in bridges], default=0.0),
        "max_bridge_wall_mm": max([b["wall_mm"] for b in bridges], default=0.0),
        "largest_air_cc_voxels": int(sizes.max()),
        "n_path_voxels": len(path_kji),
        "fill_holes_used": mask_diag.get("fill_holes_used", ""),
        "mask_fallback_reason": mask_diag.get("fallback_reason", ""),
        "traversal": cfg.traversal,
        **{f"traversal_{k}": v for k, v in traversal_diag.items() if k != "traversal"},
        **seed_diag,
    }
    return CenterlineResult(
        path_kji=path_kji,
        lumen=lumen,
        length_mm=compute_path_length_mm(path_kji, spacing_xyz),
        diag=diag,
    )


def compute_path_length_mm(
    path_kji: list[tuple[int, int, int]],
    spacing_xyz: tuple[float, float, float],
) -> float:
    """Polyline length of a dense voxel path, in millimetres."""
    if len(path_kji) < 2:
        return 0.0
    sx, sy, sz = spacing_xyz
    pts = np.asarray(path_kji, dtype=np.float64) * np.array([sz, sy, sx])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def path_to_physical_mm(
    path_kji: list[tuple[int, int, int]],
    ct: sitk.Image,
) -> np.ndarray:
    """Voxel path -> (N, 3) physical points in mm, ordered ``x, y, z``."""
    return np.array(
        [ct.TransformIndexToPhysicalPoint((int(i), int(j), int(k))) for k, j, i in path_kji],
        dtype=np.float64,
    )
