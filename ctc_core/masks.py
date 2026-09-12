"""Body and intra-abdominal air masks, and air connected components.

Ported from the in-house prototype (not distributed), which contained *two* independent
implementations of the same conceptual mask: ``_build_body_mask`` in
``auto_centerline.py`` (used by the centerline stage) and ``build_masks`` in
``phase0_fat_along_centerline.py`` (used by the fat-map stage). They disagree,
and the disagreement silently discarded most of the colon on some series -- see
``docs/FAILURE_ANALYSIS.md`` section 2.2.

Both are reproduced here as explicit, named options so the difference can be
measured rather than inherited:

``fill_holes="image26"``
    ``sitk.BinaryFillhole(..., fullyConnected=True)`` over the whole image.
    This is what the centerline stage did. ``fullyConnected=True`` makes the
    *background* 26-connected, so a cavity escapes to the exterior through any
    single diagonal gap in the surrounding wall and is then not a hole at all.

``fill_holes="image6"``
    The same filter with ``fullyConnected=False`` (6-connected background). A
    diagonal gap no longer leaks, so thin-walled cavities are filled.

``fill_holes="bbox"``
    ``scipy.ndimage.binary_fill_holes`` on the mask's own bounding box padded by
    one voxel -- 6-connected background, and additionally immune to a cavity
    that reaches the image border. This is what the fat-map stage did.

``fill_holes="slicewise"``
    2-D hole filling within each axial slice. Any 3-D variant fails whenever the
    gas cavity is connected to the exterior *somewhere* in the volume, because
    it is then not a hole by definition. Measured against HQColon, the 3-D fill
    lost the colon entirely on 5 of 26 reference series -- 1.6 to 3.3 L of gas,
    with the background component holding the colon reaching a volume face
    (52 510 mL and merged with exterior air on 0026/primary; a 3 002 mL cavity
    cut by the field of view on 0030/primary). Within an axial slice the body
    wall encircles the colon, so 2-D filling seals it: it recovers 100 % of the
    reference colon on all five, and leaves the working series unchanged.

On a truncated or thin-walled abdomen these disagree by more than two orders of
magnitude in recovered colonic gas; see ``docs/FAILURE_ANALYSIS.md`` section 2.2
and ``docs/EVALUATION_HQCOLON.md`` section 2. ``"image26"`` is retained only to
reproduce the 2026-08 batch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import SimpleITK as sitk

__all__ = [
    "BODY_THRESHOLD_HU",
    "AIR_THRESHOLD_HU",
    "build_body_mask",
    "build_body_air_masks",
    "MIN_COLON_GAS_VOXELS",
    "FillHoles",
    "build_air_mask",
    "label_air_components",
    "touches_image_border",
]

# Registered in docs/PARAMETERS.md.
BODY_THRESHOLD_HU = -300.0
AIR_THRESHOLD_HU = -700.0

# Largest contiguous intra-abdominal gas component below which there is no colon
# to trace: 150 000 voxels = 150 mL at 1.0 mm isotropic. Used by ``fill_holes
# ="auto"`` to decide that a 3-D hole fill has failed.
MIN_COLON_GAS_VOXELS = 150_000

FillHoles = Literal["auto", "slicewise", "image26", "image6", "bbox", "none"]


def _fill_holes_bbox(mask: np.ndarray) -> np.ndarray:
    """Fill cavities relative to the mask's own padded bounding box.

    Reproduces ``phase0_fat_along_centerline.fill_holes_3d`` using
    ``scipy.ndimage`` rather than an explicit BFS; the result is identical and
    far faster.
    """
    from scipy import ndimage

    if not mask.any():
        return mask.copy()

    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))

    cropped = np.pad(mask[sl], 1, mode="constant", constant_values=False)
    filled = ndimage.binary_fill_holes(cropped)

    out = mask.copy()
    out[sl] = filled[1:-1, 1:-1, 1:-1]
    return out


def _fill_holes_slicewise(mask: np.ndarray) -> np.ndarray:
    """Fill holes within each axial slice independently.

    Robust to a cavity that escapes the volume in z or merges with exterior air
    somewhere along the body, both of which make a 3-D fill a no-op. See the
    module docstring.
    """
    from scipy import ndimage

    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        sl = mask[z]
        if sl.any():
            out[z] = ndimage.binary_fill_holes(sl)
    return out


def build_body_mask(
    ct: sitk.Image,
    body_threshold: float = BODY_THRESHOLD_HU,
    closing_radius: int = 2,
    fill_holes: FillHoles = "bbox",
) -> np.ndarray:
    """Largest connected soft-tissue component, closed and hole-filled.

    Returns a boolean array in (z, y, x) order. ``fill_holes="auto"`` is not
    accepted here because the choice depends on the resulting air mask; use
    :func:`build_body_air_masks`, which is what every stage should call.
    """
    if fill_holes == "auto":
        raise ValueError('fill_holes="auto" requires build_body_air_masks()')
    body = sitk.Cast(ct > body_threshold, sitk.sitkUInt8)
    body = sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(body)) == 1, sitk.sitkUInt8)
    if closing_radius > 0:
        body = sitk.BinaryMorphologicalClosing(body, [closing_radius] * 3)

    if fill_holes in ("image26", "image6"):
        body = sitk.BinaryFillhole(body, fullyConnected=(fill_holes == "image26"))
        return sitk.GetArrayFromImage(body) > 0

    arr = sitk.GetArrayFromImage(body) > 0
    if fill_holes == "bbox":
        arr = _fill_holes_bbox(arr)
    elif fill_holes == "slicewise":
        arr = _fill_holes_slicewise(arr)
    return arr


def build_air_mask(
    ct_arr: np.ndarray,
    body: np.ndarray,
    air_threshold: float = AIR_THRESHOLD_HU,
    closing_radius: int = 1,
    reference: sitk.Image | None = None,
) -> np.ndarray:
    """Air inside the body, optionally morphologically closed.

    ``closing_radius`` bridges the thin partial-volume gaps at haustral folds
    and fluid menisci. The 2026-08 batch ran the centerline stage with
    ``closing_radius=0`` and the fat-map stage with ``1``; a radius of 0 leaves
    the colon fragmented into components the dust filter then deletes.
    """
    air = (ct_arr < air_threshold) & body
    if closing_radius > 0:
        air_img = sitk.GetImageFromArray(air.astype(np.uint8))
        if reference is not None:
            air_img.CopyInformation(reference)
        air_img = sitk.BinaryMorphologicalClosing(air_img, [closing_radius] * 3)
        air = sitk.GetArrayFromImage(air_img) > 0
    return air


def label_air_components(
    air: np.ndarray,
    connectivity: int = 26,
    dust_threshold_voxels: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Label ``air`` and drop components below ``dust_threshold_voxels``.

    Returns ``(labels, sizes)`` where ``sizes`` is indexed by label and
    ``sizes[0]`` is zeroed.

    ``connectivity`` matters beyond labelling: the Fast Marching solver used
    downstream propagates on the 6-connected stencil, so components joined only
    by a corner or edge contact under ``connectivity=26`` are not traversable.
    See ``docs/FAILURE_ANALYSIS.md`` section 2.3.
    """
    import cc3d

    labels, _ = cc3d.connected_components(
        air.astype(np.uint8), connectivity=connectivity, return_N=True
    )
    sizes = np.bincount(labels.ravel())
    if sizes.size:
        sizes[0] = 0

    if dust_threshold_voxels > 0:
        keep = np.where(sizes >= dust_threshold_voxels)[0]
        keep = keep[keep != 0]
        if keep.size == 0:
            raise RuntimeError(
                f"No air component is larger than the dust threshold "
                f"({dust_threshold_voxels} voxels)."
            )
        labels = np.where(np.isin(labels, keep), labels, 0).astype(np.uint32)

    return labels, sizes


def touches_image_border(mask: np.ndarray) -> bool:
    """True if ``mask`` reaches any face of the volume.

    A gas cavity that does is not a hole for ``sitk.BinaryFillhole``, which is
    what makes the two body-mask variants diverge.
    """
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def build_body_air_masks(
    ct: sitk.Image,
    body_threshold: float = BODY_THRESHOLD_HU,
    air_threshold: float = AIR_THRESHOLD_HU,
    body_closing_radius: int = 2,
    air_closing_radius: int = 1,
    fill_holes: FillHoles = "auto",
    connectivity: int = 6,
    min_colon_gas_voxels: int = MIN_COLON_GAS_VOXELS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Body and air masks together, with the adaptive hole-filling decision.

    This is the single mask entry point every stage should use. The in-house
    prototype carried two divergent implementations, one per stage, and the
    disagreement silently discarded most of the colon on some series; keeping
    one function is the structural fix for that, not just the parameter values.

    With ``fill_holes="auto"`` the 3-D fill is tried first and slice-wise
    filling is used only if the 3-D result contains no plausible colon. Returns
    ``(body, air, diag)`` where ``diag`` records which fill was used and why.
    """
    ct_arr = sitk.GetArrayFromImage(ct)

    def build(variant: FillHoles) -> tuple[np.ndarray, np.ndarray, int]:
        body = build_body_mask(ct, body_threshold, body_closing_radius, variant)
        air = build_air_mask(
            ct_arr, body, air_threshold, air_closing_radius, reference=ct
        )
        largest = 0
        if air.any():
            import cc3d

            labels, _ = cc3d.connected_components(
                air.astype(np.uint8), connectivity=connectivity, return_N=True
            )
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = int(sizes.max()) if sizes.size else 0
        return body, air, largest

    if fill_holes != "auto":
        body, air, largest = build(fill_holes)
        return body, air, {"fill_holes_used": fill_holes, "largest_gas_voxels": largest}

    body, air, largest = build("bbox")
    diag = {
        "fill_holes_used": "bbox",
        "largest_gas_voxels": largest,
        "largest_gas_voxels_bbox": largest,
        "fallback_reason": "",
    }
    if largest >= min_colon_gas_voxels:
        return body, air, diag

    body_sw, air_sw, largest_sw = build("slicewise")
    diag["largest_gas_voxels_slicewise"] = largest_sw
    if largest_sw > largest:
        diag["fill_holes_used"] = "slicewise"
        diag["largest_gas_voxels"] = largest_sw
        diag["fallback_reason"] = (
            f"3-D fill left only {largest} gas voxels "
            f"(< {min_colon_gas_voxels}); slice-wise gives {largest_sw}"
        )
        return body_sw, air_sw, diag

    diag["fallback_reason"] = (
        f"3-D fill left only {largest} gas voxels and slice-wise did not improve "
        f"on it ({largest_sw}); the colon is probably genuinely absent"
    )
    return body, air, diag
