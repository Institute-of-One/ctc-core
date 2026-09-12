"""Virtual gross pathology (VGP) unfold by cubemap GPU volume rendering.

Visualisation only: nothing in this module feeds a measurement. Requires VTK
(``pip install ctc-core[vgp]``); importing the module without VTK works, calling
the renderer raises.

Ported function by function from the in-house prototype
(``unfold_vgp_gpu_cube.py`` and the frame builders of ``unfold_vgp.py``); the
function bodies are unchanged. How it works:

* The CT is volume-rendered on the GPU (``vtkGPUVolumeRayCastMapper``): air is
  transparent, tissue opaque, with gradient shading.
* At each centerline station a camera sits on the centerline and renders four
  cube faces (field of view 100 degrees) looking along ``+n, +b, -n, -b`` with
  the tangent as view-up. The middle row of each face is remapped by ``tan`` to
  equal angles, with linear blending where adjacent faces overlap, giving one
  360-degree ring per station; the rings stacked along the centerline form the
  unfold. Four renders per station instead of one per angle is what makes it
  about 120 times faster than the per-angle renderer (47 min to 24 s).
* ``n`` is the anatomical anterior direction projected perpendicular to the
  tangent, so angle 0 is anterior at every station.
* The centerline is Gaussian-smoothed (``centerline_smooth_sigma_mm``, default
  2 mm, endpoints fixed) before the frames are computed. On a voxel path the
  tangent flips at every step, the frame rotates with it, rays graze the wall
  at bends and fall through, and the unfold shows black dots; smoothing removes
  the kinks and the dots with them.
* The framebuffer is read as RGBA. Pixels whose ray crossed only air
  (alpha < 0.4) are filled from their neighbours; partially accumulated pixels
  (0.4 <= alpha < 0.97) are restored by ``rgb / alpha``, since VTK returns
  premultiplied alpha.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    import vtk
    from vtk.util import numpy_support
    _HAS_VTK = True
except Exception:  # VTK is an optional extra
    vtk = None  # type: ignore[assignment]
    numpy_support = None  # type: ignore[assignment]
    _HAS_VTK = False

LOGGER = logging.getLogger(__name__)

__all__ = ["compute_vgp_unfold_gpu_cube", "ULTRA_HQ"]

# The prototype's "VGP Cube (Ultra HQ, super-sampled)" preset. The prototype
# reaches its highest quality by resampling the one CT to 0.5 mm before
# rendering ("very fine"); the preset itself adds finer ray sampling and more
# angles, and clamps tagged residue at 300 HU so the gradient does not blow up
# there.
ULTRA_HQ = {
    "num_angles": 720,
    "sample_distance_mm": 0.25,
    "face_height": 4,
    "internal_iso_spacing_mm": 0.0,
    "opacity_ramp_hu": 100.0,
    "interpolation_type": "linear",
    "clamp_high_hu": 300.0,
}
ULTRA_HQ_CT_SPACING_MM = 0.5
ULTRA_HQ_STEP_MM = 1.0


# ---------------------------------------------------------------------------
# Frames (from unfold_vgp.py)
# ---------------------------------------------------------------------------

def _anatomical_up_frames(
    centerline_xyz_mm: np.ndarray,
    up_world: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tangent, normal, binormal)`` where the normal axis at
    every centerline point is the *world* ``up_world`` direction
    projected perpendicular to the local tangent.

    Why this matters for VGP-style unfold: with parallel-transport the
    angular reference theta=0 stays anatomically arbitrary, so a haustral
    fold at a fixed circumferential location appears at the same column
    of the unfold image at all s -> all folds become straight vertical
    lines. With an anatomical up vector (default = LPS anterior, -Y), the
    angular reference is anatomically meaningful, so as the colon turns
    through flexures the same circumferential line traces a curve in the
    unfold — matching the natural look of clinical VGP viewers."""
    pts = np.asarray(centerline_xyz_mm, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("Centerline needs at least 2 points.")
    tangents = np.zeros((n, 3), dtype=np.float64)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents /= norms

    up = np.asarray(up_world, dtype=np.float64)
    up_n = float(np.linalg.norm(up))
    if up_n < 1e-9:
        up = np.array([0.0, -1.0, 0.0])
    else:
        up = up / up_n

    # Fallback axis if up is parallel to a given tangent.
    candidate_axes = np.eye(3, dtype=np.float64)
    fallback_axis = candidate_axes[int(np.argmin(np.abs(candidate_axes @ up)))]

    normals = np.zeros((n, 3), dtype=np.float64)
    binormals = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        t = tangents[i]
        proj = up - np.dot(up, t) * t
        nproj = float(np.linalg.norm(proj))
        if nproj < 0.05:
            # Tangent ~parallel to up; use fallback axis instead.
            proj = fallback_axis - np.dot(fallback_axis, t) * t
            nproj = float(np.linalg.norm(proj))
        if nproj < 1e-9:
            normals[i] = normals[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])
            binormals[i] = binormals[i - 1] if i > 0 else np.array([0.0, 0.0, 1.0])
            continue
        normal = proj / nproj
        binormal = np.cross(t, normal)
        bn = float(np.linalg.norm(binormal))
        if bn > 1e-9:
            binormal /= bn
        normals[i] = normal
        binormals[i] = binormal
    return tangents, normals, binormals


def _parallel_transport_frames(
    centerline_xyz_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tangent, normal, binormal) arrays of shape (N, 3). The
    initial reference vector is rotated minimally at each step so the
    angular axis along the colon does not spin. This keeps theta=0
    pointing at a consistent anatomical direction throughout the unfold."""
    pts = np.asarray(centerline_xyz_mm, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("Centerline needs at least 2 points.")

    # Tangents via central differences (forward/backward at endpoints)
    tangents = np.zeros((n, 3), dtype=np.float64)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents /= norms

    normals = np.zeros((n, 3), dtype=np.float64)
    binormals = np.zeros((n, 3), dtype=np.float64)

    # Initial reference: pick an axis least aligned with the first tangent
    t0 = tangents[0]
    candidate_axes = np.eye(3, dtype=np.float64)
    dots = np.abs(candidate_axes @ t0)
    init_axis = candidate_axes[int(np.argmin(dots))]
    n0 = init_axis - np.dot(init_axis, t0) * t0
    n0 /= max(np.linalg.norm(n0), 1e-9)
    b0 = np.cross(t0, n0)
    b0 /= max(np.linalg.norm(b0), 1e-9)
    normals[0] = n0
    binormals[0] = b0

    for i in range(1, n):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        # Rotation that takes t_prev -> t_curr (about axis = t_prev x t_curr).
        axis = np.cross(t_prev, t_curr)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
            continue
        axis /= axis_norm
        cos_a = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
        sin_a = axis_norm  # since both unit, |t_prev x t_curr| = sin(angle)

        # Rodrigues' rotation formula applied to normal[i-1]
        n_prev = normals[i - 1]
        n_rot = (
            n_prev * cos_a
            + np.cross(axis, n_prev) * sin_a
            + axis * (np.dot(axis, n_prev)) * (1.0 - cos_a)
        )
        # Re-orthogonalize against the new tangent (numerical safety)
        n_rot -= np.dot(n_rot, t_curr) * t_curr
        nn = float(np.linalg.norm(n_rot))
        if nn < 1e-9:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
            continue
        n_rot /= nn
        b_rot = np.cross(t_curr, n_rot)
        bn = float(np.linalg.norm(b_rot))
        if bn > 1e-9:
            b_rot /= bn
        normals[i] = n_rot
        binormals[i] = b_rot

    return tangents, normals, binormals


# ---------------------------------------------------------------------------
# Volume-side preprocessing
# ---------------------------------------------------------------------------

def _gate_ct_with_lumen_mask(
    ct_image: sitk.Image,
    lumen_mask_path: Path,
    dilation_voxels: int = 3,
    background_hu: int = -1024,
) -> sitk.Image:
    """Mask the CT outside the colon: keep voxels inside the lumen and a
    dilated wall band, replace everything else with ``background_hu``.

    Why: the radial rays from the centerline often graze peri-colonic
    tissue (mesentery, fat, vessels) before saturating opacity. That
    contamination shows up in the unfold as occasional black/dim spots
    even when the colon wall itself is clean. Gating restricts the rays
    to a band around the lumen so they cannot accumulate from
    extracolonic structures.
    """
    if not Path(lumen_mask_path).exists():
        LOGGER.warning("Lumen mask not found, skipping air-mask gating: %s", lumen_mask_path)
        return ct_image
    lumen_img = sitk.ReadImage(str(lumen_mask_path), sitk.sitkUInt8)
    # Resample the mask onto the CT grid (handles spacing / origin
    # differences caused by the optional internal CT resample).
    if (
        lumen_img.GetSize() != ct_image.GetSize()
        or lumen_img.GetSpacing() != ct_image.GetSpacing()
        or lumen_img.GetOrigin() != ct_image.GetOrigin()
    ):
        lumen_img = sitk.Resample(
            lumen_img,
            ct_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
        )
    if int(dilation_voxels) > 0:
        lumen_img = sitk.BinaryDilate(
            lumen_img, [int(dilation_voxels)] * 3
        )
    # Largest connected component of the dilated band -> drops isolated
    # extracolonic air pockets that would otherwise still be inside the
    # band and confuse the rays.
    cc = sitk.ConnectedComponent(lumen_img)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(rel, 1, 1, 1, 0)
    mask_arr = sitk.GetArrayFromImage(largest).astype(bool)
    ct_arr = sitk.GetArrayFromImage(ct_image)
    out_arr = np.where(mask_arr, ct_arr, np.int16(background_hu)).astype(ct_arr.dtype)
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(ct_image)
    LOGGER.info(
        "Air-mask gating: kept %.1f%% of voxels (lumen + %d-voxel wall band).",
        float(mask_arr.mean() * 100.0), int(dilation_voxels),
    )
    return out_img


def _anisotropic_diffusion_smooth(
    ct_image: sitk.Image,
    num_iterations: int = 4,
    conductance: float = 2.0,
    time_step: float = 0.02,
) -> sitk.Image:
    """Edge-preserving smoothing via SimpleITK CurvatureAnisotropicDiffusion.

    Why: Gaussian pre-blur is isotropic and softens the air-tissue edge
    along with the noise. CurvatureAnisotropic preserves the wall edge
    (large gradient) while smoothing within homogeneous regions
    (small gradient) -> reduces voxel-grid quantization at fold edges
    without dulling the wall. Default 4 iterations is light; bump to
    8-10 for stronger denoising."""
    pixel_id = ct_image.GetPixelID()
    ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)
    out = sitk.CurvatureAnisotropicDiffusion(
        ct_f,
        timeStep=float(time_step),
        conductanceParameter=float(conductance),
        numberOfIterations=int(num_iterations),
    )
    return sitk.Cast(out, pixel_id)


def _smooth_centerline_xyz(
    xyz_mm: np.ndarray,
    sigma_mm: float,
) -> np.ndarray:
    """Gaussian-smooth the centerline trajectory in (x, y, z) to remove
    voxel-scale kinks from skeleton extraction.

    Why: `_anatomical_up_frames` computes the tangent from neighbouring
    centerline points. If the input is the raw voxel skeleton (1-voxel
    jumps), the tangent flips abruptly at every step and the cubemap
    frame rotates with it. Adjacent s-rows of the unfold then sample
    different physical directions for the same theta -> frame-flip
    artefacts: the rays graze the wall at sharp bends and partial
    fall-through produces black dots. Smoothing with a sigma well below
    the lumen radius (~10 mm) keeps the centerline inside the lumen
    while removing the kinks.
    """
    if sigma_mm <= 0.0 or xyz_mm.shape[0] < 3:
        return xyz_mm
    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception as exc:
        LOGGER.warning("scipy.ndimage unavailable, centerline smoothing skipped: %s", exc)
        return xyz_mm
    diffs = np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1)
    avg_step = float(diffs.mean()) if diffs.size > 0 else 1.0
    if avg_step <= 0.0:
        return xyz_mm
    sigma_samples = float(sigma_mm) / avg_step
    out = np.empty_like(xyz_mm)
    for k in range(3):
        out[:, k] = gaussian_filter1d(
            xyz_mm[:, k], sigma=sigma_samples, mode="nearest"
        )
    # Pin endpoints so the smoothed centerline still starts/ends at the
    # original anatomical positions.
    out[0] = xyz_mm[0]
    out[-1] = xyz_mm[-1]
    LOGGER.info(
        "Centerline smoothing: sigma=%.2f mm (~%.2f samples, avg step=%.2f mm).",
        float(sigma_mm), sigma_samples, avg_step,
    )
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _bilateral_post_smooth(
    rgb: np.ndarray,
    sigma_color: float,
    sigma_spatial: float,
) -> np.ndarray:
    """Edge-preserving bilateral smoothing on the unfold RGB. Smooths
    within-region speckle while keeping fold edges crisp. Pass
    ``sigma_color <= 0`` or ``sigma_spatial <= 0`` to skip."""
    if sigma_color <= 0.0 or sigma_spatial <= 0.0:
        return rgb
    try:
        from skimage.restoration import denoise_bilateral
    except Exception as exc:
        LOGGER.warning("skimage.restoration unavailable, bilateral skipped: %s", exc)
        return rgb
    img_float = rgb.astype(np.float32) / 255.0
    try:
        denoised = denoise_bilateral(
            img_float,
            sigma_color=float(sigma_color),
            sigma_spatial=float(sigma_spatial),
            channel_axis=-1,
        )
    except TypeError:
        denoised = denoise_bilateral(
            img_float,
            sigma_color=float(sigma_color),
            sigma_spatial=float(sigma_spatial),
            multichannel=True,
        )
    out = (np.clip(denoised, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    LOGGER.info(
        "Bilateral post-smooth: sigma_color=%.3f sigma_spatial=%.2f.",
        float(sigma_color), float(sigma_spatial),
    )
    return out


def _alpha_aware_fill(
    rgb: np.ndarray,
    alpha: np.ndarray,
    surface_color_rgb: tuple[float, float, float],
    inpaint_alpha_max: float = 0.40,
    recover_alpha_min: float = 0.40,
    recover_alpha_max: float = 0.97,
    recover_cap_factor: float = 1.10,
    max_biharmonic_component_px: int = 64,
) -> np.ndarray:
    """Fix fall-through dim pixels using accumulated opacity.

    Three-tier strategy:
    - alpha in [recover_alpha_min, recover_alpha_max): partial fall-
      through. Recover wall color via rgb / alpha (premultiplied alpha).
    - alpha < inpaint_alpha_max, **small** connected components
      (<= max_biharmonic_component_px pixels): biharmonic inpaint, OK
      because boundary conditions cover a small region.
    - alpha < inpaint_alpha_max, **large** connected components:
      biharmonic produces unnatural rosette / disk gradients in big
      regions because Laplace's equation interpolates smoothly. Flat-
      fill with the wall color instead.
    """
    a = alpha.astype(np.float32) / 255.0
    out = rgb.copy()

    # Tier 1: partial fall-through -> alpha-divide recovery
    mask_recover = (a >= float(recover_alpha_min)) & (a < float(recover_alpha_max))
    n_recover = int(mask_recover.sum())
    if n_recover > 0:
        safe_a = np.maximum(a[mask_recover], 0.05)
        recovered = rgb[mask_recover].astype(np.float32) / safe_a[:, None]
        cap = np.array(
            [255.0 * surface_color_rgb[0],
             255.0 * surface_color_rgb[1],
             255.0 * surface_color_rgb[2]],
            dtype=np.float32,
        ) * float(recover_cap_factor)
        recovered = np.minimum(recovered, cap[None, :])
        out[mask_recover] = np.clip(recovered, 0.0, 255.0).astype(np.uint8)
        LOGGER.info(
            "Alpha-divide recovery: %d pixel(s) (alpha in [%.2f, %.2f)).",
            n_recover, recover_alpha_min, recover_alpha_max,
        )

    # Full fall-through: split by connected component size.
    mask_inpaint = a < float(inpaint_alpha_max)
    n_inpaint = int(mask_inpaint.sum())
    if n_inpaint == 0:
        return out

    wall_color_u8 = np.array(
        [int(round(255.0 * surface_color_rgb[0])),
         int(round(255.0 * surface_color_rgb[1])),
         int(round(255.0 * surface_color_rgb[2]))],
        dtype=np.uint8,
    )

    small_mask = mask_inpaint
    big_mask = np.zeros_like(mask_inpaint)
    try:
        from scipy.ndimage import label as _label
        labels_arr, n_comp = _label(mask_inpaint)
        if n_comp > 0:
            sizes = np.bincount(labels_arr.ravel())
            big_ids = np.where(sizes > int(max_biharmonic_component_px))[0]
            big_ids = big_ids[big_ids != 0]
            if big_ids.size > 0:
                big_mask = np.isin(labels_arr, big_ids)
                small_mask = mask_inpaint & ~big_mask
    except Exception as exc:
        LOGGER.warning("Component split failed: %s — using biharmonic for all.", exc)

    n_big = int(big_mask.sum())
    if n_big > 0:
        # Per-component fill: instead of one global wall_color (which
        # showed up as visible flat-pink rectangles where the fall-
        # through was large), use the mean color of each component's
        # immediate boundary ring. Each big region then gets its own
        # locally appropriate tone and dissolves into the surroundings.
        try:
            from scipy.ndimage import binary_dilation
        except Exception:
            binary_dilation = None
        n_big_comp = 0
        for comp_id in big_ids:
            comp_mask_i = labels_arr == comp_id
            boundary_mean = None
            if binary_dilation is not None:
                dilated = binary_dilation(comp_mask_i, iterations=2)
                boundary = dilated & ~comp_mask_i & ~mask_inpaint
                if boundary.any():
                    bvals = out[boundary].astype(np.float32)
                    boundary_mean = np.clip(
                        bvals.mean(axis=0), 0.0, 255.0
                    ).astype(np.uint8)
            if boundary_mean is None:
                boundary_mean = wall_color_u8
            out[comp_mask_i] = boundary_mean
            n_big_comp += 1
        LOGGER.info(
            "Big-region local-mean fill: %d pixel(s) across %d components "
            "(>%d px each) -> per-component boundary mean.",
            n_big, n_big_comp, int(max_biharmonic_component_px),
        )

    n_small = int(small_mask.sum())
    if n_small > 0:
        try:
            from skimage.restoration import inpaint as _ski_inpaint
            img_float = out.astype(np.float32) / 255.0
            try:
                filled = _ski_inpaint.inpaint_biharmonic(
                    img_float, small_mask, channel_axis=2
                )
            except TypeError:
                filled = _ski_inpaint.inpaint_biharmonic(
                    img_float, small_mask, multichannel=True
                )
            out = (np.clip(filled * 255.0, 0.0, 255.0) + 0.5).astype(np.uint8)
            LOGGER.info(
                "Small-region biharmonic fill: %d pixel(s) (alpha < %.2f, "
                "component <= %d px).",
                n_small, inpaint_alpha_max, int(max_biharmonic_component_px),
            )
        except Exception as exc:
            LOGGER.warning("Biharmonic small-fill failed: %s", exc)

    return out


def _despeckle_median(
    rgb: np.ndarray,
    size_px: int,
    spike_factor: float = 0.4,
) -> np.ndarray:
    """Conditional median despeckle: replace a pixel with the local
    median only if it is darker than ``spike_factor * local_max``
    (i.e. an isolated dark spike). Other pixels are kept untouched, so
    edges and gradients stay crisp. Without the conditional gate a
    blanket 3x3 median softens the entire image — the user's main
    objection. Pass size_px<=1 to skip."""
    if int(size_px) <= 1:
        return rgb
    try:
        from scipy.ndimage import maximum_filter, median_filter
    except Exception as exc:
        LOGGER.warning("scipy.ndimage unavailable, despeckle skipped: %s", exc)
        return rgb
    n = int(size_px)
    if n % 2 == 0:
        n += 1
    gray = rgb.astype(np.float32).mean(axis=-1)
    local_max = maximum_filter(gray, size=n)
    mask = gray < float(spike_factor) * local_max
    n_spike = int(mask.sum())
    if n_spike == 0:
        return rgb
    out = rgb.copy()
    medians = np.empty_like(rgb)
    for c in range(3):
        medians[..., c] = median_filter(rgb[..., c], size=n)
    out[mask] = medians[mask]
    LOGGER.info(
        "Conditional despeckle (median %dx%d, spike_factor=%.2f): "
        "%d pixel(s) replaced.", n, n, spike_factor, n_spike,
    )
    return out


def _denoise_nlm(rgb: np.ndarray, h_strength: float) -> np.ndarray:
    """Non-Local Means denoising. Finds similar patches across the
    image and averages them — the modern way to suppress speckle while
    preserving edges and texture. Targets the residual speckle that
    survives both alpha-aware fill and conditional median despeckle.
    Pass h_strength <= 0 to skip. Typical 0.05-0.12. Fast mode is on
    so a 480x2135 image runs in ~3-6 sec."""
    if h_strength <= 0.0:
        return rgb
    try:
        from skimage.restoration import denoise_nl_means
    except Exception as exc:
        LOGGER.warning("skimage.restoration unavailable, NLM skipped: %s", exc)
        return rgb
    img_float = rgb.astype(np.float32) / 255.0
    try:
        denoised = denoise_nl_means(
            img_float,
            h=float(h_strength),
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            channel_axis=-1,
        )
    except TypeError:
        denoised = denoise_nl_means(
            img_float,
            h=float(h_strength),
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            multichannel=True,
        )
    out = (np.clip(denoised, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    LOGGER.info("NLM denoise: h=%.3f applied.", h_strength)
    return out


def _post_blur_2d(rgb: np.ndarray, sigma_px: float) -> np.ndarray:
    """Small 2D Gaussian smoothing on the unfold image. Anti-aliases
    the voxel-aligned stair-step artifacts at fold edges (the staircase
    spans ~2-3 px in the angle direction because a 1.5 mm voxel covers
    ~8 deg at a 10 mm radius wall, while output angular resolution is
    0.75 deg/step). Sigma 0.5-1.0 px smooths the staircase without
    losing real fold structure (folds span 5-15 px). Pass 0 to skip."""
    if sigma_px <= 0.0:
        return rgb
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as exc:
        LOGGER.warning("scipy.ndimage unavailable, post blur skipped: %s", exc)
        return rgb
    blurred = gaussian_filter(
        rgb.astype(np.float32), sigma=(float(sigma_px), float(sigma_px), 0.0)
    )
    return np.clip(blurred, 0.0, 255.0).astype(np.uint8)


def _tone_map(
    rgb: np.ndarray,
    gain: float,
    gamma: float,
    shadow_lift: float = 0.0,
) -> np.ndarray:
    """Tone curve: ``shadow_lift`` brightens dark pixels only (quadratic
    falloff toward white -> highlights untouched), ``gamma < 1`` lifts
    midtones, ``gain > 1`` adds overall brightness."""
    if gain == 1.0 and gamma == 1.0 and shadow_lift == 0.0:
        return rgb
    x = rgb.astype(np.float32) / 255.0
    if shadow_lift != 0.0:
        # Add a non-negative function that peaks near x=0 and falls to 0 at x=1.
        # (1-x)^2 falls smoothly so highlights are unaffected.
        boost = float(shadow_lift) * np.maximum(1.0 - x, 0.0) ** 2
        x = x + boost
    if gamma != 1.0:
        x = np.power(np.maximum(x, 0.0), float(gamma))
    if gain != 1.0:
        x = x * float(gain)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# VTK pipeline, remap and renderer
# ---------------------------------------------------------------------------

def _build_volume_pipeline_cube(
    ct_image: sitk.Image,
    air_hu: float,
    tissue_hu: float,
    soft_tissue_hu: float,
    surface_color_rgb: tuple[float, float, float],
    sample_distance_mm: float,
    opacity_unit_distance_mm: float,
    opacity_ramp_hu: float,
    specular: float,
    specular_power: float,
    diffuse: float,
    ambient: float,
    use_jittering: bool,
    gradient_opacity_enabled: bool = True,
    interpolation_type: str = "linear",
) -> tuple[vtk.vtkVolume, vtk.vtkGPUVolumeRayCastMapper]:
    """Volume pipeline tuned for the cubemap path."""
    ct_arr = sitk.GetArrayFromImage(ct_image).astype(np.int16)
    sp = ct_image.GetSpacing()
    og = ct_image.GetOrigin()

    vtk_img = vtk.vtkImageData()
    nz, ny, nx = ct_arr.shape
    vtk_img.SetDimensions(int(nx), int(ny), int(nz))
    vtk_img.SetSpacing(float(sp[0]), float(sp[1]), float(sp[2]))
    vtk_img.SetOrigin(float(og[0]), float(og[1]), float(og[2]))
    vtk_arr = numpy_support.numpy_to_vtk(
        num_array=ct_arr.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_SHORT,
    )
    vtk_img.GetPointData().SetScalars(vtk_arr)

    cr, cg, cb = surface_color_rgb
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(-1024.0, 0.0, 0.0, 0.0)
    color_tf.AddRGBPoint(float(air_hu), cr * 0.85, cg * 0.85, cb * 0.85)
    color_tf.AddRGBPoint(float(air_hu) + 30.0, cr, cg, cb)
    color_tf.AddRGBPoint(float(tissue_hu), cr, cg, cb)
    color_tf.AddRGBPoint(float(soft_tissue_hu), cr, cg, cb)
    color_tf.AddRGBPoint(500.0, cr, cg, cb)

    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(-1024.0, 0.0)
    opacity_tf.AddPoint(float(air_hu), 0.0)
    opacity_tf.AddPoint(float(air_hu) + float(opacity_ramp_hu), 1.0)
    opacity_tf.AddPoint(float(tissue_hu), 1.0)
    opacity_tf.AddPoint(float(soft_tissue_hu), 1.0)
    opacity_tf.AddPoint(500.0, 1.0)

    # Gradient opacity: when enabled, lowers opacity in low-gradient
    # (interior) regions and emphasizes wall edges. Useful for VGP
    # unfold (depth/relief). For Virtual Endoscopy where the camera is
    # inside the lumen and looks at the wall head-on, gradient
    # contribution amplifies iso-HU contour banding — set flat (=1.0
    # everywhere) by disabling this.
    gradient_tf = vtk.vtkPiecewiseFunction()
    if gradient_opacity_enabled:
        gradient_tf.AddPoint(0.0, 0.0)
        gradient_tf.AddPoint(80.0, 1.0)
    else:
        gradient_tf.AddPoint(0.0, 1.0)
        gradient_tf.AddPoint(255.0, 1.0)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color_tf)
    prop.SetScalarOpacity(opacity_tf)
    prop.SetGradientOpacity(gradient_tf)
    prop.ShadeOn()
    prop.SetAmbient(float(ambient))
    prop.SetDiffuse(float(diffuse))
    prop.SetSpecular(float(specular))
    prop.SetSpecularPower(float(specular_power))
    interp = str(interpolation_type).lower()
    if interp == "cubic":
        # Tricubic interpolation: addresses the diagonal-fold staircase
        # that trilinear shows. The aliasing happens because trilinear
        # blends 8 neighbours (2x2x2); when a fold runs at ~45 deg to
        # the voxel axes, adjacent ray samples land in different 8-cell
        # cubes and the gradient flips voxel-by-voxel. Tricubic uses
        # 64 (4x4x4) neighbours and blends smoothly across that
        # boundary, eliminating the diagonal stair-step without
        # globally blurring the wall.
        try:
            prop.SetInterpolationTypeToCubic()
        except AttributeError:
            LOGGER.warning("Cubic interpolation not available in this VTK; using linear.")
            prop.SetInterpolationTypeToLinear()
    elif interp == "nearest":
        prop.SetInterpolationTypeToNearest()
    else:
        prop.SetInterpolationTypeToLinear()
    prop.SetScalarOpacityUnitDistance(float(opacity_unit_distance_mm))

    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_img)
    mapper.SetSampleDistance(float(sample_distance_mm))
    mapper.SetAutoAdjustSampleDistances(False)
    try:
        mapper.SetUseJittering(bool(use_jittering))
    except Exception:
        pass
    mapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    return volume, mapper


def _log_timing(step: str, t0: float) -> None:
    LOGGER.info("%s finished in %.2f sec", step, time.perf_counter() - t0)


def _build_remap_table(
    num_angles: int,
    face_width: int,
    face_fov_deg: float = 100.0,
):
    """Pre-compute equirectangular -> 4-cube-face mapping with overlap blending.

    Returns 8 arrays (length num_angles): primary_face, x_low_p, w_high_p,
    weight_p, secondary_face, x_low_s, w_high_s, weight_s.
    weight_p + weight_s == 1 everywhere.
    """
    if face_width < 4:
        raise ValueError("face_width must be >= 4 for stable equirectangular remap.")
    if face_fov_deg <= 90.0:
        raise ValueError("face_fov_deg must be > 90 for overlap blending.")

    half_fov_rad = np.deg2rad(face_fov_deg / 2.0)
    focal_pix = (face_width / 2.0) / np.tan(half_fov_rad)
    overlap_rad = half_fov_rad - np.pi / 4.0

    angles_out = np.arange(num_angles, dtype=np.float64) * (2.0 * np.pi / num_angles)
    theta_ks = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=np.float64)

    phi_all = theta_ks[:, None] - angles_out[None, :]
    phi_all = (phi_all + np.pi) % (2.0 * np.pi) - np.pi

    abs_phi = np.abs(phi_all)
    primary_face = np.argmin(abs_phi, axis=0).astype(np.int32)
    aidx = np.arange(num_angles)
    phi_primary = phi_all[primary_face, aidx]

    secondary_face = np.where(
        phi_primary > 0.0,
        (primary_face - 1) % 4,
        (primary_face + 1) % 4,
    ).astype(np.int32)
    phi_secondary = phi_all[secondary_face, aidx]

    x_primary_float = (face_width / 2.0) + focal_pix * np.tan(phi_primary) - 0.5
    phi_safe_limit = np.pi / 2.0 - 0.05
    phi_secondary_safe = np.clip(phi_secondary, -phi_safe_limit, phi_safe_limit)
    x_secondary_float = (face_width / 2.0) + focal_pix * np.tan(phi_secondary_safe) - 0.5

    boundary_dist = np.pi / 4.0 - np.abs(phi_primary)
    blend_t = np.clip(boundary_dist / overlap_rad, 0.0, 1.0)
    weight_primary = 0.5 + 0.5 * blend_t
    weight_secondary = 1.0 - weight_primary

    x_low_p = np.clip(np.floor(x_primary_float).astype(np.int32), 0, int(face_width) - 2)
    w_high_p = np.clip(x_primary_float - x_low_p, 0.0, 1.0)
    x_low_s = np.clip(np.floor(x_secondary_float).astype(np.int32), 0, int(face_width) - 2)
    w_high_s = np.clip(x_secondary_float - x_low_s, 0.0, 1.0)

    return (
        primary_face, x_low_p, w_high_p, weight_primary,
        secondary_face, x_low_s, w_high_s, weight_secondary,
    )


def compute_vgp_unfold_gpu_cube(
    ct_image: sitk.Image,
    centerline_xyz_mm: np.ndarray,
    s_mm: np.ndarray,
    out_dir: Path,
    num_angles: int = 480,
    air_hu: float = -700.0,
    tissue_hu: float = -300.0,
    soft_tissue_hu: float = -100.0,
    sample_distance_mm: float = 0.5,
    max_ray_mm: float = 80.0,
    surface_color_rgb: tuple[float, float, float] = (1.0, 0.72, 0.70),
    frame_mode: str = "anatomical",
    anatomical_up: tuple[float, float, float] = (0.0, -1.0, 0.0),
    face_height: int = 4,
    face_fov_deg: float = 100.0,
    opacity_unit_distance_mm: float = 0.4,
    opacity_ramp_hu: float = 60.0,
    specular: float = 0.35,
    specular_power: float = 30.0,
    diffuse: float = 0.75,
    ambient: float = 0.25,
    use_jittering: bool = True,
    inpaint_alpha_max: float = 0.40,
    recover_alpha_min: float = 0.40,
    recover_alpha_max: float = 0.97,
    max_biharmonic_component_px: int = 64,
    tone_gain: float = 1.00,
    tone_gamma: float = 0.92,
    tone_shadow_lift: float = 0.20,
    pre_blur_sigma_mm: float = 0.0,
    post_blur_sigma_px: float = 0.0,
    despeckle_size_px: int = 1,
    despeckle_spike_factor: float = 0.4,
    nlm_h_strength: float = 0.0,
    face_width_multiplier: int = 1,
    internal_iso_spacing_mm: float = 0.0,
    # ---- Geometric / sampling pipeline (volume-side preprocessing) ----
    lumen_mask_path: Path | None = None,
    air_mask_dilation_voxels: int = 3,
    anisotropic_iterations: int = 0,
    anisotropic_conductance: float = 2.0,
    bilateral_sigma_color: float = 0.0,
    bilateral_sigma_spatial: float = 0.0,
    centerline_smooth_sigma_mm: float = 2.0,
    interpolation_type: str = "linear",
    clamp_high_hu: float = 0.0,
    progress_every: int = 50,
) -> tuple[np.ndarray, dict]:
    """4-face cubemap GPU VGP unfold with alpha-aware fall-through fix.

    ``pre_blur_sigma_mm`` (default 0.5) applies a small Gaussian blur to
    the CT volume before rendering. This smooths the voxel-aligned
    gradient steps that produce the "stair-step black dots" at fold
    edges in the unfold (the gradient direction snaps to the voxel grid
    when the wall transitions within ~1 voxel; small blur makes the
    transition continuous over ~3 voxels and the gradient direction
    becomes smooth). Set to 0.0 to disable."""
    if not _HAS_VTK:
        raise RuntimeError("VTK is required for GPU VGP cubemap.")
    if num_angles < 8 or num_angles % 4 != 0:
        raise ValueError("num_angles must be a positive multiple of 4 (>= 8).")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if int(face_width_multiplier) < 1:
        raise ValueError("face_width_multiplier must be >= 1.")
    face_width = int(num_angles // 4) * int(face_width_multiplier)
    fb_h = max(2, int(face_height))
    middle_row = fb_h // 2

    # ---- Optional anisotropic diffusion (edge-preserving smoothing).
    #      Run BEFORE the supersample so we operate on the smaller
    #      original-spacing volume (8x faster than running after the
    #      upsample). Targets diagonal voxel-aligned aliasing in the
    #      gradient ("stair-step black lines on oblique folds") while
    #      keeping the wall edge intact.
    if int(anisotropic_iterations) > 0:
        t_aniso = time.perf_counter()
        ct_image = _anisotropic_diffusion_smooth(
            ct_image,
            num_iterations=int(anisotropic_iterations),
            conductance=float(anisotropic_conductance),
        )
        _log_timing(
            f"VGP-GPU-Cube: anisotropic diffusion "
            f"(iters={anisotropic_iterations}, c={anisotropic_conductance})",
            t_aniso,
        )

    # ---- Optional CT resample to a finer isotropic spacing.
    #      Use the requested ``interpolation_type`` for the resample:
    #      "cubic" -> sitkBSpline order 3, the actual fix for diagonal
    #      trilinear staircase artifacts (VTK's GPU mapper does not
    #      expose cubic in this build, so we bake cubic reconstruction
    #      into the supersampled grid and let the GPU do its native
    #      trilinear on top of an already-smooth source).
    if float(internal_iso_spacing_mm) > 0.0:
        sp_in = ct_image.GetSpacing()
        target = float(internal_iso_spacing_mm)
        if min(sp_in) > target * 1.05:
            t_resamp = time.perf_counter()
            new_size = [
                int(round(ct_image.GetSize()[i] * sp_in[i] / target))
                for i in range(3)
            ]
            sitk_interp = (
                sitk.sitkBSpline
                if str(interpolation_type).lower() == "cubic"
                else sitk.sitkLinear
            )
            ct_image = sitk.Resample(
                ct_image,
                size=new_size,
                outputSpacing=(target, target, target),
                outputOrigin=ct_image.GetOrigin(),
                outputDirection=ct_image.GetDirection(),
                interpolator=sitk_interp,
                defaultPixelValue=-1024,
            )
            _log_timing(
                f"VGP-GPU-Cube: CT internal resample to {target} mm "
                f"(new size={new_size}, interp={interpolation_type})", t_resamp,
            )

    # ---- Optional Gaussian pre-blur (legacy; isotropic, dulls edges).
    if float(pre_blur_sigma_mm) > 0.0:
        t_blur = time.perf_counter()
        ct_image = sitk.SmoothingRecursiveGaussian(
            ct_image, sigma=float(pre_blur_sigma_mm)
        )
        _log_timing(
            f"VGP-GPU-Cube: CT pre-blur (sigma={pre_blur_sigma_mm} mm)", t_blur
        )

    # ---- Optional HU clamp at the high end. Removes the extreme
    #      gradient created by tagged barium residue (HU ~500-1000
    #      bordering wall HU ~0). VTK's central-difference gradient
    #      goes huge at that boundary -> voxel-aligned aliasing visible
    #      as the diagonal black-stair artifact at fold edges that sit
    #      under tagged residue. Clamping >300 HU to 300 keeps wall and
    #      tagged regions both at uniform "opaque tissue", preserving
    #      the air-tissue gradient (-1024 to ~0) which is what we want.
    if float(clamp_high_hu) > 0.0:
        t_clamp = time.perf_counter()
        ct_arr = sitk.GetArrayFromImage(ct_image)
        n_clamped = int((ct_arr > clamp_high_hu).sum())
        cap = np.int16(round(float(clamp_high_hu)))
        ct_arr_clamped = np.minimum(ct_arr, cap).astype(ct_arr.dtype)
        ct_clamp = sitk.GetImageFromArray(ct_arr_clamped)
        ct_clamp.CopyInformation(ct_image)
        ct_image = ct_clamp
        _log_timing(
            f"VGP-GPU-Cube: HU clamp at {clamp_high_hu:.0f} "
            f"({n_clamped} voxel(s) capped)", t_clamp,
        )

    # ---- Optional air-mask gating (zero out extracolonic voxels). Must
    #      run AFTER iso resample / smoothing so the mask aligns with
    #      the final CT grid handed to the GPU.
    if lumen_mask_path is not None:
        t_gate = time.perf_counter()
        ct_image = _gate_ct_with_lumen_mask(
            ct_image,
            Path(lumen_mask_path),
            dilation_voxels=int(air_mask_dilation_voxels),
        )
        _log_timing(
            f"VGP-GPU-Cube: air-mask gating "
            f"(dilate={air_mask_dilation_voxels} vx)",
            t_gate,
        )

    # ---- Volume pipeline ----
    t0 = time.perf_counter()
    volume, _mapper = _build_volume_pipeline_cube(
        ct_image,
        air_hu=air_hu,
        tissue_hu=tissue_hu,
        soft_tissue_hu=soft_tissue_hu,
        surface_color_rgb=surface_color_rgb,
        sample_distance_mm=sample_distance_mm,
        opacity_unit_distance_mm=opacity_unit_distance_mm,
        opacity_ramp_hu=opacity_ramp_hu,
        specular=specular,
        specular_power=specular_power,
        diffuse=diffuse,
        ambient=ambient,
        use_jittering=use_jittering,
        interpolation_type=str(interpolation_type),
    )
    _log_timing("VGP-GPU-Cube: build volume pipeline", t0)

    # ---- Offscreen renderer with RGBA-aware framebuffer ----
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    # Transparent background so the framebuffer alpha channel reflects
    # accumulated volume opacity (not the opaque background).
    try:
        renderer.SetBackgroundAlpha(0.0)
    except AttributeError:
        pass
    renderer.AddVolume(volume)
    renderer.RemoveAllLights()
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPositional(True)
    light.SetConeAngle(180.0)
    light.SetIntensity(1.0)
    light.SetAttenuationValues(1.0, 0.0, 0.0)
    renderer.AddLight(light)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetAlphaBitPlanes(1)
    render_window.SetMultiSamples(0)
    render_window.AddRenderer(renderer)
    render_window.SetSize(int(face_width), int(fb_h))

    # ---- Smooth the centerline trajectory (de-kink) before frame
    #      computation. Voxel-scale kinks otherwise cause tangent flips
    #      at sharp colon bends -> frame jumps -> partial fall-through
    #      black dots and discontinuous unfold rows.
    if float(centerline_smooth_sigma_mm) > 0.0:
        t_sm = time.perf_counter()
        centerline_xyz_mm = _smooth_centerline_xyz(
            np.asarray(centerline_xyz_mm, dtype=np.float64),
            sigma_mm=float(centerline_smooth_sigma_mm),
        )
        _log_timing(
            f"VGP-GPU-Cube: centerline smoothing "
            f"(sigma={centerline_smooth_sigma_mm} mm)",
            t_sm,
        )

    # ---- Centerline frames ----
    t0 = time.perf_counter()
    if str(frame_mode).lower().startswith("anatom"):
        tangents, normals, binormals = _anatomical_up_frames(
            centerline_xyz_mm, up_world=anatomical_up
        )
        _log_timing(f"VGP-GPU-Cube: anatomical-up frames (up={anatomical_up})", t0)
    else:
        tangents, normals, binormals = _parallel_transport_frames(centerline_xyz_mm)
        _log_timing("VGP-GPU-Cube: parallel-transport frames", t0)

    # ---- Pre-compute remap table ----
    (
        primary_face, x_low_p, w_high_p, weight_p,
        secondary_face, x_low_s, w_high_s, weight_s,
    ) = _build_remap_table(int(num_angles), face_width, face_fov_deg=face_fov_deg)
    x_high_p = x_low_p + 1
    x_high_s = x_low_s + 1
    w_low_p = 1.0 - w_high_p
    w_low_s = 1.0 - w_high_s
    weight_p_b = weight_p[:, None]
    weight_s_b = weight_s[:, None]
    w_low_p_b = w_low_p[:, None]
    w_high_p_b = w_high_p[:, None]
    w_low_s_b = w_low_s[:, None]
    w_high_s_b = w_high_s[:, None]

    # ---- Render loop (RGBA capture) ----
    n_s = centerline_xyz_mm.shape[0]
    rgba = np.zeros((n_s, num_angles, 4), dtype=np.uint8)

    win2img = vtk.vtkWindowToImageFilter()
    win2img.SetInput(render_window)
    win2img.SetInputBufferTypeToRGBA()

    cam = renderer.GetActiveCamera()
    cam.UseHorizontalViewAngleOn()
    cam.SetViewAngle(float(face_fov_deg))

    cos_k = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
    sin_k = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)

    t0 = time.perf_counter()
    for s_idx in range(n_s):
        p = centerline_xyz_mm[s_idx]
        n_axis = normals[s_idx]
        b_axis = binormals[s_idx]
        t_axis = tangents[s_idx]

        light.SetPosition(float(p[0]), float(p[1]), float(p[2]))
        focal_for_light = p + n_axis
        light.SetFocalPoint(
            float(focal_for_light[0]), float(focal_for_light[1]), float(focal_for_light[2])
        )

        face_strips = np.zeros((4, face_width, 4), dtype=np.uint8)

        for k in range(4):
            view_dir = cos_k[k] * n_axis + sin_k[k] * b_axis
            cam.SetPosition(float(p[0]), float(p[1]), float(p[2]))
            focal = p + view_dir
            cam.SetFocalPoint(float(focal[0]), float(focal[1]), float(focal[2]))
            cam.SetViewUp(float(t_axis[0]), float(t_axis[1]), float(t_axis[2]))
            cam.SetClippingRange(0.05, float(max_ray_mm))

            render_window.Render()
            win2img.Modified()
            win2img.Update()

            img_vtk = win2img.GetOutput()
            sc = img_vtk.GetPointData().GetScalars()
            if sc is None:
                continue
            arr = numpy_support.vtk_to_numpy(sc).reshape(int(fb_h), int(face_width), 4)
            face_strips[k] = arr[middle_row]

        # Equirectangular remap with secondary-face blending (RGBA).
        prim_low = face_strips[primary_face, x_low_p, :].astype(np.float32)
        prim_high = face_strips[primary_face, x_high_p, :].astype(np.float32)
        ring_p = prim_low * w_low_p_b + prim_high * w_high_p_b
        sec_low = face_strips[secondary_face, x_low_s, :].astype(np.float32)
        sec_high = face_strips[secondary_face, x_high_s, :].astype(np.float32)
        ring_s = sec_low * w_low_s_b + sec_high * w_high_s_b
        ring = ring_p * weight_p_b + ring_s * weight_s_b
        rgba[s_idx] = np.clip(np.round(ring), 0, 255).astype(np.uint8)

        if (s_idx + 1) % progress_every == 0 or s_idx + 1 == n_s:
            elapsed = time.perf_counter() - t0
            LOGGER.info(
                "VGP-GPU-Cube: %d / %d s positions (%.0f%%)  elapsed=%.1f sec",
                s_idx + 1, n_s, 100.0 * (s_idx + 1) / n_s, elapsed,
            )
    _log_timing("VGP-GPU-Cube: render loop", t0)

    # Transpose to (angles, s, 4)
    rgba_out = np.transpose(rgba, (1, 0, 2)).copy()

    alpha_pct = float((rgba_out[..., 3] >= 247).mean() * 100.0)
    LOGGER.info("VGP-GPU-Cube: pixels with alpha>=0.97: %.1f%%", alpha_pct)

    # ---- Persist RAW RGBA so the tune dialog can re-postprocess ----
    np.save(out_dir / "unfold_vgp_gpu_cube_rgba_raw.npy", rgba_out)
    np.savez(
        out_dir / "unfold_vgp_gpu_cube_meta.npz",
        render_color_rgb=np.asarray(surface_color_rgb, dtype=np.float32),
    )

    # ---- Post-process pipeline ----
    rgb_clean = _alpha_aware_fill(
        rgba_out[..., :3],
        rgba_out[..., 3],
        surface_color_rgb=surface_color_rgb,
        inpaint_alpha_max=float(inpaint_alpha_max),
        recover_alpha_min=float(recover_alpha_min),
        recover_alpha_max=float(recover_alpha_max),
        max_biharmonic_component_px=int(max_biharmonic_component_px),
    )
    rgb_clean2 = _despeckle_median(
        rgb_clean,
        size_px=int(despeckle_size_px),
        spike_factor=float(despeckle_spike_factor),
    )
    rgb_denoised = _denoise_nlm(rgb_clean2, h_strength=float(nlm_h_strength))
    # Edge-preserving bilateral (alternative / complement to NLM).
    rgb_bilat = _bilateral_post_smooth(
        rgb_denoised,
        sigma_color=float(bilateral_sigma_color),
        sigma_spatial=float(bilateral_sigma_spatial),
    )
    rgb_toned = _tone_map(
        rgb_bilat,
        gain=float(tone_gain),
        gamma=float(tone_gamma),
        shadow_lift=float(tone_shadow_lift),
    )
    rgb_out = _post_blur_2d(rgb_toned, sigma_px=float(post_blur_sigma_px))

    # ---- Persist final ----
    np.save(out_dir / "unfold_vgp_gpu_cube_rgb.npy", rgb_out)
    np.save(out_dir / "unfold_vgp_gpu_cube_s_mm.npy", np.asarray(s_mm, dtype=np.float32))
    try:
        from PIL import Image
        Image.fromarray(rgb_out).save(out_dir / "unfold_vgp_gpu_cube.png")
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(str(out_dir / "unfold_vgp_gpu_cube.png"), rgb_out)
        except Exception as exc:
            LOGGER.warning("Could not write VGP-GPU-Cube PNG: %s", exc)

    diag = {
        "n_s": int(n_s),
        "num_angles": int(num_angles),
        "face_width": int(face_width),
        "face_height": int(fb_h),
        "face_fov_deg": float(face_fov_deg),
        "renders_per_s": 4,
        "total_renders": int(n_s) * 4,
        "sample_distance_mm": float(sample_distance_mm),
        "opacity_unit_distance_mm": float(opacity_unit_distance_mm),
        "opacity_ramp_hu": float(opacity_ramp_hu),
        "specular": float(specular),
        "specular_power": float(specular_power),
        "use_jittering": bool(use_jittering),
        "inpaint_alpha_max": float(inpaint_alpha_max),
        "recover_alpha_min": float(recover_alpha_min),
        "recover_alpha_max": float(recover_alpha_max),
        "max_biharmonic_component_px": int(max_biharmonic_component_px),
        "tone_gain": float(tone_gain),
        "tone_gamma": float(tone_gamma),
        "tone_shadow_lift": float(tone_shadow_lift),
        "pre_blur_sigma_mm": float(pre_blur_sigma_mm),
        "post_blur_sigma_px": float(post_blur_sigma_px),
        "despeckle_size_px": int(despeckle_size_px),
        "despeckle_spike_factor": float(despeckle_spike_factor),
        "nlm_h_strength": float(nlm_h_strength),
        "face_width_multiplier": int(face_width_multiplier),
        "internal_iso_spacing_mm": float(internal_iso_spacing_mm),
        "lumen_mask_gating": bool(lumen_mask_path is not None),
        "air_mask_dilation_voxels": int(air_mask_dilation_voxels),
        "anisotropic_iterations": int(anisotropic_iterations),
        "anisotropic_conductance": float(anisotropic_conductance),
        "bilateral_sigma_color": float(bilateral_sigma_color),
        "bilateral_sigma_spatial": float(bilateral_sigma_spatial),
        "centerline_smooth_sigma_mm": float(centerline_smooth_sigma_mm),
        "interpolation_type": str(interpolation_type),
        "clamp_high_hu": float(clamp_high_hu),
        "surface_color_rgb": list(surface_color_rgb),
        "frame_mode": str(frame_mode),
        "anatomical_up": list(anatomical_up),
        "ct_spacing_mm": list(ct_image.GetSpacing()),
        "ct_size": list(ct_image.GetSize()),
        "output_shape": list(rgb_out.shape),
        "alpha_full_pct": alpha_pct,
    }
    LOGGER.info(
        "VGP-GPU-Cube done: shape (angles=%d, s=%d) sample=%.2f mm CT=%s renders=%d",
        num_angles, n_s, sample_distance_mm, list(ct_image.GetSpacing()), int(n_s) * 4,
    )
    return rgb_out, diag
