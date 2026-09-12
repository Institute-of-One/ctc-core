"""DICOM input and resampling to an LPS isotropic grid.

Ported from ``centerline_from_seeds.py`` (``load_dicom_series``,
``resample_to_lps_iso``, ``get_or_make_resampled_ct``) in the in-house
prototype (not distributed). Behaviour is intentionally identical so that the
prototype's batch results can be reproduced (``CenterlineConfig.batch_2026_08``).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk

__all__ = [
    "load_dicom_series",
    "resample_to_lps_iso",
    "get_or_make_resampled_ct",
    "physical_bounds",
]


def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    """Read the first GDCM series found in ``dicom_dir``."""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not files:
        raise ValueError(f"No readable DICOM files in: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()


def physical_bounds(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned physical bounding box of ``image`` as ``(min_xyz, max_xyz)``."""
    size = image.GetSize()
    corners = list(product([0, size[0] - 1], [0, size[1] - 1], [0, size[2] - 1]))
    pts = np.array([image.TransformIndexToPhysicalPoint(c) for c in corners], dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def resample_to_lps_iso(image: sitk.Image, iso_spacing: float) -> sitk.Image:
    """Resample onto an axis-aligned isotropic grid covering the physical bounds.

    The output direction is the identity, so array axes are (z, y, x) in LPS.
    Voxels outside the source image are set to -1024 HU (air), which matters at
    the border: the body mask must not treat that padding as tissue.
    """
    min_xyz, max_xyz = physical_bounds(image)
    spacing = np.array([iso_spacing, iso_spacing, iso_spacing], dtype=np.float64)
    new_size = np.maximum(np.ceil((max_xyz - min_xyz) / spacing).astype(int) + 1, 1)

    rf = sitk.ResampleImageFilter()
    rf.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(-1024.0)
    rf.SetOutputSpacing(tuple(spacing.tolist()))
    rf.SetOutputOrigin(tuple(min_xyz.tolist()))
    rf.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    rf.SetSize([int(v) for v in new_size.tolist()])
    rf.SetOutputPixelType(sitk.sitkFloat32)
    return rf.Execute(image)


def get_or_make_resampled_ct(
    dicom_dir: Path,
    out_ct_path: Path,
    iso_spacing: float,
    reuse_resampled: bool = True,
) -> sitk.Image:
    """Return the LPS isotropic CT, reusing ``out_ct_path`` when it matches."""
    out_ct_path = Path(out_ct_path)
    if reuse_resampled and out_ct_path.exists():
        ct = sitk.ReadImage(str(out_ct_path), sitk.sitkFloat32)
        spacing_ok = np.allclose(ct.GetSpacing(), [iso_spacing] * 3, atol=1e-6)
        identity_ok = np.allclose(np.array(ct.GetDirection()), np.eye(3).ravel(), atol=1e-6)
        if spacing_ok and identity_ok:
            return ct

    ct_iso = resample_to_lps_iso(load_dicom_series(Path(dicom_dir)), iso_spacing=iso_spacing)
    out_ct_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_iso, str(out_ct_path))
    return ct_iso
