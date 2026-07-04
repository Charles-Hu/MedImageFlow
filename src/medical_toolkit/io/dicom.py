"""DICOM series utilities backed by optional SimpleITK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_toolkit.utils.optional import require


def read_dicom_series(directory: str | Path, *, series_id: str | None = None) -> Any:
    """Read one DICOM series from a directory as a SimpleITK image.

    If multiple series exist, ``series_id`` must be supplied to avoid silently
    selecting the wrong patient acquisition.

    Args:
        directory: Directory containing DICOM instances.
        series_id: Optional GDCM series identifier.

    Returns:
        A SimpleITK image containing the selected series.

    Raises:
        ImportError: If SimpleITK is not installed.
        NotADirectoryError: If ``directory`` is not a directory.
        ValueError: If no series exists, multiple series are ambiguous, or the
            requested series identifier is unavailable.
    """
    sitk = require("SimpleITK", extra="imaging")
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or [])
    if not series_ids:
        raise ValueError(f"No readable DICOM series found in {directory}")
    if series_id is None:
        if len(series_ids) != 1:
            raise ValueError(f"Found {len(series_ids)} series; choose series_id from {series_ids}")
        series_id = series_ids[0]
    if series_id not in series_ids:
        raise ValueError(f"Unknown series_id {series_id!r}; available: {series_ids}")

    filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    return reader.Execute()


def dicom_series_to_nifti(
    source: str | Path,
    destination: str | Path,
    *,
    series_id: str | None = None,
    compress: bool = True,
) -> Path:
    """Convert a DICOM series directory to a NIfTI file.

    Args:
        source: Directory containing a DICOM series.
        destination: Output path ending in ``.nii`` or ``.nii.gz``.
        series_id: Optional GDCM series identifier.
        compress: Whether SimpleITK should compress the output.

    Returns:
        The output path.

    Raises:
        ImportError: If SimpleITK is not installed.
        ValueError: If the output extension or DICOM series is invalid.
    """
    sitk = require("SimpleITK", extra="imaging")
    destination = Path(destination)
    if not (destination.name.endswith(".nii") or destination.name.endswith(".nii.gz")):
        raise ValueError("destination must end with .nii or .nii.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = read_dicom_series(source, series_id=series_id)
    sitk.WriteImage(image, str(destination), useCompression=compress)
    return destination
