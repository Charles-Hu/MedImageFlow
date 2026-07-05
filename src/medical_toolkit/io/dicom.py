"""Strict and SimpleITK-backed DICOM series utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Literal

import numpy as np

from medical_toolkit.utils.optional import require


ZSpacingSource = Literal["position", "slice_thickness"]
DICOMConversionBackend = Literal["native", "dcm2niix"]


def convert_dicom_to_nifti(
    directory: str | Path,
    *,
    backend: DICOMConversionBackend = "native",
    series_uid: str | None = None,
    z_spacing: ZSpacingSource = "position",
    geometry_tolerance: float = 1e-4,
    spacing_tolerance: float = 1e-3,
    dcm2niix_command: str | Path = "dcm2niix",
) -> tuple[Any, list[Any]]:
    """Convert one regular, single-frame DICOM series to a NIfTI image.

    This deliberately rejects incomplete or irregular geometry instead of
    inventing spatial metadata. Enhanced multi-frame DICOM, mosaics, colour
    images, gantry tilt, and multi-volume acquisitions are outside its scope.

    Args:
        directory: Directory searched recursively for DICOM instances.
        backend: Conversion implementation. ``"native"`` uses the strict
            pydicom implementation in this package; ``"dcm2niix"`` invokes
            an installed dcm2niix executable.
        series_uid: ``SeriesInstanceUID`` to select. It is required when more
            than one image series is present. Only supported by the native
            backend.
        z_spacing: Use adjacent ``ImagePositionPatient`` values (recommended),
            or use the DICOM ``SliceThickness`` value. Only supported by the
            native backend.
        geometry_tolerance: Absolute tolerance for direction cosines and
            in-plane slice displacement, in patient-coordinate units.
        spacing_tolerance: Relative tolerance used to detect irregular slice
            spacing.
        dcm2niix_command: Executable name or path used by the dcm2niix backend.

    Returns:
        A ``(nibabel.Nifti1Image, datasets)`` tuple. With the native backend,
        datasets are ordered along the third NIfTI array axis. The dcm2niix
        backend returns an empty dataset list because dcm2niix does not expose
        pydicom Dataset objects.

    Raises:
        ImportError: If a required Python package is not installed.
        NotADirectoryError: If ``directory`` is not a directory.
        FileNotFoundError: If the dcm2niix executable cannot be found.
        RuntimeError: If dcm2niix fails.
        ValueError: If an option, selection, pixel data, or geometry is invalid.
    """
    if backend not in ("native", "dcm2niix"):
        raise ValueError("backend must be 'native' or 'dcm2niix'")
    if z_spacing not in ("position", "slice_thickness"):
        raise ValueError("z_spacing must be 'position' or 'slice_thickness'")
    if geometry_tolerance <= 0 or spacing_tolerance <= 0:
        raise ValueError("geometry_tolerance and spacing_tolerance must be positive")

    nib = require("nibabel", extra="imaging")
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    if backend == "dcm2niix":
        if series_uid is not None:
            raise ValueError("series_uid is only supported by backend='native'")
        if z_spacing != "position":
            raise ValueError("z_spacing is only configurable with backend='native'")

        command = str(dcm2niix_command)
        executable = shutil.which(command)
        if executable is None:
            command_path = Path(command).expanduser()
            if not command_path.is_file():
                raise FileNotFoundError(f"dcm2niix executable not found: {command}")
            executable = str(command_path)

        with tempfile.TemporaryDirectory(prefix="medical-toolkit-dcm2niix-") as output_dir:
            result = subprocess.run(
                [
                    executable,
                    "-b",
                    "n",
                    "-z",
                    "n",
                    "-f",
                    "converted",
                    "-o",
                    output_dir,
                    str(directory),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"dcm2niix failed with exit code {result.returncode}: {detail}"
                )
            outputs = sorted(Path(output_dir).glob("*.nii"))
            if not outputs:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"dcm2niix did not produce a NIfTI file: {detail}")
            if len(outputs) != 1:
                names = [path.name for path in outputs]
                raise ValueError(
                    "dcm2niix produced multiple NIfTI files; provide a directory "
                    f"containing exactly one convertible series: {names}"
                )

            converted = nib.load(str(outputs[0]))
            # Materialize data before TemporaryDirectory removes the NIfTI file.
            data = np.asanyarray(converted.dataobj).copy()
            image = nib.Nifti1Image(
                data,
                np.asarray(converted.affine).copy(),
                header=converted.header.copy(),
            )
            return image, []

    pydicom = require("pydicom", extra="imaging")

    series: dict[str, list[Any]] = defaultdict(list)
    read_errors: list[str] = []
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        try:
            ds = pydicom.dcmread(str(path), force=False)
        except (pydicom.errors.InvalidDicomError, OSError) as exc:
            read_errors.append(f"{path}: {exc}")
            continue
        if "PixelData" not in ds:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None:
            raise ValueError(f"Image instance has no SeriesInstanceUID: {path}")
        series[str(uid)].append(ds)

    if not series:
        detail = f" First read error: {read_errors[0]}" if read_errors else ""
        raise ValueError(f"No readable DICOM image series found in {directory}.{detail}")
    if series_uid is None:
        if len(series) != 1:
            summary = {uid: len(items) for uid, items in series.items()}
            raise ValueError(f"Found multiple DICOM series; choose series_uid from {summary}")
        datasets = next(iter(series.values()))
    else:
        if series_uid not in series:
            raise ValueError(f"Unknown series_uid {series_uid!r}; available: {list(series)}")
        datasets = series[series_uid]

    required = (
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    )
    for index, ds in enumerate(datasets):
        missing = [tag for tag in required if not hasattr(ds, tag)]
        if missing:
            raise ValueError(f"DICOM instance {index} is missing required tags: {missing}")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multi-frame DICOM is not supported")
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise ValueError("Colour or multi-sample DICOM is not supported")

    def unit_vector(values: Any, label: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid {label}: expected three finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= geometry_tolerance:
            raise ValueError(f"Invalid zero-length {label}")
        return vector / norm

    first = datasets[0]
    iop = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    if iop.shape != (6,) or not np.all(np.isfinite(iop)):
        raise ValueError("ImageOrientationPatient must contain six finite values")
    column_axis = unit_vector(iop[:3], "first image-axis direction")
    row_axis = unit_vector(iop[3:], "second image-axis direction")
    if abs(float(np.dot(column_axis, row_axis))) > geometry_tolerance:
        raise ValueError("ImageOrientationPatient axes are not orthogonal")
    slice_axis = unit_vector(np.cross(column_axis, row_axis), "slice direction")

    positions: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    first_shape = (int(first.Rows), int(first.Columns))
    first_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    if (
        first_spacing.shape != (2,)
        or np.any(~np.isfinite(first_spacing))
        or np.any(first_spacing <= 0)
    ):
        raise ValueError("PixelSpacing must contain two positive finite values")

    reference_values = {
        name: str(getattr(first, name))
        for name in ("FrameOfReferenceUID", "Modality")
        if hasattr(first, name)
    }
    seen_sop_uids: set[str] = set()
    for index, ds in enumerate(datasets):
        shape = (int(ds.Rows), int(ds.Columns))
        if shape != first_shape:
            raise ValueError(f"Inconsistent image shape at instance {index}: {shape} != {first_shape}")
        spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        if spacing.shape != (2,) or not np.allclose(
            spacing,
            first_spacing,
            rtol=spacing_tolerance,
            atol=geometry_tolerance,
        ):
            raise ValueError(f"Inconsistent PixelSpacing at instance {index}")
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        if orientation.shape != (6,) or not np.allclose(
            orientation,
            iop,
            rtol=0,
            atol=geometry_tolerance,
        ):
            raise ValueError(f"Inconsistent ImageOrientationPatient at instance {index}")
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Invalid ImagePositionPatient at instance {index}")
        positions.append(position)
        for name, expected in reference_values.items():
            if str(getattr(ds, name, "")) != expected:
                raise ValueError(f"Inconsistent {name} at instance {index}")
        sop_uid = getattr(ds, "SOPInstanceUID", None)
        if sop_uid is not None:
            if str(sop_uid) in seen_sop_uids:
                raise ValueError(f"Duplicate SOPInstanceUID at instance {index}: {sop_uid}")
            seen_sop_uids.add(str(sop_uid))

    projections = np.asarray([float(np.dot(position, slice_axis)) for position in positions])
    order = np.argsort(projections)
    datasets = [datasets[int(index)] for index in order]
    positions_array = np.asarray([positions[int(index)] for index in order])
    projections = projections[order]

    if len(datasets) > 1:
        projected_steps = np.diff(projections)
        if np.any(projected_steps <= geometry_tolerance):
            raise ValueError("Duplicate or indistinguishable slice positions detected")
        position_spacing = float(np.median(projected_steps))
        if not np.allclose(
            projected_steps,
            position_spacing,
            rtol=spacing_tolerance,
            atol=geometry_tolerance,
        ):
            raise ValueError(f"Irregular slice spacing detected: {projected_steps.tolist()}")
        displacements = np.diff(positions_array, axis=0)
        in_plane = displacements - projected_steps[:, None] * slice_axis
        if np.any(np.linalg.norm(in_plane, axis=1) > geometry_tolerance):
            raise ValueError("In-plane slice displacement (for example gantry tilt) is unsupported")
    else:
        position_spacing = None

    if z_spacing == "slice_thickness":
        thicknesses = []
        for index, ds in enumerate(datasets):
            if not hasattr(ds, "SliceThickness"):
                raise ValueError(f"SliceThickness is missing at instance {index}")
            thicknesses.append(float(ds.SliceThickness))
        if any(not np.isfinite(value) or value <= 0 for value in thicknesses):
            raise ValueError("SliceThickness values must be positive and finite")
        if not np.allclose(
            thicknesses,
            thicknesses[0],
            rtol=spacing_tolerance,
            atol=geometry_tolerance,
        ):
            raise ValueError("Inconsistent SliceThickness values")
        slice_spacing = thicknesses[0]
    elif position_spacing is not None:
        slice_spacing = position_spacing
    elif hasattr(datasets[0], "SliceThickness") and float(datasets[0].SliceThickness) > 0:
        slice_spacing = float(datasets[0].SliceThickness)
    else:
        raise ValueError("A single-slice series needs SliceThickness to define z spacing")

    volume = np.empty((first_shape[1], first_shape[0], len(datasets)), dtype=np.float32)
    for index, ds in enumerate(datasets):
        pixels = np.asarray(ds.pixel_array)
        if pixels.shape != first_shape:
            raise ValueError(
                f"Decoded pixel shape at instance {index} is {pixels.shape}, "
                f"expected {first_shape}"
            )
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise ValueError(f"Invalid rescale parameters at instance {index}")
        volume[:, :, index] = (pixels.astype(np.float32) * slope + intercept).T

    affine_lps = np.eye(4, dtype=np.float64)
    # Array axes are [DICOM column, DICOM row, slice].
    affine_lps[:3, 0] = column_axis * first_spacing[1]
    affine_lps[:3, 1] = row_axis * first_spacing[0]
    affine_lps[:3, 2] = slice_axis * slice_spacing
    affine_lps[:3, 3] = positions_array[0]
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine = lps_to_ras @ affine_lps

    image = nib.Nifti1Image(volume, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm")
    return image, datasets


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
