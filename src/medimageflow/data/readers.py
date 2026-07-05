"""Path-based image readers that materialize complete NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from medimageflow.io import read_dicom_series, read_nifti
from medimageflow.utils.optional import require


@runtime_checkable
class ImageReader(Protocol):
    """Read one supported image path into a complete NumPy array."""

    def supports(self, path: Path) -> bool:
        """Return whether this reader accepts a path.

        Args:
            path: Candidate image path.

        Returns:
            Whether the reader supports the path.
        """

    def read(self, path: Path) -> NDArray[object]:
        """Read a complete image.

        Args:
            path: Supported image path.

        Returns:
            Materialized image array.
        """


class NiftiReader:
    """Read ``.nii`` and ``.nii.gz`` files with nibabel."""

    def supports(self, path: Path) -> bool:
        """Check whether a path has a NIfTI extension.

        Args:
            path: Candidate image path.

        Returns:
            Whether the path ends in ``.nii`` or ``.nii.gz``.
        """
        return path.name.endswith((".nii", ".nii.gz"))

    def read(self, path: Path) -> NDArray[object]:
        """Read a complete NIfTI volume.

        Args:
            path: NIfTI file path.

        Returns:
            Image data as a NumPy array.
        """
        array, _ = read_nifti(path)
        return array


class NumpyReader:
    """Read one complete array from a ``.npy`` file."""

    def supports(self, path: Path) -> bool:
        """Check whether a path is a NumPy array file.

        Args:
            path: Candidate image path.

        Returns:
            Whether the path ends in ``.npy``.
        """
        return path.suffix == ".npy"

    def read(self, path: Path) -> NDArray[object]:
        """Read a complete ``.npy`` array.

        Args:
            path: NumPy array file path.

        Returns:
            Loaded NumPy array.
        """
        return np.asarray(np.load(path, allow_pickle=False))


class DicomSeriesReader:
    """Read a DICOM series directory with SimpleITK."""

    def supports(self, path: Path) -> bool:
        """Check whether a path is a candidate DICOM series directory.

        Args:
            path: Candidate image path.

        Returns:
            Whether the path is a directory.
        """
        return path.is_dir()

    def read(self, path: Path) -> NDArray[object]:
        """Read a complete DICOM series directory.

        Args:
            path: Directory containing one DICOM series.

        Returns:
            Image data in SimpleITK array order.

        Raises:
            ImportError: If SimpleITK is not installed.
        """
        sitk = require("SimpleITK", extra="imaging")
        return np.asarray(sitk.GetArrayFromImage(read_dicom_series(path)))


class ReaderRegistry:
    """Select custom readers first, followed by built-in path readers."""

    def __init__(self, readers: Iterable[ImageReader] = ()) -> None:
        """Initialize an ordered image-reader registry.

        Args:
            readers: Custom readers placed before built-in readers.
        """
        self._readers = [*readers, NiftiReader(), NumpyReader(), DicomSeriesReader()]

    def read(self, value: str | Path) -> NDArray[object]:
        """Select a reader and materialize an image path.

        Args:
            value: Image path as a string or ``Path``.

        Returns:
            Complete image as a NumPy array.

        Raises:
            FileNotFoundError: If the path does not exist.
            TypeError: If a reader returns a non-NumPy value.
            ValueError: If no registered reader supports the path.
        """
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        for reader in self._readers:
            if reader.supports(path):
                array = reader.read(path)
                if not isinstance(array, np.ndarray):
                    raise TypeError(
                        f"ImageReader {type(reader).__name__} must return numpy.ndarray"
                    )
                return array
        raise ValueError(f"No ImageReader supports path: {path}")
