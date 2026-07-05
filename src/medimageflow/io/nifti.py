"""NIfTI utilities backed by optional nibabel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medimageflow.utils.optional import require


def read_nifti(path: str | Path, *, dtype: Any = np.float32) -> tuple[NDArray[Any], NDArray[Any]]:
    """Read a NIfTI volume and its affine matrix.

    Args:
        path: Input ``.nii`` or ``.nii.gz`` path.
        dtype: NumPy dtype requested from nibabel.

    Returns:
        A tuple containing the image array and its 4-by-4 affine matrix.

    Raises:
        ImportError: If nibabel is not installed.
    """
    nib = require("nibabel", extra="imaging")
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(dtype=dtype)), np.asarray(image.affine)


def write_nifti(data: ArrayLike, affine: ArrayLike, path: str | Path) -> Path:
    """Write array data and an affine matrix to a NIfTI file.

    Args:
        data: Array-like image data.
        affine: A 4-by-4 voxel-to-world affine matrix.
        path: Output path ending in ``.nii`` or ``.nii.gz``.

    Returns:
        The output path.

    Raises:
        ImportError: If nibabel is not installed.
        ValueError: If the extension or affine shape is invalid.
    """
    nib = require("nibabel", extra="imaging")
    path = Path(path)
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError("path must end with .nii or .nii.gz")
    affine_array = np.asarray(affine)
    if affine_array.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4), got {affine_array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), affine_array), str(path))
    return path
