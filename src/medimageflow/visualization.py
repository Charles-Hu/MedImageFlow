"""Utilities for displaying two- and three-dimensional images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medimageflow.utils.optional import require


def _spacing(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...] | None:
    if spacing is None:
        return None
    values = tuple(float(value) for value in spacing)
    if len(values) != ndim:
        raise ValueError(f"spacing must contain {ndim} values, got {len(values)}")
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("spacing values must be finite and greater than zero")
    return values


def _extent(shape: tuple[int, int], spacing: Sequence[float] | None) -> Any:
    if spacing is None:
        return None
    height, width = shape
    return (0.0, width * spacing[1], height * spacing[0], 0.0)


def visualization(
    image: ArrayLike,
    spacing: Sequence[float] | None = None,
    *,
    figure_name: str | int | None = None,
    cmap: str = "gray",
    show: bool = True,
) -> tuple[Any, NDArray[Any]]:
    """Display a 2D image or the three central views of a 3D image.

    Spatial dimensions use array order. Thus 2D spacing is ``(row, column)``
    and 3D spacing is ``(axis_0, axis_1, axis_2)``. Channels-last RGB and RGBA
    images are supported for both 2D images and 3D volumes.

    Args:
        image: ``(H, W)``, ``(H, W, 3/4)``, ``(D, H, W)``, or
            ``(D, H, W, 3/4)`` image data.
        spacing: Optional physical spacing for each spatial dimension. When
            omitted, axes use pixel/voxel coordinates.
        figure_name: Optional Matplotlib figure name or number. String values
            are also used as the interactive window title.
        cmap: Matplotlib colormap used for scalar images.
        show: Whether to call :func:`matplotlib.pyplot.show`.

    Returns:
        The Matplotlib figure and a one-dimensional array of axes.
    """
    array = np.asarray(image)
    is_color = array.ndim >= 3 and array.shape[-1] in (3, 4)
    spatial_ndim = array.ndim - int(is_color)
    if spatial_ndim not in (2, 3):
        raise ValueError(
            "image must be a 2D image or 3D volume, optionally with 3 or 4 "
            f"channels last; got shape {array.shape}"
        )

    physical_spacing = _spacing(spacing, spatial_ndim)
    plt = require("matplotlib.pyplot", extra="visualization")

    if spatial_ndim == 2:
        figure, axis = plt.subplots(1, 1, num=figure_name)
        kwargs = {} if is_color else {"cmap": cmap}
        axis.imshow(
            array,
            extent=_extent(array.shape[:2], physical_spacing),
            **kwargs,
        )
        axes = np.asarray([axis], dtype=object)
    else:
        figure, raw_axes = plt.subplots(1, 3, num=figure_name, figsize=(15, 5))
        axes = np.asarray(raw_axes, dtype=object).reshape(-1)
        centers = tuple(size // 2 for size in array.shape[:3])
        slices = (array[centers[0]], array[:, centers[1]], array[:, :, centers[2]])
        dimensions = ((1, 2), (0, 2), (0, 1))
        titles = ("Axial (axis 0)", "Coronal (axis 1)", "Sagittal (axis 2)")
        for axis, view, dims, title in zip(axes, slices, dimensions, titles):
            view_spacing = (
                None
                if physical_spacing is None
                else (physical_spacing[dims[0]], physical_spacing[dims[1]])
            )
            kwargs = {} if is_color else {"cmap": cmap}
            axis.imshow(view, extent=_extent(view.shape[:2], view_spacing), **kwargs)
            axis.set_title(title)
        figure.tight_layout()

    if show:
        plt.show()
    return figure, axes


def mip_visualization(
    image: ArrayLike,
    spacing: Sequence[float] | None = None,
    *,
    figure_name: str | int | None = None,
    cmap: str = "gray",
    show: bool = True,
) -> tuple[Any, NDArray[Any]]:
    """Display three maximum-intensity projections of a 3D image.

    The projections reduce spatial axes 0, 1, and 2 respectively. Spatial
    dimensions and spacing use array order. Channels-last RGB and RGBA volumes
    are supported by applying the maximum independently to each channel.

    Args:
        image: ``(D, H, W)`` or ``(D, H, W, 3/4)`` image data.
        spacing: Optional physical spacing in ``(axis_0, axis_1, axis_2)``
            order. When omitted, axes use pixel/voxel coordinates.
        figure_name: Optional Matplotlib figure name or number. String values
            are also used as the interactive window title.
        cmap: Matplotlib colormap used for scalar images.
        show: Whether to call :func:`matplotlib.pyplot.show`.

    Returns:
        The Matplotlib figure and a one-dimensional array of three axes.
    """
    array = np.asarray(image)
    is_color = array.ndim == 4 and array.shape[-1] in (3, 4)
    if not (array.ndim == 3 or is_color):
        raise ValueError(
            "image must be a 3D volume, optionally with 3 or 4 channels last; "
            f"got shape {array.shape}"
        )

    physical_spacing = _spacing(spacing, 3)
    plt = require("matplotlib.pyplot", extra="visualization")
    figure, raw_axes = plt.subplots(1, 3, num=figure_name, figsize=(15, 5))
    axes = np.asarray(raw_axes, dtype=object).reshape(-1)
    projections = tuple(np.max(array, axis=axis) for axis in range(3))
    dimensions = ((1, 2), (0, 2), (0, 1))
    titles = ("Axial MIP (axis 0)", "Coronal MIP (axis 1)", "Sagittal MIP (axis 2)")

    for axis, view, dims, title in zip(axes, projections, dimensions, titles):
        view_spacing = (
            None
            if physical_spacing is None
            else (physical_spacing[dims[0]], physical_spacing[dims[1]])
        )
        kwargs = {} if is_color else {"cmap": cmap}
        axis.imshow(view, extent=_extent(view.shape[:2], view_spacing), **kwargs)
        axis.set_title(title)

    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


__all__ = ["mip_visualization", "visualization"]
