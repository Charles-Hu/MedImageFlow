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


def _grid_shape(grid_shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    values = (grid_shape,) * ndim if isinstance(grid_shape, int) else tuple(grid_shape)
    if len(values) != ndim:
        raise ValueError(f"grid_shape must contain {ndim} values, got {len(values)}")
    if any(value < 2 for value in values):
        raise ValueError("grid_shape values must be at least 2")
    return values


def _bilinear_sample(
    array: NDArray[Any], rows: NDArray[np.float64], columns: NDArray[np.float64]
) -> NDArray[Any]:
    """Sample a 2D array on a rectilinear grid using bilinear interpolation."""
    row_floor = np.floor(rows).astype(int)
    column_floor = np.floor(columns).astype(int)
    row_ceil = np.minimum(row_floor + 1, array.shape[0] - 1)
    column_ceil = np.minimum(column_floor + 1, array.shape[1] - 1)
    row_weight = (rows - row_floor)[:, None]
    column_weight = (columns - column_floor)[None, :]
    top = (
        array[row_floor[:, None], column_floor[None, :]] * (1.0 - column_weight)
        + array[row_floor[:, None], column_ceil[None, :]] * column_weight
    )
    bottom = (
        array[row_ceil[:, None], column_floor[None, :]] * (1.0 - column_weight)
        + array[row_ceil[:, None], column_ceil[None, :]] * column_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight


def _plot_deformed_grid(
    row_displacement: NDArray[Any],
    column_displacement: NDArray[Any],
    axis: Any,
    line_collection: Any,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
    color: str,
    linewidth: float,
) -> None:
    rows = np.linspace(0.0, row_displacement.shape[0] - 1, grid_shape[0])
    columns = np.linspace(0.0, row_displacement.shape[1] - 1, grid_shape[1])
    grid_x, grid_y = np.meshgrid(columns * spacing[1], rows * spacing[0])
    distorted_x = grid_x + _bilinear_sample(column_displacement, rows, columns) * spacing[1]
    distorted_y = grid_y + _bilinear_sample(row_displacement, rows, columns) * spacing[0]
    horizontal_lines = np.stack((distorted_x, distorted_y), axis=2)
    vertical_lines = horizontal_lines.transpose(1, 0, 2)
    axis.add_collection(line_collection(horizontal_lines, color=color, linewidth=linewidth))
    axis.add_collection(line_collection(vertical_lines, color=color, linewidth=linewidth))
    axis.autoscale()
    axis.set_aspect("equal")
    axis.invert_yaxis()


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


def registration_field_visualization(
    field: ArrayLike,
    spacing: Sequence[float] | None = None,
    *,
    component_axis: int = 0,
    grid_shape: int | Sequence[int] = 20,
    slice_indices: Sequence[int] | None = None,
    figure_name: str | int | None = None,
    color: str = "C0",
    linewidth: float = 1.0,
    hide_axes: bool = True,
    show: bool = True,
) -> tuple[Any, NDArray[Any]]:
    """Display a 2D or 3D registration field as deformed grids.

    A 2D field produces one grid. A 3D field produces grids on the three
    central orthogonal planes. Displacement components follow array-axis order;
    for example, a component-first 3D field has shape ``(3, D, H, W)``.
    Field values are interpreted in voxel units and scaled by ``spacing`` when
    physical spacing is supplied.

    Args:
        field: Component-first or component-last 2D/3D displacement field.
        spacing: Optional spacing for each spatial axis.
        component_axis: Axis containing the two or three displacement components.
        grid_shape: Number of sample points along each spatial axis. An integer
            applies the same density to every axis.
        slice_indices: Optional slice indices for the three orthogonal planes,
            in spatial-axis order. By default, every plane uses its center.
        figure_name: Optional Matplotlib figure name or number.
        color: Grid line color.
        linewidth: Grid line width.
        hide_axes: Whether to hide ticks and axis frames.
        show: Whether to call :func:`matplotlib.pyplot.show`.

    Returns:
        The Matplotlib figure and a one-dimensional array of axes.
    """
    array = np.asarray(field)
    if array.ndim not in (3, 4):
        raise ValueError(f"field must be a 2D or 3D displacement field, got {array.shape}")
    if component_axis < -array.ndim or component_axis >= array.ndim:
        raise ValueError(
            f"component_axis {component_axis} is invalid for shape {array.shape}"
        )
    components_first = np.moveaxis(array, component_axis, 0)
    spatial_ndim = components_first.ndim - 1
    if components_first.shape[0] != spatial_ndim:
        raise ValueError(
            f"field must have {spatial_ndim} components, got {components_first.shape[0]}"
        )

    physical_spacing = _spacing(spacing, spatial_ndim) or (1.0,) * spatial_ndim
    samples = _grid_shape(grid_shape, spatial_ndim)
    if spatial_ndim == 2 and slice_indices is not None:
        raise ValueError("slice_indices is only supported for 3D fields")
    plt = require("matplotlib.pyplot", extra="visualization")
    collections = require("matplotlib.collections", extra="visualization")

    if spatial_ndim == 2:
        figure, axis = plt.subplots(1, 1, num=figure_name)
        axes = np.asarray([axis], dtype=object)
        _plot_deformed_grid(
            components_first[0],
            components_first[1],
            axis,
            collections.LineCollection,
            (samples[0], samples[1]),
            (physical_spacing[0], physical_spacing[1]),
            color,
            linewidth,
        )
    else:
        figure, raw_axes = plt.subplots(1, 3, num=figure_name, figsize=(15, 5))
        axes = np.asarray(raw_axes, dtype=object).reshape(-1)
        spatial_shape = components_first.shape[1:]
        if slice_indices is None:
            positions = tuple(size // 2 for size in spatial_shape)
        else:
            positions = tuple(slice_indices)
            if len(positions) != 3:
                raise ValueError(
                    f"slice_indices must contain 3 values, got {len(positions)}"
                )
            for spatial_axis, (position, size) in enumerate(zip(positions, spatial_shape)):
                if not isinstance(position, (int, np.integer)):
                    raise TypeError("slice_indices values must be integers")
                if position < 0 or position >= size:
                    raise ValueError(
                        f"slice index {position} is out of bounds for axis "
                        f"{spatial_axis} with size {size}"
                    )
        dimensions = ((1, 2), (0, 2), (0, 1))
        titles = ("Axial (axis 0)", "Coronal (axis 1)", "Sagittal (axis 2)")
        for normal, (axis, dims, title) in enumerate(zip(axes, dimensions, titles)):
            index = [slice(None)] * 4
            index[normal + 1] = positions[normal]
            plane = components_first[tuple(index)]
            _plot_deformed_grid(
                plane[dims[0]],
                plane[dims[1]],
                axis,
                collections.LineCollection,
                (samples[dims[0]], samples[dims[1]]),
                (physical_spacing[dims[0]], physical_spacing[dims[1]]),
                color,
                linewidth,
            )
            axis.set_title(title)

    for axis in axes:
        if hide_axes:
            axis.axis("off")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


__all__ = ["mip_visualization", "registration_field_visualization", "visualization"]
