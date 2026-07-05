"""Dimension-agnostic 2D/3D patch sampling and extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Sequence as TypingSequence
from dataclasses import dataclass
from itertools import product
from typing import Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

SpatialShape = tuple[int, ...]
PatchLocation = tuple[int, ...]
PatchCenter = tuple[int, ...]
CenterRange = tuple[tuple[float, float], ...]
PatchSize = Optional[Union[int, TypingSequence[Optional[int]]]]


def normalize_size(value: int | Sequence[int], spatial_dims: int, *, name: str) -> SpatialShape:
    """Normalize a scalar or per-axis integer size.

    Args:
        value: Scalar size or one size per spatial axis.
        spatial_dims: Expected number of spatial dimensions.
        name: Parameter name included in validation errors.

    Returns:
        A validated size tuple.

    Raises:
        ValueError: If the number of values is wrong or any value is not positive.
    """
    result = (value,) * spatial_dims if isinstance(value, int) else tuple(value)
    if len(result) != spatial_dims or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain {spatial_dims} positive integers, got {result}")
    return result


def normalize_patch_spec(value: PatchSize, spatial_dims: int) -> tuple[int | None, ...]:
    """Normalize a patch specification where ``None`` preserves a whole axis.

    Args:
        value: Scalar, per-axis sequence, or ``None`` patch specification.
        spatial_dims: Expected number of spatial dimensions.

    Returns:
        A tuple containing positive integers or ``None`` for full axes.

    Raises:
        ValueError: If dimensionality is wrong or a numeric size is not positive.
    """
    result = (
        (value,) * spatial_dims
        if value is None or isinstance(value, int)
        else tuple(value)
    )
    if len(result) != spatial_dims:
        raise ValueError(f"patch_size must contain {spatial_dims} values, got {result}")
    if any(item is not None and item <= 0 for item in result):
        raise ValueError(f"patch_size values must be positive integers or None, got {result}")
    return result


def resolve_patch_size(value: PatchSize, shape: SpatialShape) -> SpatialShape:
    """Replace each full-axis ``None`` with the corresponding image size.

    Args:
        value: Patch-size specification.
        shape: Spatial image shape.

    Returns:
        A concrete positive patch size for every spatial axis.
    """
    spec = normalize_patch_spec(value, len(shape))
    return tuple(bound if size is None else size for size, bound in zip(spec, shape))


def full_patch_axes(value: PatchSize, spatial_dims: int) -> tuple[int, ...]:
    """Return axes whose original extent must be preserved.

    Args:
        value: Patch-size specification.
        spatial_dims: Number of spatial dimensions.

    Returns:
        Indices of axes represented by ``None``.
    """
    return tuple(
        axis for axis, size in enumerate(normalize_patch_spec(value, spatial_dims)) if size is None
    )


def spatial_axes(
    array: NDArray[object], spatial_dims: int, channel_axis: int | None
) -> tuple[int, ...]:
    """Resolve spatial axes for channel-free or single-channel-axis arrays.

    Args:
        array: Input image array.
        spatial_dims: Expected number of spatial dimensions.
        channel_axis: Channel axis, or ``None`` to use axis zero when present.

    Returns:
        Ordered spatial-axis indices.

    Raises:
        ValueError: If the array has an unsupported number of dimensions.
    """
    if array.ndim == spatial_dims:
        return tuple(range(array.ndim))
    if array.ndim != spatial_dims + 1:
        raise ValueError(
            f"Expected {spatial_dims}D spatial data with at most one channel axis; "
            f"received shape {array.shape}"
        )
    axis = 0 if channel_axis is None else channel_axis % array.ndim
    return tuple(index for index in range(array.ndim) if index != axis)


def get_spatial_shape(
    array: NDArray[object], spatial_dims: int, channel_axis: int | None
) -> SpatialShape:
    """Extract the spatial shape from an image array.

    Args:
        array: Input image array.
        spatial_dims: Expected number of spatial dimensions.
        channel_axis: Optional channel-axis index.

    Returns:
        The image shape excluding its channel axis.
    """
    return tuple(array.shape[axis] for axis in spatial_axes(array, spatial_dims, channel_axis))


def normalize_center_range(
    value: Sequence[float] | Sequence[Sequence[float]], spatial_dims: int
) -> CenterRange:
    """Normalize per-axis ``(before, after)`` center overrun fractions.

    A single pair is broadcast to all spatial axes. Fractions are relative to
    the patch size and describe how far a center may lie outside the image.

    Args:
        value: One range pair or one pair per spatial axis.
        spatial_dims: Number of spatial dimensions.

    Returns:
        One validated ``(before, after)`` pair per spatial axis.

    Raises:
        ValueError: If shape is invalid or a fraction is outside ``[0, 1]``.
    """
    raw = tuple(value)
    result: CenterRange
    if (
        len(raw) == 2
        and isinstance(raw[0], (int, float))
        and isinstance(raw[1], (int, float))
    ):
        pair = (float(raw[0]), float(raw[1]))
        result = (pair,) * spatial_dims
    else:
        parsed: list[tuple[float, float]] = []
        for item in raw:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError(
                    f"center_range must be one pair or {spatial_dims} "
                    "(before, after) pairs"
                )
            parsed.append((float(item[0]), float(item[1])))
        result = tuple(parsed)
    if len(result) != spatial_dims or any(len(pair) != 2 for pair in result):
        raise ValueError(
            f"center_range must be one pair or {spatial_dims} (before, after) pairs"
        )
    if any(number < 0 or number > 1 for pair in result for number in pair):
        raise ValueError("center_range fractions must be between 0 and 1")
    return result


def patch_start(center: PatchCenter, patch_size: SpatialShape) -> PatchLocation:
    """Convert a voxel center to its inclusive patch start coordinate.

    Args:
        center: Integer center coordinate for every spatial axis.
        patch_size: Concrete patch size.

    Returns:
        Inclusive patch start coordinates.
    """
    return tuple(coordinate - size // 2 for coordinate, size in zip(center, patch_size))


def extract_centered_patch(
    array: NDArray[object],
    center: PatchCenter,
    patch_size: SpatialShape,
    *,
    spatial_dims: int,
    channel_axis: int | None,
    padding_mode: str = "constant",
    padding_value: float = 0,
) -> NDArray[object]:
    """Extract around a center and pad out-of-image regions via ``numpy.pad``.

    Args:
        array: Image array to crop.
        center: Integer patch center in spatial-axis order.
        patch_size: Concrete patch size.
        spatial_dims: Number of spatial dimensions.
        channel_axis: Optional channel-axis index.
        padding_mode: Padding mode accepted by ``numpy.pad``.
        padding_value: Constant value used with constant padding.

    Returns:
        A patch with exactly ``patch_size`` spatial dimensions.

    Raises:
        ValueError: If center dimensionality is wrong or a non-constant mode
            would need to pad an axis with no source voxels.
    """
    axes = spatial_axes(array, spatial_dims, channel_axis)
    shape = tuple(array.shape[axis] for axis in axes)
    if len(center) != spatial_dims:
        raise ValueError(f"center must contain {spatial_dims} coordinates")
    location = patch_start(center, patch_size)
    slices: list[slice] = [slice(None)] * array.ndim
    pad_width: list[tuple[int, int]] = [(0, 0)] * array.ndim
    for axis, start, size, bound in zip(axes, location, patch_size, shape):
        stop = start + size
        slices[axis] = slice(max(start, 0), min(stop, bound))
        pad_width[axis] = (max(0, -start), max(0, stop - bound))
    cropped = array[tuple(slices)]
    if any(cropped.shape[axis] == 0 for axis in axes) and padding_mode != "constant":
        raise ValueError(
            f"padding_mode={padding_mode!r} cannot pad a patch with no source voxels; "
            "use constant padding or reduce center_range"
        )
    if padding_mode == "constant":
        return np.pad(
            cropped,
            pad_width,
            mode="constant",
            constant_values=padding_value,
        )
    return np.pad(cropped, pad_width, mode=padding_mode)


def extract_patch(
    array: NDArray[object],
    location: PatchLocation,
    patch_size: SpatialShape,
    *,
    spatial_dims: int,
    channel_axis: int | None,
) -> NDArray[object]:
    """Extract a fully in-bounds patch by its start coordinate.

    Kept as a strict low-level helper; center-based datasets use
    :func:`extract_centered_patch`.

    Args:
        array: Image array to crop.
        location: Inclusive patch start coordinate.
        patch_size: Concrete patch size.
        spatial_dims: Number of spatial dimensions.
        channel_axis: Optional channel-axis index.

    Returns:
        The requested in-bounds patch.

    Raises:
        ValueError: If any part of the patch lies outside the image.
    """
    center = tuple(start + size // 2 for start, size in zip(location, patch_size))
    shape = get_spatial_shape(array, spatial_dims, channel_axis)
    if any(
        start < 0 or start + size > bound
        for start, size, bound in zip(location, patch_size, shape)
    ):
        raise ValueError(f"Patch at {location} with size {patch_size} exceeds shape {shape}")
    return extract_centered_patch(
        array,
        center,
        patch_size,
        spatial_dims=spatial_dims,
        channel_axis=channel_axis,
    )


class PatchSampler(Protocol):
    """Interface for deterministic or stochastic patch coordinate strategies."""

    patch_size: PatchSize

    def centers(
        self,
        shape: SpatialShape,
        count: int,
        *,
        center_mask: NDArray[object] | None = None,
        seed_offset: int = 0,
    ) -> Iterable[PatchCenter]:
        """Yield patch centers for one spatial shape.

        Args:
            shape: Spatial image shape.
            count: Number of centers to yield.
            center_mask: Optional binary mask restricting center selection.
            seed_offset: Sample-specific random-seed offset.

        Yields:
            Integer patch center coordinates.
        """


@dataclass(frozen=True)
class RandomPatchSampler:
    """Uniformly sample patch centers, optionally inside a binary ROI.

    For an even patch size, the selected center is the first voxel on the
    positive side of the geometric center. For example, a size of four uses
    offsets ``[-2, -1, 0, 1]`` around its reported center.
    """

    patch_size: PatchSize
    seed: int | None = None
    replacement: bool = True
    center_range: Sequence[float] | Sequence[Sequence[float]] = (0.0, 0.0)

    def centers(
        self,
        shape: SpatialShape,
        count: int,
        *,
        center_mask: NDArray[object] | None = None,
        seed_offset: int = 0,
    ) -> Iterable[PatchCenter]:
        """Yield uniformly sampled patch centers.

        Args:
            shape: Spatial image shape.
            count: Number of centers to generate.
            center_mask: Optional binary ROI used to select centers.
            seed_offset: Offset combined with the configured base seed.

        Yields:
            Random integer patch center coordinates.

        Raises:
            ValueError: If count, mask, center range, or replacement constraints
                are invalid.
        """
        size = resolve_patch_size(self.patch_size, shape)
        full_axes = full_patch_axes(self.patch_size, len(shape))
        if count <= 0:
            raise ValueError("count must be positive")

        seed = None if self.seed is None else np.random.SeedSequence([self.seed, seed_offset])
        rng = np.random.default_rng(seed)
        centers: NDArray[np.int64]

        if center_mask is not None:
            mask = np.asarray(center_mask)
            if mask.shape != shape:
                raise ValueError(
                    f"center_mask shape {mask.shape} does not match image shape {shape}"
                )
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError("center_mask must be binary (only 0/False and 1/True)")
            if full_axes:
                reduced_mask = np.any(mask.astype(bool, copy=False), axis=full_axes)
                reduced_candidates = np.argwhere(reduced_mask)
                remaining_axes = tuple(axis for axis in range(len(shape)) if axis not in full_axes)
                centers = np.empty((len(reduced_candidates), len(shape)), dtype=np.int64)
                for axis in full_axes:
                    centers[:, axis] = shape[axis] // 2
                for column, axis in enumerate(remaining_axes):
                    centers[:, axis] = reduced_candidates[:, column]
                candidates = centers
            else:
                candidates = np.argwhere(mask.astype(bool, copy=False))
            if len(candidates) == 0:
                raise ValueError("center_mask contains no foreground voxel")
            if not self.replacement and count > len(candidates):
                raise ValueError(
                    f"Requested {count} unique centers, but ROI contains only {len(candidates)}"
                )
            selected = rng.choice(len(candidates), size=count, replace=self.replacement)
            centers = candidates[selected]
        else:
            ranges = normalize_center_range(self.center_range, len(shape))
            lower = tuple(
                image // 2 if axis in full_axes else int(np.ceil(-before * patch))
                for axis, (image, patch, (before, _)) in enumerate(zip(shape, size, ranges))
            )
            upper = tuple(
                image // 2
                if axis in full_axes
                else int(np.floor(image - 1 + after * patch))
                for axis, (image, patch, (_, after)) in enumerate(zip(shape, size, ranges))
            )
            candidate_shape = tuple(maximum - minimum + 1 for minimum, maximum in zip(lower, upper))
            candidate_count = int(np.prod(candidate_shape))
            if not self.replacement and count > candidate_count:
                raise ValueError(
                    f"Requested {count} unique centers, but only {candidate_count} are valid"
                )
            if self.replacement:
                centers = np.column_stack(
                    [
                        rng.integers(minimum, maximum + 1, size=count)
                        for minimum, maximum in zip(lower, upper)
                    ]
                )
            else:
                flat_indices = rng.choice(candidate_count, size=count, replace=False)
                centers = np.column_stack(np.unravel_index(flat_indices, candidate_shape))
                centers += np.asarray(lower)

        for center in centers:
            yield tuple(int(coordinate) for coordinate in center)


@dataclass(frozen=True)
class GridPatchSampler:
    """Regular grid sampling that always includes each image boundary."""

    patch_size: PatchSize
    stride: int | Sequence[int] | None = None

    def centers(
        self,
        shape: SpatialShape,
        count: int,
        *,
        center_mask: NDArray[object] | None = None,
        seed_offset: int = 0,
    ) -> Iterable[PatchCenter]:
        """Yield deterministic centers from a regular in-bounds grid.

        Args:
            shape: Spatial image shape.
            count: Maximum number of centers to yield.
            center_mask: Ignored; accepted for sampler interface compatibility.
            seed_offset: Ignored; accepted for sampler interface compatibility.

        Yields:
            Integer grid center coordinates.

        Raises:
            ValueError: If patch size exceeds the image or stride is invalid.
        """
        del center_mask, seed_offset
        size = resolve_patch_size(self.patch_size, shape)
        stride = normalize_size(self.stride or size, len(shape), name="stride")
        if any(patch > image for patch, image in zip(size, shape)):
            raise ValueError(f"patch_size {size} exceeds image shape {shape}")
        per_axis: list[list[int]] = []
        for image, patch, step in zip(shape, size, stride):
            starts = list(range(0, image - patch + 1, step))
            boundary = image - patch
            if starts[-1] != boundary:
                starts.append(boundary)
            per_axis.append(starts)
        coordinates = product(*per_axis)
        for index, location in enumerate(coordinates):
            if index >= count:
                return
            yield tuple(start + patch // 2 for start, patch in zip(location, size))
