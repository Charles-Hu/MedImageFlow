"""Image similarity metrics backed by established imaging libraries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medimageflow.utils.optional import require


def _matching_arrays(
    prediction: ArrayLike, target: ArrayLike
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Convert two inputs to arrays and ensure that they have the same shape."""
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{prediction_array.shape} and {target_array.shape}"
        )
    return prediction_array, target_array


def _validated_mask(mask: ArrayLike | None, shape: tuple[int, ...]) -> NDArray[np.bool_] | None:
    """Return a nonempty boolean mask matching an image shape."""
    if mask is None:
        return None
    mask_array: NDArray[np.bool_] = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {mask_array.shape}")
    if not np.any(mask_array):
        raise ValueError("mask must contain at least one selected element")
    return mask_array


def _masked_arrays(
    prediction: ArrayLike, target: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return matching arrays, flattened to selected elements when masked."""
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    if mask_array is not None:
        return prediction_array[mask_array], target_array[mask_array]
    return prediction_array, target_array


def _masked_binary_arrays(
    prediction: ArrayLike, target: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return matching binary arrays with positions outside the mask cleared."""
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    prediction_binary = np.asarray(prediction_array, dtype=np.bool_)
    target_binary = np.asarray(target_array, dtype=np.bool_)
    if mask_array is not None:
        prediction_binary = np.logical_and(prediction_binary, mask_array)
        target_binary = np.logical_and(target_binary, mask_array)
    return prediction_binary, target_binary


def _selected_binary_arrays(
    prediction: ArrayLike, target: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return binary arrays containing only selected positions when masked."""
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    return (
        np.asarray(prediction_array, dtype=np.bool_),
        np.asarray(target_array, dtype=np.bool_),
    )


def dice(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    label: int = 1,
    mask: ArrayLike | None = None,
) -> float:
    """Return the Dice coefficient for one label in two segmentation arrays.

    This is a NumPy-facing wrapper around SimpleITK's label overlap metric.
    If the requested label is absent from both segmentations, SimpleITK's
    return-value convention is preserved.
    """
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    if mask_array is not None:
        sentinel = np.iinfo(np.int64).min
        if label == sentinel:
            sentinel = np.iinfo(np.int64).max
        prediction_array = np.where(mask_array, prediction_array, sentinel)
        target_array = np.where(mask_array, target_array, sentinel)
    sitk = require("SimpleITK", extra="imaging")
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(
        sitk.GetImageFromArray(prediction_array.astype(np.int64, copy=False)),
        sitk.GetImageFromArray(target_array.astype(np.int64, copy=False)),
    )
    return float(overlap.GetDiceCoefficient(int(label)))


def ssim(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float | None = None,
    mask: ArrayLike | None = None,
    **kwargs: Any,
) -> float:
    """Return structural similarity using ``skimage.metrics.structural_similarity``.

    Pass ``data_range`` explicitly for floating-point images. Additional keyword
    arguments are forwarded unchanged to scikit-image.
    """
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    metrics = require("skimage.metrics", extra="imaging")
    if mask_array is None:
        return float(
            metrics.structural_similarity(
                target_array, prediction_array, data_range=data_range, **kwargs
            )
        )
    if "full" in kwargs:
        raise TypeError("full is managed internally when mask is provided")
    _, similarity_map = metrics.structural_similarity(
        target_array, prediction_array, data_range=data_range, full=True, **kwargs
    )
    return float(np.mean(similarity_map[mask_array]))


def psnr(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float | None = None,
    mask: ArrayLike | None = None,
) -> float:
    """Return peak signal-to-noise ratio using scikit-image."""
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    metrics = require("skimage.metrics", extra="imaging")
    if mask_array is not None:
        prediction_array = prediction_array[mask_array]
        target_array = target_array[mask_array]
    return float(metrics.peak_signal_noise_ratio(target_array, prediction_array, data_range))


def gradient_mae(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
    mask: ArrayLike | None = None,
) -> float:
    """Return mean absolute error between Sobel gradient magnitudes.

    ``axis`` is forwarded to ``skimage.filters.sobel``. By default, all image
    axes contribute to the gradient magnitude.
    """
    prediction_array, target_array = _matching_arrays(prediction, target)
    mask_array = _validated_mask(mask, prediction_array.shape)
    filters = require("skimage.filters", extra="imaging")
    prediction_gradient = filters.sobel(prediction_array, axis=axis)
    target_gradient = filters.sobel(target_array, axis=axis)
    difference = np.abs(prediction_gradient - target_gradient)
    return float(np.mean(difference if mask_array is None else difference[mask_array]))


def gcc(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    epsilon: float = float(np.finfo(float).eps),
) -> float:
    """Return global normalized cross correlation.

    Identical nonconstant images approach one, uncorrelated images approach
    zero, and perfectly anticorrelated images approach negative one.
    """
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    prediction_float = np.asarray(prediction_array, dtype=np.float64)
    target_float = np.asarray(target_array, dtype=np.float64)

    prediction_mean = float(np.mean(prediction_float))
    target_mean = float(np.mean(target_float))
    cross = float(np.mean(prediction_float * target_float)) - prediction_mean * target_mean
    prediction_variance = float(np.mean(np.square(prediction_float))) - prediction_mean**2
    target_variance = float(np.mean(np.square(target_float))) - target_mean**2
    denominator = np.sqrt(max(prediction_variance, 0.0)) * np.sqrt(
        max(target_variance, 0.0)
    )
    correlation = cross / (denominator + epsilon)
    return float(correlation)


def _window_sum_3d(array: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Return centered 3D window sums using a cumulative-sum volume."""
    half_window = window // 2
    spatial_padding = ((half_window + 1, half_window),) * 3
    padded = np.pad(
        array,
        ((0, 0),) * (array.ndim - 3) + spatial_padding,
        mode="constant",
    )
    cumulative = np.cumsum(np.cumsum(np.cumsum(padded, axis=-3), axis=-2), axis=-1)
    x, y, z = array.shape[-3:]
    return (
        cumulative[..., window:, window:, window:]
        - cumulative[..., window:, window:, :z]
        - cumulative[..., window:, :y, window:]
        - cumulative[..., :x, window:, window:]
        + cumulative[..., window:, :y, :z]
        + cumulative[..., :x, window:, :z]
        + cumulative[..., :x, :y, window:]
        - cumulative[..., :x, :y, :z]
    )


def ncc(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    window: int = 21,
    epsilon: float = 1e-5,
    mask: ArrayLike | None = None,
) -> float:
    """Return mean squared local normalized cross correlation.

    Local sums over the final three dimensions are computed with cumulative
    sums. Higher values indicate stronger local correlation.
    """
    prediction_array, target_array = _matching_arrays(prediction, target)
    if prediction_array.ndim < 3:
        raise ValueError("ncc inputs must have at least three dimensions")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    mask_array = _validated_mask(mask, prediction_array.shape)
    prediction_float = np.asarray(prediction_array, dtype=np.float64)
    target_float = np.asarray(target_array, dtype=np.float64)

    prediction_sum = _window_sum_3d(prediction_float, window)
    target_sum = _window_sum_3d(target_float, window)
    prediction_square_sum = _window_sum_3d(np.square(prediction_float), window)
    target_square_sum = _window_sum_3d(np.square(target_float), window)
    product_sum = _window_sum_3d(prediction_float * target_float, window)
    window_size = float(window**3)

    prediction_mean = prediction_sum / window_size
    target_mean = target_sum / window_size
    cross = (
        product_sum
        - target_mean * prediction_sum
        - prediction_mean * target_sum
        + prediction_mean * target_mean * window_size
    )
    prediction_variance = (
        prediction_square_sum
        - 2.0 * prediction_mean * prediction_sum
        + np.square(prediction_mean) * window_size
    )
    target_variance = (
        target_square_sum
        - 2.0 * target_mean * target_sum
        + np.square(target_mean) * window_size
    )
    squared_correlation = np.square(cross) / (
        prediction_variance * target_variance + epsilon
    )
    selected = squared_correlation if mask_array is None else squared_correlation[mask_array]
    return float(np.mean(selected))


def _component_last_deformation(deformation: ArrayLike) -> NDArray[np.float64]:
    """Validate a 2D or 3D deformation and move its component axis last."""
    field = np.asarray(deformation, dtype=np.float64)
    if field.ndim not in (4, 5):
        raise ValueError("deformation must be a four- or five-dimensional array")
    components = field.ndim - 2
    if field.shape[-1] != components:
        if field.shape[1] != components:
            raise ValueError(
                f"deformation must have a component axis of length {components}"
            )
        field = np.moveaxis(field, 1, -1)
    return field


def _jacobian_determinant(deformation: ArrayLike) -> NDArray[np.float64]:
    """Return forward-difference Jacobian determinants for a deformation grid."""
    field = _component_last_deformation(deformation)

    field = (field + 1.0) / 2.0
    spatial_shape = field.shape[1:-1]
    scale = np.asarray(spatial_shape, dtype=np.float64).reshape(
        (1,) * (field.ndim - 1) + (len(spatial_shape),)
    )
    field = field * scale

    if field.ndim == 4:
        dy = field[:, 1:, :-1, :] - field[:, :-1, :-1, :]
        dx = field[:, :-1, 1:, :] - field[:, :-1, :-1, :]
        return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]

    dy = field[:, 1:, :-1, :-1, :] - field[:, :-1, :-1, :-1, :]
    dx = field[:, :-1, 1:, :-1, :] - field[:, :-1, :-1, :-1, :]
    dz = field[:, :-1, :-1, 1:, :] - field[:, :-1, :-1, :-1, :]

    determinant0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    determinant1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    determinant2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
    return determinant0 - determinant1 + determinant2


def negative_jacobian_percentage(
    deformation: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return the percentage of voxels with a negative Jacobian determinant.

    ``deformation`` may be a 2D or 3D field using component-last or
    component-first layout. A mask may match either the original batch/spatial
    shape or the forward-difference determinant shape.
    """
    field = _component_last_deformation(deformation)
    determinant = _jacobian_determinant(field)
    if mask is None:
        selected = determinant
    else:
        mask_array = np.asarray(mask, dtype=np.bool_)
        spatial_shape = field.shape[:-1]
        if mask_array.shape == spatial_shape:
            interior = (slice(None),) + (slice(None, -1),) * (field.ndim - 2)
            mask_array = mask_array[interior]
        elif mask_array.shape != determinant.shape:
            raise ValueError(
                "mask must match the deformation spatial shape "
                f"{spatial_shape} or determinant shape {determinant.shape}, "
                f"got {mask_array.shape}"
            )
        if not np.any(mask_array):
            raise ValueError("mask must contain at least one selected element")
        selected = determinant[mask_array]
    return float(np.mean(selected < 0.0) * 100.0)


def deformation_smoothness(
    deformation: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return first-order spatial smoothness energy of a deformation field.

    The metric is the sum, over spatial axes, of the mean squared forward
    difference of every vector component. Lower values indicate a smoother
    deformation field.
    """
    field = _component_last_deformation(deformation)
    spatial_shape = field.shape[:-1]
    mask_array = _validated_mask(mask, spatial_shape)
    smoothness = 0.0
    for spatial_axis in range(1, field.ndim - 1):
        lower = [slice(None)] * field.ndim
        upper = [slice(None)] * field.ndim
        lower[spatial_axis] = slice(None, -1)
        upper[spatial_axis] = slice(1, None)
        squared_difference = np.square(field[tuple(upper)] - field[tuple(lower)])
        if mask_array is None:
            smoothness += float(np.mean(squared_difference))
            continue

        mask_lower = [slice(None)] * mask_array.ndim
        mask_upper = [slice(None)] * mask_array.ndim
        mask_lower[spatial_axis] = slice(None, -1)
        mask_upper[spatial_axis] = slice(1, None)
        pair_mask = np.logical_and(
            mask_array[tuple(mask_lower)], mask_array[tuple(mask_upper)]
        )
        if not np.any(pair_mask):
            raise ValueError("mask must contain an adjacent pair along every spatial axis")
        smoothness += float(np.mean(squared_difference[pair_mask]))
    return smoothness


def deformation_magnitude(
    deformation: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return mean squared vector magnitude of a deformation field."""
    field = _component_last_deformation(deformation)
    mask_array = _validated_mask(mask, field.shape[:-1])
    squared_magnitude = np.sum(np.square(field), axis=-1)
    selected = squared_magnitude if mask_array is None else squared_magnitude[mask_array]
    return float(np.mean(selected))


def hd95(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    voxelspacing: float | tuple[float, ...] | None = None,
    connectivity: int = 1,
    mask: ArrayLike | None = None,
) -> float:
    """Return the 95th-percentile Hausdorff distance using MedPy.

    Inputs are interpreted as binary masks by MedPy. ``voxelspacing`` and
    ``connectivity`` are forwarded unchanged to ``medpy.metric.binary.hd95``.
    """
    prediction_array, target_array = _masked_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(
        binary_metrics.hd95(
            prediction_array,
            target_array,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
        )
    )


def assd(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    voxelspacing: float | tuple[float, ...] | None = None,
    connectivity: int = 1,
    mask: ArrayLike | None = None,
) -> float:
    """Return average symmetric surface distance using MedPy."""
    prediction_array, target_array = _masked_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(
        binary_metrics.assd(
            prediction_array,
            target_array,
            voxelspacing=voxelspacing,
            connectivity=connectivity,
        )
    )


def jaccard(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return the Jaccard coefficient (intersection over union) using MedPy."""
    prediction_array, target_array = _selected_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(binary_metrics.jc(prediction_array, target_array))


def iou(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return intersection over union; an alias interface for :func:`jaccard`."""
    return jaccard(prediction, target, mask=mask)


def sensitivity(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return binary sensitivity (recall) using MedPy."""
    prediction_array, target_array = _selected_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(binary_metrics.sensitivity(prediction_array, target_array))


def specificity(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return binary specificity using MedPy."""
    prediction_array, target_array = _selected_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(binary_metrics.specificity(prediction_array, target_array))


def precision(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return binary precision using MedPy."""
    prediction_array, target_array = _selected_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(binary_metrics.precision(prediction_array, target_array))


def ravd(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return relative absolute volume difference using MedPy."""
    prediction_array, target_array = _selected_binary_arrays(prediction, target, mask)
    binary_metrics = require("medpy.metric.binary", extra="imaging")
    return float(binary_metrics.ravd(prediction_array, target_array))


def mae(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return mean absolute error, optionally over selected elements only."""
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    return float(np.mean(np.abs(target_array - prediction_array)))


def mse(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Return mean squared error using scikit-image."""
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    metrics = require("skimage.metrics", extra="imaging")
    return float(metrics.mean_squared_error(target_array, prediction_array))


def nrmse(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    normalization: str = "euclidean",
    mask: ArrayLike | None = None,
) -> float:
    """Return normalized root mean squared error using scikit-image."""
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    metrics = require("skimage.metrics", extra="imaging")
    return float(
        metrics.normalized_root_mse(
            target_array, prediction_array, normalization=normalization
        )
    )


def nmi(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    bins: int | tuple[int, int] = 100,
    mask: ArrayLike | None = None,
) -> float:
    """Return normalized mutual information using scikit-image."""
    prediction_array, target_array = _masked_arrays(prediction, target, mask)
    metrics = require("skimage.metrics", extra="imaging")
    return float(metrics.normalized_mutual_information(target_array, prediction_array, bins=bins))


_PAIR_METRIC_NAMES = frozenset(
    {
        "assd",
        "dice",
        "gcc",
        "gradient_mae",
        "hd95",
        "iou",
        "jaccard",
        "mae",
        "mse",
        "ncc",
        "nmi",
        "nrmse",
        "precision",
        "psnr",
        "ravd",
        "sensitivity",
        "specificity",
        "ssim",
    }
)

_SINGLE_INPUT_METRIC_NAMES = frozenset(
    {
        "deformation_magnitude",
        "deformation_smoothness",
        "negative_jacobian_percentage",
    }
)

MetricCallable = Callable[..., Any]


def _resolve_metrics(
    metrics: str
    | MetricCallable
    | Sequence[str | MetricCallable]
    | Mapping[str, MetricCallable],
    supported_names: frozenset[str],
    *,
    kind: str,
) -> list[tuple[str, MetricCallable]]:
    """Resolve built-in metric names and user-provided callables."""
    if isinstance(metrics, Mapping):
        resolved = list(metrics.items())
        if any(not isinstance(name, str) or not name for name, _ in resolved):
            raise ValueError("custom metric names must be nonempty strings")
        if any(not callable(metric) for _, metric in resolved):
            raise TypeError("custom metrics must be callable")
    else:
        specs = [metrics] if isinstance(metrics, str) or callable(metrics) else list(metrics)
        resolved = []
        unsupported: list[str] = []
        for spec in specs:
            if isinstance(spec, str):
                if spec not in supported_names:
                    unsupported.append(spec)
                    continue
                resolved.append((spec, cast(MetricCallable, globals()[spec])))
            elif callable(spec):
                resolved.append((getattr(spec, "__name__", spec.__class__.__name__), spec))
            else:
                raise TypeError("metrics must be names, callables, or a name-to-callable mapping")
        if unsupported:
            raise ValueError(
                f"unsupported {kind}metrics: {sorted(unsupported)}; "
                f"supported metrics: {sorted(supported_names)}"
            )

    if not resolved:
        raise ValueError("at least one metric must be selected")
    names = [name for name, _ in resolved]
    if len(names) != len(set(names)):
        raise ValueError("metric names must be unique")
    return resolved


def _as_image_collection(value: ArrayLike | Sequence[ArrayLike]) -> tuple[list[ArrayLike], bool]:
    """Return image inputs as a list and whether the input is a collection."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        items = list(value)
        if not items:
            return [], True
        if items and all(isinstance(item, np.ndarray) for item in items):
            return items, True
    return [value], False


def _as_prediction_matrix(
    value: ArrayLike | Sequence[ArrayLike],
) -> list[list[ArrayLike]] | None:
    """Return a rectangular matrix of NumPy arrays when one is provided."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, np.ndarray)):
        return None
    rows = list(value)
    if not rows:
        return None
    matrix: list[list[ArrayLike]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, np.ndarray)):
            return None
        items = list(row)
        if not items or not all(isinstance(item, np.ndarray) for item in items):
            return None
        matrix.append(items)
    model_count = len(matrix[0])
    if any(len(row) != model_count for row in matrix):
        raise ValueError("two-dimensional prediction input must be rectangular")
    return matrix


def aggregate_metrics(
    prediction: ArrayLike | Sequence[ArrayLike],
    target: ArrayLike | Sequence[ArrayLike],
    metrics: str
    | MetricCallable
    | Sequence[str | MetricCallable]
    | Mapping[str, MetricCallable],
    *,
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    paired_t_test: bool = False,
    paired_t_test_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate selected metrics over one or more prediction/target pairs.

    A single input is paired with every item on the opposite collection. Two
    collections are paired positionally and must have equal lengths. A rectangular
    prediction matrix is interpreted as ``[patient][model]`` and paired with one
    target per patient. Collections should contain NumPy arrays.

    For prediction matrices, ``paired_t_test=True`` compares every pair of
    models across patients and every pair of patients across models using
    ``scipy.stats.ttest_rel``. Reported p-values are not corrected for multiple
    comparisons.
    """
    prediction_matrix = _as_prediction_matrix(prediction)
    targets, target_is_collection = _as_image_collection(target)
    if not targets:
        raise ValueError("prediction and target collections must not be empty")

    resolved_metrics = _resolve_metrics(metrics, _PAIR_METRIC_NAMES, kind="")
    metric_names = [name for name, _ in resolved_metrics]

    matrix_shape: tuple[int, int] | None = None
    if prediction_matrix is not None:
        if not target_is_collection or len(prediction_matrix) != len(targets):
            raise ValueError(
                "prediction matrix patient count must equal the target collection length"
            )
        matrix_shape = (len(prediction_matrix), len(prediction_matrix[0]))
        indexed_pairs = [
            (
                {
                    "patient_index": patient_index,
                    "model_index": model_index,
                    "target_index": patient_index,
                },
                prediction_item,
                targets[patient_index],
            )
            for patient_index, row in enumerate(prediction_matrix)
            for model_index, prediction_item in enumerate(row)
        ]
    else:
        predictions, prediction_is_collection = _as_image_collection(prediction)
        if not predictions:
            raise ValueError("prediction and target collections must not be empty")
        if prediction_is_collection and target_is_collection:
            if len(predictions) != len(targets):
                raise ValueError("prediction and target collections must have equal lengths")
            indexed_pairs = [
                (
                    {"prediction_index": index, "target_index": index},
                    prediction_item,
                    target_item,
                )
                for index, (prediction_item, target_item) in enumerate(
                    zip(predictions, targets)
                )
            ]
        elif prediction_is_collection:
            indexed_pairs = [
                ({"prediction_index": index, "target_index": 0}, prediction_item, targets[0])
                for index, prediction_item in enumerate(predictions)
            ]
        elif target_is_collection:
            indexed_pairs = [
                ({"prediction_index": 0, "target_index": index}, predictions[0], target_item)
                for index, target_item in enumerate(targets)
            ]
        else:
            indexed_pairs = [
                ({"prediction_index": 0, "target_index": 0}, predictions[0], targets[0])
            ]

    kwargs_by_metric = {} if metric_kwargs is None else metric_kwargs
    pair_results: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    matrix_values: dict[str, NDArray[np.float64]] = {}
    if matrix_shape is not None:
        matrix_values = {
            name: np.empty(matrix_shape, dtype=np.float64) for name in metric_names
        }
    for indices, prediction_item, target_item in indexed_pairs:
        values: dict[str, float] = {}
        for name, metric in resolved_metrics:
            kwargs = dict(kwargs_by_metric.get(name, {}))
            value = float(metric(prediction_item, target_item, **kwargs))
            values[name] = value
            values_by_metric[name].append(value)
            if matrix_shape is not None:
                matrix_values[name][indices["patient_index"], indices["model_index"]] = value
        pair_results.append({**indices, "metrics": values})

    means = {name: float(np.mean(values)) for name, values in values_by_metric.items()}
    standard_deviations = {
        name: float(np.std(values)) for name, values in values_by_metric.items()
    }
    result: dict[str, Any] = {
        "pairs": pair_results,
        "mean": means,
        "std": standard_deviations,
    }
    if matrix_shape is not None:
        patient_count, model_count = matrix_shape
        result["patient_mean"] = [
            {
                "patient_index": patient_index,
                "metrics": {
                    name: float(np.mean(values[patient_index, :]))
                    for name, values in matrix_values.items()
                },
            }
            for patient_index in range(patient_count)
        ]
        result["patient_std"] = [
            {
                "patient_index": patient_index,
                "metrics": {
                    name: float(np.std(values[patient_index, :]))
                    for name, values in matrix_values.items()
                },
            }
            for patient_index in range(patient_count)
        ]
        result["model_mean"] = [
            {
                "model_index": model_index,
                "metrics": {
                    name: float(np.mean(values[:, model_index]))
                    for name, values in matrix_values.items()
                },
            }
            for model_index in range(model_count)
        ]
        result["model_std"] = [
            {
                "model_index": model_index,
                "metrics": {
                    name: float(np.std(values[:, model_index]))
                    for name, values in matrix_values.items()
                },
            }
            for model_index in range(model_count)
        ]
        if paired_t_test:
            stats = require("scipy.stats", extra="imaging")
            test_kwargs = (
                {} if paired_t_test_kwargs is None else dict(paired_t_test_kwargs)
            )
            result["paired_t_test"] = {
                "models": [
                    {
                        "model_indices": (first, second),
                        "metrics": {
                            name: {
                                "statistic": float(test.statistic),
                                "pvalue": float(test.pvalue),
                            }
                            for name, values in matrix_values.items()
                            for test in [
                                stats.ttest_rel(
                                    values[:, first], values[:, second], **test_kwargs
                                )
                            ]
                        },
                    }
                    for first, second in combinations(range(model_count), 2)
                ],
                "patients": [
                    {
                        "patient_indices": (first, second),
                        "metrics": {
                            name: {
                                "statistic": float(test.statistic),
                                "pvalue": float(test.pvalue),
                            }
                            for name, values in matrix_values.items()
                            for test in [
                                stats.ttest_rel(
                                    values[first, :], values[second, :], **test_kwargs
                                )
                            ]
                        },
                    }
                    for first, second in combinations(range(patient_count), 2)
                ],
            }
    elif paired_t_test:
        raise ValueError("paired_t_test is only available for prediction matrices")
    return result


def aggregate_single_input_metrics(
    inputs: ArrayLike | Sequence[ArrayLike],
    metrics: str
    | MetricCallable
    | Sequence[str | MetricCallable]
    | Mapping[str, MetricCallable],
    *,
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate selected single-input metrics over one or more inputs."""
    input_items, _ = _as_image_collection(inputs)
    if not input_items:
        raise ValueError("input collection must not be empty")

    resolved_metrics = _resolve_metrics(
        metrics, _SINGLE_INPUT_METRIC_NAMES, kind="single-input "
    )
    metric_names = [name for name, _ in resolved_metrics]

    kwargs_by_metric = {} if metric_kwargs is None else metric_kwargs
    item_results: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for input_index, input_item in enumerate(input_items):
        values: dict[str, float] = {}
        for name, metric in resolved_metrics:
            kwargs = dict(kwargs_by_metric.get(name, {}))
            value = float(metric(input_item, **kwargs))
            values[name] = value
            values_by_metric[name].append(value)
        item_results.append({"input_index": input_index, "metrics": values})

    means = {name: float(np.mean(values)) for name, values in values_by_metric.items()}
    standard_deviations = {
        name: float(np.std(values)) for name, values in values_by_metric.items()
    }
    return {"items": item_results, "mean": means, "std": standard_deviations}


__all__ = [
    "aggregate_metrics",
    "aggregate_single_input_metrics",
    "assd",
    "deformation_magnitude",
    "deformation_smoothness",
    "dice",
    "gcc",
    "gradient_mae",
    "hd95",
    "iou",
    "jaccard",
    "mae",
    "mse",
    "ncc",
    "negative_jacobian_percentage",
    "nmi",
    "nrmse",
    "precision",
    "psnr",
    "ravd",
    "sensitivity",
    "specificity",
    "ssim",
]
