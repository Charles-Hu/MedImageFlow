from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from medimageflow import metrics


def test_metrics_reject_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        metrics.psnr(np.zeros((2, 2)), np.zeros((3, 3)))


def test_metrics_reject_invalid_masks() -> None:
    image = np.zeros((2, 2))
    with pytest.raises(ValueError, match="mask must have shape"):
        metrics.gradient_mae(image, image, mask=np.ones(3))
    with pytest.raises(ValueError, match="at least one"):
        metrics.dice(image, image, mask=np.zeros_like(image))


def test_ssim_and_psnr_wrap_skimage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def structural_similarity(
        target: np.ndarray, prediction: np.ndarray, **kwargs: Any
    ) -> float | tuple[float, np.ndarray]:
        calls["ssim"] = (target, prediction, kwargs)
        if kwargs.get("full"):
            return 0.75, np.arange(target.size).reshape(target.shape)
        return 0.75

    def peak_signal_noise_ratio(
        target: np.ndarray, prediction: np.ndarray, data_range: float | None
    ) -> float:
        calls["psnr"] = (target, prediction, data_range)
        return 32.0

    fake = SimpleNamespace(
        structural_similarity=structural_similarity,
        peak_signal_noise_ratio=peak_signal_noise_ratio,
    )
    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: fake)
    prediction = np.ones((8, 8))
    target = np.zeros((8, 8))

    assert metrics.ssim(prediction, target, data_range=1.0, win_size=3) == 0.75
    assert metrics.psnr(prediction, target, data_range=1.0) == 32.0
    assert calls["ssim"][2] == {"data_range": 1.0, "win_size": 3}
    assert calls["psnr"][2] == 1.0


def test_ssim_averages_only_masked_similarity_map(monkeypatch: pytest.MonkeyPatch) -> None:
    similarity_map = np.arange(64, dtype=float).reshape(8, 8)
    fake = SimpleNamespace(
        structural_similarity=lambda *args, **kwargs: (0.5, similarity_map),
    )
    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: fake)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, :2] = True

    result = metrics.ssim(np.zeros((8, 8)), np.zeros((8, 8)), data_range=1, mask=mask)

    assert result == 0.5


def test_gradient_mae_uses_sobel_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(sobel=lambda array, axis=None: np.asarray(array) * 2)
    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: fake)

    result = metrics.gradient_mae(
        np.array([0.0, 2.0]), np.array([0.0, 1.0]), mask=np.array([False, True])
    )

    assert result == 2.0


def test_gcc_matches_global_normalized_cross_correlation() -> None:
    target = np.array([1.0, 2.0, 3.0, 100.0])
    mask = np.array([True, True, True, False])

    assert metrics.gcc(target, target, mask=mask) == pytest.approx(1.0, abs=1e-12)
    assert metrics.gcc(-target, target, mask=mask) == pytest.approx(-1.0, abs=1e-12)


def test_gcc_constant_images_are_finite() -> None:
    assert metrics.gcc(np.ones(4), np.ones(4)) == 0.0


def test_ncc_identical_images_have_unit_similarity() -> None:
    image = np.arange(27, dtype=float).reshape(3, 3, 3)

    assert metrics.ncc(image, image, window=3) == pytest.approx(1.0, abs=1e-6)
    assert metrics.ncc(-image, image, window=3) == pytest.approx(1.0, abs=1e-6)


def test_ncc_supports_leading_dimensions_and_mask() -> None:
    target = np.arange(27, dtype=float).reshape(1, 1, 3, 3, 3)
    prediction = target.copy()
    prediction[..., 0, 0, 0] += 10
    mask = np.zeros_like(target, dtype=bool)
    mask[..., 2, 2, 2] = True

    result = metrics.ncc(prediction, target, window=3, mask=mask)

    assert np.isfinite(result)


@pytest.mark.parametrize("window", [0, 2])
def test_ncc_rejects_invalid_windows(window: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        metrics.ncc(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), window=window)


def _identity_deformation(size: int = 3) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, size)
    depth, height, width = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return np.stack((width, height, depth), axis=-1)[None, ...]


def test_negative_jacobian_percentage_from_grid_for_identity_and_fold() -> None:
    identity = _identity_deformation()
    folded = identity.copy()
    folded[..., 0] *= -1.0

    assert metrics.negative_jacobian_percentage_from_grid(identity) == 0.0
    assert metrics.negative_jacobian_percentage_from_grid(folded) == 100.0


def test_negative_jacobian_from_grid_supports_component_first_and_mask() -> None:
    folded = _identity_deformation(size=4)
    folded[..., 0] *= -1.0
    component_first = np.moveaxis(folded, -1, 1)
    mask = np.zeros((1, 4, 4, 4), dtype=bool)
    mask[:, :-1, :-1, :-1] = True

    result = metrics.negative_jacobian_percentage_from_grid(component_first, mask=mask)

    assert result == 100.0


def test_negative_jacobian_rejects_invalid_mask() -> None:
    with pytest.raises(ValueError, match="mask must match"):
        metrics.negative_jacobian_percentage_from_grid(
            _identity_deformation(), mask=np.ones((2, 2), dtype=bool)
        )


def _identity_deformation_2d(size: int = 4) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, size)
    height, width = np.meshgrid(coordinates, coordinates, indexing="ij")
    return np.stack((width, height), axis=-1)[None, ...]


def test_negative_jacobian_percentage_from_grid_supports_2d_deformations() -> None:
    identity = _identity_deformation_2d()
    folded = identity.copy()
    folded[..., 0] *= -1.0

    assert metrics.negative_jacobian_percentage_from_grid(identity) == 0.0
    assert metrics.negative_jacobian_percentage_from_grid(folded) == 100.0
    assert (
        metrics.negative_jacobian_percentage_from_grid(np.moveaxis(folded, -1, 1))
        == 100.0
    )


def test_negative_jacobian_percentage_from_grid_supports_2d_mask() -> None:
    folded = _identity_deformation_2d()
    folded[..., 0] *= -1.0
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[:, :-1, :-1] = True

    assert metrics.negative_jacobian_percentage_from_grid(folded, mask=mask) == 100.0


def test_negative_jacobian_percentage_from_displacement_for_identity_and_fold() -> None:
    identity_displacement = np.zeros((1, 4, 4, 2), dtype=float)
    folded_displacement = identity_displacement.copy()
    folded_displacement[..., 0] = -2.0 * np.arange(4)[None, None, :]

    assert (
        metrics.negative_jacobian_percentage_from_displacement(identity_displacement)
        == 0.0
    )
    assert (
        metrics.negative_jacobian_percentage_from_displacement(folded_displacement)
        == 100.0
    )
    assert (
        metrics.negative_jacobian_percentage_from_displacement(
            np.moveaxis(folded_displacement, -1, 1)
        )
        == 100.0
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_deformation_regularization_metrics_for_constant_field(dimensions: int) -> None:
    spatial_shape = (4,) * dimensions
    vector = np.arange(1, dimensions + 1, dtype=float)
    field = np.broadcast_to(vector, (1, *spatial_shape, dimensions))

    assert metrics.deformation_smoothness(field) == 0.0
    assert metrics.deformation_magnitude(field) == float(np.sum(np.square(vector)))


def test_deformation_smoothness_matches_axis_difference_sum() -> None:
    field = np.zeros((1, 3, 3, 2), dtype=float)
    field[..., 0] = np.arange(3)[None, :, None]
    field[..., 1] = 2.0 * np.arange(3)[None, None, :]

    result = metrics.deformation_smoothness(field)

    assert result == 2.5


def test_deformation_regularization_metrics_apply_mask() -> None:
    field = np.zeros((1, 3, 3, 2), dtype=float)
    field[:, 2, 2, :] = 100.0
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[:, :2, :2] = True

    assert metrics.deformation_smoothness(field, mask=mask) == 0.0
    assert metrics.deformation_magnitude(field, mask=mask) == 0.0


def test_dice_wraps_simpleitk(monkeypatch: pytest.MonkeyPatch) -> None:
    class Overlap:
        def Execute(self, prediction: np.ndarray, target: np.ndarray) -> None:
            self.prediction = prediction
            self.target = target
            fake.last_prediction = prediction

        def GetDiceCoefficient(self, label: int) -> float:
            assert label == 2
            assert self.prediction.dtype == np.int64
            return 0.8

    fake = SimpleNamespace(
        LabelOverlapMeasuresImageFilter=Overlap,
        GetImageFromArray=lambda array: array,
    )
    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: fake)

    assert metrics.dice([[0, 2]], [[2, 2]], label=2, mask=[[False, True]]) == 0.8
    assert fake.last_prediction[0, 0] != 0


def test_hd95_wraps_medpy_and_applies_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def hd95(
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        voxelspacing: tuple[float, ...] | None,
        connectivity: int,
    ) -> float:
        calls["arguments"] = (prediction, target, voxelspacing, connectivity)
        return 1.25

    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: SimpleNamespace(hd95=hd95))
    prediction = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 1]])
    mask = np.array([[False, True], [True, True]])

    result = metrics.hd95(
        prediction, target, voxelspacing=(0.5, 2.0), connectivity=2, mask=mask
    )

    assert result == 1.25
    np.testing.assert_array_equal(calls["arguments"][0], [[False, True], [False, False]])
    np.testing.assert_array_equal(calls["arguments"][1], [[False, False], [False, True]])
    assert calls["arguments"][2:] == ((0.5, 2.0), 2)


def test_assd_wraps_medpy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def assd(*args: Any, **kwargs: Any) -> float:
        calls["arguments"] = (args, kwargs)
        return 2.5

    monkeypatch.setattr(
        metrics, "require", lambda *args, **kwargs: SimpleNamespace(assd=assd)
    )

    result = metrics.assd([[1, 0]], [[0, 1]], voxelspacing=0.5, connectivity=2)

    assert result == 2.5
    assert calls["arguments"][1] == {"voxelspacing": 0.5, "connectivity": 2}


@pytest.mark.parametrize(
    ("metric_name", "backend_name"),
    [
        ("jaccard", "jc"),
        ("iou", "jc"),
        ("sensitivity", "sensitivity"),
        ("specificity", "specificity"),
        ("precision", "precision"),
        ("ravd", "ravd"),
    ],
)
def test_binary_metrics_wrap_medpy_and_select_mask(
    monkeypatch: pytest.MonkeyPatch, metric_name: str, backend_name: str
) -> None:
    calls: dict[str, Any] = {}

    def backend(prediction: np.ndarray, target: np.ndarray) -> float:
        calls["arrays"] = (prediction, target)
        return 0.6

    monkeypatch.setattr(
        metrics,
        "require",
        lambda *args, **kwargs: SimpleNamespace(**{backend_name: backend}),
    )
    metric = getattr(metrics, metric_name)

    result = metric(
        [[1, 0], [1, 0]],
        [[1, 1], [0, 0]],
        mask=[[False, True], [True, False]],
    )

    assert result == 0.6
    np.testing.assert_array_equal(calls["arrays"][0], [False, True])
    np.testing.assert_array_equal(calls["arrays"][1], [True, False])


def test_intensity_error_metrics_apply_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def mean_squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
        calls["mse"] = (target, prediction)
        return 4.0

    def normalized_root_mse(
        target: np.ndarray, prediction: np.ndarray, *, normalization: str
    ) -> float:
        calls["nrmse"] = (target, prediction, normalization)
        return 0.4

    fake = SimpleNamespace(
        mean_squared_error=mean_squared_error,
        normalized_root_mse=normalized_root_mse,
    )
    monkeypatch.setattr(metrics, "require", lambda *args, **kwargs: fake)
    prediction = np.array([1.0, 20.0, 5.0])
    target = np.array([3.0, 0.0, 5.0])
    mask = np.array([True, False, True])

    assert metrics.mae(prediction, target, mask=mask) == 1.0
    assert metrics.mse(prediction, target, mask=mask) == 4.0
    assert metrics.nrmse(prediction, target, normalization="min-max", mask=mask) == 0.4
    np.testing.assert_array_equal(calls["mse"][0], [3.0, 5.0])
    assert calls["nrmse"][2] == "min-max"


def test_mae_avoids_unsigned_integer_overflow() -> None:
    prediction = np.array([255], dtype=np.uint8)
    target = np.array([0], dtype=np.uint8)

    assert metrics.mae(prediction, target) == 255.0
    assert metrics.mae(target, prediction) == 255.0


def test_nmi_wraps_skimage_and_applies_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def normalized_mutual_information(
        target: np.ndarray, prediction: np.ndarray, *, bins: int | tuple[int, int]
    ) -> float:
        calls["arguments"] = (target, prediction, bins)
        return 1.5

    monkeypatch.setattr(
        metrics,
        "require",
        lambda *args, **kwargs: SimpleNamespace(
            normalized_mutual_information=normalized_mutual_information
        ),
    )

    result = metrics.nmi([1, 2, 3], [3, 2, 1], bins=32, mask=[True, False, True])

    assert result == 1.5
    np.testing.assert_array_equal(calls["arguments"][0], [3, 1])
    np.testing.assert_array_equal(calls["arguments"][1], [1, 3])
    assert calls["arguments"][2] == 32


def test_aggregate_metrics_supports_single_pair_and_multiple_metrics() -> None:
    prediction = np.array([1.0, 2.0])
    target = np.array([1.0, 4.0])

    result = metrics.aggregate_metrics(prediction, target, ["mae", "gcc"])

    assert result["pairs"][0]["prediction_index"] == 0
    assert result["pairs"][0]["target_index"] == 0
    assert result["pairs"][0]["metrics"]["mae"] == 1.0
    assert result["pairs"][0]["metrics"]["gcc"] == pytest.approx(1.0)
    assert result["mean"]["mae"] == 1.0
    assert result["mean"]["gcc"] == pytest.approx(1.0)
    assert result["std"] == {"mae": 0.0, "gcc": 0.0}


def test_aggregate_metrics_supports_one_to_many() -> None:
    prediction = np.array([0.0, 0.0])
    targets = [np.array([1.0, 1.0]), np.array([3.0, 3.0])]

    result = metrics.aggregate_metrics(prediction, targets, "mae")

    assert [pair["target_index"] for pair in result["pairs"]] == [0, 1]
    assert [pair["metrics"]["mae"] for pair in result["pairs"]] == [1.0, 3.0]
    assert result["mean"]["mae"] == 2.0


def test_aggregate_metrics_supports_many_to_one() -> None:
    predictions = [np.array([1.0]), np.array([3.0])]
    target = np.array([2.0])

    result = metrics.aggregate_metrics(predictions, target, "mae")

    assert [pair["prediction_index"] for pair in result["pairs"]] == [0, 1]
    assert result["mean"]["mae"] == 1.0


def test_aggregate_metrics_supports_equal_length_many_to_many() -> None:
    predictions = [np.array([1.0]), np.array([5.0])]
    targets = [np.array([2.0]), np.array([2.0])]

    result = metrics.aggregate_metrics(predictions, targets, "mae")

    assert [(pair["prediction_index"], pair["target_index"]) for pair in result["pairs"]] == [
        (0, 0),
        (1, 1),
    ]
    assert result["mean"]["mae"] == 2.0


def test_aggregate_metrics_supports_patient_by_model_predictions() -> None:
    predictions = [
        [np.array([0.0]), np.array([2.0]), np.array([4.0])],
        [np.array([10.0]), np.array([12.0]), np.array([14.0])],
    ]
    targets = [np.array([0.0]), np.array([10.0])]

    result = metrics.aggregate_metrics(predictions, targets, "mae")

    assert len(result["pairs"]) == 6
    assert result["pairs"][4] == {
        "patient_index": 1,
        "model_index": 1,
        "target_index": 1,
        "metrics": {"mae": 2.0},
    }
    assert result["mean"] == {"mae": 2.0}
    assert result["std"]["mae"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert result["patient_mean"] == [
        {"patient_index": 0, "metrics": {"mae": 2.0}},
        {"patient_index": 1, "metrics": {"mae": 2.0}},
    ]
    assert [item["metrics"]["mae"] for item in result["patient_std"]] == pytest.approx(
        [np.sqrt(8.0 / 3.0), np.sqrt(8.0 / 3.0)]
    )
    assert result["model_mean"] == [
        {"model_index": 0, "metrics": {"mae": 0.0}},
        {"model_index": 1, "metrics": {"mae": 2.0}},
        {"model_index": 2, "metrics": {"mae": 4.0}},
    ]
    assert [item["metrics"]["mae"] for item in result["model_std"]] == [0.0, 0.0, 0.0]


def test_aggregate_metrics_rejects_invalid_prediction_matrix() -> None:
    target = [np.zeros(1), np.zeros(1)]
    with pytest.raises(ValueError, match="rectangular"):
        metrics.aggregate_metrics(
            [[np.zeros(1)], [np.zeros(1), np.zeros(1)]], target, "mae"
        )
    with pytest.raises(ValueError, match="patient count"):
        metrics.aggregate_metrics([[np.zeros(1)]], target, "mae")


def test_aggregate_metrics_runs_optional_paired_t_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

    def ttest_rel(first: np.ndarray, second: np.ndarray, **kwargs: Any) -> Any:
        calls.append((first, second, kwargs))
        return SimpleNamespace(statistic=np.sum(first - second), pvalue=0.1)

    monkeypatch.setattr(
        metrics, "require", lambda *args, **kwargs: SimpleNamespace(ttest_rel=ttest_rel)
    )
    predictions = [
        [np.array([0.0]), np.array([1.0])],
        [np.array([10.0]), np.array([13.0])],
    ]
    targets = [np.array([0.0]), np.array([10.0])]

    result = metrics.aggregate_metrics(
        predictions,
        targets,
        "mae",
        paired_t_test=True,
        paired_t_test_kwargs={"alternative": "two-sided"},
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], [0.0, 0.0])
    np.testing.assert_array_equal(calls[0][1], [1.0, 3.0])
    assert calls[0][2] == {"alternative": "two-sided"}
    assert result["paired_t_test"]["models"] == [
        {
            "model_indices": (0, 1),
            "metrics": {"mae": {"statistic": -4.0, "pvalue": 0.1}},
        }
    ]
    assert result["paired_t_test"]["patients"] == [
        {
            "patient_indices": (0, 1),
            "metrics": {"mae": {"statistic": -2.0, "pvalue": 0.1}},
        }
    ]


def test_aggregate_metrics_only_allows_paired_t_test_for_matrix() -> None:
    with pytest.raises(ValueError, match="only available"):
        metrics.aggregate_metrics(np.zeros(2), np.zeros(2), "mae", paired_t_test=True)


def test_aggregate_metrics_forwards_metric_specific_kwargs() -> None:
    prediction = np.array([1.0, 100.0])
    target = np.array([3.0, 0.0])

    result = metrics.aggregate_metrics(
        prediction,
        target,
        "mae",
        metric_kwargs={"mae": {"mask": np.array([True, False])}},
    )

    assert result["mean"]["mae"] == 2.0


def test_aggregate_metrics_supports_custom_and_builtin_metrics() -> None:
    def maximum_error(
        prediction: np.ndarray, target: np.ndarray, *, scale: float = 1.0
    ) -> float:
        return float(np.max(np.abs(prediction - target)) * scale)

    result = metrics.aggregate_metrics(
        np.array([1.0, 4.0]),
        np.array([2.0, 2.0]),
        ["mae", maximum_error],
        metric_kwargs={"maximum_error": {"scale": 2.0}},
    )

    assert result["mean"] == {"mae": 1.5, "maximum_error": 4.0}


def test_aggregate_metrics_supports_explicit_custom_metric_names() -> None:
    result = metrics.aggregate_metrics(
        np.array([1.0]),
        np.array([4.0]),
        {"custom_distance": lambda prediction, target: np.abs(prediction - target).mean()},
    )

    assert result["mean"]["custom_distance"] == 3.0


def test_aggregate_metrics_rejects_invalid_requests() -> None:
    image = np.zeros(2)
    with pytest.raises(ValueError, match="unsupported metrics"):
        metrics.aggregate_metrics(image, image, "unknown")
    with pytest.raises(ValueError, match="equal lengths"):
        metrics.aggregate_metrics([image, image], [image], "mae")


def test_aggregate_single_input_metrics_supports_one_input() -> None:
    field = np.ones((1, 3, 3, 2), dtype=float)

    result = metrics.aggregate_single_input_metrics(
        field, ["deformation_magnitude", "deformation_smoothness"]
    )

    assert result == {
        "items": [
            {
                "input_index": 0,
                "metrics": {
                    "deformation_magnitude": 2.0,
                    "deformation_smoothness": 0.0,
                },
            }
        ],
        "mean": {"deformation_magnitude": 2.0, "deformation_smoothness": 0.0},
        "std": {"deformation_magnitude": 0.0, "deformation_smoothness": 0.0},
    }


def test_aggregate_single_input_metrics_supports_input_list_and_mean() -> None:
    fields = [
        np.ones((1, 3, 3, 2), dtype=float),
        np.full((1, 3, 3, 2), 2.0),
    ]

    result = metrics.aggregate_single_input_metrics(fields, "deformation_magnitude")

    assert [item["input_index"] for item in result["items"]] == [0, 1]
    assert [item["metrics"]["deformation_magnitude"] for item in result["items"]] == [
        2.0,
        8.0,
    ]
    assert result["mean"]["deformation_magnitude"] == 5.0
    assert result["std"]["deformation_magnitude"] == 3.0


def test_aggregate_single_input_metrics_forwards_kwargs() -> None:
    field = np.ones((1, 3, 3, 2), dtype=float)
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[:, :2, :2] = True

    result = metrics.aggregate_single_input_metrics(
        field,
        "deformation_magnitude",
        metric_kwargs={"deformation_magnitude": {"mask": mask}},
    )

    assert result["mean"]["deformation_magnitude"] == 2.0


def test_aggregate_single_input_metrics_supports_custom_metrics() -> None:
    result = metrics.aggregate_single_input_metrics(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        {"maximum": lambda value, offset=0.0: np.max(value) + offset},
        metric_kwargs={"maximum": {"offset": 1.0}},
    )

    assert [item["metrics"]["maximum"] for item in result["items"]] == [3.0, 5.0]
    assert result["mean"]["maximum"] == 4.0


def test_aggregate_single_input_metrics_rejects_pair_metric() -> None:
    with pytest.raises(ValueError, match="unsupported single-input metrics"):
        metrics.aggregate_single_input_metrics(np.zeros((1, 3, 3, 2)), "mae")
