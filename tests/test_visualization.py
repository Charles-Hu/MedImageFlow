import numpy as np
import pytest

from medimageflow import mip_visualization, visualization

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_visualization_displays_2d_rgb_as_one_image() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    figure, axes = visualization(image, show=False)

    assert len(axes) == 1
    assert np.asarray(axes[0].images[0].get_array()).shape == image.shape
    figure.clear()


def test_visualization_accepts_figure_name() -> None:
    figure, _ = visualization(
        np.zeros((8, 12)), figure_name="patient-001", show=False
    )

    assert figure.get_label() == "patient-001"
    figure.clear()


def test_visualization_displays_three_central_volume_views() -> None:
    image = np.arange(5 * 7 * 9).reshape(5, 7, 9)
    figure, axes = visualization(image, show=False)

    assert [np.asarray(axis.images[0].get_array()).shape for axis in axes] == [
        (7, 9),
        (5, 9),
        (5, 7),
    ]
    assert np.array_equal(axes[0].images[0].get_array(), image[2])
    figure.clear()


def test_visualization_supports_channels_last_rgb_volume() -> None:
    image = np.zeros((5, 7, 9, 3), dtype=np.uint8)
    figure, axes = visualization(image, show=False)

    assert [np.asarray(axis.images[0].get_array()).shape for axis in axes] == [
        (7, 9, 3),
        (5, 9, 3),
        (5, 7, 3),
    ]
    figure.clear()


def test_visualization_uses_physical_extent() -> None:
    image = np.zeros((5, 7, 9))
    figure, axes = visualization(image, spacing=(2.0, 3.0, 4.0), show=False)

    assert axes[0].images[0].get_extent() == (0.0, 36.0, 21.0, 0.0)
    assert axes[1].images[0].get_extent() == (0.0, 36.0, 10.0, 0.0)
    assert axes[2].images[0].get_extent() == (0.0, 21.0, 10.0, 0.0)
    figure.clear()


@pytest.mark.parametrize("spacing", [(1.0, 2.0), (1.0, 0.0, 2.0)])
def test_visualization_rejects_invalid_spacing(spacing: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        visualization(np.zeros((5, 7, 9)), spacing=spacing, show=False)


def test_mip_visualization_displays_three_maximum_projections() -> None:
    image = np.arange(5 * 7 * 9).reshape(5, 7, 9)
    figure, axes = mip_visualization(image, show=False)

    assert np.array_equal(axes[0].images[0].get_array(), np.max(image, axis=0))
    assert np.array_equal(axes[1].images[0].get_array(), np.max(image, axis=1))
    assert np.array_equal(axes[2].images[0].get_array(), np.max(image, axis=2))
    figure.clear()


def test_mip_visualization_rejects_2d_image() -> None:
    with pytest.raises(ValueError, match="3D volume"):
        mip_visualization(np.zeros((7, 9)), show=False)


def test_mip_visualization_uses_physical_extent() -> None:
    figure, axes = mip_visualization(
        np.zeros((5, 7, 9)), spacing=(2.0, 3.0, 4.0), show=False
    )

    assert axes[0].images[0].get_extent() == (0.0, 36.0, 21.0, 0.0)
    assert axes[1].images[0].get_extent() == (0.0, 36.0, 10.0, 0.0)
    assert axes[2].images[0].get_extent() == (0.0, 21.0, 10.0, 0.0)
    figure.clear()
