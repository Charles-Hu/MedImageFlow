import numpy as np

from medical_toolkit.data import (
    GridPatchSampler,
    PatchDataset,
    RandomPatchSampler,
    Sample,
)
from medical_toolkit.transforms import MinMaxNormalize, ZScoreNormalize


def test_image_transform_runs_before_patch_transform(tmp_path) -> None:
    events = []

    def image_transform(sample):
        events.append("image")
        return sample

    def patch_transform(sample):
        events.append("patch")
        return sample

    image_path = tmp_path / "image.npy"
    target_path = tmp_path / "target.npy"
    np.save(image_path, np.zeros((2, 8, 8)))
    np.save(target_path, np.zeros((8, 8)))
    dataset = PatchDataset(
        [Sample(paths={"image": image_path, "target": target_path})],
        GridPatchSampler((4, 4)),
        spatial_dims=2,
        channel_axis=0,
        image_transform=image_transform,
        patch_transform=patch_transform,
    )
    item = dataset[0]

    assert events == ["image", "patch"]
    assert item["image"].shape == (2, 4, 4)
    assert item["target"].shape == (4, 4)


def test_random_patch_center_is_inside_roi(tmp_path) -> None:
    roi = np.zeros((16, 16), dtype=np.uint8)
    roi[8, 10] = 1
    image_path = tmp_path / "image.npy"
    roi_path = tmp_path / "roi.npy"
    np.save(image_path, np.zeros((16, 16)))
    np.save(roi_path, roi)
    dataset = PatchDataset(
        [Sample(paths={"image": image_path, "roi": roi_path})],
        RandomPatchSampler((5, 5), seed=7),
        spatial_dims=2,
    )

    item = dataset[0]

    np.testing.assert_array_equal(item["patch_center"], [8, 10])
    assert item["roi"][2, 2] == 1


def test_outside_center_keeps_patch_shape_with_constant_padding(tmp_path) -> None:
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.ones((5, 5), dtype=np.float32))
    dataset = PatchDataset(
        [Sample(paths={"image": image_path})],
        RandomPatchSampler((5, 5), seed=1, center_range=((1.0, 0.0), (0.0, 0.0))),
        spatial_dims=2,
        padding_mode="constant",
        padding_value=0,
    )

    item = dataset[0]

    assert item["image"].shape == (5, 5)


def test_multimodal_fields_share_patch_but_use_independent_processing(tmp_path) -> None:
    ct = np.arange(64, dtype=np.float32).reshape(8, 8)
    mri = ct + 100
    ct_path = tmp_path / "ct.npy"
    mri_path = tmp_path / "mri.npy"
    np.save(ct_path, ct)
    np.save(mri_path, mri)
    dataset = PatchDataset(
        [Sample(paths={"ct": ct_path, "mri": mri_path})],
        GridPatchSampler((4, 4)),
        spatial_dims=2,
        patch_keys=("ct", "mri"),
        reference_key="ct",
        image_field_transforms={
            "ct": MinMaxNormalize(),
            "mri": ZScoreNormalize(nonzero=False),
        },
    )

    item = dataset[0]

    assert item["ct"].shape == item["mri"].shape == (4, 4)


def test_scalar_feature_has_batchable_axis_and_custom_transform(tmp_path) -> None:
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.zeros((8, 8), dtype=np.float32))
    dataset = PatchDataset(
        [Sample(paths={"image": image_path}, features={"age": 50, "sex": "F"})],
        GridPatchSampler((4, 4)),
        spatial_dims=2,
        feature_transforms={"age": lambda value: (value - 40) / 10},
    )

    item = dataset[0]

    np.testing.assert_array_equal(item["features"]["age"], [1.0])
    assert item["features"]["sex"] == "F"


def test_none_patch_axis_preserves_the_complete_image_direction(tmp_path) -> None:
    image_path = tmp_path / "volume.npy"
    np.save(image_path, np.zeros((12, 10, 8), dtype=np.float32))
    dataset = PatchDataset(
        [Sample(paths={"image": image_path})],
        RandomPatchSampler((5, None, 3), seed=3),
        spatial_dims=3,
    )

    item = dataset[0]

    assert item["image"].shape == (5, 10, 3)
    assert item["patch_center"][1] == 5
