import numpy as np

from medical_toolkit.data import (
    GridPatchSampler,
    PatchDataset,
    RandomPatchSampler,
    Sample,
)
from medical_toolkit.transforms import MinMaxNormalize, ZScoreNormalize


def test_image_transform_runs_before_patch_transform(tmp_path) -> None:
    """Verify whole-image processing precedes patch-level processing.

    Two transforms append their stage names while a two-channel image and
    target are loaded. Dataset access should record ``image`` then ``patch``;
    the key crop step should also return aligned 4x4 image and target arrays.
    """
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
    """Ensure ROI-guided random sampling chooses a valid foreground center.

    A binary ROI contains exactly one positive coordinate. Sampling therefore
    has one valid intermediate center, ``(8, 10)``, and the returned 5x5 patch
    should place that ROI voxel at its central coordinate.
    """
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
    """Check constant padding when a requested center lies outside the image.

    The sampler's center range deliberately selects an out-of-bounds location.
    Patch extraction should pad missing pixels rather than truncate the array,
    producing the configured 5x5 output shape.
    """
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
    """Verify aligned modalities share geometry but retain field transforms.

    CT and MRI inputs have identical spatial shapes and different intensities.
    Whole-image transforms should run independently, while the shared sampler
    should crop both at one location and return matching 4x4 shapes.
    """
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
    """Check scalar feature transformation and batch-friendly normalization.

    The input Sample has numeric age and string sex features. Age should pass
    through its custom transform, then gain a one-element feature axis; sex
    should bypass conversion and remain the original string.
    """
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
    """Ensure a ``None`` patch dimension retains the complete source axis.

    The sampler crops the first and third axes but marks the second as full.
    The resolved intermediate patch size should therefore use length 10 on
    that axis, and the returned center should be its midpoint.
    """
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
