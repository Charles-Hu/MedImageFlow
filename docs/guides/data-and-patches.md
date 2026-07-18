# Datasets and patch sampling

Use this guide when you need to organize cases, read image arrays, or extract
aligned patches.

## Build samples from records

`Sample.from_mapping` converts one CSV, JSON, or database record into a sample.
The mapping keys are output names and the mapping values are record fields.

```python
from medimageflow.data import Sample

sample = Sample.from_mapping(
    record,
    paths={"ct": "ct_path", "label": "label_path"},
    features={"age": "patient_age"},
    id="patient_id",
    metadata=("site",),
    base_dir="data",
)
```

## Use CSV or directory sources

CSV paths are lazy: rows are converted to `Sample` objects when indexed.
Relative image paths are resolved from the CSV file's directory unless
`base_dir` is supplied.

```python
from medimageflow.data import CSVSampleSource, DirectorySampleSource

csv_source = CSVSampleSource(
    "data/samples.csv",
    paths={"ct": "ct_path", "label": "label_path"},
    features=("age",),
    id="patient_id",
)

directory_source = DirectorySampleSource(
    "data/cases",
    paths={"ct": "{id}/ct.nii.gz", "label": "{id}/label.nii.gz"},
)
```

Directory patterns must be relative and contain exactly one `{id}` placeholder.
Missing modalities and duplicate matches are rejected during discovery.

## Apply image transforms

Use field transforms when each array can be processed independently.

```python
from medimageflow.data import MedicalImageDataset
from medimageflow.transforms import MinMaxNormalize, ZScoreNormalize

dataset = MedicalImageDataset(
    [sample],
    image_field_transforms={
        "ct": MinMaxNormalize(minimum=-1000, maximum=1000),
        "mri": ZScoreNormalize(nonzero=True),
    },
)
```

Use `image_transform` or `patch_transform` when multiple fields must share
parameters, such as a synchronized random spatial augmentation.

## Extract multimodal patches

All `patch_keys` must have the same spatial shape after any whole-image
transforms.

```python
from medimageflow.data import PatchDataset, RandomPatchSampler

dataset = PatchDataset(
    [sample],
    sampler=RandomPatchSampler((64, 64, 32), seed=42),
    spatial_dims=3,
    patch_keys=("ct", "mri", "label"),
    reference_key="ct",
    center_mask_key=None,
)
```

The returned item includes `patch_center`, `patch_location`, and `image_index`.
Patch locations may be negative when padding is needed.

## ROI center sampling

Add a binary mask to the sample and set `center_mask_key`.

```python
dataset = PatchDataset(
    [sample],
    sampler=RandomPatchSampler((64, 64, 32), seed=42),
    spatial_dims=3,
    patch_keys=("ct", "mri", "label"),
    reference_key="ct",
    center_mask_key="roi",
)
```

The ROI must be binary and match the spatial shape exactly. Set
`center_mask_key=None` if a field named `roi` should not be used for center
sampling.

## Channels and preserved axes

`spatial_dims` must be 2 or 3. Arrays may be channel-free or may have one
channel axis. Configure channel layout globally or per field:

```python
dataset = PatchDataset(
    [sample],
    sampler=RandomPatchSampler((64, 64), seed=42),
    spatial_dims=2,
    channel_axis=0,
    channel_axes={"label": None},
)
```

Set a patch-size axis to `None` to preserve the full source extent on that axis:

```python
sampler = RandomPatchSampler((96, 96, None), seed=42)
```

If preserved axes have different lengths across samples, default PyTorch
collation cannot stack them into a batch.

## Padding

Out-of-bounds patches are padded with `numpy.pad`. Modes and constant values can
be configured globally or by field.

```python
dataset = PatchDataset(
    [sample],
    sampler=RandomPatchSampler((96, 96, 64), seed=42),
    spatial_dims=3,
    padding_mode={"ct": "constant", "mri": "reflect", "label": "constant"},
    padding_value={"ct": -1000, "label": 0},
)
```
