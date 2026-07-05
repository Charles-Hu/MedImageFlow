# Medical Imaging Toolkit

**English** | [简体中文](README.zh-CN.md)

A path-first Python toolkit for medical imaging research and model training. It
provides multimodal datasets, synchronized 2D/3D patch extraction, PyTorch
DataLoader integration, optional profiling, and basic DICOM/NIfTI I/O.

> Version `0.1.0` is an early-stage release. Do not use unvalidated outputs for
> clinical decisions.

## At a Glance

The toolkit follows one path-first pipeline:

```text
Image paths + features
          ↓
        Sample
          ↓
Reader → whole-image transforms → synchronized sampling/padding
          ↓
Patch transforms → Dataset → optional PyTorch DataLoader
```

Main capabilities:

- **Organize data:** model named modalities, labels, ROIs, metadata, and
  non-image features with [`Sample`](#sample-model).
- **Read images:** use built-in NIfTI, NumPy, and DICOM
  [image readers](#image-readers), or register a custom reader.
- **Process complete images:** apply shared or modality-specific
  [whole-image transforms](#whole-image-dataset).
- **Build aligned patches:** perform synchronized 2D/3D
  [multimodal patch extraction](#multimodal-patch-dataset), including
  [ROI sampling and boundary padding](#patch-center-sampling).
- **Train and inspect:** connect to PyTorch DataLoader and optionally
  [report pipeline timing](#optional-performance-timing).
- **Convert medical formats:** use the [DICOM and NIfTI tools](#dicom-and-nifti-io),
  including strict pydicom conversion or a dcm2niix backend.

Start with [installation](#installation), then choose the detailed section
matching your workflow. Before production use, review the
[current limitations](#current-limitations).

## Installation

Python 3.9 or later is required.

```bash
# Core package and NumPy
python -m pip install -e .

# DICOM and NIfTI support
python -m pip install -e ".[imaging]"

# PyTorch DataLoader support
python -m pip install -e ".[torch]"

# All runtime features and development tools
python -m pip install -e ".[all,dev]"
```

## Sample Model

`Sample` stores image paths, trainable non-image features, an identifier, and
free-form metadata:

```python
from medical_toolkit.data import Sample

sample = Sample(
    paths={
        "ct": "data/case-001/ct.nii.gz",
        "mri": "data/case-001/mri.nii.gz",
        "mra": "data/case-001/mra.nii.gz",
        "label": "data/case-001/label.nii.gz",
        "roi": "data/case-001/roi.npy",
    },
    features={
        "age": 67,
        "sex": "F",
        "clinical_score": [0.2, 0.7, 0.1],
    },
    id="case-001",
    metadata={"site": "hospital-a"},
)
```

`id`, `metadata`, `features`, and `__timing__` are reserved image names. Paths
may be strings or `pathlib.Path` objects. The current implementation reads each
image completely inside `Dataset.__getitem__`; storage-level lazy patch I/O is
not implemented yet.

### Non-image features

`features` contains values that participate in training or inference. Numeric
scalars are automatically converted to NumPy arrays with shape `(1,)`, so the
default PyTorch collation produces `(batch_size, 1)` values.

```python
from medical_toolkit.data import MedicalImageDataset


def normalize_age(value):
    return (value - 50) / 20


dataset = MedicalImageDataset(
    samples=[sample],
    feature_transforms={
        "age": normalize_age,
        "sex": lambda value: 1 if value == "F" else 0,
    },
)

item = dataset[0]
age = item["features"]["age"]  # shape: (1,)
```

Numeric lists, tuples, and arrays remain vectors. Strings and other values stay
unchanged unless a feature transform encodes them. Values with the same feature
name must have collatable, consistent shapes across a batch.

## Whole-image Dataset

```python
from medical_toolkit.data import MedicalImageDataset
from medical_toolkit.transforms import MinMaxNormalize, ZScoreNormalize

dataset = MedicalImageDataset(
    samples=[sample],
    image_field_transforms={
        "ct": MinMaxNormalize(minimum=-1000, maximum=1000),
        "mri": ZScoreNormalize(nonzero=True),
        "mra": ZScoreNormalize(nonzero=True),
    },
)

item = dataset[0]
ct = item["ct"]
```

`image_transform` receives the complete sample dictionary and is intended for
operations that share parameters across fields. `image_field_transforms`
receives one NumPy array at a time and supports modality-specific processing.

### Custom normalization

Any callable with the signature `Callable[[numpy.ndarray], numpy.ndarray]` can
be used as an image field transform:

```python
import numpy as np


def ct_window_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array, -1000, 400)
    return (array + 1000) / 1400


dataset = MedicalImageDataset(
    samples=[sample],
    image_field_transforms={"ct": ct_window_norm},
)
```

Array transforms must return `numpy.ndarray`. Use `image_transform` or
`patch_transform` when multiple fields must share random parameters.

## Multimodal Patch Dataset

All configured image fields use one shared center and crop geometry:

```python
from medical_toolkit.data import PatchDataset, RandomPatchSampler, create_dataloader
from medical_toolkit.transforms import MinMaxNormalize, ZScoreNormalize

dataset = PatchDataset(
    samples=[sample],
    sampler=RandomPatchSampler(
        patch_size=(96, 96, 64),
        seed=42,
        replacement=True,
    ),
    spatial_dims=3,
    patch_keys=("ct", "mri", "mra", "label"),
    reference_key="ct",
    image_field_transforms={
        "ct": MinMaxNormalize(minimum=-1000, maximum=1000),
        "mri": ZScoreNormalize(nonzero=True),
        "mra": ZScoreNormalize(nonzero=True),
    },
    padding_mode={
        "ct": "constant",
        "mri": "reflect",
        "mra": "reflect",
        "label": "constant",
    },
    padding_value={"ct": -1000, "label": 0},
)

loader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
)
```

PyTorch's default `collate_fn` converts numeric NumPy arrays to tensors. Pass a
custom `collate_fn` to preserve arrays or support variable-shaped data.

### Array shapes and channels

`spatial_dims` must be 2 or 3. Each cropped image may be channel-free with shape
`(*spatial)` or contain one channel axis, such as `(C, *spatial)` or
`(*spatial, C)`. When an array has `spatial_dims + 1` dimensions and no channel
axis is specified, axis zero is used.

```python
dataset = PatchDataset(
    ...,
    channel_axis=0,
    channel_axes={"mri": -1, "label": None},
)
```

All `patch_keys` must have the same spatial shape. The toolkit does not perform
registration, resampling, orientation normalization, or spacing alignment.

### Preserve complete axes

Set one or more patch-size axes to `None` to preserve their original extent:

```python
sampler = RandomPatchSampler(
    patch_size=(96, 96, None),
    seed=42,
)
```

An input with shape `(256, 256, 80)` produces a patch with shape
`(96, 96, 80)`. `patch_size=None` preserves every spatial axis. A preserved
axis uses `image_size // 2` as its center and ignores `center_range`.

If samples have different lengths along a preserved axis, default PyTorch
collation cannot stack them. Use `batch_size=1`, resample beforehand, or provide
a custom `collate_fn`.

### Processing order

`PatchDataset` uses the following order:

1. Read every image path into a NumPy array.
2. Apply `feature_transforms`.
3. Apply the shared whole-image `image_transform`.
4. Apply independent `image_field_transforms`.
5. Sample one center and crop every `patch_key` synchronously.
6. Apply the shared `patch_transform`.
7. Apply independent `patch_field_transforms`.

The current dataset returns one patch per sample, so
`len(dataset) == len(samples)`.

## Patch Center Sampling

### Global random centers

`RandomPatchSampler` selects integer voxel centers rather than patch starts.
With the default `center_range=(0, 0)`, every voxel inside the image may be a
center, so patches near a boundary may require padding.

`center_range` extends the valid center range beyond image boundaries. Each axis
uses a `(before, after)` pair expressed as a fraction of that axis's patch size:

```python
sampler = RandomPatchSampler(
    patch_size=(64, 64, 32),
    center_range=((0.5, 0.2), (0.0, 0.0), (0.1, 0.1)),
    seed=42,
)
```

A single pair, such as `(0.1, 0.1)`, is broadcast to all axes. Fractions must be
between 0 and 1. Set `replacement=False` when requesting unique centers directly
from the sampler.

### ROI center sampling

```python
sample = Sample(
    paths={
        "ct": "data/ct.nii.gz",
        "mri": "data/mri.nii.gz",
        "sampling_roi": "data/roi.nii.gz",
    }
)

dataset = PatchDataset(
    samples=[sample],
    sampler=RandomPatchSampler((64, 64, 32), seed=42),
    spatial_dims=3,
    patch_keys=("ct", "mri"),
    reference_key="ct",
    center_mask_key="sampling_roi",
)
```

The ROI must be binary and match the spatial shape exactly. Centers are selected
from foreground voxels; `center_range` is ignored in ROI mode. Preserved axes
are projected out before ROI selection. Set `center_mask_key=None` to disable
mask-based sampling.

### Out-of-bounds padding

Patches are padded to the requested size with `numpy.pad`. Supported modes
include `constant`, `edge`, `reflect`, `symmetric`, and `wrap`. Values can be
configured globally or by image name:

```python
padding_mode={"ct": "constant", "mri": "reflect", "label": "constant"}
padding_value={"ct": -1000, "label": 0}
```

`padding_value` applies only to constant padding. If a patch has no overlap with
the source along an axis, constant padding is required.

Each returned patch includes:

- `patch_center`: integer center coordinates;
- `patch_location`: patch start coordinates, which may be negative;
- `image_index`: index of the source sample.

## Optional Performance Timing

```python
loader = create_dataloader(
    dataset,
    batch_size=2,
    num_workers=4,
    timing=True,
)
```

Before returning each batch, the loader reports:

```text
[medical_toolkit timing] iteration=1 dataloader_total=0.182341s data_read=0.241020s image_processing=0.031240s feature_processing=0.000021s patch_extraction=0.004812s patch_processing=0.012005s
```

- `dataloader_total`: wall time spent waiting for `next(loader)`;
- `data_read`: summed full-image reading time for the batch;
- `image_processing`: summed whole-image processing time;
- `feature_processing`: summed non-image feature processing time;
- `patch_extraction`: summed sampling, validation, cropping, and padding time;
- `patch_processing`: summed patch-level processing time.

Worker stages may run concurrently, so stage sums can exceed wall time. Timing
metadata remains available in `batch["__timing__"]`. Use
`timing_reporter=logger.info` to send reports to a logging system. Timing is
disabled by default.

## Image Readers

Built-in readers:

- `NiftiReader`: `.nii` and `.nii.gz`, requiring nibabel;
- `NumpyReader`: `.npy`;
- `DicomSeriesReader`: a directory containing one DICOM series, requiring
  SimpleITK.

Custom formats can implement the `ImageReader` protocol:

```python
from pathlib import Path

import numpy as np

from medical_toolkit.data import MedicalImageDataset, Sample


class CustomReader:
    def supports(self, path: Path) -> bool:
        return path.suffix == ".custom"

    def read(self, path: Path) -> np.ndarray:
        return load_custom_format(path)


dataset = MedicalImageDataset(
    samples=[Sample(paths={"ct": Path("data/ct.custom")})],
    readers=[CustomReader()],
)
```

Custom readers are checked before built-in readers and must return
`numpy.ndarray`.

## DICOM and NIfTI I/O

Install the `imaging` extra first.

```python
from medical_toolkit.io import (
    convert_dicom_to_nifti,
    dicom_series_to_nifti,
    read_dicom_series,
    read_nifti,
    write_nifti,
)

image = read_dicom_series("data/dicom_series")
dicom_series_to_nifti("data/dicom_series", "outputs/scan.nii.gz")

# Strict pydicom/nibabel conversion returning an in-memory NIfTI image.
nifti_image, ordered_datasets = convert_dicom_to_nifti("data/dicom_series")

# Optionally use SliceThickness instead of adjacent slice positions for z spacing.
nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    series_uid="1.2.840...",
    z_spacing="slice_thickness",
)

# Alternatively, delegate conversion to an installed dcm2niix executable.
nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    backend="dcm2niix",
)

volume, affine = read_nifti("data/scan.nii.gz")
write_nifti(volume, affine, "outputs/scan-copy.nii.gz")
```

If a DICOM directory contains multiple series, pass `series_id` to the
SimpleITK functions or `series_uid` to `convert_dicom_to_nifti` explicitly.
The strict converter defaults to `z_spacing="position"`, calculated from
`ImagePositionPatient`. Its `z_spacing="slice_thickness"` option uses the
`SliceThickness` tag instead. It rejects missing or inconsistent geometry,
irregular or duplicate slice positions, multi-frame data, colour images, and
in-plane slice displacement rather than silently constructing an unreliable
affine.

Set `backend="dcm2niix"` to use the external dcm2niix program instead of the
package's pydicom converter. dcm2niix must be installed separately and available
on `PATH`; use `dcm2niix_command="/path/to/dcm2niix"` for a custom location.
The input directory must produce exactly one NIfTI output. `series_uid` and
`z_spacing` are native-backend options and cannot be combined with dcm2niix.
Because dcm2niix does not expose pydicom objects, `ordered_datasets` is an empty
list with this backend.

`read_nifti` returns `(volume, affine)` and reads volume data as `float32` by
default. `write_nifti` requires a 4-by-4 affine matrix.

## Utilities

```python
from medical_toolkit.utils import find_files

paths = find_files("data", pattern="*.nii.gz", recursive=True)
```

## Development

```bash
python -m pytest
ruff check .
mypy src
```

```text
src/medical_toolkit/
├── data/        # readers, datasets, DataLoader, sampling, and extraction
├── io/          # DICOM and NIfTI
├── transforms/  # composition and intensity transforms
└── utils/       # file and optional-dependency helpers
```

## Current Limitations

- No automatic multimodal registration, resampling, orientation normalization,
  or spacing alignment.
- `PatchDataset` reads complete images before cropping; storage-level lazy patch
  I/O and caching are not implemented.
- ROI sampling is supported by `RandomPatchSampler`; `GridPatchSampler` ignores
  ROI and the dataset currently consumes only its first center.
- Each image supports at most one channel axis.
- Built-in spatial augmentation is not yet available. Shared transforms can
  integrate MONAI, TorchIO, or custom implementations.

## License

[MIT](LICENSE)
