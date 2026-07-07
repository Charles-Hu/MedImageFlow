# MedImageFlow

**English** | [简体中文](README.zh-CN.md)

A path-first Python toolkit for medical imaging research and model training. It
provides multimodal datasets, synchronized 2D/3D patch extraction, PyTorch
DataLoader integration, evaluation metrics and aggregation, optional profiling,
and basic DICOM/NIfTI I/O.

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
- **Evaluate results:** calculate segmentation, image-similarity, and deformation-field
  [metrics](#evaluation-metrics), then aggregate patients and models.
- **Visualize images:** display a 2D grayscale/RGB image or the three central
  slice or maximum-intensity-projection views of a 3D volume with optional
  physical spacing.

Start with [installation](#installation), then choose the detailed section
matching your workflow. Before production use, review the
[current limitations](#current-limitations).

## Installation

Python 3.9 or later is required.

```bash
# Core package and NumPy from PyPI
python -m pip install medimageflow

# DICOM, NIfTI, and evaluation-metric support
python -m pip install "medimageflow[imaging]"

# PyTorch DataLoader support
python -m pip install "medimageflow[torch]"

# Matplotlib-based 2D/3D visualization
python -m pip install "medimageflow[visualization]"

# All runtime features
python -m pip install "medimageflow[all]"

# Editable source installation with development tools
python -m pip install -e ".[all,dev]"
```

## Visualization

Install the `visualization` extra to use the Matplotlib-backed display helpers.
Both functions return `(figure, axes)`, accept an optional `figure_name`, and
can defer display with `show=False`.

```python
from medimageflow import mip_visualization, visualization

# Displays the three central slices in voxel coordinates.
figure, axes = visualization(volume)

# spacing follows the array axes and displays physical coordinates.
figure, axes = visualization(volume, spacing=(2.5, 0.8, 0.8))

# A channels-last RGB image is displayed as one 2D image.
figure, axes = visualization(rgb_image, figure_name="case-001")

# A 3D volume can also be displayed as three maximum-intensity projections.
figure, axes = mip_visualization(volume, spacing=(2.5, 0.8, 0.8))
```

`visualization` supports `(H, W)`, `(H, W, 3/4)`, `(D, H, W)`, and
`(D, H, W, 3/4)`. For a 3D volume, it displays the central slice perpendicular
to each spatial axis. `mip_visualization` accepts only `(D, H, W)` and
`(D, H, W, 3/4)` volumes and calculates one maximum-intensity projection along
each spatial axis. RGB/RGBA data must use channels-last layout.

Spacing follows array-axis order: `(row, column)` for 2D images and
`(axis_0, axis_1, axis_2)` for 3D volumes. When supplied, it controls the axes'
physical coordinates and aspect ratios; otherwise pixel or voxel coordinates
are used.

## Sample Model

`Sample` stores image paths, trainable non-image features, an identifier, and
free-form metadata:

```python
from medimageflow.data import Sample

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

### Building samples from records and sources

You do not need to construct a complete `Sample` list manually. The first
layer, `Sample.from_mapping()`, independently converts one CSV, JSON, or
database record. Mapping keys name the sample fields and values name the
record fields:

```python
sample = Sample.from_mapping(
    record,
    paths={"ct": "ct_path", "label": "label_path"},
    features={"age": "patient_age"},
    id="patient_id",
    metadata=("site",),
    base_dir="data",
)
```

The second layer provides indexable sources: `MappingSampleSource`,
`CSVSampleSource`, and pattern-based `DirectorySampleSource`. Samples are
constructed only when their index is accessed:

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

Directory patterns must be relative and contain exactly one `{id}`. Missing
modalities, duplicate matches, empty CSV files, and duplicate CSV IDs fail
early. Relative paths in a CSV are resolved from the CSV file's directory.

The third layer exposes matching dataset shortcuts:

```python
dataset = MedicalImageDataset.from_csv(
    "data/samples.csv",
    paths={"ct": "ct_path", "label": "label_path"},
    features=("age",),
    id="patient_id",
)

directory_dataset = MedicalImageDataset.from_directory(
    "data/cases",
    paths={"ct": "{id}/ct.nii.gz", "label": "{id}/label.nii.gz"},
)
```

The original `MedicalImageDataset([Sample(...)])` API remains valid. A custom
object implementing the `SampleSource` protocol (`__len__` and `__getitem__`)
can also be passed directly.

### Non-image features

`features` contains values that participate in training or inference. Numeric
scalars are automatically converted to NumPy arrays with shape `(1,)`, so the
default PyTorch collation produces `(batch_size, 1)` values.

```python
from medimageflow.data import MedicalImageDataset


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
from medimageflow.data import MedicalImageDataset
from medimageflow.transforms import MinMaxNormalize, ZScoreNormalize

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
from medimageflow.data import PatchDataset, RandomPatchSampler, create_dataloader
from medimageflow.transforms import MinMaxNormalize, ZScoreNormalize

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
[medimageflow timing] iteration=1 dataloader_total=0.182341s data_read=0.241020s image_processing=0.031240s feature_processing=0.000021s patch_extraction=0.004812s patch_processing=0.012005s
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

from medimageflow.data import MedicalImageDataset, Sample


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
from medimageflow.io import (
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

## Evaluation Metrics

Install the `imaging` extra to use metrics backed by SimpleITK, scikit-image,
MedPy, and SciPy. All public functions are evaluation metrics rather than
training losses.

| Category | Metrics |
| --- | --- |
| Segmentation and boundary | `dice`, `jaccard`/`iou`, `hd95`, `assd`, `sensitivity`, `specificity`, `precision`, `ravd` |
| Image similarity and error | `ssim`, `psnr`, `mae`, `mse`, `nrmse`, `nmi`, `gcc`, `ncc`, `gradient_mae` |
| Deformation quality | `negative_jacobian_percentage`, `deformation_smoothness`, `deformation_magnitude` |

Pair metrics accept NumPy-compatible prediction and target arrays. Every pair
metric accepts an optional boolean `mask`; only selected elements contribute to
the reported value. Metric-specific arguments remain available, including
`label`, `data_range`, `voxelspacing`, `connectivity`, and the local NCC window.

```python
from medimageflow.metrics import dice, hd95, ssim

dice_score = dice(prediction_label, target_label, label=1, mask=roi)
surface_distance = hd95(
    prediction_label,
    target_label,
    voxelspacing=(1.0, 1.0, 2.5),
    mask=roi,
)
structural_similarity = ssim(prediction_image, target_image, data_range=1.0, mask=roi)
```

`gcc` reports global normalized cross correlation in approximately `[-1, 1]`.
`ncc` reports mean squared local normalized cross correlation, where higher is
better. `gradient_mae`, MAE/MSE/NRMSE, HD95/ASSD, RAVD, smoothness, and magnitude
are error or distance measures where lower is generally better.

Deformation metrics support 2D and 3D fields in component-first or
component-last layout. `negative_jacobian_percentage` expects a normalized
coordinate deformation grid and reports the percentage of determinants below
zero. Smoothness and magnitude report raw field regularity statistics.

### Metric aggregation

`aggregate_metrics` accepts one prediction and one target, one-to-many,
many-to-one, or two equal-length lists. It evaluates one or more built-in or
custom metrics and returns every pair plus overall population mean and standard
deviation (`ddof=0`).

```python
from medimageflow.metrics import aggregate_metrics

summary = aggregate_metrics(
    predictions,
    targets,
    ["dice", "hd95"],
    metric_kwargs={
        "dice": {"label": 1, "mask": roi},
        "hd95": {"voxelspacing": (1.0, 1.0, 2.5), "mask": roi},
    },
)

summary["pairs"]
summary["mean"]
summary["std"]
```

A rectangular prediction input uses `[patient][model]` layout and is paired
with one target per patient. In addition to overall `mean` and `std`, the result
contains `patient_mean`, `patient_std`, `model_mean`, and `model_std`.

```python
# Ten patients, each evaluated with five models.
predictions = [[model_output[p][m] for m in range(5)] for p in range(10)]

summary = aggregate_metrics(predictions, targets, ["dice", "hd95"])
summary["model_mean"]
summary["patient_mean"]
```

Paired t-tests are deliberately disabled by default. For a prediction matrix,
enable them to compare model pairs across patients and patient pairs across
models with `scipy.stats.ttest_rel`:

```python
summary = aggregate_metrics(
    predictions,
    targets,
    "dice",
    paired_t_test=True,
    paired_t_test_kwargs={"alternative": "two-sided", "nan_policy": "omit"},
)

summary["paired_t_test"]["models"]
summary["paired_t_test"]["patients"]
```

The returned p-values are not corrected for multiple comparisons.

Single-input deformation metrics use `aggregate_single_input_metrics`, which
returns per-input values plus overall `mean` and `std`:

```python
from medimageflow.metrics import aggregate_single_input_metrics

deformation_summary = aggregate_single_input_metrics(
    deformation_fields,
    ["negative_jacobian_percentage", "deformation_magnitude"],
)
```

Both aggregators accept a callable directly, a mixture of built-in names and
callables, or an explicit `{name: callable}` mapping. `metric_kwargs` is keyed
by the final metric name.

```python
custom_summary = aggregate_metrics(
    predictions,
    targets,
    {"maximum_error": lambda prediction, target: abs(prediction - target).max()},
)
```

## Utilities

```python
from medimageflow.utils import find_files

paths = find_files("data", pattern="*.nii.gz", recursive=True)
```

## Development

```bash
python -m pytest
ruff check .
mypy src
```

```text
src/medimageflow/
├── data/        # readers, datasets, DataLoader, sampling, and extraction
├── io/          # DICOM and NIfTI
├── metrics.py   # evaluation metrics and aggregation
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

[BSD 3-Clause License](LICENSE)
