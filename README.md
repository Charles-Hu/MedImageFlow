# MedImageFlow

**English** | [简体中文](README.zh-CN.md)

Path-first medical image I/O, datasets, patch sampling, transforms, and
evaluation utilities for Python research workflows.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

MedImageFlow helps you move from image files on disk to model-ready NumPy or
PyTorch batches. It is designed for research pipelines that need to read NIfTI
or DICOM data, organize multimodal cases, crop aligned patches, and compute
common medical image metrics.

> Version `0.1.0` is an early-stage release. Do not use unvalidated outputs for
> clinical decisions.

## Why MedImageFlow

- Keep datasets path-first: samples store file paths and are materialized only
  when indexed.
- Build multimodal samples with named fields such as `ct`, `mri`, `label`, and
  `roi`.
- Extract synchronized 2D or 3D patches across aligned image fields.
- Use optional PyTorch DataLoader integration without making PyTorch a required
  dependency.
- Keep DICOM, NIfTI, visualization, and metric backends optional.

## Installation

Python 3.9 or later is required.

```bash
python -m pip install medimageflow
```

Install extras for the workflows you use:

| Extra | Use case |
| --- | --- |
| `imaging` | DICOM, NIfTI, SimpleITK, nibabel, MedPy, scikit-image, SciPy-backed metrics |
| `torch` | `create_dataloader` and PyTorch DataLoader integration |
| `visualization` | Matplotlib-based image and field visualization |
| `notebooks` | JupyterLab, IPython kernel, and dependencies used by the example notebooks |
| `dev` | Tests, Ruff, coverage, and mypy |
| `all` | All runtime optional dependencies |

```bash
python -m pip install "medimageflow[imaging,torch]"
python -m pip install "medimageflow[notebooks]"
python -m pip install -e ".[all,dev]"  # from a local checkout
```

## Quick Start

This example creates two tiny NumPy image files, builds a patch dataset, creates
a PyTorch DataLoader, and reads one batch. Install the `torch` extra first.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from medimageflow.data import (
    GridPatchSampler,
    PatchDataset,
    Sample,
    create_dataloader,
)

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    samples = []

    for index in range(2):
        image_path = root / f"case-{index:03d}-image.npy"
        label_path = root / f"case-{index:03d}-label.npy"

        np.save(image_path, np.arange(64, dtype=np.float32).reshape(8, 8) + index)
        np.save(label_path, np.eye(8, dtype=np.uint8))

        samples.append(
            Sample(
                paths={"image": image_path, "label": label_path},
                features={"age": 50 + index},
                id=f"case-{index:03d}",
            )
        )

    dataset = PatchDataset(
        samples,
        sampler=GridPatchSampler((4, 4)),
        spatial_dims=2,
        patch_keys=("image", "label"),
        reference_key="image",
    )

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    print(batch["id"])
    print(batch["image"].shape)
    print(batch["label"].shape)
    print(batch["features"]["age"].shape)
```

Expected output:

```text
['case-000', 'case-001']
torch.Size([2, 4, 4])
torch.Size([2, 4, 4])
torch.Size([2, 1])
```

## Common Workflows

### Read a NIfTI image

Use lower-level I/O when you need both array data and the affine matrix.

```python
from medimageflow.io import read_nifti

volume, affine = read_nifti("path/to/scan.nii.gz")
print(volume.shape)
print(affine.shape)
```

Requires `medimageflow[imaging]`.

### Read a DICOM series

Use `read_dicom_series` for a directory containing one SimpleITK-readable DICOM
series.

```python
from medimageflow.io import read_dicom_series

image = read_dicom_series("path/to/dicom_series")
print(image.GetSize())
```

If a directory contains multiple series, pass `series_id` explicitly.

### Convert DICOM to NIfTI

For a SimpleITK-backed conversion directly to disk:

```python
from medimageflow.io import dicom_series_to_nifti

output_path = dicom_series_to_nifti(
    "path/to/dicom_series",
    "outputs/scan.nii.gz",
)
print(output_path)
```

For strict in-memory conversion with pydicom and nibabel:

```python
from medimageflow.io import convert_dicom_to_nifti

nifti_image, ordered_datasets = convert_dicom_to_nifti("path/to/dicom_series")
print(nifti_image.shape)
print(len(ordered_datasets))
```

### Build a multimodal dataset

Each `Sample` can contain any number of named image paths plus non-image
features.

```python
from medimageflow.data import MedicalImageDataset, Sample
from medimageflow.transforms import MinMaxNormalize, ZScoreNormalize

sample = Sample(
    paths={
        "ct": "data/case-001/ct.nii.gz",
        "mri": "data/case-001/mri.nii.gz",
        "label": "data/case-001/label.nii.gz",
    },
    features={"age": 67},
    id="case-001",
)

dataset = MedicalImageDataset(
    [sample],
    image_field_transforms={
        "ct": MinMaxNormalize(minimum=-1000, maximum=1000),
        "mri": ZScoreNormalize(nonzero=True),
    },
)

item = dataset[0]
print(item["ct"].shape, item["mri"].shape, item["features"]["age"].shape)
```

### Extract synchronized patches

Use `PatchDataset` when aligned modalities must share one crop center.

```python
from medimageflow.data import PatchDataset, RandomPatchSampler

patch_dataset = PatchDataset(
    [sample],
    sampler=RandomPatchSampler((96, 96, 64), seed=42),
    spatial_dims=3,
    patch_keys=("ct", "mri", "label"),
    reference_key="ct",
    center_mask_key=None,
    padding_mode={"ct": "constant", "mri": "reflect", "label": "constant"},
    padding_value={"ct": -1000, "label": 0},
)

patch = patch_dataset[0]
print(patch["ct"].shape)
print(patch["patch_center"])
```

Set `center_mask_key` to the name of a binary ROI field, for example `"roi"`,
to sample random centers from foreground voxels.

### Evaluate a prediction

Segmentation metrics are NumPy-facing wrappers around optional imaging
libraries.

```python
from medimageflow.metrics import dice, hd95

dice_score = dice(prediction_label, target_label, label=1)
surface_distance = hd95(
    prediction_label,
    target_label,
    voxelspacing=(1.0, 1.0, 2.5),
)
```

Requires `medimageflow[imaging]`.

## Core Concepts

- **Sample:** A case-level record containing named image paths, optional
  trainable `features`, an optional `id`, and free-form `metadata`.
- **SampleSource:** An indexable object that returns `Sample` instances on
  demand. Built-in sources can map records, CSV rows, or directory patterns.
- **Feature:** Non-image data that should be returned with a sample, such as
  age or a clinical score. Numeric scalars become one-element NumPy arrays for
  batch collation.
- **Dataset:** `MedicalImageDataset` reads complete images. `PatchDataset`
  extends it by cropping one synchronized patch per sample.
- **Reader:** A path reader that materializes one image as a NumPy array.
  Built-in readers cover `.npy`, NIfTI files, and DICOM series directories.
- **Sampler / PatchSampler:** A strategy that returns patch centers.
  `RandomPatchSampler` supports random and ROI-based centers;
  `GridPatchSampler` returns deterministic grid centers.
- **Transform:** A callable used to process sample dictionaries or individual
  image arrays. Field transforms must return `numpy.ndarray`.

## Documentation

- [Interactive notebooks](examples/notebooks/)
  - [I/O quickstart](examples/notebooks/01_io_quickstart.ipynb)
  - [Dataset and DataLoader](examples/notebooks/02_dataset_and_dataloader.ipynb)
  - [Patch sampling](examples/notebooks/03_patch_sampling.ipynb)
- [Getting started](docs/getting-started.md)
- [Core concepts](docs/concepts.md)
- [Datasets and patch sampling](docs/guides/data-and-patches.md)
- [DICOM and NIfTI I/O](docs/guides/io.md)
- [Evaluation metrics](docs/guides/evaluation.md)
- [Visualization](docs/guides/visualization.md)
- [Current limitations](docs/limitations.md)

## Current Limitations

- No automatic multimodal registration, resampling, orientation normalization,
  or spacing alignment.
- `PatchDataset` reads complete images before cropping; storage-level lazy patch
  I/O and caching are not implemented.
- `PatchDataset` returns one patch per sample access.
- ROI sampling is supported by `RandomPatchSampler`; `GridPatchSampler` ignores
  ROI masks.
- Each image supports at most one channel axis.
- Built-in spatial augmentation is not yet available.

More detail is available in [docs/limitations.md](docs/limitations.md).

## Development

```bash
python -m pip install -e ".[all,dev]"
python -m pytest
ruff check .
mypy src
```

## License

[BSD 3-Clause License](LICENSE)
