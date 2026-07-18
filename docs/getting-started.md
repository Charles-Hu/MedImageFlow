# Getting started

This guide expands the README quick start and shows the smallest path from files
on disk to a batch.

## Install

```bash
python -m pip install medimageflow
python -m pip install "medimageflow[torch]"      # DataLoader support
python -m pip install "medimageflow[imaging]"    # DICOM, NIfTI, metrics
```

From a local checkout:

```bash
python -m pip install -e ".[all,dev]"
```

## Create samples

`Sample` stores paths, not arrays. Any field name can be used as long as it does
not conflict with reserved names: `id`, `metadata`, `features`, and
`__timing__`.

```python
from medimageflow.data import Sample

sample = Sample(
    paths={"image": "data/case-001/image.npy", "label": "data/case-001/label.npy"},
    features={"age": 67},
    id="case-001",
    metadata={"site": "hospital-a"},
)
```

## Build a dataset

`MedicalImageDataset` reads complete images. `.npy` files work with the core
package; NIfTI and DICOM paths require the `imaging` extra.

```python
from medimageflow.data import MedicalImageDataset

dataset = MedicalImageDataset([sample])
item = dataset[0]
print(item["image"].shape)
```

## Create a DataLoader

Install the `torch` extra to use `create_dataloader`.

```python
from medimageflow.data import create_dataloader

loader = create_dataloader(dataset, batch_size=1, shuffle=False)
batch = next(iter(loader))
```

PyTorch's default collate function converts numeric NumPy arrays into tensors.
Pass a custom `collate_fn` if you need to preserve arrays or batch variable-size
items.
