# MedImageFlow

[English](README.md) | **简体中文**

一个 path-first 的 Python 医学图像工具箱，用于医学影像读写、Dataset、patch
采样、transform 和评估指标。

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

MedImageFlow 帮助你把磁盘上的图像文件变成可用于模型训练或评估的 NumPy / PyTorch
batch。它适合需要读取 NIfTI 或 DICOM、组织多模态病例、裁剪对齐 patch，以及计算常见
医学图像指标的研究流程。

> 当前版本为 `0.1.0`，仍处于早期开发阶段。未经独立验证，请勿将输出用于临床决策。

## 为什么选择 MedImageFlow

- Dataset 以路径为起点：样本保存文件路径，只在索引访问时读取图像。
- 使用 `ct`、`mri`、`label`、`roi` 等命名字段组织多模态病例。
- 在多个已对齐图像字段之间同步裁剪 2D 或 3D patch。
- PyTorch DataLoader 是可选集成，不把 PyTorch 作为核心依赖。
- DICOM、NIfTI、可视化和指标后端都按需安装。

## 安装

项目要求 Python 3.9 或更高版本。

```bash
python -m pip install medimageflow
```

按需要安装 optional extras：

| Extra | 用途 |
| --- | --- |
| `imaging` | DICOM、NIfTI、SimpleITK、nibabel、MedPy、scikit-image、SciPy 后端指标 |
| `torch` | `create_dataloader` 与 PyTorch DataLoader 集成 |
| `visualization` | 基于 Matplotlib 的图像与形变场可视化 |
| `notebooks` | 运行示例 Notebook 所需的 JupyterLab、IPython kernel 和相关依赖 |
| `dev` | 测试、Ruff、coverage 和 mypy |
| `all` | 全部运行时可选依赖 |

```bash
python -m pip install "medimageflow[imaging,torch]"
python -m pip install "medimageflow[notebooks]"
python -m pip install -e ".[all,dev]"  # 在本地源码目录中安装
```

## 快速开始

这个示例会创建两个很小的 NumPy 图像文件，构建 patch dataset，创建 PyTorch
DataLoader，并读取一个 batch。请先安装 `torch` extra。

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

预期输出：

```text
['case-000', 'case-001']
torch.Size([2, 4, 4])
torch.Size([2, 4, 4])
torch.Size([2, 1])
```

## 常见工作流

### 读取 NIfTI 图像

当你需要同时拿到图像数组和 affine 矩阵时，使用底层 I/O 函数。

```python
from medimageflow.io import read_nifti

volume, affine = read_nifti("path/to/scan.nii.gz")
print(volume.shape)
print(affine.shape)
```

需要安装 `medimageflow[imaging]`。

### 读取 DICOM 序列

对于包含一个 SimpleITK 可读取 DICOM series 的目录，使用 `read_dicom_series`。

```python
from medimageflow.io import read_dicom_series

image = read_dicom_series("path/to/dicom_series")
print(image.GetSize())
```

如果目录里包含多个 series，需要显式传入 `series_id`。

### 将 DICOM 转换为 NIfTI

使用 SimpleITK 后端直接转换到磁盘：

```python
from medimageflow.io import dicom_series_to_nifti

output_path = dicom_series_to_nifti(
    "path/to/dicom_series",
    "outputs/scan.nii.gz",
)
print(output_path)
```

使用 pydicom 与 nibabel 进行严格的内存转换：

```python
from medimageflow.io import convert_dicom_to_nifti

nifti_image, ordered_datasets = convert_dicom_to_nifti("path/to/dicom_series")
print(nifti_image.shape)
print(len(ordered_datasets))
```

### 构建多模态 Dataset

每个 `Sample` 可以包含任意数量的命名图像路径和非图像 feature。

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

### 同步提取 Patch

当多个已对齐模态需要共享同一个裁剪中心时，使用 `PatchDataset`。

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

如果希望随机中心来自二值 ROI 前景，把 `center_mask_key` 设置为对应字段名，例如
`"roi"`。

### 评估预测结果

分割指标是面向 NumPy 输入的函数，底层依赖可选 imaging 库。

```python
from medimageflow.metrics import dice, hd95

dice_score = dice(prediction_label, target_label, label=1)
surface_distance = hd95(
    prediction_label,
    target_label,
    voxelspacing=(1.0, 1.0, 2.5),
)
```

需要安装 `medimageflow[imaging]`。

## 核心概念

- **Sample：**病例级记录，包含命名图像路径、可选的训练 `features`、可选 `id` 和自由
  形式的 `metadata`。
- **SampleSource：**按索引返回 `Sample` 的对象。内置 source 可以从记录、CSV 行或目录
  模式中按需构建样本。
- **Feature：**随样本返回的非图像数据，例如年龄或临床评分。数值标量会变成一维
  NumPy 数组，方便 batch collation。
- **Dataset：**`MedicalImageDataset` 读取完整图像；`PatchDataset` 在此基础上为每个
  样本裁剪一个同步 patch。
- **Reader：**把一个路径读取为 NumPy 数组的对象。内置 reader 支持 `.npy`、NIfTI 文件
  和 DICOM series 目录。
- **Sampler / PatchSampler：**返回 patch 中心的策略。`RandomPatchSampler` 支持随机与
  ROI 中心；`GridPatchSampler` 返回确定性的网格中心。
- **Transform：**处理完整 sample 字典或单个图像数组的 callable。字段级 transform 必须
  返回 `numpy.ndarray`。

## 详细文档

- [交互式 Notebook 示例](examples/notebooks/)
  - [I/O 快速上手](examples/notebooks/01_io_quickstart.ipynb)
  - [Dataset 与 DataLoader](examples/notebooks/02_dataset_and_dataloader.ipynb)
  - [Patch 采样](examples/notebooks/03_patch_sampling.ipynb)
- [入门指南](docs/getting-started.md)
- [核心概念](docs/concepts.md)
- [Dataset 与 patch 采样](docs/guides/data-and-patches.md)
- [DICOM 与 NIfTI I/O](docs/guides/io.md)
- [评估指标](docs/guides/evaluation.md)
- [可视化](docs/guides/visualization.md)
- [当前限制](docs/limitations.md)

## 当前限制

- 不会自动进行多模态配准、重采样、方向标准化或 spacing 对齐。
- `PatchDataset` 会先完整读取图像再裁剪；尚未实现存储层面的惰性 patch I/O 或缓存。
- `PatchDataset` 每次访问一个样本只返回一个 patch。
- ROI 采样由 `RandomPatchSampler` 支持；`GridPatchSampler` 会忽略 ROI mask。
- 每个图像最多支持一个 channel 轴。
- 暂未内置空间增强。

更多细节见 [docs/limitations.md](docs/limitations.md)。

## 开发

```bash
python -m pip install -e ".[all,dev]"
python -m pytest
ruff check .
mypy src
```

## 许可证

[BSD 3-Clause License](LICENSE)
