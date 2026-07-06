# MedImageFlow

[English](README.md) | **简体中文**

一个面向医学影像研究与模型训练的 path-first Python 工具箱。数据集保存图像路径，
在取样时读取为 NumPy 数组，并提供多模态数据集、2D/3D patch 采样、PyTorch
DataLoader 适配、evaluation metrics 与聚合，以及基础 DICOM/NIfTI I/O。

> 当前版本为 `0.1.0`，仍处于早期开发阶段。未经独立验证，请勿将输出用于临床决策。

## 快速了解

工具箱采用一条以路径为起点的数据流程：

```text
图像路径 + 非图像特征
          ↓
        Sample
          ↓
读取器 → 整图 transform → 多模态同步采样与填充
          ↓
Patch transform → Dataset → 可选 PyTorch DataLoader
```

主要功能：

- **组织数据：**使用 [`Sample`](#数据模型) 管理命名模态、label、ROI、metadata 和
  非图像特征。
- **读取图像：**使用内置 NIfTI、NumPy、DICOM [图像读取器](#图像读取器)，或注册
  自定义 reader。
- **处理整图：**执行共享或按模态独立的[整图变换](#整图-dataset)。
- **构建对齐 Patch：**进行多模态同步的 [2D/3D Patch 提取](#多模态-patch-dataset)，
  包括 [ROI 中心采样和越界填充](#patch-中心采样)。
- **训练与诊断：**接入 PyTorch DataLoader，并可选输出[数据管线耗时](#可选性能计时)。
- **转换医学影像格式：**使用 [DICOM 与 NIfTI 工具](#dicom-与-nifti)，可选择严格的
  pydicom 转换或 dcm2niix 后端。
- **评估结果：**计算分割、图像相似性和形变场[指标](#evaluation-metrics)，
  并按 patient 与模型聚合结果。

请先查看[安装说明](#安装)，再按实际任务跳转到对应章节。用于正式流程前，建议阅读
[当前限制](#当前限制)。

## 安装

项目要求 Python 3.9 或更高版本。

```bash
# 从 PyPI 安装核心功能（仅依赖 NumPy）
python -m pip install medimageflow

# DICOM、NIfTI 与 evaluation metrics
python -m pip install "medimageflow[imaging]"

# PyTorch DataLoader
python -m pip install "medimageflow[torch]"

# 全部运行时功能
python -m pip install "medimageflow[all]"

# 从源码进行可编辑安装并加入开发工具
python -m pip install -e ".[all,dev]"
```

## 数据模型

每个 `Sample` 使用字典保存任意数量的图像路径，而不是预先读取的图像数组：

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

`id` 和 `metadata` 是保留名称，不能作为 `paths` 中的字段名。路径可以是 `str` 或
`pathlib.Path`。当前实现在 `Dataset.__getitem__()` 中完整读取每个图像，然后转为
`numpy.ndarray`；尚未实现只从磁盘读取目标 patch 的惰性 I/O。

### 从记录和数据源构建样本

不必先手工构建完整的 `Sample` 列表。第一层 `Sample.from_mapping()` 可以独立把一条
CSV、JSON 或数据库记录转换为样本。映射的左侧是样本中的名称，右侧是记录字段名：

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

第二层是可索引的惰性样本源。`MappingSampleSource` 接受任意记录序列，
`CSVSampleSource` 读取表格但只在访问某行时构建 `Sample`，`DirectorySampleSource`
则用明确的 `{id}` 模式配对文件：

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

目录模式必须是相对路径且恰好包含一个 `{id}`。缺失模态、重复匹配、空 CSV 和重复
CSV ID 会立即报错。CSV 中的相对影像路径默认相对于 CSV 文件所在目录解析。

第三层提供 Dataset 快捷入口，参数与独立数据源保持一致：

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

原有的 `MedicalImageDataset([Sample(...)])` 用法仍然有效，也可以直接传入任何实现
`SampleSource` 协议（提供 `__len__` 和 `__getitem__`）的自定义对象。

### 非图像特征

`features` 用于需要参与训练或推理的非图像数据，例如年龄、性别、实验室指标和其他
临床特征。它与仅用于追踪样本信息的 `metadata` 分开，并会随每次 Dataset 返回：

```python
from medimageflow.data import MedicalImageDataset, Sample


def normalize_age(value):
    return (value - 50) / 20


sample = Sample(
    paths={"ct": "data/case-001/ct.nii.gz"},
    features={"age": 67, "sex": "F"},
)

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

数值标量会自动转成形状 `(1,)` 的 NumPy 数组。因此使用默认 PyTorch collate 时：

```python
# 当 dataset 至少包含 8 个 sample 时
batch = next(iter(create_dataloader(dataset, batch_size=8)))
batch["features"]["age"].shape  # torch.Size([8, 1])
```

数值 list、tuple 和 NumPy array 会保留为向量；字符串等其他类型保持原值，可由用户
transform 编码。`feature_transforms` 的函数签名为 `Callable[[Any], Any]`，输出必须
是 PyTorch 默认 `collate_fn` 可以组合的类型。所有 sample 中参与 batching 的同名
feature 应具有一致的 shape。

## 整图 Dataset

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

`image_transform` 接收整个 sample 字典，适合需要在多个字段之间共享参数的处理；
`image_field_transforms` 按字段接收一个 NumPy 数组，适合不同模态采用不同的强度处理。
字段 transform 必须返回 `numpy.ndarray`。

Normalization 不限于内置实现，任何签名为
`Callable[[numpy.ndarray], numpy.ndarray]` 的函数都可以直接使用：

```python
import numpy as np


def ct_window_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array, -1000, 400)
    return (array + 1000) / 1400


def percentile_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(array, (1, 99))
    return np.clip((array - low) / max(float(high - low), 1e-8), 0, 1)


dataset = MedicalImageDataset(
    samples=[sample],
    image_field_transforms={
        "ct": ct_window_norm,
        "mri": percentile_norm,
    },
)
```

同一个接口也适用于 patch 级处理：

```python
dataset = PatchDataset(
    ...,
    patch_field_transforms={"ct": custom_patch_norm},
)
```

自定义函数必须返回 `numpy.ndarray`；返回 Tensor、list 或标量会触发 `TypeError`。
需要同时处理多个模态或共享随机参数时，应使用接收完整 sample
字典的 `image_transform` 或 `patch_transform`。

## 多模态 Patch Dataset

下面的 CT、MRI、MRA 和 label 只采样一次中心，因此返回的 patch 在 voxel 坐标上完全
对应；每个模态仍可使用独立的归一化方法。

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
    padding_mode={"ct": "constant", "mri": "reflect", "mra": "reflect", "label": "constant"},
    padding_value={"ct": -1000, "label": 0},
)

loader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
)
```

PyTorch 的默认 `collate_fn` 会把数值 NumPy 数组转换为 Tensor。如需保留 NumPy，
可向 `create_dataloader` 传入自定义 `collate_fn`。

### 可选性能计时

调试数据管线时，可以通过 `timing=True` 输出每个 DataLoader iteration 的耗时：

```python
loader = create_dataloader(
    dataset,
    batch_size=2,
    num_workers=4,
    timing=True,
)

for batch in loader:
    train_step(batch)
```

每次返回 batch 前会输出类似：

```text
[medimageflow timing] iteration=1 dataloader_total=0.182341s data_read=0.241020s image_processing=0.031240s feature_processing=0.000021s patch_extraction=0.004812s patch_processing=0.012005s
```

- `dataloader_total`：主进程等待 `next(loader)` 的墙钟时间，包含 worker 等待与 collate
- `data_read`：该 batch 中所有 sample 的完整文件读取耗时之和
- `image_processing`：该 batch 中整图共享及字段独立处理耗时之和
- `feature_processing`：该 batch 中非图像 feature 预处理耗时之和
- `patch_extraction`：中心采样、空间检查、裁剪和填充耗时之和
- `patch_processing`：该 batch 中 patch 级共享及字段独立处理耗时之和

使用多个 worker 时，各 sample 会并行执行，因此阶段耗时之和可能大于
`dataloader_total`，这是正常现象。计时结果会保存在 batch 的 `__timing__` 字段中。
可以用 `timing_reporter` 将文本发送给日志系统：

```python
loader = create_dataloader(
    dataset,
    timing=True,
    timing_reporter=logger.info,
    batch_size=2,
)
```

默认 `timing=False`，不会添加计时字段或输出信息。

### 数组形状与 channel

`spatial_dims` 只能是 2 或 3。每个参与裁剪的数组可以是：

- 无 channel：`(*spatial)`
- 有一个 channel 轴：例如 `(C, *spatial)` 或 `(*spatial, C)`

数组维度必须等于 `spatial_dims` 或 `spatial_dims + 1`。存在 channel 轴但未指定时，
默认使用第 0 轴。可以全局设置 `channel_axis`，也可以按字段覆盖：

```python
dataset = PatchDataset(
    ...,
    channel_axis=0,
    channel_axes={"mri": -1, "label": None},
)
```

所有 `patch_keys` 的空间 shape 必须一致。工具箱当前不会自动完成多模态配准、重采样或
方向统一；调用方需要先确保它们位于同一个 voxel 网格。

### 在指定方向保留完整图像

将 `patch_size` 的某个空间轴设置为 `None`，可以在该方向不裁剪，直接保留原始长度：

```python
sampler = RandomPatchSampler(
    # x、y 裁剪为 96；z 方向完整保留
    patch_size=(96, 96, None),
    seed=42,
)

dataset = PatchDataset(
    samples=[sample],
    sampler=sampler,
    spatial_dims=3,
    patch_keys=("ct", "mri", "label"),
    reference_key="ct",
)
```

例如输入 shape 为 `(256, 256, 80)`，输出 patch shape 为 `(96, 96, 80)`。可以在一个
或多个方向使用 `None`，`patch_size=None` 表示所有空间方向都保留完整图像。

完整保留轴的中心固定为该轴的 `image_size // 2`，该轴不应用 `center_range`。使用 ROI
采样时，ROI 会沿完整轴投影，随机中心只在仍需裁剪的方向变化。

如果不同 sample 在完整保留轴上的长度不同，返回 patch 的 shape 也会不同。此时
PyTorch 默认 `collate_fn` 无法将它们堆叠为同一个 batch，需要设置 `batch_size=1`、
提前重采样到相同尺寸，或提供自定义 `collate_fn`。

### 数据处理顺序

PatchDataset 的执行顺序固定为：

1. reader 根据路径完整读取每个图像，并转为 NumPy
2. `feature_transforms`：非图像字段独立处理
3. `image_transform`：整图级共享处理
4. `image_field_transforms`：整图级字段独立处理
5. 采样一次中心，并同步裁剪所有 `patch_keys`
6. `patch_transform`：patch 级共享处理
7. `patch_field_transforms`：patch 级字段独立处理

共享 transform 接收整个 sample 字典，适合同步随机翻转、旋转等空间增强。若共享的
整图 transform 改变空间 shape，必须以相同方式处理所有对齐字段。

```python
dataset = PatchDataset(
    ...,
    image_transform=shared_whole_image_transform,
    image_field_transforms={"ct": ct_normalizer, "mri": mri_normalizer},
    patch_transform=shared_patch_augmentation,
    patch_field_transforms={"ct": ct_patch_transform},
)
```

## Patch 中心采样

### 整图随机中心

`RandomPatchSampler` 采样的是整数 voxel 中心，不是 patch 左上角/起点。默认情况下，
每个图像内 voxel 都可以成为中心，因此靠近边界的 patch 可能需要填充。

`center_range` 可以进一步允许中心超出图像边界。每个空间轴使用
`(before, after)`，数值是相对于该轴 patch size 的比例，范围为 `[0, 1]`：

```python
sampler = RandomPatchSampler(
    patch_size=(64, 64, 32),
    # x 轴中心范围扩展到 -0.5*64 至 (X-1)+0.2*64
    # y 轴不扩展；z 轴前后各扩展 0.1*32
    center_range=((0.5, 0.2), (0.0, 0.0), (0.1, 0.1)),
    seed=42,
)
```

传入单个 pair，例如 `center_range=(0.1, 0.1)`，会应用到所有空间轴。
`replacement=False` 可要求同一图像内的随机中心不重复。

### ROI 内随机中心

将二值 mask 放入样本，并用 `center_mask_key` 指定它：

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

ROI 必须是与空间 shape 完全一致的二值数组。中心只从非零 voxel 中抽取；此时
`center_range` 不参与候选中心计算。若不希望名为 `roi` 的字段自动用于采样，设置
`center_mask_key=None`。

当前 `PatchDataset` 对每个 sample 只返回一个 patch，因此
`len(dataset) == len(samples)`。暂不支持一张图在同一 epoch 中展开为多个 patch。

当 DataLoader 每个 epoch 遍历所有索引一次时，每个病例贡献一个 patch。使用
`RandomPatchSampler(seed=None)` 时，每次访问都会重新随机中心；设置固定 seed 后，
同一病例跨 epoch 会得到相同中心。当前尚未提供 `set_epoch()` 来生成“可复现但每个
epoch 不同”的中心。

### 越界填充

越界 patch 通过 `numpy.pad` 补齐到指定大小。`padding_mode` 支持 NumPy 提供的模式，
例如 `constant`、`edge`、`reflect`、`symmetric` 和 `wrap`。模式和值既可以全局设置，
也可以按字段设置：

```python
padding_mode={"ct": "constant", "mri": "reflect", "label": "constant"}
padding_value={"ct": -1000, "label": 0}
```

`padding_value` 仅在 `constant` 模式下使用。如果 patch 与原图在某个轴上完全没有重叠，
只能使用 `constant` 填充。

每个 patch 返回以下附加信息：

- `patch_center`：采样中心的整数坐标
- `patch_location`：patch 起始坐标，越界时可能为负数
- `image_index`：原始样本在 Dataset 中的索引

## 图像读取器

内置 reader 与对应路径：

- `NiftiReader`：`.nii`、`.nii.gz`，依赖 nibabel
- `NumpyReader`：`.npy`
- `DicomSeriesReader`：包含单个 DICOM series 的目录，依赖 SimpleITK

自定义格式可以实现 `ImageReader`：

```python
from pathlib import Path

import numpy as np

from medimageflow.data import ImageReader, MedicalImageDataset, Sample


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

自定义 reader 需要实现 `supports(path)` 和 `read(path)`，读取结果必须是
`numpy.ndarray`。reader 按注册顺序匹配，自定义 reader 优先于内置 reader。

## DICOM 与 NIfTI

需要先安装 `imaging` 可选依赖。

### DICOM 序列读取与转换

```python
from medimageflow.io import (
    convert_dicom_to_nifti,
    dicom_series_to_nifti,
    read_dicom_series,
)

image = read_dicom_series("data/dicom_series")

output_path = dicom_series_to_nifti(
    "data/dicom_series",
    "outputs/scan.nii.gz",
    compress=True,
)

# 严格转换，直接返回内存中的 Nifti1Image 和排序后的 DICOM datasets
nifti_image, ordered_datasets = convert_dicom_to_nifti("data/dicom_series")

# 可选：使用 SliceThickness 作为 z 轴层间距
nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    series_uid="1.2.840...",
    z_spacing="slice_thickness",
)

# 也可以交给已安装的 dcm2niix 转换
nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    backend="dcm2niix",
)
```

DICOM 读取基于 SimpleITK。如果目录中包含多个 series，必须显式传入 `series_id`，避免
静默选择错误序列。严格转换函数 `convert_dicom_to_nifti` 基于 pydicom 和 nibabel；多
序列目录需要传入 `series_uid`。其默认值 `z_spacing="position"` 从相邻切片的
`ImagePositionPatient` 计算 z 轴层间距；`z_spacing="slice_thickness"` 则使用
`SliceThickness`。

严格转换函数不会猜测缺失的空间信息。遇到几何信息缺失或不一致、层距不规则、重复
切片、多帧或彩色数据，以及切片平面内位移时会抛出 `ValueError`，以避免生成空间信息
不可靠的 NIfTI。

设置 `backend="dcm2niix"` 可使用外部 dcm2niix 程序，而不是项目内的 pydicom 转换
实现。dcm2niix 需要单独安装并位于 `PATH` 中；自定义位置可通过
`dcm2niix_command="/path/to/dcm2niix"` 指定。输入目录必须恰好生成一个 NIfTI；
`series_uid` 和 `z_spacing` 仅适用于原生后端，不能与 dcm2niix 同时使用。由于
dcm2niix 不提供 pydicom Dataset，使用该后端时 `ordered_datasets` 为空列表。

### NIfTI 读写

```python
from medimageflow.io import read_nifti, write_nifti

volume, affine = read_nifti("data/scan.nii.gz")
write_nifti(volume, affine, "outputs/scan-copy.nii.gz")
```

NIfTI 读写基于 nibabel。`read_nifti` 返回 `(volume, affine)`，默认将 volume 读取为
`float32`；写入时 affine 必须是 `4 x 4`。

## Evaluation Metrics

使用 SimpleITK、scikit-image、MedPy 和 SciPy 后端的指标前，需安装
`imaging` 可选依赖。所有公开函数均为 evaluation metric，不提供训练 loss。

| 类别 | 指标 |
| --- | --- |
| 分割与边界 | `dice`、`jaccard`/`iou`、`hd95`、`assd`、`sensitivity`、`specificity`、`precision`、`ravd` |
| 图像相似性与误差 | `ssim`、`psnr`、`mae`、`mse`、`nrmse`、`nmi`、`gcc`、`ncc`、`gradient_mae` |
| 形变场质量 | `negative_jacobian_percentage`、`deformation_smoothness`、`deformation_magnitude` |

成对指标接收 NumPy 兼容的 prediction 和 target 数组。每个成对指标都支持可选
布尔 `mask`，仅选中元素参与计算。同时保留 `label`、`data_range`、
`voxelspacing`、`connectivity` 和局部 NCC 窗口等指标专用参数。

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

`gcc` 返回约位于 `[-1, 1]` 的全局归一化互相关；`ncc` 返回局部归一化互相关
平方的均值，二者均是越大越好。`gradient_mae`、MAE/MSE/NRMSE、
HD95/ASSD、RAVD、smoothness 和 magnitude 是误差或距离类指标，通常越小越好。

形变场指标支持 2D/3D 及 component-first/component-last 布局。
`negative_jacobian_percentage` 期望归一化坐标形变网格，返回行列式小于零的
体素百分比。Smoothness 和 magnitude 返回形变场原始正则性统计量。

### 指标聚合

`aggregate_metrics` 支持单对单、单对多、多对单和两个等长 list。它可同时计算一个或
多个内置/自定义指标，返回每个 pair 的结果以及总体 population mean 和
standard deviation（`ddof=0`）。

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

二维 prediction 使用 `[patient][model]` 布局，并与每个 patient 的一个 target 配对。
除 overall `mean` 和 `std` 外，返回值还包含 `patient_mean`、`patient_std`、
`model_mean` 和 `model_std`。

```python
# 10 个 patient，每个 patient 有 5 个模型结果
predictions = [[model_output[p][m] for m in range(5)] for p in range(10)]

summary = aggregate_metrics(predictions, targets, ["dice", "hd95"])
summary["model_mean"]
summary["patient_mean"]
```

Paired t-test 默认关闭。对二维 prediction 显式开启后，使用 `scipy.stats.ttest_rel`
比较模型对（按 patient 配对）和 patient 对（按模型配对）：

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

返回的 p-value 未执行多重比较校正。

单输入形变场指标使用 `aggregate_single_input_metrics`，返回每个 input 的结果以及
overall `mean` 和 `std`：

```python
from medimageflow.metrics import aggregate_single_input_metrics

deformation_summary = aggregate_single_input_metrics(
    deformation_fields,
    ["negative_jacobian_percentage", "deformation_magnitude"],
)
```

两个聚合器都支持直接传入 callable、混合使用内置名称与 callable，或使用显式
`{name: callable}` 映射。`metric_kwargs` 使用最终指标名称作为 key。

```python
custom_summary = aggregate_metrics(
    predictions,
    targets,
    {"maximum_error": lambda prediction, target: abs(prediction - target).max()},
)
```

## 其他工具

```python
from medimageflow.utils import find_files

paths = find_files("data", pattern="*.nii.gz", recursive=True)
```

`find_files` 返回按路径排序的文件列表。

## 开发

```bash
python -m pytest
ruff check .
mypy src
```

项目结构：

```text
src/medimageflow/
├── data/        # reader、Dataset、DataLoader、patch 采样与裁剪
├── io/          # DICOM / NIfTI
├── metrics.py   # evaluation metrics 与聚合
├── transforms/  # 组合与强度变换
└── utils/       # 文件和可选依赖工具
```

## 当前限制

- 不自动执行多模态配准、重采样、方向统一或 spacing 对齐
- PatchDataset 当前会先完整读取各个图像，再裁剪 patch；尚未实现磁盘级局部读取与缓存
- ROI 仅用于随机中心采样；GridPatchSampler 当前只会被 Dataset 取第一个网格中心
- 每个数组最多支持一个 channel 轴
- 尚未提供内置空间增强；可通过共享 transform 接口接入 MONAI、TorchIO 或自定义实现

## 许可证

[BSD 3-Clause License](LICENSE)
