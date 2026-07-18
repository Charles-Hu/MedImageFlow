# Evaluation metrics

Install the `imaging` extra before using metrics backed by SimpleITK,
scikit-image, MedPy, or SciPy.

```bash
python -m pip install "medimageflow[imaging]"
```

## Metric groups

| Category | Metrics |
| --- | --- |
| Segmentation and boundary | `dice`, `jaccard`/`iou`, `hd95`, `assd`, `sensitivity`, `specificity`, `precision`, `ravd` |
| Image similarity and error | `ssim`, `psnr`, `mae`, `mse`, `nrmse`, `nmi`, `gcc`, `ncc`, `gradient_mae` |
| Deformation quality | `negative_jacobian_percentage_from_grid`, `negative_jacobian_percentage_from_displacement`, `deformation_smoothness`, `deformation_magnitude` |

All pair metrics accept NumPy-compatible prediction and target arrays. Pair
metrics also accept an optional boolean `mask`.

```python
from medimageflow.metrics import dice, hd95, ssim

dice_score = dice(prediction_label, target_label, label=1, mask=roi)
surface_distance = hd95(
    prediction_label,
    target_label,
    voxelspacing=(1.0, 1.0, 2.5),
    mask=roi,
)
structural_similarity = ssim(
    prediction_image,
    target_image,
    data_range=1.0,
    mask=roi,
)
```

## Aggregate metrics

`aggregate_metrics` evaluates one or more metrics over one pair, one-to-many,
many-to-one, or two equal-length lists. It returns per-pair values plus overall
population mean and standard deviation.

```python
from medimageflow.metrics import aggregate_metrics

summary = aggregate_metrics(
    predictions,
    targets,
    ["dice", "hd95"],
    metric_kwargs={
        "dice": {"label": 1},
        "hd95": {"voxelspacing": (1.0, 1.0, 2.5)},
    },
)

print(summary["pairs"])
print(summary["mean"])
print(summary["std"])
```

Rectangular predictions use `[patient][model]` layout and produce additional
patient-level and model-level summaries.

## Deformation metrics

Deformation metrics support 2D and 3D fields in component-first or
component-last layout.

- `negative_jacobian_percentage_from_grid` expects a normalized coordinate grid
  in `[-1, 1]`.
- `negative_jacobian_percentage_from_displacement` expects a displacement field
  and evaluates `phi(x) = x + u(x)`.
- `deformation_smoothness` and `deformation_magnitude` report field regularity
  statistics.

Use `aggregate_single_input_metrics` for metrics that accept only one input.
