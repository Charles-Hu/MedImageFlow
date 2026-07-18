# Visualization

Install the `visualization` extra before using Matplotlib-backed helpers.

```bash
python -m pip install "medimageflow[visualization]"
```

## Central slices and MIP

`visualization` displays a 2D grayscale/RGB image or the three central slices of
a 3D volume. `mip_visualization` displays maximum-intensity projections along
the three spatial axes.

```python
from medimageflow import mip_visualization, visualization

figure, axes = visualization(volume, spacing=(2.5, 0.8, 0.8), show=False)
figure, axes = mip_visualization(volume, spacing=(2.5, 0.8, 0.8), show=False)
```

Supported layouts include `(H, W)`, `(H, W, 3/4)`, `(D, H, W)`, and
`(D, H, W, 3/4)`. RGB/RGBA data must be channels-last.

## Registration fields

`registration_field_visualization` draws deformed grids for 2D or 3D
displacement fields.

```python
from medimageflow import registration_field_visualization

figure, axes = registration_field_visualization(
    field,
    spacing=(2.5, 0.8, 0.8),
    slice_indices=(20, 64, 64),
    show=False,
)
```

By default the function expects component-first fields such as `(2, H, W)` or
`(3, D, H, W)`. Use `component_axis=-1` for component-last arrays.
