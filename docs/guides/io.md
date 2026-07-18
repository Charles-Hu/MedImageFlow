# DICOM and NIfTI I/O

Install the `imaging` extra before using these APIs.

```bash
python -m pip install "medimageflow[imaging]"
```

## NIfTI

`read_nifti` returns `(volume, affine)` and reads data as `float32` by default.
`write_nifti` requires a 4-by-4 affine matrix and an output path ending in
`.nii` or `.nii.gz`.

```python
from medimageflow.io import read_nifti, write_nifti

volume, affine = read_nifti("data/scan.nii.gz")
write_nifti(volume, affine, "outputs/scan-copy.nii.gz")
```

## DICOM series

`read_dicom_series` returns a SimpleITK image. If multiple series exist in the
directory, pass `series_id`.

```python
from medimageflow.io import read_dicom_series

image = read_dicom_series("data/dicom_series", series_id="1.2.840...")
```

`DicomSeriesReader` is the dataset reader for DICOM directories. It materializes
the SimpleITK image with `sitk.GetArrayFromImage(...)`, which uses SimpleITK
array order, typically `(Z, Y, X)`.

## Convert DICOM to NIfTI

Use `dicom_series_to_nifti` for SimpleITK-backed conversion directly to disk.

```python
from medimageflow.io import dicom_series_to_nifti

dicom_series_to_nifti("data/dicom_series", "outputs/scan.nii.gz")
```

Use `convert_dicom_to_nifti` for strict pydicom/nibabel conversion that returns
an in-memory NIfTI image and ordered pydicom datasets.

```python
from medimageflow.io import convert_dicom_to_nifti

nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    series_uid="1.2.840...",
    z_spacing="position",
)
```

The strict converter rejects missing or inconsistent geometry, irregular or
duplicate slice positions, enhanced multi-frame DICOM, colour images, and
in-plane slice displacement.

Set `backend="dcm2niix"` to delegate conversion to an installed `dcm2niix`
executable:

```python
nifti_image, ordered_datasets = convert_dicom_to_nifti(
    "data/dicom_series",
    backend="dcm2niix",
)
```

With this backend, `ordered_datasets` is empty because dcm2niix does not expose
pydicom objects.

## Axis order

Be explicit when mixing formats:

- `read_nifti` returns nibabel data in NIfTI array order.
- `read_dicom_series` returns a SimpleITK image.
- `DicomSeriesReader` returns `sitk.GetArrayFromImage(...)` arrays in SimpleITK
  order.
