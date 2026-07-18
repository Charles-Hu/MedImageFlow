# Example data

This directory is reserved for local tutorial data used by the notebooks in
`examples/notebooks/`.

Do not commit patient data or identifiable clinical files to this repository.
Keep local data fully anonymized and verify that its license permits tutorial
or research use.

Recommended layout:

```text
examples/data/
├── example_image.nii.gz
├── example_label.nii.gz
└── dicom_series/
    ├── IM0001.dcm
    ├── IM0002.dcm
    └── ...
```

Supported formats in the notebooks:

- NIfTI: `.nii` or `.nii.gz`
- NumPy arrays: `.npy`
- DICOM: a directory containing one anonymized DICOM series

The notebooks can generate synthetic NumPy data for dataset and patch-sampling
tutorials. The I/O notebook can also generate a synthetic NIfTI file when
`medimageflow[notebooks]` or compatible imaging dependencies are installed.

DICOM data is optional. If `examples/data/dicom_series/` is missing, the DICOM
section of the I/O notebook prints a clear message and skips that part.

Acceptable public sample data should come from a reputable source, be fully
de-identified, and include a license that allows redistribution or local use.
