# Core concepts

## Sample

`Sample` is the case-level record. It contains a mapping of image names to paths,
optional trainable `features`, an optional `id`, and optional `metadata`.
MedImageFlow reads image paths only when a dataset item is requested.

Reserved image names are `id`, `metadata`, `features`, and `__timing__`.

## SampleSource

A `SampleSource` is any indexable object with `__len__` and `__getitem__` that
returns `Sample` objects. Built-in implementations include:

- `MappingSampleSource` for record sequences.
- `CSVSampleSource` for CSV files.
- `DirectorySampleSource` for matching explicit `{id}` path patterns.

`MedicalImageDataset.from_csv(...)` and
`MedicalImageDataset.from_directory(...)` are convenience constructors around
these sources.

## Feature

`features` are non-image values that should participate in training or
inference. Numeric scalars are converted to one-element NumPy arrays, so default
PyTorch collation produces shapes such as `(batch_size, 1)`.

Use `metadata` for values that only identify or describe a case and should not
be treated as training features.

## Dataset

`MedicalImageDataset` reads every path in a sample and returns a dictionary
containing `id`, `metadata`, `features`, and loaded image arrays.

`PatchDataset` extends the whole-image dataset by selecting one patch center and
cropping all selected `patch_keys` with the same geometry.

## Reader

A reader implements `supports(path)` and `read(path)`. Custom readers are
checked before built-in readers. Built-ins support:

- `.npy` files through `NumpyReader`.
- `.nii` and `.nii.gz` through `NiftiReader`.
- DICOM series directories through `DicomSeriesReader`.

Readers return arrays only. Use `medimageflow.io` functions when affine or other
spatial metadata matters.

## Sampler

A patch sampler returns integer voxel centers. `RandomPatchSampler` can sample
globally or from a binary ROI. `GridPatchSampler` returns deterministic
in-bounds grid centers and ignores ROI masks.

`PatchDataset` currently consumes one center per dataset item.

## Transform

Sample transforms receive and return the complete item dictionary. Field
transforms receive one image array and must return `numpy.ndarray`.

Processing order in `PatchDataset` is:

1. Read paths.
2. Apply feature transforms.
3. Apply shared whole-image transform.
4. Apply image field transforms.
5. Sample and crop patches.
6. Apply shared patch transform.
7. Apply patch field transforms.
