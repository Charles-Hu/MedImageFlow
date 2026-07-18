# Current limitations

MedImageFlow is an early-stage research toolkit. These limits are intentional
or currently unresolved.

- The toolkit does not perform automatic registration, resampling, orientation
  normalization, or spacing alignment. Aligned multimodal inputs must already
  share the intended voxel grid before synchronized patch extraction.
- `PatchDataset` reads complete images before cropping. Storage-level lazy patch
  I/O and caching are not implemented.
- `PatchDataset` returns one patch per sample access. It does not yet expand one
  source image into multiple patches inside the dataset length.
- `RandomPatchSampler(seed=None)` draws new random centers on each access. A
  fixed seed makes the same sample index deterministic across epochs; there is
  no `set_epoch()` API yet for reproducible epoch-varying centers.
- ROI sampling is implemented by `RandomPatchSampler`. `GridPatchSampler`
  accepts the sampler protocol arguments but ignores ROI masks.
- Each image array may have at most one channel axis.
- Built-in readers return arrays only. Use lower-level I/O functions when
  affine, spacing, origin, or direction metadata is required.
- DICOM conversion is deliberately conservative. The strict converter rejects
  enhanced multi-frame DICOM, colour images, irregular geometry, duplicate or
  ambiguous slice locations, and in-plane slice displacement.
- Built-in spatial augmentation is not available. Use shared sample transforms
  to integrate MONAI, TorchIO, or project-specific augmentation code.
