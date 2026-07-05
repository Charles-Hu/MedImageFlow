"""Path-first whole-image and patch-based datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Union

import numpy as np
from numpy.typing import NDArray

from medical_toolkit.data.patch import (
    PatchSampler,
    extract_centered_patch,
    full_patch_axes,
    get_spatial_shape,
    patch_start,
    resolve_patch_size,
)
from medical_toolkit.data.readers import ImageReader, ReaderRegistry

SampleDict = dict[str, Any]
Transform = Callable[[SampleDict], SampleDict]
ArrayTransform = Callable[[NDArray[Any]], NDArray[Any]]
FeatureTransform = Callable[[Any], Any]
FieldSelection = Union[Sequence[str], Mapping[str, str]]
TIMING_KEY = "__timing__"


def _field_mapping(selection: FieldSelection | None) -> dict[str, str]:
    """Normalize a field-name sequence or output-to-input mapping."""
    if selection is None:
        return {}
    if isinstance(selection, Mapping):
        return dict(selection)
    if isinstance(selection, str):
        raise TypeError("field selection must be a sequence or mapping, not a string")
    return {name: name for name in selection}


@dataclass(frozen=True)
class Sample:
    """Paths for any number of named, spatially aligned medical images.

    Names such as ``ct``, ``mri``, ``label``, and ``roi`` have no intrinsic
    special meaning except when selected by dataset configuration.
    """

    paths: Mapping[str, str | Path]
    features: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        paths: Mapping[str, str],
        features: FieldSelection | None = None,
        id: str | None = None,
        metadata: FieldSelection | None = None,
        base_dir: str | Path | None = None,
    ) -> Sample:
        """Build a sample by selecting fields from one record.

        Mapping arguments use ``{output_name: record_field}``. A sequence used
        for ``features`` or ``metadata`` preserves the record field names.

        Args:
            record: Source record, such as a CSV row or database result.
            paths: Sample path names mapped to fields in ``record``.
            features: Feature fields to select and optionally rename.
            id: Record field used as the sample identifier.
            metadata: Metadata fields to select and optionally rename.
            base_dir: Directory prepended to relative paths.

        Returns:
            A sample containing the selected record values.

        Raises:
            KeyError: If a selected field is absent from ``record``.
            TypeError: If a selected path is not a string or ``Path``.
        """

        def select(selection: FieldSelection | None) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for output_name, record_field in _field_mapping(selection).items():
                if record_field not in record:
                    raise KeyError(f"Record field {record_field!r} is missing")
                output[output_name] = record[record_field]
            return output

        if not paths:
            raise ValueError("paths must contain at least one field mapping")
        selected_paths: dict[str, Path] = {}
        root = Path(base_dir) if base_dir is not None else None
        for output_name, record_field in paths.items():
            if record_field not in record:
                raise KeyError(f"Record path field {record_field!r} is missing")
            value = record[record_field]
            if not isinstance(value, (str, Path)):
                raise TypeError(f"Record path field {record_field!r} must be str or Path")
            path = Path(value)
            selected_paths[output_name] = root / path if root and not path.is_absolute() else path

        identifier = None
        if id is not None:
            if id not in record:
                raise KeyError(f"Record ID field {id!r} is missing")
            identifier = str(record[id])
        return cls(
            paths=selected_paths,
            features=select(features),
            id=identifier,
            metadata=select(metadata),
        )


class SampleSource(Protocol):
    """Indexable source that returns samples on demand."""

    def __len__(self) -> int:
        """Return the number of available samples."""
        ...

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at ``index``."""
        ...


class MedicalImageDataset:
    """Whole-image dataset with shared and per-field processing hooks."""

    def __init__(
        self,
        samples: Sequence[Sample] | SampleSource,
        *,
        readers: Sequence[ImageReader] = (),
        image_transform: Transform | None = None,
        image_field_transforms: Mapping[str, ArrayTransform] | None = None,
        feature_transforms: Mapping[str, FeatureTransform] | None = None,
        timing: bool = False,
    ) -> None:
        """Initialize a path-based whole-image dataset.

        Args:
            samples: Samples containing image paths and non-image features.
            readers: Custom readers checked before the built-in readers.
            image_transform: Transform applied to the complete sample dictionary.
            image_field_transforms: Per-image transforms keyed by image name.
            feature_transforms: Per-feature transforms keyed by feature name.
            timing: Whether to collect per-sample stage timings.
        """
        self.samples = samples
        self.readers = ReaderRegistry(readers)
        self.image_transform = image_transform
        self.image_field_transforms = dict(image_field_transforms or {})
        self.feature_transforms = dict(feature_transforms or {})
        self.timing = timing

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        paths: Mapping[str, str],
        features: FieldSelection | None = None,
        id: str | None = None,
        metadata: FieldSelection | None = None,
        base_dir: str | Path | None = None,
        encoding: str = "utf-8-sig",
        **dataset_options: Any,
    ) -> MedicalImageDataset:
        """Create a dataset backed by lazily converted CSV records."""
        from medical_toolkit.data.sources import CSVSampleSource

        source = CSVSampleSource(
            csv_path,
            paths=paths,
            features=features,
            id=id,
            metadata=metadata,
            base_dir=base_dir,
            encoding=encoding,
        )
        return cls(source, **dataset_options)

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        paths: Mapping[str, str],
        **dataset_options: Any,
    ) -> MedicalImageDataset:
        """Create a dataset by matching ``{id}`` path patterns below a root."""
        from medical_toolkit.data.sources import DirectorySampleSource

        return cls(DirectorySampleSource(root, paths=paths), **dataset_options)

    def enable_timing(self, enabled: bool = True) -> None:
        """Enable or disable per-sample stage timing metadata.

        Args:
            enabled: Whether timing collection should be enabled.
        """
        self.timing = enabled

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            The dataset length.
        """
        return len(self.samples)

    def _load(self, index: int) -> SampleDict:
        """Read every image belonging to one sample.

        Args:
            index: Sample index.

        Returns:
            A dictionary containing loaded arrays, raw features, and metadata.

        Raises:
            ValueError: If no paths are supplied or an image name is reserved.
            FileNotFoundError: If an image path does not exist.
        """
        source = self.samples[index]
        values = dict(source.paths)
        if not values:
            raise ValueError("Sample must contain at least one image path")
        reserved = {"id", "metadata", "features", TIMING_KEY}
        conflicts = reserved.intersection(values)
        if conflicts:
            raise ValueError(f"Reserved image names are not allowed: {sorted(conflicts)}")
        item: SampleDict = {
            "id": source.id or str(index),
            "metadata": dict(source.metadata),
            "features": dict(source.features),
        }
        item.update({key: self.readers.read(value) for key, value in values.items()})
        return item

    @staticmethod
    def _apply_field_transforms(
        item: SampleDict, transforms: Mapping[str, ArrayTransform]
    ) -> SampleDict:
        """Apply independent transforms to named image arrays.

        Args:
            item: Sample dictionary containing image arrays.
            transforms: Array transforms keyed by image name.

        Returns:
            A shallow copy containing transformed arrays, or the original item
            when no transforms are configured.

        Raises:
            KeyError: If a transform references a missing image.
            TypeError: If a transform does not return a NumPy array.
        """
        if not transforms:
            return item
        output = dict(item)
        for key, transform in transforms.items():
            if key not in output:
                raise KeyError(f"No array named {key!r} is available for its field transform")
            result = transform(np.asarray(output[key]))
            if not isinstance(result, np.ndarray):
                raise TypeError(f"Field transform for {key!r} must return numpy.ndarray")
            output[key] = result
        return output

    @staticmethod
    def _make_batchable_feature(value: Any) -> Any:
        """Give numeric scalars a feature axis while preserving general values.

        Args:
            value: Feature value produced by an optional feature transform.

        Returns:
            A batch-friendly value. Numeric scalars become one-element arrays.
        """
        if isinstance(value, np.ndarray):
            return value.reshape(1) if value.ndim == 0 else value
        if isinstance(value, Number):
            return np.asarray([value])
        if isinstance(value, (list, tuple)):
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                return array
        return value

    def _prepare_features(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Transform and normalize all non-image features for collation.

        Args:
            features: Raw feature mapping from a sample.

        Returns:
            Prepared features keyed by their original names.

        Raises:
            KeyError: If a configured transform references a missing feature.
        """
        missing = tuple(key for key in self.feature_transforms if key not in features)
        if missing:
            raise KeyError(f"Feature transforms reference missing features: {missing}")
        output: dict[str, Any] = {}
        for key, value in features.items():
            transform = self.feature_transforms.get(key)
            result = transform(value) if transform else value
            output[key] = self._make_batchable_feature(result)
        return output

    def __getitem__(self, index: int) -> SampleDict:
        """Load and process one complete sample.

        Args:
            index: Sample index.

        Returns:
            The processed sample dictionary, optionally including timing data.
        """
        started = perf_counter() if self.timing else 0.0
        item = self._load(index)
        read_time = perf_counter() - started if self.timing else 0.0
        feature_started = perf_counter() if self.timing else 0.0
        item["features"] = self._prepare_features(item["features"])
        feature_time = perf_counter() - feature_started if self.timing else 0.0
        started = perf_counter() if self.timing else 0.0
        if self.image_transform:
            item = self.image_transform(item)
        item = self._apply_field_transforms(item, self.image_field_transforms)
        if self.timing:
            item[TIMING_KEY] = {
                "data_read": read_time,
                "image_processing": perf_counter() - started,
                "feature_processing": feature_time,
                "patch_extraction": 0.0,
                "patch_processing": 0.0,
            }
        return item


class PatchDataset(MedicalImageDataset):
    """Synchronously crop any number of aligned 2D or 3D arrays.

    Processing order is full image read -> feature transforms -> shared image
    transform -> per-field image transforms -> one shared center/crop -> shared
    patch transform -> per-field patch transforms.
    """

    def __init__(
        self,
        samples: Sequence[Sample] | SampleSource,
        sampler: PatchSampler,
        *,
        spatial_dims: int,
        channel_axis: int | None = None,
        readers: Sequence[ImageReader] = (),
        image_transform: Transform | None = None,
        image_field_transforms: Mapping[str, ArrayTransform] | None = None,
        feature_transforms: Mapping[str, FeatureTransform] | None = None,
        patch_transform: Transform | None = None,
        patch_field_transforms: Mapping[str, ArrayTransform] | None = None,
        patch_keys: Sequence[str] | None = None,
        reference_key: str | None = None,
        center_mask_key: str | None = "roi",
        channel_axes: Mapping[str, int | None] | None = None,
        padding_mode: str | Mapping[str, str] = "constant",
        padding_value: float | Mapping[str, float] = 0,
        timing: bool = False,
    ) -> None:
        """Initialize a synchronized multimodal patch dataset.

        Args:
            samples: Samples containing aligned image paths and features.
            sampler: Strategy used to select one shared patch center.
            spatial_dims: Number of spatial dimensions; must be 2 or 3.
            channel_axis: Default channel axis for arrays with channels.
            readers: Custom image readers checked before built-in readers.
            image_transform: Shared whole-image sample transform.
            image_field_transforms: Independent whole-image array transforms.
            feature_transforms: Independent non-image feature transforms.
            patch_transform: Shared transform applied after patch extraction.
            patch_field_transforms: Independent transforms applied to patches.
            patch_keys: Image names to crop. All arrays are used when omitted.
            reference_key: Image name used to determine the spatial shape.
            center_mask_key: Binary mask used for ROI center sampling.
            channel_axes: Per-image channel-axis overrides.
            padding_mode: Global or per-image NumPy padding mode.
            padding_value: Global or per-image constant padding value.
            timing: Whether to collect per-sample stage timings.

        Raises:
            ValueError: If ``spatial_dims`` is not 2 or 3.
        """
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        super().__init__(
            samples,
            readers=readers,
            image_transform=image_transform,
            image_field_transforms=image_field_transforms,
            feature_transforms=feature_transforms,
            timing=timing,
        )
        self.sampler = sampler
        self.spatial_dims = spatial_dims
        self.channel_axis = channel_axis
        self.channel_axes = dict(channel_axes or {})
        self.patch_transform = patch_transform
        self.patch_field_transforms = dict(patch_field_transforms or {})
        self.patch_keys = tuple(patch_keys) if patch_keys is not None else None
        self.reference_key = reference_key
        self.center_mask_key = center_mask_key
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __len__(self) -> int:
        """Return the number of samples and therefore returned patches.

        Returns:
            The dataset length.
        """
        return len(self.samples)

    def _axis_for(self, key: str) -> int | None:
        """Resolve the channel axis for one image.

        Args:
            key: Image name.

        Returns:
            The image-specific channel axis or the dataset default.
        """
        return self.channel_axes.get(key, self.channel_axis)

    @staticmethod
    def _setting_for(key: str, setting: Any, default: Any) -> Any:
        """Resolve a global or per-image configuration value.

        Args:
            key: Image name.
            setting: Scalar setting or mapping keyed by image name.
            default: Fallback used when a mapping omits ``key``.

        Returns:
            The resolved setting.
        """
        return setting.get(key, default) if isinstance(setting, Mapping) else setting

    def __getitem__(self, index: int) -> SampleDict:
        """Load a sample and extract one synchronized multimodal patch.

        Args:
            index: Sample index.

        Returns:
            A sample dictionary containing cropped images, prepared features,
            patch coordinates, metadata, and optional timing information.

        Raises:
            KeyError: If a requested patch image is missing.
            ValueError: If image shapes are not aligned or configuration is invalid.
            IndexError: If the sampler does not return a center.
        """
        item = super().__getitem__(index)
        timing = item.pop(TIMING_KEY, None)
        extraction_started = perf_counter() if timing is not None else 0.0
        available_keys = tuple(key for key, value in item.items() if isinstance(value, np.ndarray))
        patch_keys = self.patch_keys or available_keys
        if not patch_keys:
            raise ValueError("No arrays are available for patch extraction")
        missing = tuple(key for key in patch_keys if key not in item)
        if missing:
            raise KeyError(f"Patch arrays are missing: {missing}")

        reference_key = self.reference_key
        if reference_key is None:
            candidates = tuple(key for key in patch_keys if key != self.center_mask_key)
            if not candidates:
                raise ValueError("A non-mask reference array is required")
            reference_key = "image" if "image" in candidates else candidates[0]
        if reference_key not in patch_keys:
            raise ValueError(f"reference_key {reference_key!r} must be included in patch_keys")

        reference = np.asarray(item[reference_key])
        shape = get_spatial_shape(reference, self.spatial_dims, self._axis_for(reference_key))
        for key in patch_keys:
            field_shape = get_spatial_shape(
                np.asarray(item[key]), self.spatial_dims, self._axis_for(key)
            )
            if field_shape != shape:
                raise ValueError(
                    f"Array {key!r} has spatial shape {field_shape}, expected aligned shape {shape}"
                )

        center_mask = None
        if self.center_mask_key is not None and self.center_mask_key in item:
            center_mask = np.asarray(item[self.center_mask_key])
        centers = self.sampler.centers(
            shape,
            1,
            center_mask=center_mask,
            seed_offset=index,
        )
        try:
            center = next(iter(centers))
        except StopIteration as error:
            raise IndexError("Sampler did not return a patch center") from error

        full_axes = full_patch_axes(self.sampler.patch_size, self.spatial_dims)
        center = tuple(
            shape[axis] // 2 if axis in full_axes else coordinate
            for axis, coordinate in enumerate(center)
        )
        patch_size = resolve_patch_size(self.sampler.patch_size, shape)
        output = dict(item)
        for key in patch_keys:
            mode = self._setting_for(key, self.padding_mode, "constant")
            value = self._setting_for(key, self.padding_value, 0)
            output[key] = extract_centered_patch(
                np.asarray(item[key]),
                center,
                patch_size,
                spatial_dims=self.spatial_dims,
                channel_axis=self._axis_for(key),
                padding_mode=mode,
                padding_value=value,
            )
        output["patch_center"] = np.asarray(center, dtype=np.int64)
        output["patch_location"] = np.asarray(patch_start(center, patch_size), dtype=np.int64)
        output["image_index"] = index
        extraction_time = perf_counter() - extraction_started if timing is not None else 0.0
        processing_started = perf_counter() if timing is not None else 0.0
        if self.patch_transform:
            output = self.patch_transform(output)
        output = self._apply_field_transforms(output, self.patch_field_transforms)
        if timing is not None:
            timing["patch_extraction"] = extraction_time
            timing["patch_processing"] = perf_counter() - processing_started
            output[TIMING_KEY] = timing
        return output
