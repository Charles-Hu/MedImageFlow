from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medical_toolkit.data import (
    CSVSampleSource,
    DirectorySampleSource,
    GridPatchSampler,
    MappingSampleSource,
    MedicalImageDataset,
    PatchDataset,
    Sample,
)


def test_sample_from_mapping_selects_renames_and_resolves_fields(tmp_path: Path) -> None:
    """Verify the complete record-to-Sample mapping contract.

    The input record contains differently named path, feature, ID, and metadata
    fields. The key intermediate result is that the relative image path is
    resolved below ``base_dir`` while non-path values retain their types. The
    output should contain only selected fields under their requested names.
    """
    sample = Sample.from_mapping(
        {
            "case": 7,
            "image_path": "images/7.npy",
            "patient_age": 62,
            "hospital": "site-a",
        },
        paths={"image": "image_path"},
        features={"age": "patient_age"},
        id="case",
        metadata=("hospital",),
        base_dir=tmp_path,
    )

    assert sample.paths == {"image": tmp_path / "images/7.npy"}
    assert sample.features == {"age": 62}
    assert sample.id == "7"
    assert sample.metadata == {"hospital": "site-a"}


def test_sample_from_mapping_preserves_names_for_sequence_selection() -> None:
    """Check sequence-based feature and metadata selection.

    The input uses tuples instead of rename mappings. During conversion each
    selected field should map to itself. The resulting Sample should preserve
    the original names and values, with an absent ID represented by ``None``.
    """
    sample = Sample.from_mapping(
        {"path": "scan.npy", "age": 40, "site": "north", "unused": True},
        paths={"ct": "path"},
        features=("age",),
        metadata=("site",),
    )

    assert sample.paths == {"ct": Path("scan.npy")}
    assert sample.features == {"age": 40}
    assert sample.metadata == {"site": "north"}
    assert sample.id is None


def test_sample_from_mapping_does_not_prefix_absolute_paths(tmp_path: Path) -> None:
    """Ensure ``base_dir`` is applied only to relative input paths.

    The record supplies one absolute and one relative path. The intermediate
    path resolution should leave the absolute path untouched and prefix only
    the relative path. Both resolved paths should appear in the final Sample.
    """
    absolute = tmp_path / "absolute.npy"
    sample = Sample.from_mapping(
        {"absolute": absolute, "relative": "relative.npy"},
        paths={"ct": "absolute", "label": "relative"},
        base_dir=tmp_path / "base",
    )

    assert sample.paths["ct"] == absolute
    assert sample.paths["label"] == tmp_path / "base/relative.npy"


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"paths": {}}, ValueError, "at least one"),
        ({"paths": {"image": "missing"}}, KeyError, "path field"),
        ({"paths": {"image": "path"}, "features": ("missing",)}, KeyError, "missing"),
        ({"paths": {"image": "path"}, "metadata": ("missing",)}, KeyError, "missing"),
        ({"paths": {"image": "path"}, "id": "missing"}, KeyError, "ID field"),
        ({"paths": {"image": "path"}, "features": "age"}, TypeError, "selection"),
    ],
)
def test_sample_from_mapping_rejects_invalid_field_configuration(
    kwargs: dict[str, Any], error_type: type[Exception], message: str
) -> None:
    """Cover missing fields, empty paths, and ambiguous string selections.

    Each parameter supplies a malformed configuration against a record with a
    valid ``path`` field. Conversion should stop at the relevant validation
    step and raise the documented exception instead of creating a partial
    Sample or silently treating a string as a sequence of characters.
    """
    record = {"path": "scan.npy", "age": 50}

    with pytest.raises(error_type, match=message):
        Sample.from_mapping(record, **kwargs)


def test_sample_from_mapping_rejects_non_path_values() -> None:
    """Verify that arbitrary record values cannot enter ``Sample.paths``.

    The selected record field contains an integer rather than ``str`` or
    ``Path``. Path conversion is the key validation step and should raise a
    TypeError before a Sample is returned.
    """
    with pytest.raises(TypeError, match="must be str or Path"):
        Sample.from_mapping({"path": 123}, paths={"image": "path"})


def test_mapping_source_is_lazy_and_delegates_indexing() -> None:
    """Confirm that MappingSampleSource accesses only the requested record.

    The custom input records all requested indices. Calling ``len`` must not
    touch any record; requesting index 1 should record exactly ``[1]`` and
    produce the corresponding Sample ID and path.
    """

    class Records:
        def __init__(self) -> None:
            self.requested: list[int] = []

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, Any]:
            self.requested.append(index)
            return {"id": index, "path": f"{index}.npy"}

    records = Records()
    source = MappingSampleSource(records, paths={"image": "path"}, id="id")

    assert len(source) == 2
    assert records.requested == []
    assert source[1] == Sample(paths={"image": Path("1.npy")}, id="1")
    assert records.requested == [1]


def test_mapping_source_defers_record_validation_until_access() -> None:
    """Check that lazy conversion also defers per-record mapping errors.

    Construction receives a record missing the configured path field and
    should still succeed because no record has been requested. Accessing index
    zero is the key conversion point and should then raise the missing-field
    KeyError.
    """
    source = MappingSampleSource([{"id": "a"}], paths={"image": "path"}, id="id")

    assert len(source) == 1
    with pytest.raises(KeyError, match="path field"):
        source[0]


def test_mapping_source_supports_negative_indices_and_reports_out_of_range() -> None:
    """Ensure source indexing follows the wrapped sequence's semantics.

    A two-record list is valid input. Index ``-1`` should resolve to the last
    record and produce ID ``b``; index 2 should propagate the underlying
    IndexError without wrapping it in an unrelated conversion exception.
    """
    source = MappingSampleSource(
        [{"id": "a", "path": "a.npy"}, {"id": "b", "path": "b.npy"}],
        paths={"image": "path"},
        id="id",
    )

    assert source[-1].id == "b"
    with pytest.raises(IndexError):
        source[2]


def test_csv_source_maps_all_field_groups_and_uses_csv_directory(tmp_path: Path) -> None:
    """Verify CSV parsing, field renaming, and default relative path handling.

    The CSV contains one row with a relative image path and all Sample field
    groups. After source construction, indexed conversion should resolve the
    path against the CSV directory, preserve CSV values as strings, and map
    each selected column to the expected output name.
    """
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "patient,image,years,hospital\na,images/a.npy,51,site-a\n",
        encoding="utf-8",
    )
    source = CSVSampleSource(
        csv_path,
        paths={"ct": "image"},
        features={"age": "years"},
        id="patient",
        metadata={"site": "hospital"},
    )

    sample = source[0]
    assert sample.paths == {"ct": tmp_path / "images/a.npy"}
    assert sample.features == {"age": "51"}
    assert sample.id == "a"
    assert sample.metadata == {"site": "site-a"}


def test_csv_source_honors_base_dir_and_encoding(tmp_path: Path) -> None:
    """Check explicit path roots and non-default CSV encodings together.

    A Latin-1 CSV contains a non-ASCII metadata value and a relative path.
    Decoding should preserve ``Montréal`` while path resolution should use the
    explicit base directory instead of the CSV directory.
    """
    csv_path = tmp_path / "samples.csv"
    csv_path.write_bytes("id,image,site\na,a.npy,Montréal\n".encode("latin-1"))
    image_root = tmp_path / "external"
    source = CSVSampleSource(
        csv_path,
        paths={"image": "image"},
        id="id",
        metadata=("site",),
        base_dir=image_root,
        encoding="latin-1",
    )

    assert source[0].paths == {"image": image_root / "a.npy"}
    assert source[0].metadata == {"site": "Montréal"}


def test_csv_source_accepts_utf8_bom_with_default_encoding(tmp_path: Path) -> None:
    """Ensure the default ``utf-8-sig`` encoding removes a CSV BOM.

    The first header includes a UTF-8 byte-order mark. The intermediate
    DictReader keys should effectively become ``id`` and ``image``; selecting
    those fields should produce a normal Sample rather than a missing-ID error.
    """
    csv_path = tmp_path / "samples.csv"
    csv_path.write_bytes(b"\xef\xbb\xbfid,image\na,a.npy\n")

    source = CSVSampleSource(csv_path, paths={"image": "image"}, id="id")

    assert source[0].id == "a"
    assert source[0].paths["image"] == tmp_path / "a.npy"


@pytest.mark.parametrize("content", ["", "id,image\n"])
def test_csv_source_rejects_files_without_data_rows(tmp_path: Path, content: str) -> None:
    """Cover completely empty and header-only CSV inputs.

    Neither input contains a usable sample record. CSV parsing should yield no
    records and source construction should raise ValueError immediately, so a
    zero-length dataset is not created accidentally.
    """
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="no data records"):
        CSVSampleSource(csv_path, paths={"image": "image"})


def test_csv_source_rejects_missing_and_duplicate_ids(tmp_path: Path) -> None:
    """Validate ID-column presence and uniqueness at source construction.

    The first CSV lacks the configured ID column and should raise KeyError.
    The second repeats ID ``a`` and should raise ValueError. Both checks must
    occur before indexing because IDs define sample identity globally.
    """
    missing = tmp_path / "missing.csv"
    missing.write_text("image\na.npy\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("id,image\na,a.npy\na,b.npy\n", encoding="utf-8")

    with pytest.raises(KeyError, match="ID field"):
        CSVSampleSource(missing, paths={"image": "image"}, id="id")
    with pytest.raises(ValueError, match="duplicate"):
        CSVSampleSource(duplicate, paths={"image": "image"}, id="id")


def test_csv_source_preserves_row_order_and_allows_no_id_mapping(tmp_path: Path) -> None:
    """Check stable CSV ordering when no identity field is requested.

    Two rows deliberately use the same unrelated group value. Since no ID
    mapping is configured, uniqueness validation should not apply. Access by
    index should preserve file order and both Samples should have ``id=None``.
    """
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "image,group\nsecond.npy,same\nfirst.npy,same\n",
        encoding="utf-8",
    )
    source = CSVSampleSource(csv_path, paths={"image": "image"})

    assert source[0].paths["image"].name == "second.npy"
    assert source[1].paths["image"].name == "first.npy"
    assert source[0].id is None
    assert source[1].id is None


@pytest.mark.parametrize(
    "patterns",
    [
        {},
        {"image": "case/image.npy"},
        {"image": "{id}/{id}.npy"},
    ],
)
def test_directory_source_rejects_invalid_pattern_sets(
    tmp_path: Path, patterns: dict[str, str]
) -> None:
    """Validate empty patterns and incorrect ``{id}`` placeholder counts.

    Each supplied pattern set violates the directory-source grammar: no paths,
    no placeholder, or two placeholders. Construction should reject all three
    before scanning the directory, preventing ambiguous grouping rules.
    """
    with pytest.raises(ValueError):
        DirectorySampleSource(tmp_path, paths=patterns)


def test_directory_source_rejects_absolute_pattern(tmp_path: Path) -> None:
    """Ensure patterns cannot escape or replace the configured root.

    The input pattern is absolute even though it contains one ``{id}`` token.
    Validation should fail before globbing, keeping all discovery scoped below
    the supplied root directory.
    """
    pattern = str(tmp_path / "{id}/image.npy")

    with pytest.raises(ValueError, match="relative"):
        DirectorySampleSource(tmp_path, paths={"image": pattern})


def test_directory_source_rejects_missing_root_and_empty_matches(tmp_path: Path) -> None:
    """Distinguish an invalid root from a valid root with no matching samples.

    A nonexistent root should raise NotADirectoryError during root validation.
    The existing but empty root should pass that step, then raise ValueError
    after discovery finds no paths matching the configured pattern.
    """
    with pytest.raises(NotADirectoryError):
        DirectorySampleSource(
            tmp_path / "missing",
            paths={"image": "{id}/image.npy"},
        )
    with pytest.raises(ValueError, match="No samples match"):
        DirectorySampleSource(tmp_path, paths={"image": "{id}/image.npy"})


def test_directory_source_pairs_paths_and_sorts_sample_ids(tmp_path: Path) -> None:
    """Verify deterministic multimodal pairing independent of creation order.

    Cases are created in reverse lexical order with both image and label
    files. Discovery should group files by the captured ``{id}``, sort IDs,
    and return two Samples whose path dictionaries contain correctly paired
    files from the same case directory.
    """
    for identifier in ("case-b", "case-a"):
        case = tmp_path / identifier
        case.mkdir()
        (case / "image.npy").touch()
        (case / "label.npy").touch()
    source = DirectorySampleSource(
        tmp_path,
        paths={"image": "{id}/image.npy", "label": "{id}/label.npy"},
    )

    assert [source[index].id for index in range(len(source))] == ["case-a", "case-b"]
    assert source[0].paths == {
        "image": tmp_path / "case-a/image.npy",
        "label": tmp_path / "case-a/label.npy",
    }


def test_directory_source_supports_prefix_suffix_patterns(tmp_path: Path) -> None:
    """Check ID extraction when ``{id}`` appears inside a filename.

    Files use ``scan-<id>-ct.npy`` and ``scan-<id>-label.npy`` patterns rather
    than per-case directories. Matchers should capture only the middle ID,
    pair ``alpha`` paths, and return that exact identifier in the Sample.
    """
    (tmp_path / "scan-alpha-ct.npy").touch()
    (tmp_path / "scan-alpha-label.npy").touch()
    source = DirectorySampleSource(
        tmp_path,
        paths={
            "ct": "scan-{id}-ct.npy",
            "label": "scan-{id}-label.npy",
        },
    )

    assert len(source) == 1
    assert source[0].id == "alpha"
    assert source[0].paths["label"].name == "scan-alpha-label.npy"


def test_directory_source_rejects_incomplete_samples(tmp_path: Path) -> None:
    """Prevent partially matched modalities from becoming valid Samples.

    The case contains an image but no required label. Discovery should first
    identify ``case-a`` from the image, then detect the absent label during
    completeness validation and raise ValueError listing missing paths.
    """
    case = tmp_path / "case-a"
    case.mkdir()
    (case / "image.npy").touch()

    with pytest.raises(ValueError, match="missing paths"):
        DirectorySampleSource(
            tmp_path,
            paths={"image": "{id}/image.npy", "label": "{id}/label.npy"},
        )


def test_directory_source_supports_negative_and_out_of_range_indices(tmp_path: Path) -> None:
    """Confirm list-like indexing after directory discovery.

    Two complete cases are discovered and sorted. Negative index ``-1`` should
    produce the last Sample, while an index equal to the source length should
    propagate IndexError from the internal record sequence.
    """
    for identifier in ("a", "b"):
        case = tmp_path / identifier
        case.mkdir()
        (case / "image.npy").touch()
    source = DirectorySampleSource(tmp_path, paths={"image": "{id}/image.npy"})

    assert source[-1].id == "b"
    with pytest.raises(IndexError):
        source[2]


def test_dataset_from_csv_loads_images_features_metadata_and_transforms(tmp_path: Path) -> None:
    """Exercise all three layers in one whole-image dataset workflow.

    A CSV row points to a real NumPy image and maps feature and metadata fields.
    The factory should create a CSV source, lazy Sample conversion should retain
    those fields, and Dataset access should read the image and apply the feature
    transform. The final item should contain doubled age ``[102]`` and image data.
    """
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.arange(4).reshape(2, 2))
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "id,image,age,site\ncase-a,image.npy,51,north\n",
        encoding="utf-8",
    )
    dataset = MedicalImageDataset.from_csv(
        csv_path,
        paths={"image": "image"},
        features=("age",),
        id="id",
        metadata=("site",),
        feature_transforms={"age": lambda value: int(value) * 2},
    )

    item = dataset[0]
    np.testing.assert_array_equal(item["image"], np.arange(4).reshape(2, 2))
    np.testing.assert_array_equal(item["features"]["age"], [102])
    assert item["id"] == "case-a"
    assert item["metadata"] == {"site": "north"}


def test_dataset_from_directory_loads_paired_arrays(tmp_path: Path) -> None:
    """Combine directory discovery with multimodal Dataset image loading.

    One directory case contains an image and label NumPy array. Discovery must
    pair them under one ID, and Dataset access must pass both paths through the
    reader registry. The output should expose both original arrays and the
    identifier captured by the directory pattern.
    """
    case = tmp_path / "case-a"
    case.mkdir()
    np.save(case / "image.npy", np.ones((2, 2)))
    np.save(case / "label.npy", np.zeros((2, 2)))
    dataset = MedicalImageDataset.from_directory(
        tmp_path,
        paths={"image": "{id}/image.npy", "label": "{id}/label.npy"},
    )

    item = dataset[0]
    np.testing.assert_array_equal(item["image"], np.ones((2, 2)))
    np.testing.assert_array_equal(item["label"], np.zeros((2, 2)))
    assert item["id"] == "case-a"


def test_patch_dataset_from_csv_combines_factory_source_and_patch_logic(tmp_path: Path) -> None:
    """Verify the inherited CSV factory can construct a PatchDataset safely.

    The CSV source resolves one 4x4 array. Factory options include the required
    patch sampler and spatial dimension, so ``cls(source, **options)`` should
    create a PatchDataset rather than a base dataset. Access should return a
    deterministic 2x2 grid patch while retaining the CSV-provided ID.
    """
    np.save(tmp_path / "image.npy", np.arange(16).reshape(4, 4))
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("id,image\ncase-a,image.npy\n", encoding="utf-8")
    dataset = PatchDataset.from_csv(
        csv_path,
        paths={"image": "image"},
        id="id",
        sampler=GridPatchSampler((2, 2)),
        spatial_dims=2,
    )

    assert isinstance(dataset, PatchDataset)
    item = dataset[0]
    assert item["image"].shape == (2, 2)
    assert item["id"] == "case-a"


def test_dataset_keeps_custom_source_lazy_until_item_access(tmp_path: Path) -> None:
    """Ensure direct SampleSource injection is not eagerly converted to a list.

    MappingSampleSource wraps records that log index access and points to a real
    image. Dataset construction and ``len(dataset)`` should leave the log empty;
    requesting item zero should log one access and then load the expected array.
    """
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.ones((1, 1)))

    class Records:
        def __init__(self) -> None:
            self.requested: list[int] = []

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, Any]:
            self.requested.append(index)
            return {"path": image_path}

    records = Records()
    source = MappingSampleSource(records, paths={"image": "path"})
    dataset = MedicalImageDataset(source)

    assert len(dataset) == 1
    assert records.requested == []
    np.testing.assert_array_equal(dataset[0]["image"], np.ones((1, 1)))
    assert records.requested == [0]
