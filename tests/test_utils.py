from medical_toolkit.utils import find_files


def test_find_files_is_sorted(tmp_path) -> None:
    (tmp_path / "b.nii.gz").touch()
    (tmp_path / "a.nii.gz").touch()
    assert [path.name for path in find_files(tmp_path)] == ["a.nii.gz", "b.nii.gz"]

