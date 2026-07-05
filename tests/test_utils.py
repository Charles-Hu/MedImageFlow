from medical_toolkit.utils import find_files


def test_find_files_is_sorted(tmp_path) -> None:
    """Ensure file discovery returns deterministic lexical ordering.

    Files are created in reverse name order. The intermediate directory scan
    must normalize that filesystem-dependent order, and the output should list
    ``a.nii.gz`` before ``b.nii.gz``.
    """
    (tmp_path / "b.nii.gz").touch()
    (tmp_path / "a.nii.gz").touch()
    assert [path.name for path in find_files(tmp_path)] == ["a.nii.gz", "b.nii.gz"]
