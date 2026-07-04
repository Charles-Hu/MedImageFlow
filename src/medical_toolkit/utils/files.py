from __future__ import annotations

from pathlib import Path


def find_files(root: str | Path, pattern: str = "*.nii.gz", *, recursive: bool = True) -> list[Path]:
    """Find files deterministically under a directory.

    Args:
        root: Directory to search.
        pattern: Glob pattern used to select files.
        recursive: Whether to search subdirectories recursively.

    Returns:
        A sorted list of matching file paths.
    """
    root = Path(root)
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in iterator if path.is_file())
