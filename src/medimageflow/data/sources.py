"""Indexable sources that construct samples only when accessed."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medimageflow.data.dataset import FieldSelection, Sample


class MappingSampleSource:
    """Convert indexable mapping records to samples on demand."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        paths: Mapping[str, str],
        features: FieldSelection | None = None,
        id: str | None = None,
        metadata: FieldSelection | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.records = records
        self.paths = dict(paths)
        self.features = features
        self.id_field = id
        self.metadata = metadata
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        return Sample.from_mapping(
            self.records[index],
            paths=self.paths,
            features=self.features,
            id=self.id_field,
            metadata=self.metadata,
            base_dir=self.base_dir,
        )


class CSVSampleSource(MappingSampleSource):
    """Read CSV records once and convert individual rows on access."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        paths: Mapping[str, str],
        features: FieldSelection | None = None,
        id: str | None = None,
        metadata: FieldSelection | None = None,
        base_dir: str | Path | None = None,
        encoding: str = "utf-8-sig",
    ) -> None:
        csv_path = Path(csv_path)
        with csv_path.open(newline="", encoding=encoding) as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError(f"CSV contains no data records: {csv_path}")
        if id is not None:
            identifiers: set[str] = set()
            duplicates: set[str] = set()
            for record in records:
                identifier = record.get(id)
                if identifier is None:
                    raise KeyError(f"CSV ID field {id!r} is missing")
                if identifier in identifiers:
                    duplicates.add(identifier)
                identifiers.add(identifier)
            if duplicates:
                raise ValueError(f"CSV contains duplicate sample IDs: {sorted(duplicates)}")
        root = Path(base_dir) if base_dir is not None else csv_path.parent
        super().__init__(
            records,
            paths=paths,
            features=features,
            id=id,
            metadata=metadata,
            base_dir=root,
        )


class DirectorySampleSource:
    """Discover samples from explicit relative patterns containing ``{id}``."""

    def __init__(self, root: str | Path, *, paths: Mapping[str, str]) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if not paths:
            raise ValueError("paths must contain at least one named pattern")
        for pattern in paths.values():
            if pattern.count("{id}") != 1 or Path(pattern).is_absolute():
                raise ValueError("each path pattern must be relative and contain one {id}")
        self.patterns = dict(paths)
        self._records = self._discover()

    @staticmethod
    def _matcher(pattern: str) -> re.Pattern[str]:
        prefix, suffix = pattern.replace("\\", "/").split("{id}")
        return re.compile(f"^{re.escape(prefix)}(?P<id>.+?){re.escape(suffix)}$")

    def _discover(self) -> list[tuple[str, dict[str, Path]]]:
        discovered: dict[str, dict[str, Path]] = {}
        for name, pattern in self.patterns.items():
            matcher = self._matcher(pattern)
            glob_pattern = pattern.replace("{id}", "*")
            for candidate in sorted(self.root.glob(glob_pattern)):
                relative = candidate.relative_to(self.root).as_posix()
                match = matcher.fullmatch(relative)
                if match is None:
                    continue
                identifier = match.group("id")
                fields = discovered.setdefault(identifier, {})
                if name in fields:
                    raise ValueError(f"Multiple paths match {name!r} for sample {identifier!r}")
                fields[name] = candidate

        expected = set(self.patterns)
        if not discovered:
            raise ValueError(f"No samples match the configured patterns below {self.root}")
        records: list[tuple[str, dict[str, Path]]] = []
        for identifier, fields in sorted(discovered.items()):
            missing = expected.difference(fields)
            if missing:
                raise ValueError(f"Sample {identifier!r} is missing paths: {sorted(missing)}")
            records.append((identifier, fields))
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        identifier, paths = self._records[index]
        return Sample(paths=paths, id=identifier)
