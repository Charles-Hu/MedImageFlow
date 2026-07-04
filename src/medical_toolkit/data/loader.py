"""Optional PyTorch DataLoader factory and iteration timing wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from time import perf_counter
from typing import Any

from medical_toolkit.data.dataset import TIMING_KEY
from medical_toolkit.utils.optional import require


def _metric_total(value: Any) -> float:
    """Reduce a collated timing value to a floating-point total.

    Args:
        value: Scalar, sequence, NumPy array, or tensor-like timing value.

    Returns:
        The sum represented as a Python float.
    """
    if isinstance(value, (list, tuple)):
        return sum(_metric_total(item) for item in value)
    total = value.sum() if hasattr(value, "sum") else value
    total = total.item() if hasattr(total, "item") else total
    return float(total)


def _stage_totals(batch: Any) -> dict[str, float]:
    """Extract aggregate stage timings from a collated batch.

    Args:
        batch: Batch returned by a DataLoader iterator.

    Returns:
        Stage names mapped to aggregate durations in seconds.
    """
    if not isinstance(batch, Mapping):
        return {}
    timing = batch.get(TIMING_KEY)
    if not isinstance(timing, Mapping):
        return {}
    return {str(key): _metric_total(value) for key, value in timing.items()}


class _TimedIterator:
    def __init__(self, iterator: Iterator[Any], reporter: Callable[[str], None]) -> None:
        """Initialize a timed iterator.

        Args:
            iterator: Underlying DataLoader iterator.
            reporter: Callback receiving one formatted timing line per batch.
        """
        self._iterator = iterator
        self._reporter = reporter
        self._iteration = 0

    def __iter__(self) -> _TimedIterator:
        """Return this iterator.

        Returns:
            This timed iterator instance.
        """
        return self

    def __next__(self) -> Any:
        """Fetch, report, and return the next batch.

        Returns:
            The next collated batch.

        Raises:
            StopIteration: When the underlying iterator is exhausted.
        """
        started = perf_counter()
        batch = next(self._iterator)
        total = perf_counter() - started
        self._iteration += 1
        stages = _stage_totals(batch)
        self._reporter(
            "[medical_toolkit timing] "
            f"iteration={self._iteration} "
            f"dataloader_total={total:.6f}s "
            f"data_read={stages.get('data_read', 0.0):.6f}s "
            f"image_processing={stages.get('image_processing', 0.0):.6f}s "
            f"feature_processing={stages.get('feature_processing', 0.0):.6f}s "
            f"patch_extraction={stages.get('patch_extraction', 0.0):.6f}s "
            f"patch_processing={stages.get('patch_processing', 0.0):.6f}s"
        )
        return batch


class TimedDataLoader:
    """Proxy that reports wall-clock and Dataset stage times for every batch."""

    def __init__(self, loader: Any, reporter: Callable[[str], None] = print) -> None:
        """Initialize a DataLoader timing proxy.

        Args:
            loader: Underlying PyTorch DataLoader.
            reporter: Callback receiving formatted timing messages.
        """
        self.loader = loader
        self.reporter = reporter

    def __iter__(self) -> _TimedIterator:
        """Create a new timed iterator.

        Returns:
            A timing-aware iterator over the underlying DataLoader.
        """
        return _TimedIterator(iter(self.loader), self.reporter)

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            Length reported by the underlying DataLoader.
        """
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying DataLoader.

        Args:
            name: Attribute name.

        Returns:
            The delegated attribute value.

        Raises:
            AttributeError: If the underlying DataLoader lacks the attribute.
        """
        return getattr(self.loader, name)


def create_dataloader(
    dataset: Any,
    *,
    timing: bool = False,
    timing_reporter: Callable[[str], None] = print,
    **kwargs: Any,
) -> Any:
    """Create a PyTorch DataLoader, optionally reporting each iteration's timing.

    PyTorch's default collation converts numeric NumPy arrays to tensors. Pass a
    custom ``collate_fn`` in ``kwargs`` to preserve arrays or support custom batch
    structures.

    Args:
        dataset: Dataset passed to PyTorch DataLoader.
        timing: Whether to collect and report per-iteration timings.
        timing_reporter: Callback receiving formatted timing messages.
        **kwargs: Additional keyword arguments forwarded to PyTorch DataLoader.

    Returns:
        A PyTorch DataLoader, or a ``TimedDataLoader`` when timing is enabled.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    torch_data = require("torch.utils.data", extra="torch")
    if hasattr(dataset, "enable_timing"):
        dataset.enable_timing(timing)
    loader = torch_data.DataLoader(dataset, **kwargs)
    return TimedDataLoader(loader, timing_reporter) if timing else loader
