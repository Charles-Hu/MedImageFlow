from collections.abc import Callable, Iterable
from typing import Any


class Compose:
    """Apply transforms sequentially to a dictionary sample."""

    def __init__(self, transforms: Iterable[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        """Initialize an ordered transform composition.

        Args:
            transforms: Sample transforms applied from left to right.
        """
        self.transforms = tuple(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply every configured transform to a sample.

        Args:
            sample: Input sample dictionary.

        Returns:
            The sequentially transformed sample.
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample
