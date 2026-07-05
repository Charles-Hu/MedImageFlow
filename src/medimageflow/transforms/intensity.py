from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MinMaxNormalize:
    """Scale one NumPy array to [0, 1], with optional fixed intensity bounds."""

    minimum: float | None = None
    maximum: float | None = None
    clip: bool = True
    epsilon: float = 1e-8

    def __call__(self, array: NDArray[Any]) -> NDArray[Any]:
        """Apply min-max normalization to an array.

        Args:
            array: Input intensity array.

        Returns:
            A float32 array normalized by the configured bounds.
        """
        result = np.asarray(array, dtype=np.float32)
        minimum = float(result.min()) if self.minimum is None else self.minimum
        maximum = float(result.max()) if self.maximum is None else self.maximum
        result = (result - minimum) / max(maximum - minimum, self.epsilon)
        return np.clip(result, 0.0, 1.0) if self.clip else result


@dataclass
class ZScoreNormalize:
    """Z-score normalize one NumPy array, optionally over nonzero voxels."""

    nonzero: bool = True
    epsilon: float = 1e-8

    def __call__(self, array: NDArray[Any]) -> NDArray[Any]:
        """Apply z-score normalization to an array.

        Args:
            array: Input intensity array.

        Returns:
            A float32 z-score-normalized array.
        """
        result = np.asarray(array, dtype=np.float32).copy()
        mask = result != 0 if self.nonzero else np.ones(result.shape, dtype=bool)
        if not np.any(mask):
            return result
        values = result[mask]
        result[mask] = (values - values.mean()) / max(float(values.std()), self.epsilon)
        return result


@dataclass
class NormalizeIntensity:
    """Sample-level compatibility transform; prefer ``ZScoreNormalize`` per field."""

    key: str = "image"
    nonzero: bool = True
    epsilon: float = 1e-8

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Normalize one named field in a sample dictionary.

        Args:
            sample: Sample containing the configured field key.

        Returns:
            A shallow copy containing the normalized field.

        Raises:
            KeyError: If the configured field is missing.
        """
        output = dict(sample)
        output[self.key] = ZScoreNormalize(self.nonzero, self.epsilon)(
            np.asarray(sample[self.key])
        )
        return output
