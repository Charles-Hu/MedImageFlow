import numpy as np

from medical_toolkit.transforms import Compose, NormalizeIntensity


def test_normalize_intensity_preserves_zero_background() -> None:
    sample = {"image": np.array([0.0, 1.0, 2.0, 3.0])}
    result = Compose([NormalizeIntensity()])(sample)
    assert result["image"][0] == 0
    np.testing.assert_allclose(result["image"][1:].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(result["image"][1:].std(), 1.0, atol=1e-6)

