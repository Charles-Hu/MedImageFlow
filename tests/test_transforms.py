import numpy as np

from medimageflow.transforms import Compose, NormalizeIntensity


def test_normalize_intensity_preserves_zero_background() -> None:
    """Verify foreground normalization leaves a zero-valued background intact.

    The input contains zeros surrounding positive foreground voxels. The
    transform should calculate statistics from nonzero values only, preserve
    background zeros during the key masking step, and return normalized
    foreground values with approximately zero mean and unit variance.
    """
    sample = {"image": np.array([0.0, 1.0, 2.0, 3.0])}
    result = Compose([NormalizeIntensity()])(sample)
    assert result["image"][0] == 0
    np.testing.assert_allclose(result["image"][1:].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(result["image"][1:].std(), 1.0, atol=1e-6)
