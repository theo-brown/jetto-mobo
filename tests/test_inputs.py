import numpy as np
import pytest

from jetto_mobo.inputs import plasma_profile


def test_plasma_profile():
    def wrong_signature(x):
        pass

    with pytest.raises(AttributeError):
        plasma_profile(wrong_signature)

    def no_types(xrho, parameters):
        pass

    with pytest.raises(AttributeError):
        plasma_profile(no_types)

    def no_return(xrho: np.ndarray, parameters: np.ndarray):
        pass

    with pytest.raises(AttributeError):
        plasma_profile(no_return)

    def good_signature(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        pass

    plasma_profile(good_signature)
