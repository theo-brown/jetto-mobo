import numpy as np
import pytest
from jetto_tools.results import JettoResults

from jetto_mobo.objectives import objective


def test_objective_unweighted():
    def wrong_signature(x):
        pass

    with pytest.raises(AttributeError):
        objective(wrong_signature)

    def no_types(results):
        pass

    with pytest.raises(AttributeError):
        objective(no_types)

    def no_return(results: JettoResults):
        pass

    with pytest.raises(AttributeError):
        objective(no_return)

    def good_signature(results: JettoResults) -> np.ndarray:
        pass

    objective(good_signature)


def test_objective_weighted():
    def wrong_signature(x):
        pass

    with pytest.raises(AttributeError):
        objective(weights=True)(wrong_signature)

    def no_types(results, weights):
        pass

    with pytest.raises(AttributeError):
        objective(weights=True)(no_types)

    def no_return(results: JettoResults, weights: np.ndarray):
        pass

    with pytest.raises(AttributeError):
        objective(weights=True)(no_return)

    def good_signature(results: JettoResults, weights: np.ndarray) -> np.ndarray:
        pass

    objective(weights=True)(good_signature)
