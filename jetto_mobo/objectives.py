# TODO: write module-level docstring
from inspect import Parameter, Signature
from typing import Callable, Optional

import numpy as np
from jetto_tools.results import JettoResults

objective_signature = Signature(
    parameters=[
        Parameter(
            name="results",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=JettoResults,
        )
    ],
    return_annotation=np.ndarray,
)

weighted_objective_signature = Signature(
    parameters=[
        Parameter(
            name="results",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=JettoResults,
        ),
        Parameter(
            name="weights",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=np.ndarray,
        ),
    ],
    return_annotation=np.ndarray,
)


def objective(f: Optional[Callable] = None, weights: bool = False) -> Callable:
    """Decorator to ensure that the objective function has the correct signature.

    If ``weights=False`` (default), objective functions should be of the form ``f(results: JettoResults) -> np.ndarray``.
    If ``weights=True``, objective functions should be of the form ``f(results: JettoResults, weights: np.ndarray) -> np.ndarray``.

    Raises
    ------
    AttributeError
        If profile signature is not of the correct form; only ``(results: JettoResults) -> np.ndarray`` and ``(results: JettoResults, weights: np.ndarray) -> np.ndarray`` are supported.
    """
    if f is None and weights == True:

        def weighted_objective(f):
            f_signature = Signature.from_callable(f)
            if f_signature != weighted_objective_signature:
                raise AttributeError(
                    f"Functions decorated with @objective(weights=True) must have signature {weighted_objective_signature},"
                    f" but function {f.__name__} has signature {f_signature}."
                )
            return f

        return weighted_objective
    elif f is not None and weights == False:
        f_signature = Signature.from_callable(f)
        if f_signature != objective_signature:
            raise AttributeError(
                f"Functions decorated with @objective must have signature {objective_signature},"
                f" but function {f.__name__} has signature {f_signature}."
            )
        return f
    else:
        raise AttributeError(
            "Only decorating functions with @objective or @objective(weights=True) is supported."
        )
