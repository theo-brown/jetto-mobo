from inspect import Parameter, Signature
from typing import Callable

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


def objective(f: Callable) -> Callable:
    """Decorator to ensure that the objective function has the correct signature.

    Objective functions should be of the form f(results: JettoResults) -> np.ndarray.
    """
    f_signature = Signature.from_callable(f)
    if f_signature != objective_signature:
        raise AttributeError(
            f"Functions decorated with @objective must have signature {objective_signature},"
            f" but function {f.__name__} has signature {f_signature}."
        )
    return f
