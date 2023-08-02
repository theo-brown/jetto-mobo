<<<<<<< HEAD
# TODO: write module-level docstring
=======
>>>>>>> 8a3aa1aaa673e79b915d04775bc75d3b5492d118
from inspect import Parameter, Signature
from typing import Callable

import numpy as np

profile_signature = Signature(
    parameters=[
        Parameter(
            name="xrho",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=np.ndarray,
        ),
        Parameter(
            name="parameters",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=np.ndarray,
        ),
    ],
    return_annotation=np.ndarray,
)


<<<<<<< HEAD
def plasma_profile(f: Callable) -> Callable:
    """Decorator to ensure that a function representing a parameterised plasma profile has the correct signature.

    Profile functions should be of the form ``f(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray``.

    Raises
    ------
    AttributeError
        If the signature of the decorated function is not ``(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray``.
=======
def profile(f: Callable) -> Callable:
    """Decorator to ensure that a function representing a parameterised plasma profile has the correct signature.

    Profile functions should be of the form f(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray.
>>>>>>> 8a3aa1aaa673e79b915d04775bc75d3b5492d118
    """
    f_signature = Signature.from_callable(f)
    if f_signature != profile_signature:
        raise AttributeError(
            f"Functions decorated with @profile must have signature {profile_signature},"
            f" but function {f.__name__} has signature {f_signature}."
        )
    return f
