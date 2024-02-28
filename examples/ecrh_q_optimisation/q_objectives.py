from typing import Union

import numpy as np
from jetto_tools.results import JettoResults
from netCDF4 import Dataset

from jetto_mobo.objectives import objective


def soft_hat(
    x: Union[float, np.ndarray],
    x_lower: float = 0,
    y_lower: float = 1e-3,
    x_plateau_start: float = 0,
    x_plateau_end: float = 0,
    x_upper: float = 0,
    y_upper: float = 1e-3,
) -> np.ndarray:
    """
    Smooth tophat function.

    Passes through (x_lower, y_lower), (x_plateau_start, 1), (x_plateau_end, 1), (x_upper, y_upper).
    Squared exponential decay from 0 to x_plateau_start and from x_plateau_end to infinity, with rate of decay such that y=y_lower at x=x_lower and y=y_upper at x=x_upper.

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Input value
    x_lower : float, optional
        x-value at which y=y_lower (default: 0)
    y_lower : float, optional
        y-value at x=x_lower (default: 1e-3)
    x_plateau_start : float, optional
        x-value at which the plateau starts (default: 0)
    x_plateau_end : float, optional
        x-value at which the plateau ends (default: 0)
    x_upper : float, optional
        x-value at which y=y_upper (default: 0)
    y_upper : float, optional
        y-value at x=x_upper (default: 1e-3)

    Returns
    -------
    np.ndarray
        Smooth objective value
    """
    k_lower = -np.log(y_lower) / np.power(x_lower - x_plateau_start, 2)
    k_upper = -np.log(y_upper) / np.power(x_upper - x_plateau_end, 2)
    return np.piecewise(
        x,
        [
            x < x_plateau_start,
            (x >= x_plateau_start) & (x <= x_plateau_end),
            x > x_plateau_end,
        ],
        [
            lambda x: np.exp(-k_lower * np.power(x - x_plateau_start, 2)),
            1,
            lambda x: np.exp(-k_upper * np.power(x - x_plateau_end, 2)),
        ],
    )


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp)


def q0_close_to_qmin(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    """1 if q0 = qmin, decaying to 0.5 at ||q0 - qmin|| = 2"""
    distance = np.abs(timetraces["Q0"][-1].data - timetraces["QMIN"][-1].data)
    return soft_hat(
        distance,
        x_lower=-1,  # Not used, as 0 < x < 1
        y_lower=1e-3,  # Not used, as 0 < x < 1
        x_plateau_start=0,
        x_plateau_end=0,
        x_upper=2,
        y_upper=0.5,
    )


def qmin_close_to_centre(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    """1 if argmin(q) = 0, decaying to 1e-3 at ||argmin(q)|| = 1"""
    return soft_hat(
        timetraces["ROQM"][-1].data,
        x_lower=-1,  # Not used, as 0 < x < 1
        y_lower=1e-3,  # Not used, as 0 < x < 1
        x_plateau_start=0,
        x_plateau_end=0,
        x_upper=1,
        y_upper=1e-3,
    )


def qmin_in_safe_region(profiles: Dataset, timetraces: Dataset) -> float:
    """1 if qmin between 2.2 and 2.5, decaying to 0.5 at 2 and 3"""
    return soft_hat(
        timetraces["QMIN"][-1].data,
        x_lower=2.2,
        y_lower=0.5,
        x_plateau_start=2.2,
        x_plateau_end=2.5,
        x_upper=3,
        y_upper=0.5,
    )


def q_increasing(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    """1 if q is increasing at every radial point, decaying to 1e-3 if q is non-increasing at every radial point"""
    is_increasing = np.gradient(profiles["Q"][-1].data) > 0
    return soft_hat(
        np.mean(is_increasing),  # Fraction of curve where q is increasing
        x_lower=0,
        y_lower=1e-3,
        x_plateau_start=1,
        x_plateau_end=1,
        x_upper=2,  # Not used, as 0 < x < 1
        y_upper=1e-3,  # Not used, as 0 < x < 1
    )


def maximise_radius_at_which_q_is_value(
    profiles: Dataset, timetraces: Dataset, value: float
) -> np.ndarray:
    """1 if q=value at r>0.8 decaying to 0.5 if q=value at r=0.5

    Note that this excludes points where q=value but r < argmin(q)."""
    xrho = profiles["XRHO"][-1].data
    condition_1 = profiles["Q"][-1].data >= value
    condition_2 = xrho >= timetraces["ROQM"][-1].data
    i = np.where(condition_1 & condition_2)[0][0]
    radius_of_q_is_value = xrho[i]
    return soft_hat(
        radius_of_q_is_value,
        x_lower=0.5,
        y_lower=0.5,
        x_plateau_start=0.8,
        x_plateau_end=1,
        x_upper=2,  # Not used, as 0 < x < 1
        y_upper=2,  # Not used, as 0 < x < 1
    )


@objective
def q_vector_objective(results: JettoResults) -> np.ndarray:
    """
    Vector of 6 objective functions relating to the shape of the q-profile.

    Returns
    -------
    np.ndarray
        Vector of objective values:
        - Reduced 'height' of reversed shear at axis (q0 close to qmin)
        - Reduced 'width' of reversed shear at axis (qmin close to r=0)
        - Monotonic q (q increasing at every radial point)
        - qmin in safe region (2 < qmin < 3)
        - Maximise radius at which q=3
        - Maximise radius at which q=4
    """
    profiles = results.load_profiles()
    timetraces = results.load_timetraces()

    return q_vector_objective_from_cdf(profiles, timetraces)


def q_vector_objective_from_cdf(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    return np.array(
        [
            q0_close_to_qmin(profiles, timetraces),
            qmin_close_to_centre(profiles, timetraces),
            q_increasing(profiles, timetraces),
            qmin_in_safe_region(profiles, timetraces),
            maximise_radius_at_which_q_is_value(profiles, timetraces, 3),
            maximise_radius_at_which_q_is_value(profiles, timetraces, 4),
        ]
    )


def q_constraints(results: JettoResults) -> np.ndarray:
    """Vector of constraints on the q profile.

    Constraint functions are of the form ``g(x) <= 0`` (i.e. negative if the constraint is satisfied).

    Returns
    -------
    np.ndarray
        Vector of constraint values:
        - qmin > 2
        - qmin < 3
    """
    return q_constraints_from_cdf(results.load_profiles(), results.load_timetraces())


def q_constraints_from_cdf(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    return np.array(
        [
            2 - timetraces["QMIN"][-1].data,
            timetraces["QMIN"][-1].data - 3,
        ]
    )


@objective(weights=True)
def q_scalar_objective(results: JettoResults, weights: np.ndarray) -> np.ndarray:
    """
    Weighted sum of q_vector_objective.
    """
    v = q_vector_objective(results)
    return v @ weights


def q_scalar_objective_from_cdf(
    profiles: Dataset, timetraces: Dataset, weights: np.ndarray
) -> np.ndarray:
    v = q_vector_objective_from_cdf(profiles, timetraces)
    return v @ weights
