import numpy as np
from netCDF4 import Dataset


def proximity_of_q0_to_qmin(profiles: Dataset, timetraces: Dataset):
    """exp(-|| q(0) - min(q) ||)"""
    distance = np.abs(timetraces["Q0"][-1].data - timetraces["QMIN"][-1].data)
    return np.exp(-distance)


def proximity_of_qmin_to_target(
    profiles: Dataset,
    timetraces: Dataset,
    target: float = 2.2,
    minimum: float = 2,
):
    """exp(-|| min(q) - target ||)"""
    qmin = timetraces["QMIN"][-1].data
    if qmin < minimum:
        return 0
    else:
        distance = np.abs(qmin - target)
        return np.exp(-distance)


def proximity_of_argmin_q_to_axis(profiles: Dataset, timetraces: Dataset):
    """1 - argmin(q)"""
    return 1 - timetraces["ROQM"][-1].data


def q_increasing(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where q is increasing, weighted by area"""
    q = profiles["Q"][-1].data
    # Shift q so that area is guaranteed to be positive
    shifted_q = q - np.min(q)
    dq = np.gradient(shifted_q)
    increasing_q_area = np.sum(shifted_q[dq > 0])
    non_increasing_q_area = np.sum(shifted_q[dq <= 0])
    return increasing_q_area / (increasing_q_area + non_increasing_q_area)


def dq_increasing(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where dq is increasing, weighted by area"""
    dq = np.gradient(profiles["Q"][-1].data)
    # Shift dq so that area is guaranteed to be positive
    shifted_dq = dq - np.min(dq)
    ddq = np.gradient(shifted_dq)
    increasing_dq_area = np.sum(shifted_dq[ddq > 0])
    non_increasing_dq_area = np.sum(shifted_dq[ddq <= 0])
    return increasing_dq_area / (increasing_dq_area + non_increasing_dq_area)


def rho_of_q_value(profiles: Dataset, timetraces: Dataset, value: float):
    """rho at first point where q>=value and r >= argmin(q)"""
    xrho = profiles["XRHO"][-1].data
    condition_1 = profiles["Q"][-1].data >= value
    condition_2 = xrho >= timetraces["ROQM"][-1].data
    i = np.where(condition_1 & condition_2)[0][0]
    return xrho[i]


from jetto_tools.results import JettoResults

from jetto_mobo.objectives import objective


@objective
def q_vector_objective(results: JettoResults) -> np.ndarray:
    """
    Vector of 7 objective functions relating to the shape of the q-profile.

    Returns
    -------
    np.ndarray
        Vector of objective values:
        - Proximity of q0 to qmin
        - Proximity of qmin to 2.2
        - Proximity of argmin(q) to axis
        - Fraction of curve where q is increasing, weighted by area
        - Fraction of curve where dq is increasing, weighted by area
        - rho at first point where q>=3 and r >= argmin(q)
        - rho at first point where q>=4 and r >= argmin(q)
    """
    profiles = results.load_profiles()
    timetraces = results.load_timetraces()

    return np.array(
        [
            proximity_of_q0_to_qmin(profiles, timetraces),
            proximity_of_qmin_to_target(profiles, timetraces, target=2.2),
            proximity_of_argmin_q_to_axis(profiles, timetraces),
            q_increasing(profiles, timetraces),
            dq_increasing(profiles, timetraces),
            rho_of_q_value(profiles, timetraces, value=3),
            rho_of_q_value(profiles, timetraces, value=4),
        ]
    )


@objective(weights=True)
def q_scalar_objective(results: JettoResults, weights: np.ndarray) -> np.ndarray:
    """
    Weighted sum of q_vector_objective.
    """
    v = q_vector_objective(results)
    return np.mean(weights * v)
