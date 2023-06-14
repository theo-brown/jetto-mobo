from typing import Optional

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


def q_increasing_area_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where q is increasing, weighted by area"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    increasing_q_area = np.sum(q[dq > 0])
    non_increasing_q_area = np.sum(q[dq <= 0])
    return increasing_q_area / (increasing_q_area + non_increasing_q_area)


def q_increasing_index_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where q is increasing, weighted by the number of locations"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    return np.sum(dq > 0) / len(dq)


def q_increasing_magntiude_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where q is increasing, weighted by the magnitude of the increase"""
    q = profiles["Q"][-1].data

    dq = np.gradient(q)
    if np.all(dq > 0):
        return 1  # Perfectly increasing
    else:
        # Find the indices where q is non-increasing
        non_increasing_indices = np.where(dq <= 0)[0]

    # Find the start and end indices of each non-increasing region
    start_indices = [non_increasing_indices[0]]
    end_indices = []
    for i in range(1, len(non_increasing_indices)):
        if non_increasing_indices[i] != non_increasing_indices[i - 1] + 1:
            end_indices.append(non_increasing_indices[i - 1])
            start_indices.append(non_increasing_indices[i])
    end_indices.append(non_increasing_indices[-1])

    # Compute the total change in q over each non-increasing region
    total_change_in_q = 0
    for start, end in zip(start_indices, end_indices):
        # Compute the change in q over this region, and add it to the total
        total_change_in_q += q[end] - q[start]

    # Total change in q is negative, so this will be [0,1]
    return np.exp(total_change_in_q)


def dq_increasing_area_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where dq is increasing,  weighted by area"""
    dq = np.gradient(profiles["Q"][-1].data)
    ddq = np.gradient(dq)
    increasing_dq_area = np.sum(dq[ddq > 0])
    non_increasing_dq_area = np.sum(dq[ddq <= 0])
    return increasing_dq_area / (increasing_dq_area + non_increasing_dq_area)


def dq_increasing_index_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where dq is increasing,  weighted by the number of locations"""
    dq = np.gradient(profiles["Q"][-1].data)
    ddq = np.gradient(dq)
    return np.sum(ddq > 0) / len(ddq)


def dq_increasing_magnitude_weighted(profiles: Dataset, timetraces: Dataset):
    """Fraction of curve where dq is increasing, weighted by the magnitude of the increase"""
    q = profiles["Q"][-1].data

    dq = np.gradient(q)
    ddq = np.gradient(dq)
    if np.all(ddq > 0):
        return 1  # Perfectly increasing
    else:
        # Find the indices where dq is non-increasing
        non_increasing_indices = np.where(ddq <= 0)[0]

    # Find the start and end indices of each non-increasing region
    start_indices = [non_increasing_indices[0]]
    end_indices = []
    for i in range(1, len(non_increasing_indices)):
        if non_increasing_indices[i] != non_increasing_indices[i - 1] + 1:
            end_indices.append(non_increasing_indices[i - 1])
            start_indices.append(non_increasing_indices[i])
    end_indices.append(non_increasing_indices[-1])

    # Compute the total change in dq over each non-increasing region
    total_change_in_dq = 0
    for start, end in zip(start_indices, end_indices):
        # Compute the change in q over this region, and add it to the total
        total_change_in_dq += dq[end] - dq[start]

    # Total change in q is negative, so this will be [0,1]
    return np.exp(total_change_in_dq)


# We can use this for a proxy for gradient at the given q value
def rho_of_q_value(profiles: Dataset, timetraces: Dataset, value: float):
    """rho at first point where q>=value and r >= argmin(q)"""
    xrho = profiles["XRHO"][-1].data
    condition_1 = profiles["Q"][-1].data >= value
    condition_2 = xrho >= timetraces["ROQM"][-1].data
    i = np.where(condition_1 & condition_2)[0][0]
    return xrho[i]


# The issue with this as an objective is that you can end up rewarding bumpy q profiles (eg PL50-9.19)
def gradient_of_q_at_value(profiles: Dataset, timetraces: Dataset, value=3):
    """1 - exp(-dq) at first point where q>=value and r >= argmin(q)"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    condition_1 = q >= value
    condition_2 = np.arange(len(q)) >= np.argmin(q)
    # Get index of element where both conditions are met
    i = np.where(condition_1 & condition_2)[0][0]

    if dq[i] < 0:
        return 0
    else:
        return 1 - np.exp(-dq[i])


def vector_objective(
    profiles: Dataset,
    timetraces: Dataset,
) -> np.ndarray:
    """Vector objective function for safety factor profile.

    Parameters
    ----------
    profiles : Dataset

    timetraces : Dataset


    Returns
    -------
    np.ndarray
        Vector of costs of safety factor profile.
    """
    return np.array(
        [
            proximity_of_q0_to_qmin(profiles, timetraces),
            proximity_of_qmin_to_target(profiles, timetraces, target=2),
            proximity_of_argmin_q_to_axis(profiles, timetraces),
            q_increasing_index_weighted(profiles, timetraces),
            dq_increasing_index_weighted(profiles, timetraces),
            rho_of_q_value(profiles, timetraces, value=3),
            rho_of_q_value(profiles, timetraces, value=4),
        ]
    )


def scalar_objective(
    profiles: Dataset, timetraces: Dataset, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    v = vector_objective(profiles, timetraces)
    if weights is None:
        return np.mean(v)
    else:
        # TODO: softmax?
        return np.mean(weights * v)
