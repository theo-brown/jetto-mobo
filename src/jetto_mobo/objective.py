import numpy as np
from netCDF4 import Dataset


def proximity_of_q0_to_qmin(profiles: Dataset, timetraces: Dataset):
    """exp(-|| q(0) - min(q) ||)"""
    distance = np.abs(timetraces["Q0"][-1].data - timetraces["QMIN"][-1].data)
    return np.exp(-distance)


def proximity_of_qmin_to_target(profiles: Dataset, timetraces: Dataset, target=2.2):
    """exp(-|| min(q) - target ||)"""
    distance = np.abs(timetraces["QMIN"][-1].data - target)
    return np.exp(-distance)


def proximity_of_argmin_q_to_axis(profiles: Dataset, timetraces: Dataset):
    """1 - argmin(q)"""
    return 1 - timetraces["ROQM"][-1].data


def q_increasing(profiles: Dataset, timetraces: Dataset):
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    increasing_q_area = np.sum(q[dq > 0])
    non_increasing_q_area = np.sum(q[dq <= 0])
    return increasing_q_area / (increasing_q_area + non_increasing_q_area)


def dq_increasing(profiles: Dataset, timetraces: Dataset):
    dq = np.gradient(profiles["Q"][-1].data)
    ddq = np.gradient(dq)
    increasing_dq_area = np.sum(dq[ddq > 0])
    non_increasing_dq_area = np.sum(dq[ddq <= 0])
    return increasing_dq_area / (increasing_dq_area + non_increasing_dq_area)


def rho_of_q_value(profiles: Dataset, timetraces: Dataset, value: float):
    """rho at first point where q>=value and r >= argmin(q)"""
    xrho = profiles["XRHO"][-1].data
    condition_1 = profiles["Q"][-1].data >= value
    condition_2 = xrho >= timetraces["ROQM"][-1].data
    i = np.where(condition_1 & condition_2)[0][0]
    return xrho[i]


# def gradient_of_q_at_value(profiles: Dataset, timetraces: Dataset, value=3):
#     """dq at first point where q>=value and r >= argmin(q)"""
#     q = profiles["Q"][-1].data
#     dq = np.gradient(q)
#     condition_1 = q >= value
#     condition_2 = np.arange(len(q)) >= np.argmin(q)
#     # Get index of element where both conditions are met
#     i = np.where(condition_1 & condition_2)[0][0]
#     return dq[i]


# def proximity_of_internal_inductance_to_target(profiles: Dataset, timetraces: Dataset, target=0.25):
#     """exp(-|| li - target ||)"""
#     return np.exp(-np.abs(timetraces["LI"][-1].data - target)))


def vector_objective(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
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
            proximity_of_qmin_to_target(profiles, timetraces, target=2.2),
            proximity_of_argmin_q_to_axis(profiles, timetraces),
            q_increasing(profiles, timetraces),
            dq_increasing(profiles, timetraces),
            rho_of_q_value(profiles, timetraces, value=3),
            rho_of_q_value(profiles, timetraces, value=4),
        ]
    )


def scalar_objective(profiles: Dataset, timetraces: Dataset) -> np.ndarray:
    return np.sum(vector_objective(profiles, timetraces))
