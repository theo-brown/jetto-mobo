import netCDF4
import numpy as np

from jetto_mobo.configuration import bo_objective


@bo_objective
def proximity_of_q0_to_qmin(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    """exp(-|| q(0) - min(q) ||)"""
    distance = np.abs(timetraces["Q0"][-1].data - timetraces["QMIN"][-1].data)
    return np.exp(-distance)


@bo_objective
def proximity_of_qmin_to_target(
    profiles: netCDF4.Dataset,
    timetraces: netCDF4.Dataset,
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


@bo_objective
def proximity_of_argmin_q_to_axis(
    profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset
):
    """1 - argmin(q)"""
    return 1 - timetraces["ROQM"][-1].data


@bo_objective
def q_increasing(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    """Fraction of curve where q is increasing, weighted by area"""
    q = profiles["Q"][-1].data
    # Shift q so that area is guaranteed to be positive
    shifted_q = q - np.min(q)
    dq = np.gradient(shifted_q)
    increasing_q_area = np.sum(shifted_q[dq > 0])
    non_increasing_q_area = np.sum(shifted_q[dq <= 0])
    return increasing_q_area / (increasing_q_area + non_increasing_q_area)


@bo_objective
def dq_increasing(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    """Fraction of curve where dq is increasing, weighted by area"""
    dq = np.gradient(profiles["Q"][-1].data)
    # Shift dq so that area is guaranteed to be positive
    shifted_dq = dq - np.min(dq)
    ddq = np.gradient(shifted_dq)
    increasing_dq_area = np.sum(shifted_dq[ddq > 0])
    non_increasing_dq_area = np.sum(shifted_dq[ddq <= 0])
    return increasing_dq_area / (increasing_dq_area + non_increasing_dq_area)


@bo_objective
def rho_of_q_value(
    profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset, value: float
):
    """rho at first point where q>=value and r >= argmin(q)"""
    xrho = profiles["XRHO"][-1].data
    condition_1 = profiles["Q"][-1].data >= value
    condition_2 = xrho >= timetraces["ROQM"][-1].data
    i = np.where(condition_1 & condition_2)[0][0]
    return xrho[i]


# The issue with this as an objective is that you can end up rewarding bumpy q profiles (eg PL50-9.19)
@bo_objective
def gradient_of_q_at_value(
    profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset, value=3
):
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
