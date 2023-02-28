from typing import Iterable, Optional, Tuple

import netCDF4
import numpy as np
from scipy.interpolate import CubicSpline


def rho_of_q_value(
    profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset, value: float
):
    """
    Gives the value of normalised rho where q first passes through the specified
    flux surface in the region rho = [roqm,1.0]. If qmin > value, 0 is
    returned. If qmax < value, 1 is returned.
    """
    qmin = timetraces["QMIN"][-1].data
    q = profiles["Q"][-1].data
    xrho = profiles["XRHO"][-1].data
    # Check if qmin gets down below flux_surface.
    if qmin > value:
        return 0
    # Calculate the bin number for the minimum q.
    i_qmin = np.argmin(np.array(q))
    # Find the first bin greater than flux_surface.
    for i_bin, q_value in enumerate(q[i_qmin:]):
        if q_value > value:
            return xrho[i_qmin + i_bin]
    # If we haven't returned before now it must mean that we are never above
    # flux_surface.
    return 1


def monotonic_fraction_q(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    """
    Returns the fraction of the range of rho between roqm and 0.9 for which q is
    monotonically increasing.
    """
    # Read in useful values.
    q = profiles["Q"][-1].data
    xrho = profiles["XRHO"][-1].data
    # Calculate useful bin numbers.
    i_0d9 = np.argmin(np.abs(np.array(xrho) - 0.9))  # xrho bin where xrho = 0.9
    i_qmin = np.argmin(np.array(q))  # xrho bin where q is the minimum
    # Calculate the monotonic fraction of q.
    frac_mono = sum(q[i] <= q[i + 1] for i in range(i_qmin, i_0d9)) / (i_0d9 - i_qmin)
    # Return the monotonic fraction.
    return frac_mono


def monotonic_fraction_q1(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    """
    Returns the fraction of the range of rho between 0.0 and 0.9 for which
    dq/drho is monotonically increasing.
    """
    q = profiles["Q"][-1].data
    xrho = profiles["XRHO"][-1].data
    # Calculate useful bin numbers.
    i_0d9 = np.argmin(np.abs(np.array(xrho) - 0.9))  # xrho bin where xrho = 0.9
    # Calculate the monotonic fraction of q.
    frac_mono = sum((q[i] - q[i - 1]) <= (q[i + 1] - q[i]) for i in range(1, i_0d9)) / (
        i_0d9 - 1
    )
    # Return the monotonic fraction.
    return frac_mono


def q0_qmin(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return timetraces["Q0"][-1].data - timetraces["QMIN"][-1].data


def qmin(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return timetraces["QMIN"][-1].data


def rho_qmin(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return timetraces["ROQM"][-1].data


def scale_to_target(
    value: float,
    possible_range: Tuple[Optional[float], Optional[float]],
    target_range: Tuple[Optional[float], Optional[float]],
):
    # Check if there were any errors in evaluating the function.
    if value is None:
        # There was. Pass the issue forwards.
        return None
    # Error checking.
    # If the value is outside what is thought to be the possible range, return 0
    if (possible_range[0] is not None and value < possible_range[0]) or (
        possible_range[1] is not None and value > possible_range[1]
    ):
        return 0
    # Initialise the storage for the relavent boundaries of where the fitness
    # should equal 1, and where it should equal 0.
    zero_at = None
    one_at = None
    # Figure out what should be stored in these values.
    if target_range[0] is not None and value < target_range[0]:
        # We are below the target range, use the low end values.
        zero_at = possible_range[0]
        one_at = target_range[0]
    elif target_range[1] is not None and value > target_range[1]:
        # We are above the target range, use the high end values.
        zero_at = possible_range[1]
        one_at = target_range[1]
    else:
        # We must be inside the target range.
        return 1.0
    # Note that one_at must now set as we are outside of the range it defines.
    # zero_at may still be none.
    if zero_at is None:
        # If unbound, return an exponential decrease.
        return np.exp(-abs(value - one_at))
    else:
        # If bound, return a linear decrease.
        return abs((value - zero_at) / (one_at - zero_at))


def scalar_objective(
    profiles: netCDF4.Dataset,
    timetraces: netCDF4.Dataset,
    weights: Optional[Iterable[float]] = [0.5, 5, 6, 1, 2, 10, 10],
):
    v = vector_objective(profiles, timetraces)
    if weights is not None:
        w = np.asarray(weights)
        normalised_weights = w / w.sum()
        return normalised_weights @ v
    else:
        return v.sum()


def vector_objective(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return (
        scale_to_target(
            q0_qmin(profiles, timetraces),
            possible_range=(0.0, None),
            target_range=(0.0, 2.0),
        ),
        scale_to_target(
            qmin(profiles, timetraces),
            possible_range=(0.0, None),
            target_range=(2.25, 2.35),
        ),
        scale_to_target(
            rho_qmin(profiles, timetraces),
            possible_range=(0.0, 1.0),
            target_range=(0.0, 0.0),
        ),
        scale_to_target(
            rho_of_q_value(profiles, timetraces, 3),
            possible_range=(0.0, 1.0),
            target_range=(1.0, 1.0),
        ),
        scale_to_target(
            rho_of_q_value(profiles, timetraces, 4),
            possible_range=(0.0, 1.0),
            target_range=(1.0, 1.0),
        ),
        scale_to_target(
            monotonic_fraction_q(profiles, timetraces),
            possible_range=(0.0, 1.0),
            target_range=(1.0, 1.0),
        ),
        scale_to_target(
            monotonic_fraction_q1(profiles, timetraces),
            possible_range=(0.0, 1.0),
            target_range=(1.0, 1.0),
        ),
    )


def piecewise_linear(xrho: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")

    on_axis_peak_power = parameters[0]
    on_axis_descent_end_power = parameters[1] * on_axis_peak_power
    minimum_power = parameters[4]
    minimum_shaper_power = (
        parameters[3] * minimum_power + (1 - parameters[3]) * on_axis_descent_end_power
    )
    off_axis_peak_power = parameters[8]
    off_axis_shaper_2_power = (
        parameters[7] * off_axis_peak_power + (1 - parameters[7]) * minimum_power
    )
    off_axis_shaper_1_power = (
        parameters[6] * off_axis_shaper_2_power + (1 - parameters[6]) * minimum_power
    )
    turn_off_shaper_power = parameters[10] * off_axis_peak_power
    turn_off_power = 0
    on_axis_peak_xrho = 0
    on_axis_descent_end_xrho = parameters[2]
    minimum_xrho = parameters[5]
    minimum_shaper_xrho = (minimum_xrho + on_axis_descent_end_xrho) / 2
    off_axis_peak_xrho = minimum_xrho + parameters[9]
    turn_off_xrho = off_axis_peak_xrho + parameters[11]
    off_axis_shaper_1_xrho = 2 / 3 * minimum_xrho + 1 / 3 * off_axis_peak_xrho
    off_axis_shaper_2_xrho = 1 / 3 * minimum_xrho + 2 / 3 * off_axis_peak_xrho
    turn_off_shaper_xrho = 1 / 2 * off_axis_peak_xrho + 1 / 2 * turn_off_xrho

    xdata = [
        on_axis_peak_xrho,
        on_axis_descent_end_xrho,
        minimum_shaper_xrho,
        minimum_xrho,
        off_axis_shaper_1_xrho,
        off_axis_shaper_2_xrho,
        off_axis_peak_xrho,
        turn_off_shaper_xrho,
        turn_off_xrho,
    ]
    ydata = [
        on_axis_peak_power,
        on_axis_descent_end_power,
        minimum_shaper_power,
        minimum_power,
        off_axis_shaper_1_power,
        off_axis_shaper_2_power,
        off_axis_peak_power,
        turn_off_shaper_power,
        turn_off_power,
    ]

    # Check that the xrho values are monotonically increasing.
    if not all([xdata[i] < xdata[i + 1] for i in range(len(xdata) - 1)]):
        raise RuntimeError(
            f"Xrho coordinates are not monotonically increasing: {xdata}"
        )

    # Storage for output profiles.
    qece = np.zeros(len(xrho))

    # Linearly interpolate between each of the points.
    for i_point in range(len(xdata) - 1):
        # Get just the pair of points.
        points_xrho = [xdata[i_point], xdata[i_point + 1]]
        points_power = [ydata[i_point], ydata[i_point + 1]]
        # Fit with a straight line between them.
        cs = CubicSpline(
            points_xrho, points_power, bc_type=[(2, 0), (2, 0)], extrapolate=False
        )
        # Set the values in the profile.
        qece = np.array([v if v > 0 else qece[i] for i, v in enumerate(cs(xrho))])

    return qece
