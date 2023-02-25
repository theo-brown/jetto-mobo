from typing import Iterable

import netCDF4
import numpy as np


def rho_of_q_value(
    profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset, value: float
):
    """
    Gives the value of normalised rho where q first passes through the specified
    flux surface in the region rho = [roqm,1.0]. If qmin > value, 0 is
    returned. If qmax < value, 1 is returned.
    """
    qmin = timetraces["QMIN"][-1].data
    roqm = timetraces["ROQM"][-1].data
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
    roqm = timetraces["ROQM"][-1].data
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


def scalar_cost_function(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return (
        0.5 * q0_qmin(profiles, timetraces)
        + 5 * qmin(profiles, timetraces)
        + 6 * rho_qmin(profiles, timetraces)
        + 1
        - rho_of_q_value(profiles, timetraces, 3)
        + 2 * (1 - rho_of_q_value(profiles, timetraces, 4))
        + 10 * (1 - monotonic_fraction_q(profiles, timetraces))
        + 10 * (1 - monotonic_fraction_q1(profiles, timetraces))
    )


def vector_cost_function(profiles: netCDF4.Dataset, timetraces: netCDF4.Dataset):
    return (
        0.5 * q0_qmin(profiles, timetraces),
        5 * qmin(profiles, timetraces),
        6 * rho_qmin(profiles, timetraces),
        10 * (1 - monotonic_fraction_q(profiles, timetraces)),
        10 * (1 - monotonic_fraction_q1(profiles, timetraces)),
        1 - rho_of_q_value(profiles, timetraces, 3),
        2 * (1 - rho_of_q_value(profiles, timetraces, 4)),
    )


def piecewise_linear(x: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")

    # On axis peak
    on_axis_peak_x = 0
    on_axis_peak_y = parameters[0]

    # On axis peak shaper
    on_axis_peak_end_x = parameters[1]
    on_axis_peak_end_y = parameters[2] * on_axis_peak_y

    # Minimum
    minimum_x = parameters[3]
    minimum_y = parameters[4]

    # Minimum shaper
    minimum_shaper_x = (minimum_x + on_axis_peak_end_x) / 2
    minimum_shaper_y = parameters[5] * minimum_y

    # Off-axis peak
    off_axis_peak_x = (minimum_x + parameters[6]) / 2
    off_axis_peak_y = parameters[7]

    # Off-axis shaper 2
    off_axis_shaper_2_x = (minimum_x + 2 * off_axis_peak_x) / 3
    off_axis_shaper_2_y = (
        parameters[8] * off_axis_peak_y + (1 - parameters[8]) * minimum_y
    )

    # Off-axis shaper 1
    off_axis_shaper_1_x = (2 * minimum_x + off_axis_peak_x) / 3
    off_axis_shaper_1_y = (
        parameters[9] * off_axis_shaper_2_y + (1 - parameters[9]) * minimum_y
    )

    # Turn-off
    turn_off_x = off_axis_peak_x + parameters[11]
    turn_off_y = 0

    # Turn-off shaper
    turn_off_shaper_x = (off_axis_peak_x + turn_off_x) / 2
    turn_off_shaper_y = parameters[10] * off_axis_peak_y

    # Collect into array
    node_xs = [
        on_axis_peak_x,
        on_axis_peak_end_x,
        minimum_shaper_x,
        minimum_x,
        off_axis_shaper_1_x,
        off_axis_shaper_2_x,
        off_axis_peak_x,
        turn_off_shaper_x,
        turn_off_x,
    ]
    node_ys = [
        on_axis_peak_y,
        on_axis_peak_end_y,
        minimum_shaper_y,
        minimum_y,
        off_axis_shaper_1_y,
        off_axis_shaper_2_y,
        off_axis_peak_y,
        turn_off_shaper_y,
        turn_off_y,
    ]

    return np.interp(x, node_xs, node_ys)
