import numpy as np
from scipy.interpolate import CubicSpline

from jetto_mobo.inputs import plasma_profile


# TODO: docstring
@plasma_profile
def marsden_piecewise_linear(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """
    12-parameter piecewise linear profile, originally designed for ECRH profiles by Stephen Marsden.

    Parameters
    ----------
    xrho : np.ndarray
        Normalised radial position.
    parameters : np.ndarray
        Profile parameters:
        - parameters[0]: on-axis peak power
        - parameters[1]: on-axis descent end power
        - parameters[2]: on-axis descent end xrho
        - parameters[3]: minimum power fraction
        - parameters[4]: minimum power
        - parameters[5]: minimum xrho
        - parameters[6]: off-axis shaper 1 fraction
        - parameters[7]: off-axis shaper 2 fraction
        - parameters[8]: off-axis peak power
        - parameters[9]: off-axis peak xrho
        - parameters[10]: turn-off shaper fraction
        - parameters[11]: turn-off xrho

    Raises
    ------
    ValueError
        If the number of parameters is not 12.
    """
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")

    lower_bounds = np.array([0, 0.05, 0.01, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    upper_bounds = np.array([1, 1, 0.09, 1, 1, 0.29, 0.9, 0.9, 1, 0.75, 0.9, 0.45])
    is_outside_bounds = (parameters < lower_bounds) | (parameters > upper_bounds)
    if np.any(is_outside_bounds):
        raise ValueError(
            f"Parameter(s) outside of bounds at indices {np.nonzero(is_outside_bounds)}"
        )

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

    # Linearly interpolate between each of the points.
    qece = np.zeros(len(xrho))
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


marsden_piecewise_linear_bounds = np.array(
    [
        [0, 0.05, 0.01, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [1, 1, 0.09, 1, 1, 0.29, 0.9, 0.9, 1, 0.75, 0.9, 0.45],
    ]
)
