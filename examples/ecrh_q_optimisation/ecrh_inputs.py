from typing import Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import comb

from jetto_mobo.inputs import plasma_profile


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


def bernstein(i: Union[int, np.ndarray], n: int, t: np.ndarray) -> np.ndarray:
    """
    Evaluate the Bernstein basis polynomials at t.

    $$
    b_{i,n}(t) = \sum_{i=0}^n \binom{n}{i} t^i (1-t)^{n-i}
    $$
    """
    return comb(n, i) * np.power.outer(t, i) * np.power.outer((1 - t), (n - i))


def bezier_parametric(t: np.ndarray, control_points: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    t : np.ndarray
        Array of parametric coordinate values, length M.
    control_points : np.ndarray
        Array of shape (N, 2), giving N control points in the form [[x1, y1], ..., [xN, yN]].

    Returns
    -------
    np.ndarray
        Bezier curve defined by the control points, evaluated at t. Shape (M, 2), where [:, 0] is x and [:, 1] y coordinates.
    """
    # Check control points
    if not len(control_points.shape) == 2 or not control_points.shape[1] == 2:
        raise ValueError(
            f"control_points must be an array of shape (N, 2) (got {control_points.shape})."
        )

    n = control_points.shape[0]

    # Evaluate basis functions
    i = np.arange(n)
    basis_functions = bernstein(i, n - 1, t)

    # Evaluate bezier curve
    return basis_functions @ control_points


def bezier_x(
    x: np.ndarray, control_points: np.ndarray, parametric_resolution: int = int(1e3)
) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates to evaluate the bezier curve at.
    control_points : np.ndarray
        Array of shape (N, 2), giving N control points in the form [[x1, y1], ..., [xN, yN]].
    parametric_resolution : int, default int(1e3)
        Number of points to evaluate the bezier curve at to obtain the implicit/parametric coordinates.

    Returns
    -------
    np.ndarray
        Array of y coordinates, evaluated at x.
    """
    t = np.linspace(0, 1, parametric_resolution)
    b = bezier_parametric(t, control_points)
    implicit_points_x = b[:, 0]
    implicit_points_y = b[:, 1]

    # Interpolate into x coordinates
    t_x = np.interp(
        x, implicit_points_x, t
    )  # t_x is the implicit/parametric value corresponding to x
    b = bezier_parametric(t_x, control_points)
    x_ = b[:, 0]
    y = b[:, 1]

    return y


on_axis_smoothness = 0.05
off_axis_smoothness = 0.05


@plasma_profile
def constrained_bezier_profile(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """
    Bezier curve evaluated at x, with added hyperparameters to control the behaviour at the ends of the curve.

    Parameters are of the form [y0, x1, y1, ..., xn].
    Control points are [[0, y0], [on_axis_smoothness, y_0], [x1, y1], ..., [xn - off_axis_smoothness, 0], [xn, 0]].
    """
    y0 = parameters[0]
    x_parameters = parameters[1:-1:2]
    y_parameters = parameters[2:-1:2]
    xn = parameters[-1]

    control_points_x = np.concatenate(
        [
            [0, on_axis_smoothness],
            # Sort the x coordinates of the control points so that they are monotonically increasing in x.
            # np.sort(x_parameters),
            # OR
            # Product the x coordinates of the control points so that they are monotonically increasing in x.
            # x coordinates will be [x0*x1*...*xn, x1*...*xn, ..., xn]
            # np.cumprod(x_parameters[::-1])[::-1],
            # OR
            # Use the x coordinates of the control points as they are.
            x_parameters * xn,
            [xn - off_axis_smoothness, xn],
        ]
    )
    control_points_y = np.concatenate([[y0, y0], y_parameters, [0, 0]])
    if not control_points_x.shape == control_points_y.shape:
        raise ValueError(
            f"control_points_x and control_points_y must have the same shape (got {control_points_x.shape} and {control_points_y.shape})."
        )
    control_points = np.array([control_points_x, control_points_y]).T
    return bezier_x(xrho, control_points, parametric_resolution=int(1e3))


@plasma_profile
def bezier_profile(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """
    Bezier curve evaluated at x.

    Parameters are of the form [y0, x1, y1, ..., xn].
    Control points are [[0, y0], [x1, y1], ..., [xn, 0]].
    """
    y0 = parameters[0]
    x_parameters = parameters[1:-1:2]
    y_parameters = parameters[2:-1:2]
    xn = parameters[-1]

    control_points_x = np.concatenate([[0], x_parameters * xn, [xn]])
    control_points_y = np.concatenate([[y0], y_parameters, [0]])
    if not control_points_x.shape == control_points_y.shape:
        raise ValueError(
            f"control_points_x and control_points_y must have the same shape (got {control_points_x.shape} and {control_points_y.shape})."
        )
    control_points = np.array([control_points_x, control_points_y]).T
    return bezier_x(xrho, control_points, parametric_resolution=int(1e3))
