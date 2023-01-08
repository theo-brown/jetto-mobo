from typing import Iterable, Tuple
import numpy as np


def _gaussian(
    x: Iterable[float], mean: float, variance: float, amplitude: float = 1
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * (x - mean) ** 2 / variance)


def sum_of_gaussians(
    x: Iterable[float],
    means: Iterable[float],
    variances: Iterable[float],
    amplitudes: Iterable[float],
) -> np.ndarray:
    return np.sum(
        [_gaussian(x, m, v, a) for m, v, a in zip(means, variances, amplitudes)], axis=0
    )


def marsden_linear(x: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
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

    return np.interp(x, node_xs, node_ys), [(x, y) for x, y in zip(node_xs, node_ys)]


if __name__ == "__main__":
    import plotly.graph_objects as go

    rng = np.random.default_rng()

    x = np.linspace(0, 1, 100)

    # Piecewise
    figure = go.Figure()
    for _ in range(5):
        y, nodes = marsden_linear(x, parameters=rng.random(12))
        figure.add_trace(go.Scatter(x=x, y=y))
    figure.show()

    # Sum of Gaussians
    y = sum_of_gaussians(x, means=[0.1, 0.6], variances=[0.001, 0.1], amplitudes=[5, 2])
    figure = go.Figure(go.Scatter(x=x, y=y))
    figure.show()
