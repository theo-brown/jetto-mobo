import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import hex_to_rgb

colours = plotly.colors.qualitative.G10
uncontrolled_colour = plotly.colors.qualitative.T10[-1]
benchmark_colour = plotly.colors.qualitative.T10[-1]
pid_colour = colours[0]
sac_colour = colours[1]
ppo_colour = colours[7]  # 4
bo_sog2_colour = colours[5]
bo_mpl_colour = colours[1]
bo_pl_colour = colours[2]
bo_sog_colour = colours[3]


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude]
    )


def discrete_colorscale(n, colorscale="viridis"):
    values = np.linspace(0, 1, n + 1)
    colors = plotly.colors.sample_colorscale(
        colorscale, n, low=0, high=1, colortype="rgb"
    )
    lower_edges = [(values[i], colors[i]) for i in range(n)]
    upper_edges = [(values[i + 1], colors[i]) for i in range(n)]
    return [x for pair in zip(lower_edges, upper_edges) for x in pair]


def colour_to_rgba(colour: str, alpha: float):
    if colour.startswith("#"):
        r, g, b = hex_to_rgb(colour)
    elif colour.startswith("rgb"):
        r, g, b = colour[4:-1].split(",")
    elif colour.startswith("rgba"):
        r, g, b = colour[5:-1].split(",")
    else:
        raise ValueError(f"Unknown colour format: {colour}")
    return f"rgba({r},{g},{b},{alpha})"


def save_figure(
    figure: go.Figure,
    filename: str,
    linewidth=1,
    fontsize=22,
    width=1,
    height=1,
    title_standoff=20,
    margin={"l": 20, "r": 20, "b": 20, "t": 20, "pad": 0},
):
    figure.update_layout(
        font={"size": fontsize},
        title=None,
        template="simple_white",
        margin=margin,
    )
    figure.update_xaxes(linewidth=linewidth)
    if title_standoff is not None:
        figure.update_yaxes(linewidth=linewidth, title_standoff=title_standoff)
    else:
        figure.update_yaxes(linewidth=linewidth)
    pio.write_image(
        figure, f"{filename}.svg", format="svg", width=width * 1280, height=height * 800
    )


def get_pareto_dominant_mask(
    objective_values: np.ndarray, allow_zero: bool = False
) -> np.ndarray:
    """Compute a mask that selects only Pareto-optimal solutions

    Parameters
    ----------
    objective_values : np.ndarray
        An n_points x n_objectives array.

    allow_zero : bool
        If False, points with a value of zero for any objective are excluded.

    Returns
    -------
    is_dominant
        An n_points boolean array, True where points are Pareto optimal.
    """
    is_dominant = np.zeros(objective_values.shape[0], dtype=bool)
    for i, objective_value in enumerate(objective_values):
        strictly_better_in_one_objective = (objective_values > objective_value).any(
            axis=1
        )
        at_least_as_good_in_all_objectives = (objective_values >= objective_value).all(
            axis=1
        )
        # A point is Pareto-dominated if there's a point that we could move to that
        # improves performance in one objective without losing performance in any
        # other objective
        # A point is Pareto-dominant if there are no points that dominate it
        is_dominant[i] = ~np.any(
            at_least_as_good_in_all_objectives & strictly_better_in_one_objective
        )

    if allow_zero:
        return is_dominant
    else:
        return is_dominant & np.all(objective_values > 0, axis=1)
