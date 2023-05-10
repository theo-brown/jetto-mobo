import logging
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
from matplotlib import colormaps
from plotly.colors import hex_to_rgb


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        record.elapsed_time = datetime.utcfromtimestamp(
            record.relativeCreated / 1000
        ).strftime("%H:%M:%S")
        return super().format(record)


def get_logger(
    name: Optional[str] = None, level: Optional[int] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(
            ElapsedTimeFormatter(
                "t+%(elapsed_time)s:%(name)s:%(levelname)s %(message)s"
            )
        )
        logger.addHandler(handler)
    if level is not None:
        logger.setLevel(level)
    return logger


def pad_1d(a: Iterable[Optional[np.ndarray]], pad_value: float = np.nan) -> np.ndarray:
    """Pad a ragged sequence of 1D arrays."""
    row_length = max([len(r) if r is not None else 0 for r in a])
    a_padded = np.full((len(a), row_length), np.nan)
    for i, r in enumerate(a):
        if r is not None:
            a_padded[i, : len(r)] = np.array(r)
    return a_padded


def rgba_colormap(
    x: float, min_x: float, max_x: float, colormap_name: str, alpha: float = 1
) -> str:
    colormap = colormaps[colormap_name]
    color = colormap((x - min_x) / (max_x - min_x))
    return f"rgba({int(255 * color[0])}, {int(255 * color[1])}, {int(255 * color[2])}, {alpha})"


def hex_to_rgba(hex_colour: str, alpha: float):
    r, g, b = hex_to_rgb(hex_colour)
    return f"rgba({r},{g},{b},{alpha})"


def rgb_to_rgba(rgb_colour: str, alpha: float):
    return f"rgba{rgb_colour[3:-1]}, {alpha})"


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
