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
