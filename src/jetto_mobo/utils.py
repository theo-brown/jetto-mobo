import logging
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
from matplotlib import colormaps


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        record.elapsed_time = datetime.utcfromtimestamp(
            record.relativeCreated / 1000
        ).strftime("%H:%M:%S")
        return super().format(record)


def pad_1d(a: Iterable[Optional[np.ndarray]], pad_value: float = np.nan) -> np.ndarray:
    """Pad a ragged sequence of 1D arrays."""
    row_length = max([len(r) if r is not None else 0 for r in a])
    a_padded = np.full((len(a), row_length), np.nan)
    for i, r in enumerate(a):
        if r is not None:
            a_padded[i, : len(r)] = np.array(r)
    return a_padded


def rgba_colormap(x: float, min_x: float, max_x: float, colormap_name: str) -> str:
    colormap = colormaps[colormap_name]
    color = colormap((x - min_x) / (max_x - min_x))
    return f"rgba({int(255 * color[0])}, {int(255 * color[1])}, {int(255 * color[2])}, {color[3]})"
