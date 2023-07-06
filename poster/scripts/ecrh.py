import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import utils
from plotly.colors import DEFAULT_PLOTLY_COLORS

from jetto_mobo.inputs.ecrh import (
    sum_of_gaussians_fixed_log_means,
    unit_interval_logspace,
)

x = np.linspace(0, 1, int(1e5))

xmax1 = 0.5
v1 = np.array([0.006, 0.005, 0.0003, 0.02, 0.008])
a1 = np.array([1, 0.2, 0, 2, 1])
profile_1 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.5,
    variances=np.array([0.006, 0.005, 0.0003, 0.02, 0.008]),
    amplitudes=np.array([1, 0.2, 0, 2, 1]),
)
xmax2 = 0.7
v2 = np.array([0.012, 0.006, 0.006, 0.03, 0.006])
a2 = np.array([1.2, 0.18, 0.79, 0.8, 0.2])
profile_2 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.7,
    variances=np.array([0.012, 0.006, 0.006, 0.03, 0.006]),
    amplitudes=np.array([1.2, 0.18, 0.79, 0.8, 0.2]),
)
xmax3 = 0.88
v3 = np.array([0.003, 0.003, 0.002, 0.009, 0.001])
a3 = np.array([0.2, 0.1, 0.4, 0.60, 0.94])
profile_3 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.88,
    variances=np.array([0.003, 0.003, 0.002, 0.009, 0.001]),
    amplitudes=np.array([0.2, 0.1, 0.4, 0.60, 0.94]),
)

figure = go.Figure(
    [
        go.Scatter(
            x=x,
            y=profile_1 / np.max(profile_1),
            line_width=3,
            line_color=DEFAULT_PLOTLY_COLORS[0],
        ),
        go.Scatter(
            x=x,
            y=profile_2 / np.max(profile_2),
            line_width=3,
            line_color=DEFAULT_PLOTLY_COLORS[1],
        ),
        go.Scatter(
            x=x,
            y=profile_3 / np.max(profile_3),
            line_width=3,
            line_color=DEFAULT_PLOTLY_COLORS[2],
        ),
    ]
)
x1 = unit_interval_logspace(5) * xmax1
figure.add_trace(
    go.Scatter(
        x=x1,
        y=[profile_1[np.argmin(np.abs(xi - x))] for xi in x1] / np.max(profile_1),
        mode="markers",
        marker_size=10,
        marker_color=DEFAULT_PLOTLY_COLORS[0],
    )
)
x2 = unit_interval_logspace(5) * xmax2
figure.add_trace(
    go.Scatter(
        x=x2,
        y=[profile_2[np.argmin(np.abs(xi - x))] for xi in x2] / np.max(profile_2),
        mode="markers",
        marker_size=10,
        marker_color=DEFAULT_PLOTLY_COLORS[1],
    )
)
x3 = unit_interval_logspace(5) * xmax3
figure.add_trace(
    go.Scatter(
        x=x3,
        y=[profile_3[np.argmin(np.abs(xi - x))] for xi in x3] / np.max(profile_3),
        mode="markers",
        marker_size=10,
        marker_color=DEFAULT_PLOTLY_COLORS[2],
    )
)
figure.update_layout(
    xaxis_title="Normalised radius",
    yaxis_title="Normalised QECE",
    xaxis_linewidth=3,
    yaxis_linewidth=3,
    xaxis_range=[0, 1],
    yaxis_range=[0, 1.02],
    template="simple_white",
    font_size=28,
    showlegend=False,
    margin={"l": 15, "r": 15, "t": 15, "b": 15},
)
figure.write_image("../images/ecrh_profiles.svg", format="svg", width=710, height=495)
