import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import utils

from jetto_mobo.old import genetic_algorithm

x = np.linspace(0, 1, int(1e5))

lower_bounds = np.array([0, 0.05, 0.01, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
upper_bounds = np.array([1, 1, 0.09, 1, 1, 0.29, 0.9, 0.9, 1, 0.75, 0.9, 0.45])

figure = go.Figure()
for i in range(3):
    parameters = np.random.rand(12)
    # Turn parameters into between lower and upper bounds.
    parameters = lower_bounds + parameters * (upper_bounds - lower_bounds)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=genetic_algorithm.piecewise_linear(x, parameters),
            line_width=3,
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
figure.write_image(
    "../images/ecrh_profiles_old.svg", format="svg", width=710, height=495
)
