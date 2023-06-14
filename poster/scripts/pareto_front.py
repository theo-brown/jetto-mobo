import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import utils

# pio.kaleido.scope.mathjax = None

# Generate random data between 0 and 1
n_points = 20
np.random.seed(42)
x = np.random.rand(n_points)
y = np.random.rand(n_points)

# Find the Pareto front
is_dominant = utils.get_pareto_dominant_mask(np.array([x, y]).T)

# Plot
figure = go.Figure()
nondominant_colour = utils.colours[0]
dominant_colour = utils.colours[1]

# Add Pareto dominance rectangles
figure.add_shape(
    type="rect",
    x0=x[is_dominant][5],
    y0=y[is_dominant][5],
    x1=2,
    y1=2,
    fillcolor=utils.colour_to_rgba(dominant_colour, 0.7),
    line_color=dominant_colour,
    line_width=5,
    layer="below",
)
figure.add_shape(
    type="rect",
    x0=x[~is_dominant][2],
    y0=y[~is_dominant][2],
    x1=2,
    y1=2,
    fillcolor=utils.colour_to_rgba(nondominant_colour, 0.7),
    line_color=nondominant_colour,
    line_width=5,
    layer="below",
)
figure.add_traces(
    [
        go.Scatter(
            x=x[~is_dominant],
            y=y[~is_dominant],
            mode="markers",
            marker_color=nondominant_colour,
            marker_size=15,
            showlegend=False,
        ),
        go.Scatter(
            x=x[is_dominant],
            y=y[is_dominant],
            mode="markers",
            marker_color=dominant_colour,
            marker_size=15,
            showlegend=False,
        ),
    ]
)
figure.update_layout(
    xaxis_title="Objective 1",
    yaxis_title="Objective 2",
    xaxis_linewidth=3,
    yaxis_linewidth=3,
    xaxis_tickvals=[],
    yaxis_tickvals=[],
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    template="simple_white",
    font_size=30,
)
figure.write_image("../images/pareto_front.svg", format="svg", width=600, height=500)
