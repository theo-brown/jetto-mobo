import numpy as np
import plotly.graph_objects as go
import utils
from plotly.subplots import make_subplots

# Generate random data
n_points = 20
np.random.seed(42)
x = np.random.rand(n_points) * 100  # x is quantity, 0-100%
y = np.random.rand(n_points) * 100  # y is quality, 0-100%

# Find the Pareto front
is_dominant = utils.get_pareto_dominant_mask(np.array([x, y]).T)

# # Plot
# figure = go.Figure()
nondominant_colour = utils.colours[0]
dominant_colour = utils.colours[1]

# figure.add_traces(
#     [
#         go.Scatter(
#             x=x[~is_dominant],
#             y=y[~is_dominant],
#             mode="markers",
#             marker_color=nondominant_colour,
#             marker_size=8,
#             name="Not Pareto dominant",
#         ),
#         go.Scatter(
#             x=x[is_dominant],
#             y=y[is_dominant],
#             mode="markers",
#             marker_color=dominant_colour,
#             marker_size=8,
#             name="Pareto dominant",
#         ),
#     ]
# )
# figure.update_layout(
#     xaxis_title="Robustness",
#     yaxis_title="Responsiveness",
#     xaxis_tickmode="array",
#     xaxis_tickvals=np.arange(0, 101, 25),
#     xaxis_ticktext=[f"{i}%" for i in np.arange(0, 101, 25)],
#     yaxis_tickmode="array",
#     yaxis_tickvals=np.arange(0, 101, 25),
#     yaxis_ticktext=[f"{i}%" for i in np.arange(0, 101, 25)],
#     xaxis_range=[0, 100],
#     yaxis_range=[0, 100],
#     template="simple_white",
#     font_size=16,
#     legend_orientation="h",
#     legend_y=1,
#     legend_yanchor="bottom",
#     legend_x=1,
#     legend_xanchor="right",
#     margin=dict(b=0, l=0, r=0, t=0),
# )
# figure.write_image("../images/pareto_front.svg", height=400, width=800)

# # Masking
# figure.add_shape(
#     type="rect",
#     x0=-1,
#     y0=-1,
#     x1=25,
#     y1=200,
#     fillcolor="grey",
#     line_color="grey",
#     line_width=5,
#     layer="below",
# )

# figure.add_shape(
#     type="rect",
#     x0=-1,
#     y0=-1,
#     x1=200,
#     y1=50,
#     fillcolor="grey",
#     line_color="grey",
#     line_width=5,
#     layer="below",
# )
# figure.write_image("../images/masked_pareto_front.svg", height=400, width=800)

# Weighting transform
weighting_figure = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"rowspan": 2}, {}], [None, {}]],
    subplot_titles=["", "w = [5, 1]", "w = [2, 1]", ""],
    column_titles=["Multi-objective", "Single-objective"],
    vertical_spacing=0.2,
)
ordered_indices = x[is_dominant].argsort()
labels = ["A", "B", "C", "D", "E", "F"]


def weighted_points(weight_x, weight_y):
    weights = np.array([weight_x, weight_y])
    normalised_weights = weights / np.sum(weights)

    return (
        normalised_weights[0] * x[is_dominant][ordered_indices]
        + normalised_weights[1] * y[is_dominant][ordered_indices]
    )


weighting_figure.add_traces(
    [
        go.Scatter(
            x=x[~is_dominant],
            y=y[~is_dominant],
            mode="markers+text",
            marker_color=nondominant_colour,
            marker_size=8,
        ),
        go.Scatter(
            x=x[is_dominant][ordered_indices],
            y=y[is_dominant][ordered_indices],
            marker_size=8,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker_color=dominant_colour,
        ),
        go.Bar(
            x=weighted_points(5, 1)[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=utils.colours[2],
        ),
        go.Bar(
            x=weighted_points(2, 1)[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=utils.colours[3],
        ),
    ],
    rows=[1, 1, 1, 2],
    cols=[1, 1, 2, 2],
)

weighting_figure.update_xaxes(
    title="Robustness",
    tickmode="array",
    tickvals=np.arange(0, 101, 25),
    ticktext=[f"{i}%" for i in np.arange(0, 101, 25)],
    range=[0, 100],
    row=1,
    col=1,
)
weighting_figure.update_yaxes(
    title="Responsiveness",
    tickmode="array",
    tickvals=np.arange(0, 101, 25),
    ticktext=[f"{i}%" for i in np.arange(0, 101, 25)],
    range=[0, 105],
    row=1,
    col=1,
)
for row in [1, 2]:
    weighting_figure.update_xaxes(
        title="Weighted score" if row == 2 else "",
        tickmode="array",
        tickvals=np.arange(0, 101, 25),
        ticktext=[f"{i}%" for i in np.arange(0, 101, 25)],
        range=[0, 90],
        row=row,
        col=2,
    )

weighting_figure.update_layout(
    template="simple_white",
    font_size=16,
    showlegend=False,
    margin=dict(b=0, l=0, r=0, t=50),
)

weighting_figure.for_each_annotation(
    lambda a: a.update(font_size=20, yshift=22)
    if a.text in ["Multi-objective", "Single-objective"]
    else ()
)
weighting_figure.write_image(
    "../images/weighted_multiobjective.svg", height=500, width=800
)
