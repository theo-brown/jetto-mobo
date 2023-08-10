import h5py
import numpy as np
import plotly
import plotly.io as pio
import utils
from netCDF4 import Dataset
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from jetto_mobo.old.objective import vector_objective
from jetto_mobo.old.utils import get_pareto_dominant_mask

# disable mathjax
pio.kaleido.scope.mathjax = None

# Load data from MOBO
n_steps = 5
batch_size = 5
n_objectives = 7
n_parameters = 12
n_radial_points = 150

parameters = np.zeros((n_steps * batch_size, n_parameters))
objective_values = np.zeros((n_steps * batch_size, n_objectives))
converged_ecrh = np.zeros((n_steps * batch_size, 150))
converged_q = np.zeros((n_steps * batch_size, 150))

with h5py.File("../../data/mobo/ga_pl/bayesopt.hdf5") as hdf5:
    for i in np.arange(n_steps):
        for j in np.arange(batch_size):
            if not np.any(np.isnan(hdf5[f"bayesopt/{i+1}/value"][j])):
                objective_values[i * batch_size + j] = hdf5[f"bayesopt/{i+1}/value"][j]
                parameters[i * batch_size + j] = hdf5[
                    f"bayesopt/{i+1}/ecrh_parameters"
                ][j]
                converged_ecrh[i * batch_size + j] = hdf5[
                    f"bayesopt/{i+1}/converged_ecrh"
                ][j]
                converged_q[i * batch_size + j] = hdf5[f"bayesopt/{i+1}/converged_q"][j]

# Get Pareto optimal solutions
pareto_dominant_mask = get_pareto_dominant_mask(objective_values, allow_zero=False)

# Generate figure
figure = make_subplots(
    rows=2,
    cols=2,
    specs=[[{}, {"rowspan": 2, "type": "polar", "b": 0.1}], [{}, {}]],
    horizontal_spacing=0.13,
    vertical_spacing=0.12,
)


def add_solution_to_figure(
    objective_value, ecrh, q, name, colour, dash=False, fill_alpha=0.1
):
    figure.add_traces(
        [
            go.Scatterpolar(
                r=np.concatenate([objective_value, [objective_value[0]]]),
                theta=[
                    "Proximity<br>of q<sub>0</sub> to q<sub>min</sub>",
                    "Proximity of<br>q<sub>min</sub> to 2.2",
                    "Proximity of<br>q<sub>min</sub> to centre",
                    "Monotonicity<br>of q",
                    "Monotonicity<br>of ∇q",
                    "Location<br>of q = 3",
                    "Location<br>of q = 4",
                    "Proximity<br>of q<sub>0</sub> to q<sub>min</sub>",
                ],
                legendgroup=name,
                name=name,
                showlegend=False,
                # fill="toself",
                # fillcolor=utils.colour_to_rgba(colour, fill_alpha),
                # line_dash="dash" if dash else None,
                line_color=colour,
                line_width=3,
                marker_size=10,
            ),
            go.Scatter(
                x=np.linspace(0, 1, n_radial_points),
                y=ecrh,
                legendgroup=name,
                name=name,
                showlegend=False,
                mode="lines",
                line_dash="dash" if dash else None,
                line_color=colour,
                line_width=3,
            ),
            go.Scatter(
                x=np.linspace(0, 1, n_radial_points),
                y=q,
                legendgroup=name,
                name=name,
                showlegend=True,
                mode="lines",
                line_dash="dash" if dash else None,
                line_color=colour,
                line_width=3,
            ),
            go.Scatter(
                x=[np.linspace(0, 1, n_radial_points)[np.argmin(q)]],
                y=[np.min(q)],
                legendgroup=name,
                name=f"{name} - minimum",
                mode="markers",
                line_color=colour,
                showlegend=False,
                marker_size=10,
            ),
        ],
        rows=[1, 1, 2, 2],
        cols=[2, 1, 1, 1],
    )


# Plot Bayesopt results
solution_1 = np.nonzero(pareto_dominant_mask)[0][0]
solution_5 = np.nonzero(pareto_dominant_mask)[0][4]
solution_8 = np.nonzero(pareto_dominant_mask)[0][7]
add_solution_to_figure(
    objective_values[solution_1],
    converged_ecrh[solution_1],
    converged_q[solution_1],
    name=f"Solution 1/8",
    colour=utils.colours[0],
)
add_solution_to_figure(
    objective_values[solution_5],
    converged_ecrh[solution_5],
    converged_q[solution_5],
    name=f"Solution 5/8",
    colour=utils.colours[1],
)
add_solution_to_figure(
    objective_values[solution_8],
    converged_ecrh[solution_8],
    converged_q[solution_8],
    name=f"Solution 8/8",
    colour=utils.colours[2],
)

# Layout settings
figure.update_layout(
    template="simple_white",
)
figure.update_layout(
    polar_radialaxis_range=[0, 1.01],
    polar_radialaxis_showgrid=True,
    polar_radialaxis_gridcolor="grey",
    polar_angularaxis_showgrid=True,
    polar_angularaxis_showline=False,
    polar_angularaxis_gridcolor="grey",
    legend_orientation="h",
    legend_x=0,
    legend_y=1,
    legend_xanchor="left",
    legend_yanchor="bottom",
    margin={"l": 20, "r": 110, "b": 20, "t": 20, "pad": 0},
    font_size=22,
)
figure.add_annotation(
    text="Vector objective values",
    xref="paper",
    yref="paper",
    x=0.93,
    y=-0.17,
    showarrow=False,
    font_size=26,
)
figure.update_yaxes(title="QECE [W]", row=1, col=1, linewidth=3)
figure.update_yaxes(range=[1.8, None], title="Safety factor", row=2, col=1, linewidth=3)
figure.update_xaxes(title="Normalised radius", row=2, col=1, range=[0, 1], linewidth=3)
figure.update_xaxes(row=1, col=2, linewidth=3)
figure.write_image(
    "../images/mobo_old.svg",
    width=1280,
    height=600,
)
