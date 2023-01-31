import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jetto_mobo.utils import rgba_colormap


def solution_batch(group: h5py.Group):
    figure = make_subplots(1, 2)
    converged_cost = group["cost"][~np.isnan(group["cost"])]
    max_cost = np.max(converged_cost)
    min_cost = np.min(converged_cost)

    for i in range(len(group["ecrh_parameters"])):
        if np.any(np.isnan(group["cost"][i])):
            continue

        color = rgba_colormap(group["cost"][i][0], min_cost, max_cost, "viridis")
        figure.add_traces(
            [
                # go.Scatter(
                #     x=np.linspace(0, 1, len(group["target_ecrh"][i])),
                #     # Rescale target_ecrh to match the converged value
                #     y=group["target_ecrh"][i] * np.max(group["converged_ecrh"][i]),
                #     name=str(i),
                #     line_dash="dot",
                #     line_color=color,
                # ),
                go.Scatter(
                    x=np.linspace(0, 1, len(group["converged_ecrh"][i])),
                    y=group["converged_ecrh"][i],
                    name=str(i),
                    line_color=color,
                    legendgroup=str(i),
                    hovertemplate=f"Cost: {group['cost'][i][0]:.2f}",
                ),
                go.Scatter(
                    x=np.linspace(0, 1, len(group["converged_q"][i])),
                    y=group["converged_q"][i],
                    name=str(i),
                    line_color=color,
                    legendgroup=str(i),
                    showlegend=False,
                    hovertemplate=f"Cost: {group['cost'][i][0]:.2f}",
                ),
            ],
            rows=[1, 1],
            cols=[1, 2],
        )
    # Add colorbar
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            showlegend=False,
            marker=dict(
                colorscale="viridis",
                showscale=True,
                cmin=min_cost,
                cmax=max_cost,
                colorbar_title="Cost",
                colorbar_titleside="top",
                colorbar_outlinewidth=0,
                colorbar_tickwidth=1,
            ),
            hoverinfo="none",
        )
    )
    figure.update_yaxes(title="ECRH power density", row=1, col=1)
    figure.update_yaxes(title="Safety factor", row=1, col=2)
    figure.update_xaxes(title="Normalised radius")
    figure.update_layout(
        template="simple_white",
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        legend_title="Member",
        legend_x=1.08,
        font_size=17,
    )

    return figure


def optimisation_progress(hdf5_file: h5py.File):
    figure = go.Figure()
    optimisation_step = np.arange(len(hdf5_file["bayesopt"]))
    bayesopt_cost = np.empty(len(hdf5_file["bayesopt"]))

    for bayesopt_step in hdf5_file["bayesopt"]:
        if "cost" in hdf5_file["bayesopt"][bayesopt_step]:
            cost = np.nan_to_num(hdf5_file["bayesopt"][bayesopt_step]["cost"], nan=1e3)
            bayesopt_cost[int(bayesopt_step)] = np.min(cost)
        else:
            bayesopt_cost[int(bayesopt_step)] = None
    figure.add_trace(
        go.Scatter(
            x=optimisation_step,
            y=bayesopt_cost,
            name="BayesOpt",
        )
    )

    initial_cost = np.nan_to_num(hdf5_file["initialisation/cost"], nan=1e3)
    figure.add_trace(
        go.Scatter(
            x=optimisation_step,
            y=np.ones(optimisation_step.shape) * np.min(initial_cost),
            name="Random initialisation",
            mode="lines",
            line_color="black",
            line_dash="dot",
        )
    )

    figure.update_layout(
        template="simple_white",
        margin={"l": 0, "r": 0, "b": 50, "t": 0, "pad": 0},
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1,
        legend_xanchor="center",
        legend_x=0.5,
        font_size=17,
        xaxis_title="Optimisation step",
        yaxis_title="Cost",
    )
    return figure


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["solution", "progress"],
    )
    parser.add_argument(
        "input_file",
        type=str,
    )
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as f:
        if args.mode == "solution":
            solution_batch(f["initialisation"]).show()
            if "bayesopt" in f.keys():
                for i in range(len(f["bayesopt"])):
                    solution_batch(f[f"bayesopt/{i}"]).show()
        elif args.mode == "progress":
            optimisation_progress(f).show()
