from typing import Iterable

import numpy as np
import plotly
import plotly.graph_objects as go
from jetto_tools.results import JettoResults
from plotly.subplots import make_subplots

from jetto_mobo import ecrh


def plot_piecewise_ecrh_batch(
    parameters: Iterable[Iterable[float]], costs: Iterable[Iterable[float]]
):
    f = go.Figure()
    x = np.linspace(0, 1, 100)
    for p, c in zip(parameters, costs):
        f.add_trace(
            go.Scatter(
                x=x, y=ecrh.piecewise_linear(x, p), mode="lines", name=f"Cost = {c}"
            )
        )
    return f


def plot_jetto(jetto_dir: str):
    profiles = JettoResults(path=jetto_dir).load_profiles()
    xrho = profiles["XRHO"][-1].data
    safety_factor = profiles["Q"][-1].data
    ecrh_power_density = profiles["QECE"][-1].data
    figure = make_subplots(1, 2)

    figure.add_traces(
        [
            go.Scatter(
                x=xrho,
                y=ecrh_power_density,
                name="ECRH",
                line_color=plotly.colors.qualitative.Set1[0],
            ),
            go.Scatter(
                x=xrho,
                y=safety_factor,
                name="q",
                showlegend=False,
                line_color=plotly.colors.qualitative.Set1[0],
            ),
        ],
        rows=[1, 1],
        cols=[1, 2],
    )
    return figure


def plot_bayesopt_progress(hdf5_file: str):
    import h5py

    with h5py.File(hdf5_file, "r") as f:
        cost_figure = go.Figure()
        optimisation_step = np.arange(len(f["bayesopt"]))
        bayesopt_cost = np.empty(len(f["bayesopt"]))

        for bayesopt_step in f["bayesopt"]:
            if "cost" in f["bayesopt"][bayesopt_step]:
                cost = np.nan_to_num(f["bayesopt"][bayesopt_step]["cost"], nan=1e3)
                bayesopt_cost[int(bayesopt_step)] = np.min(cost)
            else:
                bayesopt_cost[int(bayesopt_step)] = None
        cost_figure.add_trace(
            go.Scatter(
                x=optimisation_step,
                y=bayesopt_cost,
                name="BayesOpt",
            )
        )

        initial_cost = np.nan_to_num(f["initialisation/cost"], nan=1e3)
        cost_figure.add_trace(
            go.Scatter(
                x=optimisation_step,
                y=np.ones(optimisation_step.shape) * np.min(initial_cost),
                name="Random initialisation",
                mode="lines",
                line_color="black",
                line_dash="dot",
            )
        )

        cost_figure.update_layout(
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
    return cost_figure


if __name__ == "__main__":
    plot_bayesopt_progress("data/2023-01-27-173809.hdf5").show()

    # from plotly.subplots import make_subplots
    # from utils import load_tensor

    # # ecrh_parameters = load_tensor("jetto_mobo.hdf5", "initialisation/ecrh_parameters")
    # # costs = load_tensor("jetto_mobo.hdf5", "initialisation/cost")
    # # plot_piecewise_ecrh_batch(ecrh_parameters, costs).show()
    # # ecrh_parameters = load_tensor("jetto_mobo.hdf5", "bayesopt/0/ecrh_parameters")
    # # costs = load_tensor("jetto_mobo.hdf5", "bayesopt/0/cost")
    # # plot_piecewise_ecrh_batch(ecrh_parameters, costs).show()
    # # ecrh_parameters = load_tensor("jetto_mobo.hdf5", "bayesopt/1/ecrh_parameters")
    # # costs = [None]*len(ecrh_parameters)
    # # plot_piecewise_ecrh_batch(ecrh_parameters, costs).show()

    # for directory in [
    #     *[f"jetto/runs/initial/{i}" for i in range(5)],
    #     *[f"jetto/runs/bayesopt/0/{i}" for i in range(5)],
    #     *[f"jetto/runs/bayesopt/1/{i}" for i in range(5)],
    # ]:
    #     plot_jetto(directory).show()
