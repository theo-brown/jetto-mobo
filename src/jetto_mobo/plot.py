import os

import h5py
import numpy as np
import plotly
import plotly.graph_objects as go
from jetto_tools.results import JettoResults
from plotly.subplots import make_subplots

from jetto_mobo.objective import scalar_cost_function, vector_cost_function
from jetto_mobo.utils import rgba_colormap


def get_benchmark_data(path: str):
    benchmark = JettoResults(path=path)
    profiles = benchmark.load_profiles()
    timetraces = benchmark.load_timetraces()
    return {
        "scalar_cost": scalar_cost_function(profiles, timetraces),
        "vector_cost": vector_cost_function(profiles, timetraces),
        "ecrh": profiles["QECE"][-1],
        "q": profiles["Q"][-1],
        "xrho": profiles["XRHO"][-1],
    }


def animation(
    input_filename: str,
    output_filename: str,
    title: str,
    benchmark_path: str = "./data/benchmark",
    cost_range=[0, 30],
):
    benchmark = get_benchmark_data(benchmark_path)

    with h5py.File(input_filename, "r") as hdf5_file:

        def get_traces(group: str):
            min_cost = float(np.min(np.nan_to_num(hdf5_file[group]["cost"], nan=1e3)))
            q_traces = []
            ecrh_traces = []
            for i in range(batch_size):
                # Cost and colour
                cost_i = float(hdf5_file[group]["cost"][i])
                if np.isnan(cost_i):
                    continue
                colour = rgba_colormap(
                    cost_i,
                    cost_range[0],
                    cost_range[1],
                    "viridis",
                    alpha=1 if cost_i == min_cost else 0.3,
                )

                # Q profile
                converged_q_i = hdf5_file[group]["converged_q"][i]
                q_traces.append(
                    go.Scatter(
                        x=np.linspace(0, 1, len(converged_q_i)),
                        y=converged_q_i,
                        mode="lines",
                        line_color=colour,
                        showlegend=False,
                    )
                )

                # ECRH profile
                converged_ecrh_i = hdf5_file[group]["converged_ecrh"][i]
                ecrh_traces.append(
                    go.Scatter(
                        x=np.linspace(0, 1, len(converged_ecrh_i)),
                        y=converged_ecrh_i,
                        mode="lines",
                        line_color=colour,
                        showlegend=False,
                    )
                )
            return min_cost, q_traces, ecrh_traces

        n_completed_bayesopt_steps = hdf5_file["/"].attrs["n_completed_bayesopt_steps"]
        batch_size = hdf5_file["/"].attrs["batch_size"]

        step = np.arange(n_completed_bayesopt_steps + 1)
        cost = np.empty(len(step))

        # Initialisation
        cost[0], initial_q_traces, initial_ecrh_traces = get_traces("initialisation")
        initial_cost_trace = go.Scatter(
            x=step,
            y=np.ones(len(step)) * cost[0],
            mode="lines",
            line_color="black",
            line_dash="dot",
            name="Random initialisation",
        )

        # Benchmark
        benchmark_cost_trace = go.Scatter(
            x=step,
            y=np.ones(len(step)) * benchmark["scalar_cost"],
            name="SPR45-v9",
            legendgroup="SPR45-v9",
            mode="lines",
            line_dash="dash",
            line_color="black",
        )
        benchmark_ecrh_trace = go.Scatter(
            x=benchmark["xrho"],
            y=benchmark["ecrh"],
            name="SPR45-v9",
            legendgroup="SPR45-v9",
            mode="lines",
            line_dash="dash",
            showlegend=False,
            line_color="black",
        )
        benchmark_q_trace = go.Scatter(
            x=benchmark["xrho"],
            y=benchmark["q"],
            name="SPR45-v9",
            legendgroup="SPR45-v9",
            mode="lines",
            line_dash="dash",
            showlegend=False,
            line_color="black",
        )

        frames = []
        for bayesopt_step in np.arange(0, n_completed_bayesopt_steps + 1):
            figure = make_subplots(
                rows=2, cols=2, specs=[[{"colspan": 2}, {}], [{}, {}]]
            )

            # Benchmark data
            figure.add_traces(
                [
                    initial_cost_trace,
                    benchmark_cost_trace,
                    benchmark_ecrh_trace,
                    benchmark_q_trace,
                ],
                rows=[1, 1, 2, 2],
                cols=[1, 1, 1, 2],
            )

            if bayesopt_step == 0:
                # Initialisation data
                figure.add_traces(
                    initial_ecrh_traces,
                    rows=[2] * len(initial_ecrh_traces),
                    cols=[1] * len(initial_ecrh_traces),
                )
                figure.add_traces(
                    initial_q_traces,
                    rows=[2] * len(initial_q_traces),
                    cols=[2] * len(initial_q_traces),
                )
            else:
                # Bayesopt data
                cost[bayesopt_step], q_traces, ecrh_traces = get_traces(
                    f"bayesopt/{bayesopt_step}"
                )
                figure.add_traces(
                    ecrh_traces,
                    rows=[2] * len(ecrh_traces),
                    cols=[1] * len(ecrh_traces),
                )
                figure.add_traces(
                    q_traces, rows=[2] * len(q_traces), cols=[2] * len(q_traces)
                )

            # Cost trace
            figure.add_trace(
                go.Scatter(
                    x=step[: bayesopt_step + 1],
                    y=cost[: bayesopt_step + 1],
                    mode="lines+markers",
                    showlegend=False,
                    line_color=plotly.colors.qualitative.Bold[2],
                ),
                row=1,
                col=1,
            )

            # Colorbar
            figure.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        colorscale="viridis",
                        showscale=True,
                        cmin=cost_range[0],
                        cmax=cost_range[1],
                        colorbar_title="Cost",
                        colorbar_titleside="top",
                        colorbar_outlinewidth=0,
                        colorbar_tickwidth=1,
                        colorbar_len=0.5,
                        colorbar_yanchor="bottom",
                        colorbar_y=0,
                        colorbar_xanchor="left",
                        colorbar_x=1.02,
                    ),
                    hoverinfo="none",
                )
            )

            figure.update_xaxes(
                title_text="Optimisation step",
                row=1,
                col=1,
                range=[0, n_completed_bayesopt_steps],
            )
            figure.update_yaxes(title_text="Minimum cost", row=1, col=1)
            figure.update_xaxes(
                title_text="Normalised radius", row=2, col=1, range=[0, 1]
            )
            figure.update_yaxes(
                title_text="ECRH power density",
                row=2,
                col=1,
                range=[0, 1.25 * np.max(benchmark["ecrh"])],
            )
            figure.update_xaxes(
                title_text="Normalised radius", row=2, col=2, range=[0, 1]
            )
            figure.update_yaxes(
                title_text="Safety factor",
                row=2,
                col=2,
                range=[0, 1.1 * np.max(benchmark["q"])],
            )
            figure.update_layout(
                template="simple_white",
                margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
                font_size=20,
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1,
                legend_xanchor="right",
                legend_x=1,
                title_text=title,
                title_x=0.5,
            )

            frames.append(figure)

    os.makedirs(output_filename)
    for i, frame in enumerate(frames):
        frame.write_image(
            f"{output_filename}/{i:02}.png", width=1920, height=1080, scale=1
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
    )
    parser.add_argument(
        "title",
        type=str,
    )
    parser.add_argument(
        "output_file",
        type=str,
    )
    args = parser.parse_args()

    animation(args.input_file, args.output_file, args.title)
