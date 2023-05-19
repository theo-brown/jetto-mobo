import h5py
import numpy as np
import plotly.graph_objects as go

from jetto_mobo.utils import colours, hex_to_rgba, pad_1d

figure = go.Figure()


def add_trace(ensemble_dir, colour, name):
    max_values = []

    for ensemble_member in range(1, 11):
        hdf5_file = f"{ensemble_dir}/{ensemble_member}/bayesopt.hdf5"
        with h5py.File(hdf5_file) as h5:
            optimisation_step = np.arange(
                h5["/"].attrs["n_completed_bayesopt_steps"] + 1
            )
            max_value = np.zeros(len(optimisation_step))
            for i, step in enumerate(optimisation_step):
                if i == 0:
                    max_value[:] = np.max(np.nan_to_num(h5["initialisation/value"], 0))
                else:
                    value = np.max(np.nan_to_num(h5[f"bayesopt/{step}/value"], 0))
                    if value > max_value[i - 1]:
                        max_value[i:] = value
        max_values.append(max_value)

    max_values = pad_1d(max_values)
    optimisation_step = np.arange(max_values.shape[1])
    mean = np.nanmean(max_values, axis=0)
    sd = np.nanstd(max_values, axis=0)
    max_ = np.nanmax(max_values, axis=0)
    min_ = np.nanmin(max_values, axis=0)

    figure.add_traces(
        [
            go.Scatter(
                x=optimisation_step,
                y=mean + 2 * sd,
                name=f"{name}: &mu; + 2&#963;",
                # y=max_,
                # name="Max",
                mode="lines",
                line_color=hex_to_rgba(colour, 0.1),
                showlegend=False,
                legendgroup=name,
            ),
            go.Scatter(
                x=optimisation_step,
                y=mean - 2 * sd,
                name=f"{name}: &mu; - 2&#963",
                # y=min_,
                # name="Min",
                mode="lines",
                fill="tonexty",
                line_color=hex_to_rgba(colour, 0.1),
                fillcolor=hex_to_rgba(colour, 0.1),
                showlegend=False,
                legendgroup=name,
            ),
            go.Scatter(
                x=optimisation_step,
                y=mean,
                mode="lines",
                name=f"{name}: &mu;",
                line_color=colour,
                showlegend=False,
                legendgroup=name,
            ),
            go.Scatter(
                x=[optimisation_step[0]],
                y=[mean[0]],
                mode="lines",
                name=name,
                line_color=colour,
                showlegend=True,
                legendgroup=name,
            ),
        ]
    )


# Add genetic algorithm trace
with h5py.File("../data/genetic_algorithm/2023-02-10/output.hdf5") as h5:
    optimisation_step = np.asarray(list(h5.keys()))
    max_objective_value = [max(h5[f"{i}/value"]) for i in optimisation_step]
    figure.add_trace(
        go.Scatter(
            x=optimisation_step,
            y=max_objective_value,
            line_color=colours[0],
            mode="lines",
            name="GA, M-PL",
        )
    )

# Add BO traces
add_trace("../data/sobo/ga_pl", colours[1], "BO, M-PL")
add_trace("../data/sobo/pl", colours[2], "BO, PL")
add_trace("../data/sobo/sog", colours[3], "BO, SoG")
add_trace("../data/sobo/sog_fm", colours[5], "BO, SoG-FM")

figure.update_layout(template="simple_white", showlegend=True)
figure.update_xaxes(
    title="Optimisation step", range=[0, 20], tickmode="linear", tick0=0, dtick=5
)
figure.update_yaxes(title="Maximum objective value achieved")
figure.write_image("./optimisation_comparison.png")
