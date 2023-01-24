from typing import Iterable

import ecrh
import numpy as np
import plotly.graph_objects as go
import utils


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


if __name__ == "__main__":
    from utils import load_tensor

    ecrh_parameters = load_tensor("jetto_mobo.hdf5", "initialisation/ecrh_parameters")
    # costs = load_tensor("jetto_mobo.hdf5", "initialisation/costs")
    f = plot_piecewise_ecrh_batch(ecrh_parameters, [None] * ecrh_parameters.shape[0])
    f.show()
