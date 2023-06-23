import numpy as np
import plotly.graph_objects as go
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.models import FixedNoiseGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots


def true_f(x):
    freq = 5
    amp = 5
    amp_offset = 2
    return amp * np.sin(freq * x * np.pi) / (freq * x * np.pi) + amp_offset


x = torch.tensor([-0.18, -0.62, 0.35, 0.43, 0.6], dtype=torch.double).reshape(-1, 1)
y = true_f(x)
x_eval = torch.linspace(-1, 1, int(1e4)).reshape(-1, 1)

model = FixedNoiseGP(x, y, torch.ones_like(y) * 1e-3)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

acqf = ExpectedImprovement(model, best_f=0.0)
acqf_values = acqf(x_eval.unsqueeze(1))

candidates, values = optimize_acqf(
    acqf, q=1, bounds=torch.tensor([[0.0], [1.0]]), num_restarts=10, raw_samples=512
)

posterior = model.posterior(x_eval)

x_eval = x_eval.squeeze().detach().cpu().numpy()
mean = posterior.mean.squeeze().detach().cpu().numpy()
lower, upper = posterior.confidence_region()
lower = lower.squeeze().detach().cpu().numpy()
upper = upper.squeeze().detach().cpu().numpy()

x = x.squeeze().detach().cpu().numpy()
y = y.squeeze().detach().cpu().numpy()

candidates = candidates.detach().cpu().squeeze().numpy()
values = values.detach().cpu().squeeze().numpy()

figure = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05
)
confidence_interval_color = "rgba(55, 126, 284, 0.2)"
mean_color = DEFAULT_PLOTLY_COLORS[0]
observations_color = "rgba(228, 26, 28, 1)"
figure.add_traces(
    [
        go.Scatter(
            x=x_eval,
            y=lower,
            name="Lower",
            line_color=confidence_interval_color,
            line_width=0,
        ),
        go.Scatter(
            x=x_eval,
            y=upper,
            name="Upper",
            fill="tonexty",
            fillcolor=confidence_interval_color,
            line_color=confidence_interval_color,
            line_width=0,
        ),
        go.Scatter(
            x=x_eval,
            y=true_f(x_eval),
            name="True function",
            line_color="darkgrey",
            line_dash="dash",
            line_width=3,
        ),
        go.Scatter(
            x=x_eval,
            y=mean,
            name="Mean",
            line_color=mean_color,
            line_width=3,
        ),
        go.Scatter(
            x=x.squeeze(),
            y=y.squeeze(),
            mode="markers",
            name="Observations",
            marker_color=observations_color,
            marker_size=15,
        ),
    ],
    rows=[1, 1, 1, 1, 1],
    cols=[1, 1, 1, 1, 1],
)
figure.add_trace(
    go.Scatter(
        x=x_eval,
        y=acqf_values.detach().cpu().numpy(),
        line_width=3,
        line_color=DEFAULT_PLOTLY_COLORS[4],
    ),
    row=2,
    col=1,
)
figure.add_vline(
    x=candidates,
    line_color=observations_color,
    name="Next point to trial",
    line_width=3,
    opacity=1,
    row=1,
    col=1,
)
figure.add_vline(
    x=candidates,
    line_color=observations_color,
    name="Next point to trial",
    line_width=3,
    opacity=1,
    row=2,
    col=1,
)
figure.update_xaxes(
    showticklabels=False, linewidth=3, tickmode="array", tickvals=[], row=1, col=1
)
figure.update_xaxes(
    showticklabels=False, linewidth=3, tickmode="array", tickvals=[], row=2, col=1
)
figure.update_yaxes(
    title="GP",
    showticklabels=False,
    linewidth=3,
    tickmode="array",
    tickvals=[],
    row=1,
    col=1,
)
figure.update_yaxes(
    title="ACQF",
    showticklabels=False,
    linewidth=3,
    tickmode="array",
    tickvals=[],
    row=2,
    col=1,
)
figure.update_layout(
    xaxis_range=[-0.75, 0.75],
    template="simple_white",
    showlegend=False,
    margin={"l": 15, "r": 15, "t": 15, "b": 15},
    font_size=24,
)
figure.write_image("../images/gp_and_acqf.svg", width=400, height=425)
figure.show()
