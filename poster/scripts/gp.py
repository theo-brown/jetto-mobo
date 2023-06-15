import numpy as np
import plotly.graph_objects as go
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

torch.manual_seed(4)
x = torch.tensor([0.8127, 0.4157, 0.4234], dtype=torch.double).unsqueeze(1)
y = torch.sin(x * (2 * np.pi)) + torch.randn_like(x) * 0.2
x_eval = torch.linspace(0, 1, int(1e4)).reshape(-1, 1)

model = SingleTaskGP(x, y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

acqf = ExpectedImprovement(model, best_f=0.0)
acqf_values = acqf(x_eval.unsqueeze(1))

candidates, values = optimize_acqf(
    acqf, q=1, bounds=torch.tensor([[0.0], [1.0]]), num_restarts=5, raw_samples=20
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

figure = go.Figure()
confidence_interval_color = "rgba(55, 126, 284, 0.1)"
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
            y=np.sin(x_eval * (2 * np.pi)),
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
            x=x,
            y=y,
            mode="markers",
            name="Observations",
            marker_color=observations_color,
            marker_size=15,
        ),
    ]
)
figure.add_vline(
    x=candidates,
    line_color=observations_color,
    name="Next point to trial",
    line_width=3,
    opacity=1,
)
figure.update_layout(
    xaxis_tickvals=[],
    yaxis_tickvals=[],
    xaxis_range=[0, 1],
    template="simple_white",
    showlegend=False,
    xaxis_linewidth=3,
    yaxis_linewidth=3,
    margin={"l": 15, "r": 15, "t": 15, "b": 15},
)
figure.write_image("../images/gp.svg", width=350, height=250)

figure = go.Figure(
    data=[
        go.Scatter(
            x=x_eval,
            y=acqf_values.detach().cpu().numpy(),
            line_width=3,
            line_color=DEFAULT_PLOTLY_COLORS[4],
        ),
    ],
    layout=go.Layout(
        template="simple_white",
        xaxis_tickvals=[],
        yaxis_tickvals=[],
        xaxis_range=[0, 1],
        showlegend=False,
        xaxis_linewidth=3,
        yaxis_linewidth=3,
        margin={"l": 15, "r": 15, "t": 15, "b": 15},
    ),
)
figure.add_vline(
    x=candidates,
    line_color=observations_color,
    name="Next point to trial",
    line_width=3,
    opacity=1,
)
figure.write_image("../images/acqf_plot.svg", width=350, height=125)
