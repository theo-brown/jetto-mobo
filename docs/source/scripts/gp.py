import plotly
import plotly.graph_objects as go
import torch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(5)

confidence_interval_color = "rgba(55, 126, 284, 0.1)"
mean_color = plotly.colors.DEFAULT_PLOTLY_COLORS[0]
observations_color = "rgba(228, 26, 28, 1)"

x_eval = torch.linspace(0, 1, int(1e3), dtype=torch.double)
n_observations = 3
observed_xs = torch.rand((n_observations, 1), dtype=torch.double)
observed_ys = torch.rand((n_observations, 1), dtype=torch.double)

figure = go.Figure(layout_template="simple_white")

bo_model = FixedNoiseGP(
    observed_xs,
    observed_ys,
    torch.full_like(observed_xs, 1e-5),
)
mll = ExactMarginalLogLikelihood(bo_model.likelihood, bo_model)
fit_gpytorch_mll(mll)
posterior = bo_model(x_eval.unsqueeze(-1))
lower, upper = posterior.confidence_region()
mean = posterior.mean

figure.add_traces(
    [
        go.Scatter(
            x=x_eval,
            y=lower.detach().squeeze(),
            mode="lines",
            line_color=confidence_interval_color,
            fillcolor=confidence_interval_color,
            showlegend=False,
            legendgroup="confidence_interval",
            name="95% confidence interval ",
        ),
        go.Scatter(
            x=x_eval,
            y=upper.detach().squeeze(),
            mode="lines",
            line_color=confidence_interval_color,
            fillcolor=confidence_interval_color,
            fill="tonexty",
            legendgroup="confidence_interval",
            name="95% confidence interval",
        ),
        go.Scatter(
            x=x_eval,
            y=posterior.mean.detach().squeeze(),
            line_color=mean_color,
            name="Posterior mean",
        ),
        go.Scatter(
            x=observed_xs.squeeze(),
            y=observed_ys.squeeze(),
            mode="markers",
            marker_color=observations_color,
            name="Observations",
        ),
    ]
)
figure.add_annotation(
    x=observed_xs[1].item(),
    y=observed_ys[1].item(),
    text="Low uncertainty<br>near observed points",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    axref="x",
    ayref="y",
    ax=0.2,
    ay=1.4,
)
figure.add_annotation(
    x=x_eval[-100].item(),
    y=lower[-100].item(),
    text="High uncertainty away<br>from observed points",
    showarrow=True,
    arrowhead=2,
    arrowsize=2,
    axref="x",
    ayref="y",
    ax=0.7,
    ay=-2.2,
)
figure.update_layout(
    template="simple_white",
    xaxis_visible=False,
    yaxis_visible=False,
    legend_orientation="h",
    legend_x=1,
    legend_y=1,
    legend_xanchor="right",
    legend_yanchor="bottom",
    margin=dict(b=0, l=0, r=0, t=0),
    font_size=16,
)
figure.write_image("../images/gp.svg", width=800, height=400)
