import plotly.graph_objects as go
import pytest
import torch
from botorch.utils.transforms import normalize

from jetto_mobo.configuration import SurrogateConfig
from jetto_mobo.surrogate import fit_surrogate_model

# Generate dummy data
dummy_input = torch.rand(10, 2)
dummy_1d_output = (dummy_input[:, 0] + dummy_input[:, 1]).unsqueeze(-1)
dummy_2d_output = dummy_input

model = fit_surrogate_model(
    dummy_input, dummy_2d_output, SurrogateConfig(model="ModelListGP")
)
x1_eval = torch.linspace(0, 1, 50)
x2_eval = torch.linspace(0, 1, 50)
X = torch.stack(torch.meshgrid(x1_eval, x2_eval), dim=-1).reshape(-1, 2)
posterior = model.posterior(X)
mean = posterior.mean.detach().numpy()

print(X[:, 0].shape)
print(X[:, 1].shape)
print(mean.shape)

go.Figure(
    go.Heatmap(
        x=X[:, 0],
        y=X[:, 1],
        z=mean[:, 0],
    )
).show()

go.Figure(
    go.Heatmap(
        x=X[:, 0],
        y=X[:, 1],
        z=mean[:, 1],
    )
).show()
