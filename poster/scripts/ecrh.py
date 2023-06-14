import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import utils

from jetto_mobo.inputs.ecrh import sum_of_gaussians_fixed_log_means

x = np.linspace(0, 1, int(1e5))

profile_1 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.5,
    variances=np.array([0.006, 0.005, 0.0003, 0.02, 0.008]),
    amplitudes=np.array([1, 0.2, 0, 2, 1]),
)
profile_2 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.7,
    variances=np.array([0.012, 0.006, 0.006, 0.03, 0.006]),
    amplitudes=np.array([1.2, 0.18, 0.79, 0.8, 0.2]),
)
profile_3 = sum_of_gaussians_fixed_log_means(
    x=x,
    xmax=0.88,
    variances=np.array([0.003, 0.003, 0.002, 0.009, 0.001]),
    amplitudes=np.array([0.2, 0.1, 0.4, 0.60, 0.94]),
)

figure = go.Figure(
    [
        go.Scatter(
            x=x,
            y=profile_1 / np.max(profile_1),
            line_width=3,
        ),
        go.Scatter(
            x=x,
            y=profile_2 / np.max(profile_2),
            line_width=3,
        ),
        go.Scatter(
            x=x,
            y=profile_3 / np.max(profile_3),
            line_width=3,
        ),
    ]
)
figure.update_layout(
    xaxis_title="Normalised radius",
    yaxis_title="Normalised QECE",
    xaxis_linewidth=3,
    yaxis_linewidth=3,
    xaxis_range=[0, 1],
    yaxis_range=[0, 1.01],
    template="simple_white",
    font_size=30,
    showlegend=False,
)
figure.write_image("../images/ecrh_profiles.svg", format="svg", width=800, height=600)
