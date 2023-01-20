import os
from typing import Iterable

import ecrh
import jetto_singularity
import jetto_tools
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP  # Maybe use a different one?
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import MarginalLogLikelihood

# Constants
N_INITIAL_POINTS = 5
N_OPTIMISATION_STEPS = 5
# BATCH_SIZE
# NUM_RESTARTS
# RAW_SAMPLES
N_PARAMETERS = 12  # ecrh.piecewise_linear has 12 parameters

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Gather initial random points, launching processes in parallel
x_train = torch.rand((N_INITIAL_POINTS, N_PARAMETERS))
processes = []
for i in range(N_INITIAL_POINTS):
    config_directory = f"jetto/runs/initial_{i}"
    ecrh.create_config(
        "jetto/templates/spr45-v9",
        config_directory,
        lambda xrho: ecrh.piecewise_linear(xrho, x_train[i]),
    )
    processes.append(
        jetto_singularity.run("jetto/images/sim/v220922.sif", config_directory)
    )

# Wait for all processes to complete
for p in processes:
    p.wait()

y_train = torch.tensor(
    [
        objective.combined_cost_function(f"jetto/runs/initial_{i}")
        for i in range(N_INITIAL_POINTS)
    ],
    device=device,
    dtype=dtype,
)

# Initialise surrogate model
model = SingleTaskGP(x_train, y_train)

# Optimisation
for i in range(N_OPTIMISATION_STEPS):
    # Fit the model
    mll = MarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Select next input points
    candidates, _ = optimize_acqf(
        acq_function=qNoisyExpectedImprovement(
            model=model,
            X_baseline=x_train,
            # sampler=None,
        ),
        bounds=torch.tensor(
            [[0] * N_PARAMETERS, [1] * N_PARAMETERS], dtype=dtype, device=device
        ),
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Observe new values
    new_x = candidates.detach()
    config_directory = f"jetto/runs/bayesopt_{i}"
    ecrh.create_config(
        "jetto/templates/spr45-v9",
        config_directory,
        lambda xrho: ecrh.piecewise_linear(xrho, new_x),
    )
    process = jetto_singularity.run("jetto/images/sim/v220922.sif", config_directory)
    process.wait()
    new_y = torch.tensor(
        objective.combined_cost_function(config_directory),
        dtype=dtype,
        device=device,
    )

    # Store the new data
    x_train = torch.cat([x_train, new_x])
    y_train = torch.cat([y_train, new_y])

    # Update the model with the new data
    state_dict = model.state_dict()
    model = SingleTaskGP(x_train, y_train)
    model.load_state_dict(state_dict)
