import os
from typing import Iterable

import ecrh
import h5py
import jetto_singularity
import jetto_tools
import objective
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP  # Maybe use a different one?
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

# Constants
N_INITIAL_POINTS = 3
N_PARAMETERS = 12  # ecrh.piecewise_linear has 12 parameters
BATCH_SIZE = 2
NUM_RESTARTS = 10
RAW_SAMPLES = 512
N_MC_SAMPLES = 256

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Load data
output_file = h5py.File("jetto_mobo.hdf5", "r+")
x_train = torch.tensor(
    output_file["initialisation/ecrh_parameters"][:], device=device, dtype=dtype
)
y_train = torch.tensor(
    output_file["initialisation/cost"][:], device=device, dtype=dtype
)

# Initialise surrogate model
model = SingleTaskGP(x_train, y_train)

# Fit the model
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Define the acquisition function
# Choice of sampler:
# Sobol is a quasirandom number generation scheme, ie generates low-discrepancy numbers
# (on average, samples are evenly distributed to cover the space)
# BoTorch recommends using Sobol because it produces lower variance gradient estimates with much fewer samples [https://botorch.org/docs/samplers]
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_MC_SAMPLES]))
qNEI = qNoisyExpectedImprovement(
    model=model,
    X_baseline=x_train,
    sampler=sampler,
)

bounds = torch.tensor(
    [[0] * N_PARAMETERS, [1] * N_PARAMETERS], dtype=dtype, device=device
)

# Select next input points
# Uses scipy.minimize(method='L-BFGS-B') to maximise the acqf
# L-BFGS is https://epubs.siam.org/doi/abs/10.1137/0916069
# Uses multistart:
# - draw NUM_RESTARTS random samples from feasible region
# - Perform local search around sampled points
# - Take best result from all sampled points
# Does this jointly over the whole q-batch to reduce wallclock time
# For large q, might need to swap to sequential rather than joint optimisation
# https://botorch.org/v/0.1.1/docs/optimization
candidates, _ = optimize_acqf(
    acq_function=qNEI,
    bounds=bounds,
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,  # This is to do with multistart optimisation
    raw_samples=RAW_SAMPLES,  # Number of samples for initialisation
    options={"batch_limit": 5, "maxiter": 200},  # Passed to scipy
)

print(candidates)
