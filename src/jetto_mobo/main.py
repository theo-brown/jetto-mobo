import os
from typing import Iterable

import ecrh
import h5py
import objective
import plot
import torch
import utils
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP  # Maybe use a different one?
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

# Constants
N_PARAMETERS = 12  # ecrh.piecewise_linear has 12 parameters
N_INITIAL_EXPLORATION_POINTS = 5
N_BAYESOPT_STEPS = 1
BATCH_SIZE = 5  # Number of parallel JETTO to run
NUM_RESTARTS = 10  # Used in acqf optimisation
RAW_SAMPLES = 512  # Used in acqf optimisation
N_MC_SAMPLES = 256  # Used in acqf optimisation
OUTPUT_FILENAME = "jetto_mobo.hdf5"

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Gather initial data
if os.path.exists("jetto/runs/initial"):
    ecrh_parameters = utils.load_tensor(
        OUTPUT_FILENAME, "initialisation/ecrh_parameters"
    )
    cost = utils.load_tensor(OUTPUT_FILENAME, "initialisation/cost")
else:
    ecrh_parameters = torch.rand(
        [N_INITIAL_EXPLORATION_POINTS, N_PARAMETERS], dtype=dtype, device=device
    )
    utils.save_tensor(
        OUTPUT_FILENAME, "initialisation/ecrh_parameters", ecrh_parameters
    )
    cost = torch.tensor(
        ecrh.get_cost(
            ecrh_parameters,
            "jetto/runs/initial",
            ecrh.piecewise_linear,
            objective.scalar_cost_function,
        ),
        device=device,
        dtype=dtype,
    )
    utils.save_tensor(OUTPUT_FILENAME, "initialisation/cost", cost)

# Bayesian optimisation
for i in range(N_BAYESOPT_STEPS):
    # Initialise surrogate model
    # BoTorch performs maximisation, so need to use -cost
    model = SingleTaskGP(ecrh_parameters, -cost)

    # Fit the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Define the acquisition function
    # Choice of sampler:
    # Sobol is a quasirandom number generation scheme - generates low-discrepancy sequences
    # (low-discrepancy = on average, samples are evenly distributed to cover the space)
    # BoTorch recommends using Sobol because it produces lower variance gradient estimates with much fewer samples [https://botorch.org/docs/samplers]
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_MC_SAMPLES]))
    qNEI = qNoisyExpectedImprovement(
        model=model,
        X_baseline=ecrh_parameters,
        sampler=sampler,
    )

    bounds = torch.tensor(
        [[0] * N_PARAMETERS, [1] * N_PARAMETERS], dtype=dtype, device=device
    )

    # Select next ECRH parameters
    # Use multistart optimisation:
    # - Draw RAW_SAMPLES from the domain (uses the sampler defined in the acqf)
    # - Calculate value of acqf, a, at each RAW_SAMPLES point
    # - Weight the RAW_SAMPLES by w = exp(eta * (a - mean(a))/std(a)) where eta is some temperature parameter
    # - Draw NUM_RESTARTS from RAW_SAMPLES according to w
    # - Perform local qNEI maximisation using scipy.minimize(method='L-BFGS-B') around each of NUM_RESTARTS points
    # - Take largest qNEI from all NUM_RESTARTS points
    # Does this jointly over the whole q-batch to reduce wallclock time
    # General idea: RAW_SAMPLES is cheaper than NUM_RESTARTS because no local optimisation performed
    # Performing the pre-sampling and weighting ensures that your initial points are already fairly good
    # For large q, might need to swap to sequential rather than joint optimisation
    # For explanation, see
    # https://botorch.org/v/0.1.1/docs/optimization
    # https://github.com/pytorch/botorch/issues/366#issuecomment-581951153
    new_ecrh_parameters, _ = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=BATCH_SIZE,  # Number of final points to generate
        raw_samples=RAW_SAMPLES,  # Number of points to sample from acqf
        num_restarts=NUM_RESTARTS,  # Number of starting points for multistart optimisation
        options={
            "batch_limit": 5,  # Batch size for local optimisation
            "maxiter": 200,  # Max number of local optimisation iterations per batch
        },
    )
    new_ecrh_parameters = new_ecrh_parameters.detach()
    utils.save_tensor(
        OUTPUT_FILENAME,
        f"bayesopt/{i}/ecrh_parameters",
        new_ecrh_parameters,
    )
    f = plot.plot_piecewise_ecrh_batch(
        new_ecrh_parameters.cpu().numpy(), [-1] * len(new_ecrh_parameters)
    )
    f.show()

    # Observe cost values
    new_cost = torch.tensor(
        ecrh.get_cost(
            new_ecrh_parameters.cpu().numpy(),
            f"jetto/runs/bayesopt/{i}",
            ecrh.piecewise_linear,
            objective.scalar_cost_function,
        ),
        device=device,
        dtype=dtype,
    )
    utils.save_tensor(
        OUTPUT_FILENAME,
        f"bayesopt/{i}/cost",
        new_cost,
    )

    # Update
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    cost = torch.cat([cost, new_cost])
