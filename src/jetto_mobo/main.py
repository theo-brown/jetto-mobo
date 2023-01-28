import argparse
import logging
import os
from datetime import datetime
from typing import Optional, Union

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP  # Maybe use a different one?
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from jetto_mobo import ecrh, objective, utils

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default="YYYY-MM-DD-hhmmss",
    help="Directory to store results in; a directory with the specified path will be created if it does not already exist.",
)
parser.add_argument(
    "--n_bayesopt_steps", type=int, default=6, help="Number of BayesOpt steps."
)
parser.add_argument(
    "--batch_size", type=int, default=5, help="Number of parallel JETTO runs."
)
parser.add_argument(
    "--n_restarts",
    type=int,
    default=10,
    help="Number of points for multistart optimisation.",
)
parser.add_argument(
    "--raw_samples",
    type=int,
    default=512,
    help="Number of samples to draw from acquisition function.",
)
parser.add_argument(
    "--n_sobol_samples",
    type=int,
    default=256,
    help="Passed to SobolQMCNormalSampler as `sample_shape`.",
)
parser.add_argument(
    "--ecrh_function",
    type=str,
    choices=["piecewise_linear"],
    default="piecewise_linear",
    help="ECRH function to use.",
)
parser.add_argument(
    "--cost_function",
    type=str,
    choices=["scalar"],
    default="scalar",
    help="Cost function to use.",
)
parser.add_argument(
    "--jetto_fail_cost",
    type=float,
    default=1e3,
    help="Value of cost function if JETTO fails.",
)
parser.add_argument(
    "--jetto_timelimit",
    type=float,
    default=-1,
    help="Maximum number of seconds to wait for JETTO to complete. If < 0, run until complete.",
)
args = parser.parse_args()

if args.output_dir == "YYYY-MM-DD-hhmmss":
    output_dir = datetime.now().strftime("%Y-%m-%d-%H%M%S")
else:
    output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = f"{output_dir}/output.hdf5"

if args.ecrh_function == "piecewise_linear":
    ecrh_function = ecrh.piecewise_linear
    n_ecrh_parameters = 12

if args.cost_function == "scalar":
    cost_function = objective.scalar_cost_function
    cost_dimension = 1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Gather initial data
# TODO: handle if none converged
logger.info("Gathering initial data...")
ecrh_parameters = torch.rand(
    [args.batch_size, n_ecrh_parameters], dtype=dtype, device=device
)
utils.save_tensor(output_filename, "initialisation/ecrh_parameters", ecrh_parameters)
cost = torch.tensor(
    ecrh.get_cost(
        ecrh_parameters=ecrh_parameters,
        directory=f"{output_dir}/jetto/initial",
        ecrh_function=ecrh_function,
        cost_function=cost_function,
        timelimit=args.jetto_timelimit,
    ),
    device=device,
    dtype=dtype,
)
utils.save_tensor(output_filename, "initialisation/cost", cost)
# ecrh_parameters = utils.load_tensor(OUTPUT_FILENAME, "initialisation/ecrh_parameters", device=device, dtype=dtype)
# cost = utils.load_tensor(OUTPUT_FILENAME, "initialisation/cost", device=device, dtype=dtype)

# Bayesian optimisation
logger.info("Starting BayesOpt...")
for i in range(args.n_bayesopt_steps):
    # If a run failed, it will produce a NaN cost.
    # To enable us to perform gradient-based optimisation, we instead set the cost to a very large number.
    cost[cost.isnan()] = args.jetto_fail_cost

    # Initialise surrogate model
    # BoTorch performs maximisation, so need to use -cost
    logger.info("Fitting surrogate model to observed costs...")
    model = SingleTaskGP(ecrh_parameters, -cost)
    # Fit the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Define the acquisition function
    # Choice of sampler:
    # Sobol is a quasirandom number generation scheme - generates low-discrepancy sequences
    # (low-discrepancy = on average, samples are evenly distributed to cover the space)
    # BoTorch recommends using Sobol because it produces lower variance gradient estimates with much fewer samples [https://botorch.org/docs/samplers]
    qNEI = qNoisyExpectedImprovement(
        model=model,
        X_baseline=ecrh_parameters,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([args.n_sobol_samples])),
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
    logger.info("Selecting candidates using acquisition function...")
    new_ecrh_parameters, _ = optimize_acqf(
        acq_function=qNEI,
        bounds=torch.tensor(
            [[0] * n_ecrh_parameters, [1] * n_ecrh_parameters],
            dtype=dtype,
            device=device,
        ),
        q=args.batch_size,  # Number of final points to generate
        raw_samples=args.raw_samples,  # Number of points to sample from acqf
        num_restarts=args.n_restarts,  # Number of starting points for multistart optimisation
        options={
            "batch_limit": 5,  # Batch size for local optimisation
            "maxiter": 200,  # Max number of local optimisation iterations per batch
        },
    )
    new_ecrh_parameters = new_ecrh_parameters.detach()
    utils.save_tensor(
        output_filename,
        f"bayesopt/{i}/ecrh_parameters",
        new_ecrh_parameters,
    )

    # Observe cost values
    logger.info("Calculating cost of candidate points...")
    new_cost = torch.tensor(
        ecrh.get_cost(
            ecrh_parameters=new_ecrh_parameters.cpu().numpy(),
            directory=f"{output_dir}/jetto/bayesopt/{i}",
            ecrh_function=ecrh_function,
            cost_function=cost_function,
            timelimit=args.jetto_timelimit,
        ),
        device=device,
        dtype=dtype,
    )
    utils.save_tensor(
        output_filename,
        f"bayesopt/{i}/cost",
        new_cost,
    )

    # Update
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    cost = torch.cat([cost, new_cost])
