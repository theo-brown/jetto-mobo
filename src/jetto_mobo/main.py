import argparse
import json
import logging
import os
from datetime import datetime

import h5py
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP  # Maybe use a different one?
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from jetto_mobo import ecrh, objective, utils

#################
# PROGRAM SETUP #
#################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default=datetime.now().strftime("%Y-%m-%d-%H%M%S"),
    help="Directory to store results in (default: ./YYYY-MM-DD-HHMMSS)",
)
parser.add_argument(
    "--n_bayesopt_steps",
    type=int,
    default=3,
    help="Number of BayesOpt steps (default: 3).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,
    help="Number of parallel JETTO runs (default: 5).",
)
parser.add_argument(
    "--n_restarts",
    type=int,
    default=10,
    help="Number of points for multistart optimisation (default: 10).",
)
parser.add_argument(
    "--raw_samples",
    type=int,
    default=512,
    help="Number of samples to draw from acquisition function (default: 512).",
)
parser.add_argument(
    "--n_sobol_samples",
    type=int,
    default=256,
    help="Passed to SobolQMCNormalSampler as `sample_shape` (default: 256).",
)
parser.add_argument(
    "--ecrh_function",
    type=str,
    choices=[
        "piecewise_linear",
        "sum_of_gaussians",
        "piecewise_linear_2",
        # "cubic_spline",
    ],
    default="piecewise_linear",
    help="ECRH function to use (default: 'piecewise_linear').",
)
parser.add_argument(
    "--ecrh_function_config",
    type=str,
    default="{}",
    help="Config JSON passed to ECRH function, used to set fixed (non-optimisable) ECRH parameters (default: '{}').",
)
parser.add_argument(
    "--cost_function",
    type=str,
    choices=["scalar", "vector"],
    default="scalar",
    help="Cost function to use (default: 'scalar').",
)
parser.add_argument(
    "--jetto_fail_cost",
    type=float,
    default=1e3,
    help="Value of cost function if JETTO fails (default: 1e3).",
)
parser.add_argument(
    "--jetto_timelimit",
    type=float,
    default=-1,
    help="Maximum number of seconds to wait for JETTO to complete; if < 0, run until complete (default: -1).",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume optimisation from `output_dir`.",
)
args = parser.parse_args()
output_file = f"{args.output_dir}/bayesopt.hdf5"

# Attributes to save/load from file
file_attrs = [
    "n_bayesopt_steps",
    "batch_size",
    "n_restarts",
    "raw_samples",
    "n_sobol_samples",
    "ecrh_function",
    "ecrh_function_config",
    "cost_function",
    "jetto_fail_cost",
    "jetto_timelimit",
]

if args.resume:
    # Load args from file
    with h5py.File(output_file, "r") as f:
        for arg in file_attrs:
            setattr(args, arg, f["/"].attrs[arg])
else:
    # Create directory
    os.makedirs(args.output_dir)
    # Save args to file
    with h5py.File(output_file, "w") as f:
        for arg in file_attrs:
            f["/"].attrs[args] = getattr(args, arg)

# Set ECRH function
ecrh_function_config = json.loads(args.ecrh_function_config)
if args.ecrh_function == "piecewise_linear":
    n_ecrh_parameters = 12
    ecrh_function = ecrh.piecewise_linear
elif args.ecrh_function == "piecewise_linear_2":
    n_ecrh_parameters = 12
    ecrh_function = ecrh.piecewise_linear_2
# elif ecrh_function == "cubic_spline":
#     n_nodes = ecrh_function_config.get("n", 5)
#     n_ecrh_parameters = n_nodes * 2
#     ecrh_function = ecrh.cubic_spline
elif args.ecrh_function == "sum_of_gaussians":
    n_gaussians = ecrh_function_config.get("n", 5)
    variance = ecrh_function_config.get("variance", 0.0025)
    n_ecrh_parameters = n_gaussians * 2

    def ecrh_function(x, params):
        return ecrh.sum_of_gaussians(
            x,
            params[:n_gaussians],  # means
            [variance] * n_gaussians,  # variances
            params[n_gaussians:],  # amplitudes
        )


# Set cost function
if args.cost_function == "scalar":
    cost_function = objective.scalar_cost_function
    cost_dimension = 1
elif args.cost_function == "vector":
    cost_function = objective.vector_cost_function
    cost_dimension = 8

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime} [{levelname}] {message}",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

##########################
# ACQUIRE INITIAL POINTS #
##########################
if args.resume:
    logger.info("Loading initialisation data from file...")

    # TODO: also load completed bayesopt runs
    ecrh_parameters = torch.tensor(
        utils.load_from_hdf5(output_file, "initialisation/ecrh_parameters"),
        device=device,
        dtype=dtype,
    )
    cost = torch.tensor(
        utils.load_from_hdf5(output_file, "initialisation/cost"),
        device=device,
        dtype=dtype,
    )
else:
    logger.info("Gathering initial data...")

    ecrh_parameters = (
        torch.rand([args.batch_size, n_ecrh_parameters], dtype=dtype, device=device),
    )
    utils.save_to_hdf5(output_file, "initialisation/ecrh_parameters", ecrh_parameters)
    converged_ecrh, converged_q, cost = ecrh.get_batch_cost(
        ecrh_parameters=ecrh_parameters.detach().cpu().numpy(),
        batch_directory=f"{args.output_dir}/initialisation",
        ecrh_function=ecrh_function,
        cost_function=cost_function,
    )
    # TODO: handle if none converged?
    utils.save_to_hdf5(output_file, "initialisation/converged_ecrh", converged_ecrh)
    utils.save_to_hdf5(output_file, "initialisation/converged_q", converged_q)
    utils.save_to_hdf5(output_file, "initialisation/cost", cost)
    cost = torch.tensor(cost, dtype=dtype, device=device)

##############################
# BAYESIAN OPTIMISATION LOOP #
##############################
logger.info("Starting BayesOpt...")
for i in range(args.n_bayesopt_steps):
    # If a run failed, it will produce a NaN cost.
    # To enable us to perform gradient-based optimisation,
    # we instead set the cost to a very large number.
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
    # BoTorch recommends using Sobol because it produces lower variance gradient estimates
    # with much fewer samples [https://botorch.org/docs/samplers]
    qNEI = qNoisyExpectedImprovement(
        model=model,
        X_baseline=ecrh_parameters,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([args.n_sobol_samples])),
    )

    # Select next ECRH parameters
    # Use multistart optimisation:
    # - Draw RAW_SAMPLES from the domain (uses the sampler defined in the acqf)
    # - Calculate value of acqf, a, at each RAW_SAMPLES point
    # - Weight the RAW_SAMPLES by w = exp(eta * (a - mean(a))/std(a))
    #   where eta is some temperature parameter
    # - Draw NUM_RESTARTS from RAW_SAMPLES according to w
    # - Perform local qNEI maximisation using scipy.minimize(method='L-BFGS-B')
    #   around each of NUM_RESTARTS points
    # - Take largest qNEI from all NUM_RESTARTS points
    # Does this jointly over the whole q-batch to reduce wallclock time
    # General idea: RAW_SAMPLES is cheaper than NUM_RESTARTS
    # because no local optimisation is performed.
    # Performing the pre-sampling and weighting ensures that your initial
    # points are already fairly good.
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
    new_ecrh_parameters_numpy = new_ecrh_parameters.detach().cpu().numpy()
    utils.save_to_hdf5(
        output_file,
        f"bayesopt/{i}/ecrh_parameters",
        new_ecrh_parameters_numpy,
    )

    # Observe cost values
    logger.info("Calculating cost of candidate points...")
    converged_ecrh, converged_q, new_cost = ecrh.get_batch_cost(
        ecrh_parameters=new_ecrh_parameters_numpy,
        batch_directory=f"{args.output_dir}/initialisation",
        ecrh_function=ecrh_function,
        cost_function=cost_function,
    )
    utils.save_to_hdf5(output_file, f"bayesopt/{i}/converged_ecrh", converged_ecrh)
    utils.save_to_hdf5(output_file, f"bayesopt/{i}/converged_q", converged_q)
    utils.save_to_hdf5(output_file, f"bayesopt/{i}/cost", new_cost)

    # Update
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    cost = torch.cat([cost, torch.tensor(new_cost, dtype=dtype, device=device)])
