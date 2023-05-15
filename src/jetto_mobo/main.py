import argparse
import json
import logging
import os
from datetime import datetime

import h5py
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from jetto_mobo import ecrh, genetic_algorithm, objective, utils

# TODO: Do we need to standardize the outputs?
# from botorch.models.transforms.outcome import Standardize


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
    help="Number of BayesOpt steps to run (default: 3).",
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
    "--acqf_optimisation_mode",
    type=str,
    choices=["sequential", "joint"],
    default="joint",
    help="Whether to use sequential or joint optimisation for generating candidates from the acquisition function.",
)
parser.add_argument(
    "--ecrh_function",
    type=str,
    choices=[
        "ga_piecewise_linear",
        "piecewise_linear",
        "sum_of_gaussians",
        "sum_of_gaussians_fixed_means",
    ],
    default="ga_piecewise_linear",
    help="ECRH function to use (default: 'ga_piecewise_linear').",
)
parser.add_argument(
    "--ecrh_function_config",
    type=str,
    default="{}",
    help="Config JSON passed to ECRH function, used to set fixed (non-optimisable) ECRH parameters (default: '{}').",
)
parser.add_argument(
    "--value_function",
    type=str,
    choices=["ga_scalar", "scalar", "vector", "ga_vector"],
    default="ga_scalar",
    help="Value function to use (default: 'ga_scalar').",
)
parser.add_argument(
    "--jetto_fail_value",
    type=float,
    default=0,
    help="Value of objective function if JETTO fails (default: 0).",
)
parser.add_argument(
    "--reference_values",
    type=float,
    nargs="?",
    default=None,
    help="Reference value of objective function for each vector element.",
)
parser.add_argument(
    "--jetto_timelimit",
    type=float,
    default=10400,
    help="Maximum number of seconds to wait for JETTO to complete; if < 0, run until complete (default: 10400).",
)
parser.add_argument(
    "--jetto_template",
    type=str,
    choices=["spr45", "spr54", "spr45-qlknn"],
    default="spr45",
    help="JETTO template to use (default: spr45).",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume optimisation from `output_dir`.",
)
args = parser.parse_args()

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Attributes to save/load from file
output_file = f"{args.output_dir}/bayesopt.hdf5"
file_attrs = [
    "batch_size",
    "n_restarts",
    "raw_samples",
    "n_sobol_samples",
    "ecrh_function",
    "ecrh_function_config",
    "value_function",
    "jetto_fail_value",
    "jetto_timelimit",
]

if args.resume:
    # Load args from file
    with h5py.File(output_file, "r") as f:
        for arg in file_attrs:
            setattr(args, arg, f["/"].attrs[arg])
        n_completed_bayesopt_steps = f["/"].attrs["n_completed_bayesopt_steps"]
else:
    # Create directory
    os.makedirs(args.output_dir)
    # Save args to file
    with h5py.File(output_file, "w") as f:
        for arg in file_attrs:
            f["/"].attrs[arg] = getattr(args, arg)
        n_completed_bayesopt_steps = 0
        f["/"].attrs["n_completed_bayesopt_steps"] = n_completed_bayesopt_steps

# Set ECRH function
ecrh_function_config = json.loads(args.ecrh_function_config)
if args.ecrh_function == "ga_piecewise_linear":
    n_ecrh_parameters = 12
    ecrh_function = genetic_algorithm.piecewise_linear
    ecrh_parameter_bounds = torch.tensor(
        [
            [0, 0.05, 0.01, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 0.09, 1, 1, 0.29, 0.9, 0.9, 1, 0.75, 0.9, 0.45],
        ],
        dtype=dtype,
        device=device,
    )
elif args.ecrh_function == "piecewise_linear":
    n_ecrh_parameters = 12
    ecrh_function = ecrh.piecewise_linear
    ecrh_parameter_bounds = torch.tensor(
        [[0] * n_ecrh_parameters, [1] * n_ecrh_parameters],
        dtype=dtype,
        device=device,
    )
elif args.ecrh_function == "sum_of_gaussians":
    n_gaussians = ecrh_function_config.get("n", 5)
    n_ecrh_parameters = 2 * n_gaussians
    variance = ecrh_function_config.get("variance", 0.0025)
    ecrh_function = lambda x, p: ecrh.sum_of_gaussians(x, p, variance)
    ecrh_parameter_bounds = torch.tensor(
        [[0] * n_ecrh_parameters, [1] * n_ecrh_parameters],
        dtype=dtype,
        device=device,
    )
elif args.ecrh_function == "sum_of_gaussians_fixed_means":
    n_gaussians = ecrh_function_config.get("n", 5)
    n_ecrh_parameters = 2 * n_gaussians + 1
    ecrh_function = lambda x, p: ecrh.sum_of_gaussians_fixed_means(
        x,
        xmax=p[0],
        variances=p[1 : n_gaussians + 1],
        amplitudes=p[n_gaussians + 1 :],
        min_variance=5e-4,
        max_variance=1e-2,
        spacing="log",
        variance_scaling="log",
    )
    ecrh_parameter_bounds = torch.tensor(
        [[0] * n_ecrh_parameters, [1] * n_ecrh_parameters],
        dtype=dtype,
        device=device,
    )

# Set objective/value function
if args.value_function == "ga_scalar":
    value_function = genetic_algorithm.scalar_objective
    n_objectives = 1
elif args.value_function == "scalar":
    value_function = objective.scalar_objective
    n_objectives = 1
elif args.value_function == "ga_vector":
    value_function = genetic_algorithm.vector_objective
    n_objectives = 7
    if args.reference_values is None:
        reference_values = np.zeros(n_objectives)
    elif len(args.reference_values) != n_objectives:
        raise ValueError(
            f"Received {len(args.reference_values)} values for --reference_values, expected {n_objectives}."
        )
    else:
        reference_values = args.reference_values
elif args.value_function == "vector":
    value_function = objective.vector_objective
    n_objectives = 7
    if args.reference_values is None:
        reference_values = np.zeros(n_objectives)
    elif len(args.reference_values) != n_objectives:
        raise ValueError(
            f"Received {len(args.reference_values)} values for --reference_values, expected {n_objectives}."
        )
    else:
        reference_values = args.reference_values

# Set up logging
logger = utils.get_logger("jetto-mobo", level=logging.INFO)
logger.info(f"Started at {datetime.now().strftime('%H:%M:%S')}.")
logger.info(
    "Running with args:\n" + "\n".join(f"{k}={v}" for k, v in vars(args).items())
)

##########################
# ACQUIRE INITIAL POINTS #
##########################
if args.resume:
    logger.info("Loading initialisation data from file...")

    with h5py.File(output_file, "r") as f:
        ecrh_parameters = torch.tensor(
            np.concatenate(
                (
                    f["initialisation/ecrh_parameters"][:],
                    *[
                        f[f"bayesopt/{i+1}/ecrh_parameters"][:]
                        for i in range(n_completed_bayesopt_steps)
                    ],
                )
            ),
            device=device,
            dtype=dtype,
        )
        value = torch.tensor(
            np.concatenate(
                (
                    f["initialisation/value"][:],
                    *[
                        f[f"bayesopt/{i+1}/value"][:]
                        for i in range(n_completed_bayesopt_steps)
                    ],
                )
            ),
            device=device,
            dtype=dtype,
        )
else:
    logger.info("Gathering initial data...")
    ecrh_parameters = draw_sobol_samples(
        ecrh_parameter_bounds, n=1, q=args.batch_size
    ).squeeze()
    ecrh_parameters_numpy = ecrh_parameters.detach().cpu().numpy()
    converged_ecrh, converged_q, value = ecrh.get_batch_value(
        ecrh_parameters=ecrh_parameters_numpy,
        batch_directory=f"{args.output_dir}/initialisation",
        ecrh_function=ecrh_function,
        value_function=value_function,
        timelimit=args.jetto_timelimit,
        jetto_template=f"jetto/templates/{args.jetto_template}",
    )
    if np.all(np.isnan(value)):
        # TODO: retry rather than exit
        raise RuntimeError(
            "Failed to generate initial values; all initial points failed to converge."
        )

    with h5py.File(output_file, "a") as f:
        f["initialisation/ecrh_parameters"] = ecrh_parameters_numpy
        f["initialisation/converged_ecrh"] = converged_ecrh
        f["initialisation/converged_q"] = converged_q
        f["initialisation/value"] = value
    value = torch.tensor(value, dtype=dtype, device=device)

##############################
# BAYESIAN OPTIMISATION LOOP #
##############################
for i in np.arange(
    n_completed_bayesopt_steps + 1,
    args.n_bayesopt_steps + n_completed_bayesopt_steps + 1,
):
    logger.info(f"BayesOpt iteration {i}:")

    # If a run failed, it will produce a NaN value.
    # To enable us to perform gradient-based optimisation,
    # we set the corresponding value to a very small number
    # Note, we could instead drop these runs completely
    value[value.isnan()] = args.jetto_fail_value

    # Initialise surrogate model
    logger.info("Fitting surrogate model to observed values...")
    if args.value_function == "vector":
        # TODO: check whether we should be using a list of single task GPs or a multi-task GP
        # ModelListGP is a collection of SingleTaskGP that model each element of the vector objective independently
        model = ModelListGP(
            *[
                SingleTaskGP(
                    normalize(ecrh_parameters, ecrh_parameter_bounds),
                    value[:, i].unsqueeze(1),
                )
                for i in range(n_objectives)
            ]
        )
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        model = SingleTaskGP(normalize(ecrh_parameters, ecrh_parameter_bounds), value)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Define the acquisition function
    # Choice of sampler:
    # Sobol is a quasirandom number generation scheme - generates low-discrepancy sequences
    # (low-discrepancy = on average, samples are evenly distributed to cover the space)
    # BoTorch recommends using Sobol because it produces lower variance gradient estimates
    # with much fewer samples [https://botorch.org/docs/samplers]
    if args.value_function == "vector":
        acquisition_function = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference_values,
            X_baseline=normalize(ecrh_parameters, ecrh_parameter_bounds),
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([args.n_sobol_samples])
            ),
        )
    else:
        acquisition_function = qNoisyExpectedImprovement(
            model=model,
            X_baseline=normalize(ecrh_parameters, ecrh_parameter_bounds),
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([args.n_sobol_samples])
            ),
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
    candidates, _ = optimize_acqf(
        acq_function=acquisition_function,
        bounds=ecrh_parameter_bounds,
        q=args.batch_size,  # Number of final points to generate
        raw_samples=args.raw_samples,  # Number of points to sample from acqf
        num_restarts=args.n_restarts,  # Number of starting points for multistart optimisation
        sequential=True if args.acqf_optimisation_mode == "sequential" else False,
        options={
            # TODO Add to args
            "batch_limit": 5,  # Batch size for local optimisation
            "maxiter": 200,  # Max number of local optimisation iterations per batch
        },
    )
    new_ecrh_parameters = unnormalize(candidates.detach(), ecrh_parameter_bounds)

    # TODO Get templates from previous points

    # Observe values
    logger.info("Calculating value of candidate points...")
    # TODO: handle if new_value are all Nones
    converged_ecrh, converged_q, new_value = ecrh.get_batch_value(
        ecrh_parameters=new_ecrh_parameters.cpu().numpy(),
        batch_directory=f"{args.output_dir}/bayesopt/{i}",
        ecrh_function=ecrh_function,
        value_function=value_function,
        timelimit=args.jetto_timelimit,
        jetto_template=f"jetto/templates/{args.jetto_template}",
    )

    # Update logged data
    # TODO tidy resume if computing value failed
    n_completed_bayesopt_steps += 1
    with h5py.File(output_file, "a") as f:
        f[f"bayesopt/{i}/ecrh_parameters"] = new_ecrh_parameters.cpu().numpy()
        f[f"bayesopt/{i}/converged_ecrh"] = converged_ecrh
        f[f"bayesopt/{i}/converged_q"] = converged_q
        f[f"bayesopt/{i}/value"] = new_value
        f["/"].attrs["n_completed_bayesopt_steps"] = n_completed_bayesopt_steps

    # Update tensors for next BayesOpt
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    value = torch.cat([value, torch.tensor(new_value, dtype=dtype, device=device)])
