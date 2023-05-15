import argparse
import json
import logging
import os
from datetime import datetime

import h5py
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from jetto_mobo import ecrh, genetic_algorithm, objective, utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default=datetime.now().strftime("%Y-%m-%d-%H%M%S"),
    help="Directory to store results in (default: ./YYYY-MM-DD-HHMMSS)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="data/sobo/ga_piecewise_linear.pt",
    help="Model checkpoint to load (default: data/sobo/ga_piecewise_linear.pt)",
)
parser.add_argument(
    "--n_points",
    type=int,
    default=30,
    help="Number of points to generate (default: 30)",
)
parser.add_argument(
    "--jetto_template",
    type=str,
    choices=["spr45", "spr54", "spr45-qlknn"],
    default="spr45",
    help="JETTO template to use (default: spr45).",
)
args = parser.parse_args()

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Load model
checkpoint = torch.load(args.checkpoint)
ecrh_parameter_bounds = checkpoint["ecrh_parameter_bounds"].to(device)
ecrh_parameters = checkpoint["ecrh_parameters"].to(device)
observed_values = checkpoint["observed_values"].to(device)

model = SingleTaskGP(normalize(ecrh_parameters, ecrh_parameter_bounds), observed_values)
model.load_state_dict(checkpoint["model"])

# Optimise acquisition function
acquisition_function = PosteriorMean(model)
candidates, predicted_values = optimize_acqf(
    acq_function=acquisition_function,
    bounds=ecrh_parameter_bounds,
    q=args.n_points,  # Number of final points to generate
    raw_samples=512,  # Number of points to sample from acqf
    num_restarts=10,  # Number of starting points for multistart optimisation
    options={
        "batch_limit": 5,  # Batch size for local optimisation
        "maxiter": 200,  # Max number of local optimisation iterations per batch
    },
)
new_ecrh_parameters = unnormalize(candidates.detach(), ecrh_parameter_bounds)

# Observe true values
converged_ecrh, converged_q, observed_values = ecrh.get_batch_value(
    ecrh_parameters=new_ecrh_parameters.cpu().numpy(),
    batch_directory=f"{args.output_dir}/candidates",
    ecrh_function=genetic_algorithm.piecewise_linear,
    value_function=genetic_algorithm.scalar_objective,
    timelimit=10400,
    jetto_template=f"jetto/templates/{args.jetto_template}",
)

# Save output
with h5py.File(os.path.join(args.output_dir, "output.hdf5"), "a") as f:
    f[f"ecrh_parameters"] = new_ecrh_parameters.cpu().numpy()
    f[f"converged_ecrh"] = converged_ecrh
    f[f"converged_q"] = converged_q
    f[f"predicted_value"] = predicted_values.cpu().numpy()
    f[f"observed_value"] = observed_values
    f["/"].attrs["training_data"] = checkpoint["training_data"]
    f["/"].attrs["n_points"] = args.n_points
    f["/"].attrs["jetto_template"] = args.jetto_template
