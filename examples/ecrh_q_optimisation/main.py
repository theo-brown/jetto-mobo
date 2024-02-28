import argparse
import asyncio
import logging
import sys
from os import sched_getaffinity
from pathlib import Path
from typing import Tuple

import h5py
import jetto_tools
import numpy as np
import torch
from ecrh_inputs import (
    bezier_profile,
    constrained_bezier_profile,
    marsden_piecewise_linear,
    marsden_piecewise_linear_bounds,
)
from q_objectives import q_vector_objective
from utils import write_to_file

from jetto_mobo import acquisition, simulation, surrogate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

parser = argparse.ArgumentParser()
parser.add_argument("--jetto_template", type=str, default="../../jetto/templates/spr45")
parser.add_argument(
    "--jetto_image", type=str, default="../../jetto/images/sim.v220922.sif"
)
parser.add_argument("--jetto_timelimit", type=int, default=10400)
parser.add_argument("--jetto_fail_value", type=int, default=0)
parser.add_argument("--n_objectives", type=int, default=6)
parser.add_argument("--n_xrho_points", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--initial_batch_size", type=int, default=30)
parser.add_argument("--n_iterations", type=int, default=16)
parser.add_argument("--reference_values", type=float, nargs="+", default=None)
parser.add_argument(
    "--device",
    type=str,
    default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)
parser.add_argument("--dtype", type=torch.dtype, default=torch.float64)
parser.add_argument(
    "--output_dir", type=Path, default=Path("data/piecewise_linear_mobo")
)
parser.add_argument(
    "--parameterisation",
    choices=["piecewise_linear", "bezier"],
    default="piecewise_linear",
)
parser.add_argument("--n_parameters", type=int, default=12)
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)

if args.parameterisation == "piecewise_linear":
    ecrh_function = marsden_piecewise_linear
    parameter_bounds = torch.tensor(marsden_piecewise_linear_bounds)
elif args.parameterisation == "bezier":
    ecrh_function = constrained_bezier_profile
    parameter_bounds = torch.tensor([[0] * args.n_parameters, [1] * args.n_parameters])
elif args.parameterisation == "bezier2":
    ecrh_function = bezier_profile
    parameter_bounds = torch.tensor([[0] * args.n_parameters, [1] * args.n_parameters])


if args.reference_values is None:
    reference_values = torch.tensor([0.0] * args.n_objectives)
elif len(args.reference_values) == args.n_objectives:
    reference_values = torch.tensor(args.reference_values)
else:
    raise ValueError(
        f"Length of reference values ({len(args.reference_values)}) does not match number of objectives ({args.n_objectives})."
    )


def evaluate(
    ecrh_parameters_batch: np.ndarray,
    batch_directory: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a batch of ECRH parameters.

    Runs JETTO in parallel for each set of ECRH parameters, and returns the converged ECRH profiles and objective values.

    Parameters
    ----------
    ecrh_parameters_batch: np.ndarray
    batch_directory: Path

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    configs = {}

    for i, ecrh_parameters in enumerate(ecrh_parameters_batch):
        # Initialise config object
        config_directory = Path(f"{batch_directory}/candidate_{i}")
        config_object = simulation.create_config(
            template=args.jetto_template, directory=config_directory
        )

        # Set the ECRH function
        exfile = jetto_tools.binary.read_binary_file(config_object.exfile)
        exfile["QECE"][0] = ecrh_function(
            xrho=exfile["XRHO"][0], parameters=ecrh_parameters
        )
        jetto_tools.binary.write_binary_exfile(exfile, config_object.exfile)

        # Store in dict
        # Currently this is necessary as the JettoTools RunConfig does not store the directory path
        configs[config_object] = config_directory

    # Run asynchronously in parallel
    batch_output = asyncio.run(
        simulation.run_many(
            jetto_image=args.jetto_image,
            run_configs=configs,
            timelimit=args.jetto_timelimit,
        )
    )

    # Parse outputs
    converged_ecrh = []
    converged_q = []
    objective_values = []
    for results in batch_output:
        if results is not None:
            try:
                profiles = results.load_profiles()
            except:
                logger.warning(
                    "Failed to load profiles; using objective values for failed JETTO run."
                )
                converged_ecrh.append(
                    np.full(args.n_xrho_points, args.jetto_fail_value)
                )
                converged_q.append(np.full(args.n_xrho_points, args.jetto_fail_value))
                objective_values.append(
                    np.full(args.n_objectives, args.jetto_fail_value)
                )
            else:
                converged_ecrh.append(profiles["QECE"][-1])
                converged_q.append(profiles["Q"][-1])
                objective_values.append(q_vector_objective(results))
        else:
            converged_ecrh.append(np.full(args.n_xrho_points, args.jetto_fail_value))
            converged_q.append(np.full(args.n_xrho_points, args.jetto_fail_value))
            objective_values.append(np.full(args.n_objectives, args.jetto_fail_value))

    # Compress outputs
    for _, config_directory in configs.items():
        simulation.compress_jetto_dir(config_directory, delete=True)

    return (
        np.array(converged_ecrh),
        np.array(converged_q),
        np.array(objective_values),
    )


if args.resume:
    # TODO: Update to permit batched initialisation data
    logger.info("Resuming from file...")
    with h5py.File(args.output_dir / "results.h5", "r") as f:
        # Read initialisation data
        ecrh_parameters = [torch.tensor(f["initialisation/ecrh_parameters"][:])]
        objective_values = [torch.tensor(f["initialisation/objective_values"][:])]
        # Read any additional optimisation steps
        completed_optimisation_steps = len(f.keys()) - 1
        if completed_optimisation_steps > 0:
            for i in range(1, completed_optimisation_steps + 1):
                logger.info(f"Loading data from optimisation step {i}...")
                ecrh_parameters.append(
                    torch.tensor(f[f"optimisation_step_{i}/ecrh_parameters"][:])
                )
                objective_values.append(
                    torch.tensor(f[f"optimisation_step_{i}/objective_values"][:])
                )
        # Concatenate
        ecrh_parameters = torch.cat(ecrh_parameters).to(
            device=args.device, dtype=args.dtype
        )
        objective_values = torch.cat(objective_values).to(
            device=args.device, dtype=args.dtype
        )
else:
    # Generate initial data
    logger.info("Generating initial data...")

    # Sobol sampling for initial candidates
    ecrh_parameters = acquisition.generate_initial_candidates(
        bounds=parameter_bounds,
        n=args.initial_batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    # Get number of cores
    n_cores = len(sched_getaffinity(0))
    # Evaluate initial candidates in batches of size n_cores
    for batch_index, candidate_index in enumerate(
        range(0, args.initial_batch_size, n_cores)
    ):
        logger.info(f"Evaluating initial candidates (batch {batch_index})...")
        ecrh_parameter_batch = (
            ecrh_parameters[candidate_index : candidate_index + n_cores]
            .detach()
            .cpu()
            .numpy()
        )
        # Save initial candidates to file
        write_to_file(
            output_file=args.output_dir / "results.h5",
            root_label=f"initialisation/batch_{batch_index}",
            ecrh_parameters_batch=ecrh_parameter_batch,
            preconverged_ecrh=np.array(
                [
                    ecrh_function(
                        xrho=np.linspace(0, 1, args.n_xrho_points), parameters=p
                    )
                    for p in ecrh_parameter_batch
                ]
            ),
        )
        (
            converged_ecrh,
            converged_q,
            objective_values,
        ) = evaluate(
            ecrh_parameters_batch=ecrh_parameter_batch,
            batch_directory=args.output_dir / f"0_initialisation_batch_{batch_index}",
        )
        # Save evaluated results to file
        write_to_file(
            output_file=args.output_dir / "results.h5",
            root_label=f"initialisation/batch_{batch_index}",
            converged_ecrh=converged_ecrh,
            converged_q=converged_q,
            objective_values=objective_values,
        )

    # Rearrange the file structure
    with h5py.File(args.output_dir / "results.h5", "a") as h5file:
        # Create new datasets
        h5file.create_dataset(
            "initialisation/converged_ecrh",
            (args.initial_batch_size, args.n_xrho_points),
        )
        h5file.create_dataset(
            "initialisation/converged_q", (args.initial_batch_size, args.n_xrho_points)
        )
        h5file.create_dataset(
            "initialisation/ecrh_parameters",
            (args.initial_batch_size, args.n_parameters),
        )
        h5file.create_dataset(
            "initialisation/objective_values",
            (args.initial_batch_size, args.n_objectives),
        )
        h5file.create_dataset(
            "initialisation/preconverged_ecrh",
            (args.initial_batch_size, args.n_xrho_points),
        )

        # Copy data across
        for batch_index in range(args.initial_batch_size // n_cores):
            lower_index = batch_index * n_cores
            upper_index = lower_index + n_cores
            h5file["initialisation/converged_ecrh"][lower_index:upper_index] = h5file[
                f"initialisation/batch_{batch_index}/converged_ecrh"
            ]
            h5file["initialisation/converged_q"][lower_index:upper_index] = h5file[
                f"initialisation/batch_{batch_index}/converged_q"
            ]
            h5file["initialisation/ecrh_parameters"][lower_index:upper_index] = h5file[
                f"initialisation/batch_{batch_index}/ecrh_parameters"
            ]
            h5file["initialisation/objective_values"][lower_index:upper_index] = h5file[
                f"initialisation/batch_{batch_index}/objective_values"
            ]
            h5file["initialisation/preconverged_ecrh"][
                lower_index:upper_index
            ] = h5file[f"initialisation/batch_{batch_index}/preconverged_ecrh"]

            del h5file[f"initialisation/batch_{batch_index}"]

        # Load back into a tensor
        ecrh_parameters = torch.tensor(
            h5file["initialisation/ecrh_parameters"][:],
            device=args.device,
            dtype=args.dtype,
        )
        objective_values = torch.tensor(
            h5file["initialisation/objective_values"][:],
            device=args.device,
            dtype=args.dtype,
        )

    # Initialise optimisation step
    completed_optimisation_steps = 0

# Train surrogate model
logger.info("Fitting surrogate model...")
objective_values = torch.tensor(objective_values)
model = surrogate.fit_surrogate_model(
    X=ecrh_parameters,
    X_bounds=parameter_bounds,
    Y=objective_values,
    device=args.device,
    dtype=args.dtype,
    mode="joint",
)

for optimisation_step in range(
    completed_optimisation_steps + 1,
    completed_optimisation_steps + 1 + args.n_iterations,
):
    # Generate trial candidates
    logger.info(f"Optimisation step {optimisation_step}")
    logger.info("Generating trial candidates...")
    new_ecrh_parameters = acquisition.generate_trial_candidates(
        observed_inputs=ecrh_parameters,
        bounds=parameter_bounds,
        model=model,
        acquisition_function=acquisition.qNoisyExpectedHypervolumeImprovement,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        mode="sequential" if args.batch_size > 5 else "joint",
        acqf_kwargs={
            "ref_point": reference_values,
            "alpha": args.alpha,
            "prune_baseline": True,
        },
    )
    write_to_file(
        output_file=args.output_dir / "results.h5",
        root_label=f"optimisation_step_{optimisation_step}",
        ecrh_parameters_batch=new_ecrh_parameters.detach().cpu().numpy(),
        preconverged_ecrh=np.array(
            [
                ecrh_function(xrho=np.linspace(0, 1, args.n_xrho_points), parameters=p)
                for p in new_ecrh_parameters.detach().cpu().numpy()
            ]
        ),
    )

    # Evaluate candidates
    (
        converged_ecrh,
        converged_q,
        new_objective_values,
    ) = evaluate(
        ecrh_parameters_batch=new_ecrh_parameters.detach().cpu().numpy(),
        batch_directory=args.output_dir / str(optimisation_step),
    )
    write_to_file(
        output_file=args.output_dir / "results.h5",
        root_label=f"optimisation_step_{optimisation_step}",
        converged_ecrh=converged_ecrh,
        converged_q=converged_q,
        objective_values=new_objective_values,
    )

    # Update surrogate model
    logger.info("Fitting surrogate model...")
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    objective_values = torch.cat(
        [objective_values, torch.tensor(new_objective_values)]
    )  # Have to convert new_objective_values to tensor, because it is a np.ndarray output from reading JettoResults
    model = surrogate.fit_surrogate_model(
        X=ecrh_parameters,
        X_bounds=parameter_bounds,
        Y=objective_values,
        device=args.device,
        dtype=args.dtype,
        mode="joint",
    )
