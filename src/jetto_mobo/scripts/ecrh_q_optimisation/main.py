import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import jetto_tools
import numpy as np
import torch
from ecrh_inputs import marsden_piecewise_linear, marsden_piecewise_linear_bounds
from q_objectives import q_vector_objective

from jetto_mobo import acquisition, simulation, surrogate

parser = argparse.ArgumentParser()
parser.add_argument("--jetto_template", type=str, default="jetto/templates/spr45")
parser.add_argument("--jetto_image", type=str, default="jetto/images/sim.v220922.sif")
parser.add_argument("--jetto_timelimit", type=int, default=10400)
parser.add_argument("--jetto_fail_value", type=int, default=0)
parser.add_argument("--n_objectives", type=int, default=7)
parser.add_argument("--n_xrho_points", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--n_iterations", type=int, default=16)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--dtype", type=str, default="torch.float64")
parser.add_argument("--output_directory", type=str, default="data/ecrh_q_optimisation")
args = parser.parse_args()


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
        exfile["QECE"][0] = marsden_piecewise_linear(
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
            # Save converged profiles
            profiles = results.load_profiles()
            converged_ecrh.append(profiles["QECE"][-1])
            converged_q.append(profiles["Q"][-1])
            # Save objective value
            objective_values.append(q_vector_objective(results))
        else:
            converged_ecrh.append(np.full(args.n_xrho, args.jetto_fail_value))
            converged_q.append(np.full(args.n_xrho, args.jetto_fail_value))
            objective_values.append(np.full(args.n_objectives, args.jetto_fail_value))

    return (
        np.array(converged_ecrh),
        np.array(converged_q),
        np.array(objective_values),
    )


def write_to_file(
    root_label: str,
    ecrh_parameters_batch: Optional[np.ndarray] = None,
    preconverged_ecrh: Optional[np.ndarray] = None,
    converged_ecrh: Optional[np.ndarray] = None,
    converged_q: Optional[np.ndarray] = None,
    objective_values: Optional[np.ndarray] = None,
):
    """
    Utility function for writing data to hdf5 file.

    Creates a file in ``output_directory/results.h5`` if it does not exist, and writes the given data to the file under the given root label.
    """
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    with h5py.File(args.output_directory / "results.h5", "a") as f:
        if ecrh_parameters_batch is not None:
            f.create_dataset(
                f"{root_label}/ecrh_parameters", data=ecrh_parameters_batch
            )
        if preconverged_ecrh is not None:
            f.create_dataset(f"{root_label}/preconverged_ecrh", data=preconverged_ecrh)
        if converged_ecrh is not None:
            f.create_dataset(f"{root_label}/converged_ecrh", data=converged_ecrh)
        if converged_q is not None:
            f.create_dataset(f"{root_label}/converged_q", data=converged_q)
        if objective_values is not None:
            f.create_dataset(f"{root_label}/objective_values", data=objective_values)


# Bounds have to be tensor for fitting surrogate
parameter_bounds = torch.tensor(marsden_piecewise_linear_bounds)


# Generate initial data
print("Generating initial data...")
# Sobol sampling for initial candidates
ecrh_parameters = acquisition.generate_initial_candidates(
    bounds=parameter_bounds,
    n=args.batch_size,
    device=args.device,
    dtype=args.dtype,
)
# Save initial candidates to file
write_to_file(
    "initialisation",
    ecrh_parameters_batch=ecrh_parameters.detach().cpu().numpy(),
    preconverged_ecrh=np.array(
        [
            marsden_piecewise_linear(xrho=np.linspace(0, 1, args.n_xrho), parameters=p)
            for p in ecrh_parameters.detach().cpu().numpy()
        ]
    ),
)
# Evaluate initial candidates
(
    converged_ecrh,
    converged_q,
    objective_values,
) = evaluate(
    ecrh_parameters.detach().cpu().numpy(), args.output_directory / "0_initialisation"
)
# Save evaluated results to file
write_to_file(
    "initialisation",
    converged_ecrh=converged_ecrh,
    converged_q=converged_q,
    objective_values=objective_values,
)
# Train surrogate model
objective_values = torch.tensor(objective_values)
model = surrogate.fit_surrogate_model(
    X=ecrh_parameters,
    X_bounds=parameter_bounds,
    Y=objective_values,
    device=args.device,
    dtype=args.dtype,
    mode="joint",
)

for optimisation_step in range(1, args.n_iterations + 1):
    # Generate trial candidates
    print(f"Optimisation step {optimisation_step}")
    new_ecrh_parameters = acquisition.generate_trial_candidates(
        observed_inputs=ecrh_parameters,
        bounds=parameter_bounds,
        model=model,
        acquisition_function=acquisition.qNoisyExpectedHypervolumeImprovement,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        mode="sequential",
        acqf_kwargs={"ref_point": torch.zeros(objective_values.shape[1])},
    )
    write_to_file(
        f"optimisation_step_{optimisation_step}",
        ecrh_parameters_batch=new_ecrh_parameters.detach().cpu().numpy(),
        preconverged_ecrh=np.array(
            [
                marsden_piecewise_linear(
                    xrho=np.linspace(0, 1, args.n_xrho), parameters=p
                )
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
        new_ecrh_parameters.detach().cpu().numpy(),
        args.output_directory / str(optimisation_step),
    )
    write_to_file(
        f"optimisation_step_{optimisation_step}",
        converged_ecrh=converged_ecrh,
        converged_q=converged_q,
        objective_values=new_objective_values,
    )

    # Update surrogate model
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
