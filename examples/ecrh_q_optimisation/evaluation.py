import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import jetto_tools
import numpy as np

from jetto_mobo import simulation

from .ecrh_inputs import marsden_piecewise_linear
from .q_objectives import q_vector_objective


def evaluate(
    ecrh_parameters_batch: np.ndarray,
    batch_directory: Path,
    jetto_template: Path,
    jetto_image: Path,
    jetto_timelimit: int,
    jetto_fail_value: float,
    n_xrho: int,
    n_objectives: int,
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
            template=jetto_template, directory=config_directory
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
            jetto_image=jetto_image,
            run_configs=configs,
            timelimit=jetto_timelimit,
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
            converged_ecrh.append(np.full(n_xrho, jetto_fail_value))
            converged_q.append(np.full(n_xrho, jetto_fail_value))
            objective_values.append(np.full(n_objectives, jetto_fail_value))

    return (
        np.array(converged_ecrh),
        np.array(converged_q),
        np.array(objective_values),
    )


def write_to_file(
    output_file: Path,
    root_label: str,
    ecrh_parameters_batch: Optional[np.ndarray] = None,
    preconverged_ecrh: Optional[np.ndarray] = None,
    converged_ecrh: Optional[np.ndarray] = None,
    converged_q: Optional[np.ndarray] = None,
    objective_values: Optional[np.ndarray] = None,
):
    """
    Utility function for writing data to hdf5 file.

    Creates ``output_file`` if it does not exist, and writes the given data to the file under the given root label.
    """
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)
    with h5py.File(output_file, "a") as f:
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
