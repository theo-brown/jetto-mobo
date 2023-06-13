import argparse

import netCDF4
import numpy as np
import torch

from jetto_mobo.acquisition import (
    generate_initial_candidates,
    generate_trial_candidates,
)
from jetto_mobo.configuration import load_config
from jetto_mobo.evaluation import evaluate
from jetto_mobo.inputs.ecrh import (
    create_jetto_config,
    sum_of_gaussians,
    sum_of_gaussians_fixed_log_means,
)
from jetto_mobo.objectives.q import (
    dq_increasing,
    proximity_of_argmin_q_to_axis,
    proximity_of_q0_to_qmin,
    proximity_of_qmin_to_target,
    q_increasing,
    rho_of_q_value,
)
from jetto_mobo.surrogate import fit_surrogate_model


def evaluate_ecrh_q(ecrh_parameters, config):
    # Create JETTO config files
    jetto_config_directories = [
        config.output_directory / "0" / str(j)
        for j in range(config.acquisition.batch_size)
    ]
    for j in range(config.acquisition.batch_size):
        create_jetto_config(
            ecrh_parameters=dict(
                zip(config.ecrh.parameter_bounds.keys(), ecrh_parameters[j, :])
            ),
            jetto_config_directory=jetto_config_directories[j],
            simulation_config=config.simulation,
            ecrh_config=config.ecrh,
        )
    # Run JETTO simulations
    objective_values = evaluate(
        jetto_config_directories=jetto_config_directories,
        objective_config=config.objective,
        simulation_config=config.simulation,
    )
    return objective_values


if __name__ == "__main__":
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file (YAML)")
    args = parser.parse_args()
    config = load_config(args.config_file)

    # Generate initial data
    ecrh_parameters = generate_initial_candidates(
        bounds=config.ecrh.parameter_bounds.values(),
        acquisition_config=config.acquisition,
    )
    objective_values = evaluate_ecrh_q(ecrh_parameters, config)

    for i in range(1, n_optimisation_steps + 1):
        # Fit model
        model = fit_surrogate_model(
            ecrh_parameters,
            config.input_.parameter_bounds.values(),
            objective_values,
            config.surrogate,
        )

        # Generate new trial points
        candidates = generate_trial_candidates(
            ecrh_parameters,
            config.input_.parameter_bounds.values(),
            config.acquisition,
            model,
        )

        # Evaluate trial points
        new_objective_values = evaluate_ecrh_q(candidates, config)

        # Update data
        ecrh_parameters = torch.cat([ecrh_parameters, candidates], dim=0)
        objective_values = torch.cat([objective_values, new_objective_values], dim=0)
