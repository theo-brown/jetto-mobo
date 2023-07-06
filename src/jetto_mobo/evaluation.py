import pathlib
from typing import Iterable, Optional, Tuple

import netCDF4
import torch

from jetto_mobo.configuration import ObjectiveConfig, SimulationConfig
from jetto_mobo.simulation import simulate


####################
# HELPER FUNCTIONS #
####################
def evaluate(
    jetto_config_directories: Iterable[pathlib.Path],
    objective_config: ObjectiveConfig,
    simulation_config: SimulationConfig,
) -> torch.Tensor:
    simulation_results = simulate(simulation_config, jetto_config_directories)
    objective_values = torch.tensor(
        [
            get_objective_function_value(profiles, timetraces, objective_config)
            for profiles, timetraces in simulation_results
        ]
    )
    return objective_values


def get_objective_function_value(
    profiles: Optional[netCDF4.Dataset],
    timetraces: Optional[netCDF4.Dataset],
    objective_config: ObjectiveConfig,
) -> torch.Tensor:
    objective_values = torch.tensor(
        [
            component.function(profiles, timetraces, **component.kwargs)
            if profiles is not None and timetraces is not None
            else component.value_on_failure
            for component in objective_config.objective_components
        ]
    )

    objective_values = objective_values * torch.softmax(
        [component.weight for component in objective_config.objective_components]
    )

    if objective_config.scalar:
        return torch.mean(objective_values)
    else:
        return objective_values
