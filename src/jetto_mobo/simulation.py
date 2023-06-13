import asyncio
import pathlib
from typing import Iterable, Optional, Tuple

import netCDF4

from jetto_mobo.configuration import SimulationConfig
from jetto_mobo.jetto_container import run_many


def simulate(
    simulation_config: SimulationConfig,
    jetto_config_directories: Iterable[pathlib.Path],
) -> Iterable[Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]]:
    return run_many(
        jetto_image=simulation_config.jetto_image,
        config_directories=jetto_config_directories,
        timelimit=simulation_config.timelimit,
    )
