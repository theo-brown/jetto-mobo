import asyncio
import logging
from typing import Iterable, Optional, Tuple, Union

import netCDF4
from jetto_tools.results import JettoResults

from jetto_mobo import utils

# Set up logging
logger = logging.getLogger("jetto-subprocess")
handler = logging.StreamHandler()
handler.setFormatter(
    utils.ElapsedTimeFormatter("%(name)s:t+%(elapsed_time)s:%(levelname)s %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


async def run(
    jetto_image: str,
    config_directory: str,
    timelimit: Optional[Union[int, float]] = None,
) -> Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]:
    """Run a JETTO Singularity container with the given config in an async subprocess.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config_directory : str
        Path to a directory containing a JETTO configuration; output files will overwrite files in this directory.

    timelimit : Optional[Union[int, float]], default=None
        Maximum number of seconds to wait for JETTO to complete. If `None` or < 0, run until complete.

    Returns
    -------
    Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]
        If JETTO converged in the timelimit (return code 0), returns `(profiles, timetraces)`; else returns `(None, None)`.
    """
    if timelimit < 0:
        timelimit = None
        
    # Run name is only used internally in this container, so it doesn't matter what it's called
    run_name = "run"
    process = await asyncio.create_subprocess_exec(
        "singularity",
        "exec",
        "--cleanenv",  # Run in a clean environment (no env variables etc)
        "--bind",
        "/tmp",
        "--bind",
        f"{config_directory}:/jetto/runs/{run_name}",  # Bind the output directory to the container's jetto run directory
        jetto_image,  # Container image
        # Command to execute in container:
        "rjettov",
        "-x64",
        "-S",
        "-p0",
        "-n1",
        run_name,
        "build",
        "docker",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logger.info(f"Starting JETTO in {config_directory}.")
    try:
        await asyncio.wait_for(process.communicate(), timelimit)
        logger.info(
            f"JETTO in {config_directory} terminated with return code {process.returncode}."
        )
    except asyncio.exceptions.TimeoutError:
        logger.info(
            f"JETTO in {config_directory} terminated: time limit ({timelimit}s) exceeded)."
        )
        process.kill()

    if process.returncode == 0:
        results = JettoResults(path=config_directory)
        profiles = results.load_profiles()
        timetraces = results.load_timetraces()
        return profiles, timetraces
    else:
        return None, None


async def run_many(
    jetto_image: str,
    config_directories: Iterable[str],
    timelimit: Optional[Union[int, float]] = None,
) -> Iterable[Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]]:
    """Asynchronously run multiple JETTO runs, using `jetto_subprocess.run()`.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config : Iterable[str, str]
        List of config directories.
        Values must be a path to a directory containing a JETTO configuration to load for each process;
        output files will overwrite files infiles this directory.
    timelimit : Optional[Union[int, float]], default = None
        Maximum number of seconds to wait for JETTO to complete. If `None`, run until complete.

    Returns
    -------
    Iterable[Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]]
        List of (`profiles`, `timetraces`) for each JETTO subprocess.
    """
    return await asyncio.gather(
        *[
            run(jetto_image, config_directory, timelimit)
            for config_directory in config_directories
        ]
    )
