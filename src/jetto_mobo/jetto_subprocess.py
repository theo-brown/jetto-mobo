import asyncio
import logging
from typing import Mapping, Optional, Tuple, Union

import netCDF4
from jetto_tools.results import JettoResults


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
        Maximum number of seconds to wait for JETTO to complete. If `None`, run until complete.

    Returns
    -------
    Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]
        If JETTO converged in the timelimit (return code 0), returns `(profiles, timetraces)`; else returns `(None, None)`.
    """
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
    )
    logging.info(f"Starting JETTO in {config_directory} with PID {process.pid}.")
    try:
        await asyncio.wait_for(process.communicate(), timelimit)
    except TimeoutError:
        logging.info(
            f"JETTO (output={config_directory}, PID={process.pid}) cancelled (time limit of {timelimit}s exceeded)."
        )
    logging.info(
        f"JETTO (output={config_directory}, PID={process.pid}) terminated with return code {process.returncode}."
    )

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
            run(jetto_image, config_directory, run_name, timelimit)
            for config_directory in config_directories
        ]
    )
