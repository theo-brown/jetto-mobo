import asyncio
import logging
import os
from typing import Iterable, Optional, Tuple, Union
from uuid import uuid4

import netCDF4
from jetto_tools.results import JettoResults

from jetto_mobo import utils


async def run(
    jetto_image: str,
    config_directory: str,
    timelimit: Optional[Union[int, float]] = None,
    container_id: Optional[str] = None,
) -> Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]:
    """Run JETTO in an async subprocess, using a new container with the given config.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config_directory : str
        Path to a directory containing a JETTO configuration; output files will overwrite files in this directory.
    timelimit : Optional[Union[int, float]], default=None
        Maximum number of seconds to wait for JETTO to complete. If `None` or < 0, run until complete.
    container_id : Optional[str], default=None
        ID to give the Singularity container for this run. If `None`, Singularity container will be given a new UUID.

    Returns
    -------
    Tuple[Optional[netCDF4.Dataset], Optional[netCDF4.Dataset]]
        If JETTO converged within the timelimit (return code 0), returns `(profiles, timetraces)`; else returns `(None, None)`.
    """
    if container_id is None:
        container_id = str(uuid4())
    if timelimit < 0:
        timelimit = None
    # Run name is only used internally in this container, so it doesn't matter what it's called
    run_name = "run_name"

    logger = utils.get_logger(
        name=f"jetto-mobo.container.{container_id}", level=logging.INFO
    )

    # Start a container
    # logger.info(f"Creating container...")
    create_container = await asyncio.create_subprocess_exec(
        "singularity",
        "instance",
        "start",
        "--cleanenv",  # Run in a clean environment (no env variables etc)
        "--bind",
        "/tmp",
        "--bind",
        f"{config_directory}:/jetto/runs/{run_name}",  # Bind the output directory to the container's jetto run directory
        jetto_image,
        container_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await create_container.wait()

    # Exec JETTO in the container
    logger.info("Starting JETTO container...")
    run_jetto = await asyncio.create_subprocess_exec(
        "singularity",
        "exec",
        # Container to execute command in
        f"instance://{container_id}",
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
    try:
        await asyncio.wait_for(run_jetto.communicate(), timelimit)
        timeout = False
    except asyncio.TimeoutError:
        timeout = True
    finally:
        # Close the container
        # logger.info("Closing container...")
        delete_container = await asyncio.create_subprocess_exec(
            "singularity",
            "instance",
            "stop",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await delete_container.wait()

    logger.info(
        f"JETTO container terminated with return code {run_jetto.returncode}"
        + (f" (timed out after {timelimit}s)." if timeout else ".")
    )

    if run_jetto.returncode == 0 and not timeout:
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
            run(
                jetto_image,
                config_directory,
                timelimit,
                container_id=os.path.split(config_directory)[1],
            )
            for config_directory in config_directories
        ]
    )
