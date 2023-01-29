import asyncio
import logging
from typing import Mapping, Optional, Union


async def new(
    jetto_image: str,
    config_directory: str,
    run_name: str = "run",
    timelimit: Optional[Union[int, float]] = None,
) -> asyncio.subprocess.Process:
    """Create a new async subprocess that starts a JETTO Singularity container with the given config.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config_directory : str
        Path to a directory containing a JETTO configuration; output files will overwrite files in this directory.
    run_name : str, default="run"
        Name of run; used internally in JETTO.
    timelimit : Optional[Union[int, float]], default=None
        Maximum number of seconds to wait for JETTO to complete. If `None`, run until complete.

    Returns
    -------
    asyncio.subprocess.Process
        Subprocess of JETTO Singularity container.
    """
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
    logging.info(
        f"Starting JETTO {run_name} in {config_directory} with PID {process.pid}."
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timelimit)
    except TimeoutError:
        logging.info(
            f"JETTO {run_name} (output={config_directory}, PID={process.pid}) cancelled (time limit of {timelimit}s exceeded)."
        )
    logging.info(
        f"JETTO {run_name} (output={config_directory}, PID={process.pid}) terminated with return code {process.returncode}."
    )
    if stdout is not None:
        stdout = stdout.decode()
    if stderr is not None:
        stderr = stderr.decode()
    return stdout, stderr, process.returncode


async def run_many(
    jetto_image: str,
    config: Mapping[str, str],
    timelimit: Optional[Union[int, float]] = None,
) -> list[(str, str, int)]:
    """Asynchronously run multiple JETTO runs, using `jetto_subprocess.new()`.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config : Mapping[str, str]
        A mapping of run names to config directories.
        Keys are used to internally name the run.
        Values must be a path to a directory containing a JETTO configuration to load for each process;
        output files will overwrite files in this directory.
    timelimit : Optional[Union[int, float]], default = None
        Maximum number of seconds to wait for JETTO to complete. If `None`, run until complete.

    Returns
    -------
    list[(str, str, int)]
        List of (stdout, stderr, return_code) for each JETTO subprocess.
    """
    return await asyncio.gather(
        *[
            new(jetto_image, config_directory, run_name, timelimit)
            for run_name, config_directory in config.items()
        ]
    )


def is_converged(directory: str) -> bool:
    """Check if the JETTO run in `directory` is completed and converged.

    Parameters
    ----------
    directory : str
        Path to a JETTO results directory.

    Returns
    -------
    bool
        True iff `{directory}/jetto.status` exists and is `Status : Completed successfully`.
    """
    status_file = f"{directory}/jetto.status"
    if os.path.exists(status_file):
        with open(status_file) as f:
            if f.read().strip() == "Status : Completed successfully":
                return True
    return False
