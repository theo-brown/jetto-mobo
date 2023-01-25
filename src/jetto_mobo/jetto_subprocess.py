import asyncio
from typing import Mapping


async def new(
    jetto_image: str, config_directory: str, run_name: str = "run"
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
    print(f"Starting JETTO {run_name} in {config_directory} with PID {process.pid}.")
    stdout, stderr = await process.communicate()
    print(f"JETTO {run_name} in {config_directory} with PID {process.pid} completed.")
    if stdout is not None:
        stdout = stdout.decode()
    if stderr is not None:
        stderr = stderr.decode()
    return stdout, stderr, process.returncode


async def run_many(
    jetto_image: str, config: Mapping[str, str]
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

    Returns
    -------
    list[(str, str, int)]
        List of (stdout, stderr, return_code) for each JETTO subprocess.
    """
    return await asyncio.gather(
        *[
            new(jetto_image, config_directory, run_name)
            for run_name, config_directory in config.items()
        ]
    )
