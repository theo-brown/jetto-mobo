# TODO: write module-level docstring
import asyncio
import logging
import os
import shutil
import sys
import tarfile
from os import PathLike
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union
from uuid import uuid4

import jetto_tools
from jetto_tools.config import RunConfig
from jetto_tools.results import JettoResults

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


async def run(
    run_config: RunConfig,
    run_directory: Union[os.PathLike, str],
    jetto_image: Union[os.PathLike, str],
    timelimit: Optional[Union[int, float]] = None,
    container_id: Optional[str] = None,
    prepend_commands: Iterable[str] = [],
    rjettov_args: Iterable[str] = [
        "-x64",
        "-S",
        "-p0",
        "-n1",
    ],
) -> Optional[JettoResults]:
    """Run JETTO in an async subprocess, using a new container with the given config.

    .. note::

        As this is an *async* function, it must be run using ``asyncio``. For example::

            asyncio.run(
                simulation.run(
                    run_config=my_run_config,
                    run_directory=".",
                    jetto_image="jetto.sif",
                )
            )

    Parameters
    ----------
    run_config : RunConfig
        JETTO configuration to load for this run.
    run_directory : Union[os.PathLike, str]
        Path to the directory to store JETTO output files in.
    jetto_image : Union[os.PathLike, str]
        Path to the JETTO .sif Singularity container image.
    timelimit : Optional[Union[int, float]], default=None
        Maximum number of seconds to wait for JETTO to complete. If ``None`` or < 0, run until complete.
    container_id : Optional[str], default=None
        ID to give the Singularity container for this run. If `None`, Singularity container will be given a new UUID.
    prepend_commands : Iterable[str], default=[]
        Commands to prepend before the `singularity` command. Can be used to launch using ``srun`` or similar.
    rjettov_args : Iterable[str], default=["-x64", "-S", "-p0", "-n1"]
        Additional arguments to pass to ``rjettov``.

    Returns
    -------
    Optional[JettoResults]
        The results of the JETTO run, or ``None`` if the run timed out or otherwise failed.
    """
    if container_id is None:
        container_id = str(uuid4())
    if timelimit is not None and timelimit < 0:
        timelimit = None
    jetto_image = Path(jetto_image)
    run_directory = Path(run_directory)

    # Create run directory
    run_config.export(run_directory)

    # Run name is only used internally in this container, so it doesn't matter what it's called
    run_name = "run_name"

    logger.info(f"Launching container {container_id}...")
    with open(run_directory / "singularity.log", "a") as log_file:
        # Start a container
        # Note: the JETTO image currently has a bug in the runscript that means that `singularity run` doesn't work,
        # so instead we have to start a container and then execute JETTO in it.
        create_container = await asyncio.create_subprocess_exec(
            *prepend_commands,
            "singularity",
            "instance",
            "start",
            "--cleanenv",  # Run in a clean environment (no env variables etc)
            "--bind",
            "/tmp",
            "--bind",
            f"{run_directory}:/jetto/runs/{run_name}",  # Bind the output directory to the container's jetto run directory
            jetto_image,
            container_id,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        await create_container.wait()

        # Exec JETTO in the container
        run_jetto = await asyncio.create_subprocess_exec(
            *prepend_commands,
            "singularity",
            "exec",
            # Container to execute command in
            f"instance://{container_id}",
            # Command to execute in container:
            "rjettov",
            *rjettov_args,
            run_name,
            "build",
            "docker",
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(run_jetto.communicate(), timelimit)
            timeout = False
        except asyncio.TimeoutError:
            timeout = True
        finally:
            # Close the container
            delete_container = await asyncio.create_subprocess_exec(
                *prepend_commands,
                "singularity",
                "instance",
                "stop",
                container_id,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
            )
            await delete_container.wait()

    failed = run_jetto.returncode != 0 or timeout
    logger.info(
        f"Container {container_id} finished with return code {run_jetto.returncode}."
    )
    return JettoResults(path=run_directory) if not failed else None


async def run_many(
    jetto_image: Union[os.PathLike, str],
    run_configs: Mapping[RunConfig, Union[os.PathLike, str]],
    timelimit: Optional[Union[int, float]] = None,
) -> Iterable[Optional[JettoResults]]:
    """Asynchronously run multiple JETTO runs, using ``jetto_mobo.simulation.run()``.

    .. note::

        As this is an *async* function, it must be run using ``asyncio``. For example::

            asyncio.run(
                simulation.run_many(
                    jetto_image=jetto_image, run_configs=configs, timelimit=jetto_timelimit
                )
            )


    Parameters
    ----------
    jetto_image : Union[os.PathLike, str]
        Path to the JETTO .sif Singularity container image.
    run_configs : Mapping[RunConfig, Union[os.PathLike, str]]
        A mapping from JETTO configurations to the directories to store their output files in, such as
        ``{config1: "/home/runs/1", config2: "/home/runs/run2"}``. The directories will be created if they don't exist.
    timelimit : Optional[Union[int, float]], default = None
        Maximum number of seconds to wait for JETTO to complete. If ``None``, run until complete.

    Returns
    -------
    Iterable[Optional[JettoResults]]
        The results of each JETTO run, or ``None`` if the run timed out or otherwise failed.
    """
    return await asyncio.gather(
        *[
            run(
                run_config=run_config,
                run_directory=run_directory,
                jetto_image=jetto_image,
                timelimit=timelimit,
                container_id=Path(run_directory).stem,
            )
            for run_config, run_directory in run_configs.items()
        ]
    )


def compress_jetto_dir(directory: Union[str, os.PathLike], delete: bool = True):
    """Compress a JettoResults directory into a tar.bz2 archive.

    If ``delete=True``, all uncompressed files will be deleted except from ``*.CDF``, ``*.log``, and ``*.bz2``.

    Parameters
    ----------
    directory: Union[str, os.PathLike]
        Path to JettoResults directory. Output tar file ``jetto_results.tar.bz2`` will be created in this directory.
    delete: bool, default=True
        Whether to delete uncompressed files.
    """
    directory = Path(directory)
    jetto_tar_file = directory / "jetto_results.tar.bz2"
    if jetto_tar_file.exists():
        raise OSError(f"{jetto_tar_file} already exists.")

    # Compress output files into a tarball
    with tarfile.open(jetto_tar_file, "w:bz2") as tar:
        tar.add(directory, arcname="")

    if delete:
        # Delete uncompressed files
        for f in os.listdir(directory):
            if os.path.splitext(f)[1] not in [".CDF", ".log", ".bz2"]:
                path = os.path.join(directory, f)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)


def create_config(
    template: Union[PathLike, str],
    directory: Union[PathLike, str],
) -> jetto_tools.config.RunConfig:
    """Create a JETTO config directory from the given template.

    Ensures that the returned config's exfile attribute is set to the correct path.

    Parameters
    ----------
    template : Union[PathLike, str]
        Path to a JETTO template directory.
    directory : Union[PathLike, str]
        Path to a directory where the config will be created.

    Returns
    -------
    jetto_tools.config.RunConfig
        The created config object, representing the JETTO data in the given directory.
    """
    template_dir = Path(template)
    config_dir = Path(directory)
    template_ = jetto_tools.template.from_directory(template_dir)
    config = jetto_tools.config.RunConfig(template_)
    config.export(config_dir)
    config.exfile = Path(config_dir / "jetto.ex")
    return config
