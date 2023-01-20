import subprocess


def run(jetto_image: str, config_directory: str) -> subprocess.Popen:
    """Run a JETTO Singularity container using the given config.

    Parameters
    ----------
    jetto_image : str
        Path to the JETTO .sif Singularity container image.
    config_directory : str
        Path to a directory containing a JETTO configuration; output files will overwrite files in this directory.

    Returns
    -------
    subprocess.Popen
        Subprocess of JETTO Singularity container.
    """    
    return subprocess.Popen(
        [
            "singularity",
            "exec",
            "--cleanenv",  # Run in a clean environment (no env variables etc)
            "--bind",
            "/tmp",
            "--bind",
            f"{config_directory}:/jetto/runs/run1",  # Bind the output directory to the container's jetto run directory
            jetto_image,  # Container image
            # Command to execute in container:
            "rjettov",
            "-x64",
            "-S",
            "-p0",
            "-n1",
            "run1",  # Run name
            "build",
            "docker",
        ]
    )

