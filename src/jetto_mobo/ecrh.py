import asyncio
import os
from typing import Callable, Iterable, Union

import jetto_subprocess
import jetto_tools
import numpy as np


def _gaussian(
    x: Iterable[float], mean: float, variance: float, amplitude: float = 1
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * (x - mean) ** 2 / variance)


def sum_of_gaussians(
    x: Iterable[float],
    means: Iterable[float],
    variances: Iterable[float],
    amplitudes: Iterable[float],
) -> np.ndarray:
    if len(means) != len(variances) or len(means) != len(amplitudes):
        raise ValueError(
            "means, variances, amplitudes should be same length;"
            f"got lengths [{len(means)}, {len(variances)}, {len(amplitudes)}]"
        )

    return np.sum(
        [_gaussian(x, m, v, a) for m, v, a in zip(means, variances, amplitudes)], axis=0
    )


def piecewise_linear(x: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")

    # On axis peak
    on_axis_peak_x = 0
    on_axis_peak_y = parameters[0]

    # On axis peak shaper
    on_axis_peak_end_x = parameters[1]
    on_axis_peak_end_y = parameters[2] * on_axis_peak_y

    # Minimum
    minimum_x = parameters[3]
    minimum_y = parameters[4]

    # Minimum shaper
    minimum_shaper_x = (minimum_x + on_axis_peak_end_x) / 2
    minimum_shaper_y = parameters[5] * minimum_y

    # Off-axis peak
    off_axis_peak_x = (minimum_x + parameters[6]) / 2
    off_axis_peak_y = parameters[7]

    # Off-axis shaper 2
    off_axis_shaper_2_x = (minimum_x + 2 * off_axis_peak_x) / 3
    off_axis_shaper_2_y = (
        parameters[8] * off_axis_peak_y + (1 - parameters[8]) * minimum_y
    )

    # Off-axis shaper 1
    off_axis_shaper_1_x = (2 * minimum_x + off_axis_peak_x) / 3
    off_axis_shaper_1_y = (
        parameters[9] * off_axis_shaper_2_y + (1 - parameters[9]) * minimum_y
    )

    # Turn-off
    turn_off_x = off_axis_peak_x + parameters[11]
    turn_off_y = 0

    # Turn-off shaper
    turn_off_shaper_x = (off_axis_peak_x + turn_off_x) / 2
    turn_off_shaper_y = parameters[10] * off_axis_peak_y

    # Collect into array
    node_xs = [
        on_axis_peak_x,
        on_axis_peak_end_x,
        minimum_shaper_x,
        minimum_x,
        off_axis_shaper_1_x,
        off_axis_shaper_2_x,
        off_axis_peak_x,
        turn_off_shaper_x,
        turn_off_x,
    ]
    node_ys = [
        on_axis_peak_y,
        on_axis_peak_end_y,
        minimum_shaper_y,
        minimum_y,
        off_axis_shaper_1_y,
        off_axis_shaper_2_y,
        off_axis_peak_y,
        turn_off_shaper_y,
        turn_off_y,
    ]

    return np.interp(x, node_xs, node_ys)


def create_config(
    template_directory: str,
    config_directory: str,
    ecrh_function: Callable[[Iterable[float]], Iterable[float]],
) -> None:
    """Create a new JETTO run configuration with the specified ECRH heating profile.

    Parameters
    ----------
    template_directory : str
        Directory of JETTO template run, used as a base to create the new configuration.
    config_directory : str
        Directory to store the new configuration in.
    ecrh_function : Callable[[Iterable[float]], Iterable[float]]
        A function that maps from normalised radius (XRHO) to ECRH power (QECE).
    """

    # Read exfile from template
    template = jetto_tools.template.from_directory(template_directory)
    exfile = jetto_tools.binary.read_binary_file(template.extra_files["jetto.ex"])

    # Modify the exfile with the new ECRH profile
    exfile["QECE"][0] = ecrh_function(exfile["XRHO"][0])

    # Save the exfile
    modified_exfile_path = f"{config_directory}/jetto.ex"
    jetto_tools.binary.write_binary_exfile(exfile, modified_exfile_path)

    # Create a new config with the modified exfile
    config = jetto_tools.config.RunConfig(template)
    config.exfile = modified_exfile_path
    config.export(config_directory)


def get_cost(
    ecrh_parameters: np.ndarray,
    directory: str,
    ecrh_function: Callable[(np.ndarray, np.ndarray), np.ndarray],
    cost_function: Callable[str, Union[float, np.ndarray]],
    jetto_template: str = "jetto/templates/spr45-v9",
    jetto_image: str = "jetto/images/sim.v220922.sif",
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    ecrh_parameters : np.ndarray
        A `b x n` array, where `b` is the number of JETTO runs (i.e. batch size) and `n` is the number of ECRH parameters.
    directory : str
        Root directory to store JETTO files in. Output of each run will be stored in 'directory/{i}', for i=1,...,n.
    ecrh_function : Callable[(Iterable[float], Iterable[float]), Iterable[float]]
        A function that takes normalised radius (XRHO) as first argument, `n` ECRH parameters as second argument, and returns ECRH power (QECE).
    cost_function : Callable[str, Union[float, Iterable[float]]]
        A function that takes a JETTO output directory and returns a scalar or vector cost associated with the ECRH profile.
    jetto_template : str, default "jetto/templates/spr45-v9"
        Directory of JETTO template run, used as a base to create the new JETTO configurations.
    jetto_image : str, default "jetto/images/sim.v220922.sif"
        Path to the JETTO .sif Singularity container image.

    Returns
    -------
    np.ndarray
        The value of cost_function when the ECRH is set according to the parameters.
        If cost_function returns a scalar, returns a `b x 1` array.
        If cost_function returns a vector of length `c`, returns a `b x c` array.
    """
    config = {}
    for i in range(ecrh_parameters.shape[0]):
        run_directory = f"{directory}/{i}"
        os.makedirs(run_directory)

        create_config(
            jetto_template,
            run_directory,
            lambda xrho: ecrh_function(xrho, ecrh_parameters[i]),
        )

        config[str(i)] = run_directory

    asyncio.run(jetto_subprocess.run_many(jetto_image, config))

    return np.array(
        [cost_function(output_directory) for output_directory in config.values()]
    )
