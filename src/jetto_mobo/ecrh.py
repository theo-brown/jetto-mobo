import asyncio
import os
from typing import Callable, Iterable, Optional, Tuple

import jetto_tools
import netCDF4
import numpy as np
from jetto_tools.results import JettoResults

from jetto_mobo import jetto_subprocess, utils

# from scipy.interpolate import CubicSpline


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


def piecewise_linear_2(x: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
    """Simple 6-segment piecewise linear function.

    Parameters
    ----------
    x : Iterable[float]
        Input array; currently needs to be [0, 1].
    parameters : Iterable[float]
        12 parameters, each [0, 1].

    Returns
    -------
    np.ndarray
        6-segment piecewise linear function

    Raises
    ------
    ValueError
        If `len(parameters) != 12`.
    """
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")
    padded_parameters = np.pad(parameters, 1, "constant", constant_values=0)
    node_xs = padded_parameters[:7]
    node_ys = padded_parameters[7:]
    return np.interp(x, node_xs, node_ys)


# def cubic_spline(x: Iterable[float], parameters: Iterable[float]):
#     if len(parameters) % 2 != 0:
#         raise ValueError("Must have an even number of parameters.")
#     padded_parameters = np.pad(parameters, 1, "constant", constant_values=0)
#     n_nodes = len(padded_parameters) // 2
#     node_xs = padded_parameters[:n_nodes]
#     node_ys = padded_parameters[n_nodes:]
#     # Spline requires increasing x
#     sorted_indices = np.argsort(node_xs)
#     f = CubicSpline(node_xs[sorted_indices], node_ys[sorted_indices], bc_type="clamped")
#     return f(x)


def create_config(
    template_directory: str,
    config_directory: str,
    ecrh_function: Callable[[Iterable[float]], Iterable[float]],
) -> None:
    """Create a new JETTO run configuration with the specified ECRH heating profile.

    Parameters
    ----------
    template_directory : str
        Directory of JETTO template run, used as a base to create the new configuration. Note: the template must have
        `PTOTEC` set in `jetto.in`, which is used to normalise the ECRH power.
    config_directory : str
        Directory to store the new configuration in.
    ecrh_function : Callable[[Iterable[float]], Iterable[float]]
        A function that maps from normalised radius (XRHO, [0,1]) to normalised ECRH power (QECE, [0,1]).
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


def get_batch_cost(
    ecrh_parameters: np.ndarray,
    batch_directory: str,
    ecrh_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cost_function: Callable[[netCDF4.Dataset, netCDF4.Dataset], np.ndarray],
    timelimit: Optional[int] = None,
    jetto_template: str = "jetto/templates/spr45-v9",
    jetto_image: str = "jetto/images/sim.v220922.sif",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the given cost function for each element in a batch array of ECRH parameters.

    This function uses `jetto_subprocess.run_many` to asynchronously run a batch of JETTO runs, one for each element of the batch.

    Parameters
    ----------
    ecrh_parameters : np.ndarray
        A `b x n` array, where `b` is the number of JETTO runs (i.e. batch size) and `n` is the number of ECRH parameters.
    batch_directory : str
        Root directory to store JETTO files in. Output of each run will be stored in '{directory}/{i}', for i=1,...,n.
    ecrh_function : Callable[(Iterable[float], Iterable[float]), Iterable[float]]
        A function that takes normalised radius (XRHO) as first argument, `n` ECRH parameters as second argument, and returns ECRH power (QECE).
    cost_function : Callable[[netCDF4.Dataset, netCDF4.Dataset], np.ndarray]
        A function that takes JETTO `profiles` and `timetraces` datasets, and returns a cost associated with the ECRH profile. Output shape `(c,)`.
    timelimit : Optional[int], default None
        Time limit in seconds for JETTO runs. If None, no timelimit imposed.
    jetto_template : str, default "jetto/templates/spr45-v9"
        Directory of JETTO template run, used as a base to create the new JETTO configurations.
    jetto_image : str, default "jetto/images/sim.v220922.sif"
        Path to the JETTO .sif Singularity container image.

    Returns
    -------
    np.ndarray
        A `b x X` array containing the converged ECRH profile.
        If JETTO run `i` failed, elements `[i, :]`  will be np.nan.
    np.ndarray
        A `b x X` array containing the converged Q profile.
        If JETTO run `i` failed, elements `[i, :]`  will be np.nan.
    np.ndarray
        A `b x c` array containing the value of `cost_function` for the given converged outputs..
        If JETTO run `i` failed, elements `[i, :]`  will be np.nan.
    """
    batch_size = ecrh_parameters.shape[0]
    config_directories = [f"{batch_directory}/{i}" for i in range(batch_size)]
    for i, config_directory in enumerate(config_directories):
        os.makedirs(config_directory)
        create_config(
            jetto_template,
            config_directory,
            lambda xrho: ecrh_function(xrho, ecrh_parameters[i]),
        )

    # Run asynchronously in parallel
    batch_output = asyncio.run(
        jetto_subprocess.run_many(jetto_image, config_directories, timelimit)
    )

    # Parse outputs
    converged_inputs = []
    converged_outputs = []
    costs = []
    for i, (profiles, timetraces) in enumerate(batch_output):
        if profiles is not None:
            # Load data
            results = JettoResults(path=f"{batch_directory}/{i}")
            profiles = results.load_profiles()
            timetraces = results.load_timetraces()
            # Save to arrays
            converged_inputs.append(profiles["QECE"][-1])
            converged_outputs.append(profiles["Q"][-1])
            costs.append(cost_function(profiles, timetraces))
        else:
            converged_inputs.append(None)
            converged_outputs.append(None)
            costs.append(None)

    return (
        utils.pad_1d(converged_inputs),
        utils.pad_1d(converged_outputs),
        utils.pad_1d(costs),
    )
