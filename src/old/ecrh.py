import asyncio
import os
from typing import Callable, Iterable, Optional, Tuple

import jetto_tools
import netCDF4
import numpy as np
from jetto_tools.results import JettoResults
from scipy.interpolate import interp1d

from jetto_mobo import jetto_container, utils


def _gaussian(
    x: Iterable[float], mean: float, variance: float, amplitude: float = 1
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * (x - mean) ** 2 / variance)


def sum_of_gaussians(
    x: Iterable[float], parameters: Iterable[float], variance: float = 0.0025
) -> np.ndarray:
    if len(parameters) % 2 != 0:
        raise ValueError("Must have an even number of parameters.")

    n_gaussians = len(parameters) // 2

    means = np.cumprod(parameters[:n_gaussians])
    amplitudes = parameters[n_gaussians:]

    return np.sum(
        [_gaussian(x, m, variance, a) for m, a in zip(means, amplitudes)], axis=0
    )


def piecewise_linear(x: Iterable[float], parameters: Iterable[float]) -> np.ndarray:
    """Piecewise linear, with points evenly spaced in `[0, parameters[0]]`, with ys specified by `[*parameters[1:], 0]`.

    Parameters
    ----------
    x : Iterable[float]
        Input array; currently needs to be [0, 1].
    parameters : Iterable[float]
        Parameters, each [0, 1].

    Returns
    -------
    np.ndarray
        Piecewise linear function evaluated on x.

    Raises
    ------
    ValueError
        If `len(parameters) < 2`.
    """
    if len(parameters) < 2:
        raise ValueError(f"Expected at least 2 parameters, got {len(parameters)}.")

    end_node_x = parameters[0]
    node_ys = np.concatenate([parameters[1:], [0]])
    node_xs = np.linspace(0, end_node_x, len(parameters))
    return np.interp(x, node_xs, node_ys)


def piecewise_linear_decreasing(
    x: Iterable[float], parameters: Iterable[float]
) -> np.ndarray:
    """Decreasing piecewise linear function, with points evenly spaced in `[0, parameters[0]]`, with ys specified by `[*parameters[1:], 0]`.

    Parameters
    ----------
    x : Iterable[float]
        Input array; currently needs to be [0, 1].
    parameters : Iterable[float]
        parameters, each [0, 1].

    Returns
    -------
    np.ndarray
        Piecewise linear function evaluated on x.

    Raises
    ------
    ValueError
        If `len(parameters) < 2`.
    """
    if len(parameters) < 2:
        raise ValueError(f"Expected at least 2 parameters, got {len(parameters)}.")

    end_node_x = parameters[0]
    node_xs = np.linspace(0, end_node_x, len(parameters))

    node_ys = np.cumprod([*parameters[1:], 0])

    return np.interp(x, node_xs, node_ys)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_cubic_spline(x: Iterable[float], parameters: Iterable[float]):
    if len(parameters) % 2 != 0:
        raise ValueError("Must have an even number of parameters.")
    n_nodes = (len(parameters) + 2) // 2

    node_xs = np.concatenate([[0], parameters[: n_nodes - 2], [1]])
    node_ys = parameters[n_nodes - 2 :]

    spline = interp1d(node_xs, node_ys, kind="cubic")
    return _sigmoid(spline(x))


def _squared_exponential_kernel(x1, x2, l, scale_factor):
    return scale_factor**2 * np.exp(-((x1 - x2) ** 2) / (2 * l**2))


def _covariance_matrix(x, kernel):
    return kernel(x[:, None], x[None, :])


def gp(x: Iterable[float], parameters: Iterable[float]):
    lengthscale = parameters[0]
    y0 = parameters[1]
    xN = parameters[-1]
    node_xs = parameters[2:-1:2]
    node_ys = parameters[3:-1:2]
    K = _covariance_matrix(
        x=node_xs,
        kernel=lambda x1, x2: _squared_exponential_kernel(x1, x2, lengthscale, 1),
    )
    coefficients = np.linalg.inv(K.T * K) @ (K @ node_ys)
    return np.sum(
        [coefficients[i] * kernel(x, node_xs[i]) for i in range(len(node_xs))], axis=0
    )


def unit_interval_logspace(N: int) -> np.ndarray:
    """Generate N logarithmically spaced points in the unit interval [0, 1].

    Parameters
    ----------
    N : int
        NUmber of points to generate

    Returns
    -------
    np.ndarray
        Array of N points logarithmically spaced in the unit interval [0, 1].
    """
    return (np.logspace(1, 2, N) - 10) / 90


def sum_of_gaussians_fixed_log_means(
    x: np.ndarray, xmax: float, variances: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    n_gaussians = len(variances)
    means = unit_interval_logspace(n_gaussians) * xmax
    # Enforce symmetry about 0
    # The first element of means is 0, so we mirror around it
    means = np.concatenate([-np.flip(means[1:]), means])
    variances = np.concatenate([np.flip(variances[1:]), variances])
    amplitudes = np.concatenate([np.flip(amplitudes[1:]), amplitudes])
    return np.sum(
        [
            _gaussian(x, mean, variance, amplitude)
            for mean, variance, amplitude in zip(means, variances, amplitudes)
        ],
        axis=0,
    )


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


def get_batch_value(
    ecrh_parameters: np.ndarray,
    batch_directory: str,
    ecrh_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    value_function: Callable[[netCDF4.Dataset, netCDF4.Dataset], np.ndarray],
    timelimit: Optional[int] = None,
    jetto_template: str = "jetto/templates/spr45",
    jetto_image: str = "jetto/images/sim.v220922.sif",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the given value function for each element in a batch array of ECRH parameters.

    This function uses `jetto_container.run_many` to asynchronously run a batch of JETTO runs, one for each element of the batch.

    Parameters
    ----------
    ecrh_parameters : np.ndarray
        A `b x n` array, where `b` is the number of JETTO runs (i.e. batch size) and `n` is the number of ECRH parameters.
    batch_directory : str
        Root directory to store JETTO files in. Output of each run will be stored in '{directory}/{i}', for i=1,...,n.
    ecrh_function : Callable[(Iterable[float], Iterable[float]), Iterable[float]]
        A function that takes normalised radius (XRHO) as first argument, `n` ECRH parameters as second argument, and returns ECRH power (QECE).
    value_function : Callable[[netCDF4.Dataset, netCDF4.Dataset], np.ndarray]
        A function that takes JETTO `profiles` and `timetraces` datasets, and returns a value associated with the ECRH profile. Output shape `(c,)`.
    timelimit : Optional[int], default None
        Time limit in seconds for JETTO runs. If None, no timelimit imposed.
    jetto_template : str, default "jetto/templates/spr45"
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
        jetto_container.run_many(jetto_image, config_directories, timelimit)
    )

    # Parse outputs
    converged_inputs = []
    converged_outputs = []
    values = []
    for i, (profiles, timetraces) in enumerate(batch_output):
        if profiles is not None:
            # Load data
            results = JettoResults(path=f"{batch_directory}/{i}")
            profiles = results.load_profiles()
            timetraces = results.load_timetraces()
            # Save to arrays
            converged_inputs.append(profiles["QECE"][-1])
            converged_outputs.append(profiles["Q"][-1])
            values.append(np.atleast_1d(value_function(profiles, timetraces)))
        else:
            converged_inputs.append([None])
            converged_outputs.append([None])
            values.append([None])

    return (
        utils.pad_1d(converged_inputs),
        utils.pad_1d(converged_outputs),
        utils.pad_1d(values),
    )
