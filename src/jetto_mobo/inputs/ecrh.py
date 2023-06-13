import asyncio
import os
import pathlib
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import jetto_tools
import numpy as np

from jetto_mobo.configuration import InputConfig, SimulationConfig, bo_input


def _gaussian(
    x: Iterable[float], mean: float, variance: float, amplitude: float = 1
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * (x - mean) ** 2 / variance)


@bo_input
def sum_of_gaussians(
    x: Iterable[float],
    means: Iterable[float],
    variances: Iterable[float],
    amplitudes: Iterable[float],
) -> np.ndarray:
    """Sum of Gaussians.

    This parameterisation is symmetric about 0.

    Parameters
    ----------
    x : Iterable[float]
        Normalised radius, in [0, 1].
    means : Iterable[float]
        Means of each Gaussian. Must have length N, where N is the number of Gaussians.
    variances : Iterable[float]
        Variance of each Gaussian. Must have length N, where N is the number of Gaussians.
    amplitudes : Iterable[float]
        Amplitude of each Gaussian. Must have length N, where N is the number of Gaussians.

    Returns
    -------
    np.ndarray
        Sum of Gaussians.
    """
    if len(means) != len(variances) or len(means) != len(amplitudes):
        raise ValueError(
            "Must have the same number of means, variances, and amplitudes."
        )

    # Avoid double-counting Gaussians at 0
    mean_is_zero = np.isclose(means, 0)
    if np.any(mean_is_zero):
        means_ = np.concatenate([-means[~mean_is_zero], means])
        variances_ = np.concatenate([variances[~mean_is_zero], variances])
        amplitudes_ = np.concatenate([amplitudes[~mean_is_zero], amplitudes])
    else:
        means_ = np.concatenate([-means, means])
        variances_ = np.concatenate([variances, variances])
        amplitudes_ = np.concatenate([amplitudes, amplitudes])

    return np.sum(
        [
            _gaussian(x, mean, variance, amplitude)
            for mean, variance, amplitude in zip(means_, variances_, amplitudes_)
        ],
        axis=0,
    )


def unit_interval_logspace(N: int) -> np.ndarray:
    """Generate N logarithmically spaced points in the unit interval [0, 1].

    Parameters
    ----------
    N : int
        Number of points to generate

    Returns
    -------
    np.ndarray
        Array of N points logarithmically spaced in the unit interval [0, 1].
    """
    return (np.logspace(1, 2, N) - 10) / 90


@bo_input
def sum_of_gaussians_fixed_log_means(
    x: np.ndarray, xmax: float, variances: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    """Sum of Gaussians, with means log-spaced in the interval [0, xmax].

    Variance and amplitude of each Gaussian is a free parameter.
    This profile is constrained such that the gradient x=0 is 0.

    Parameters
    ----------
    x : np.ndarray
        Normalised radius, in the interval [0, 1].
    xmax : float
        Location of the Gaussian furthest from x=0, in the interval [0, 1].
    variances : np.ndarray
        Variance of each Gaussian. Must have length N, where N is the number of Gaussians.
    amplitudes : np.ndarray
        Amplitude of each Gaussian. Must have length N, where N is the number of Gaussians.

    Returns
    -------
    np.ndarray
        Sum of Gaussians, with means log-spaced in the interval [0, xmax].
    """
    n_gaussians = len(variances)
    means = unit_interval_logspace(n_gaussians) * xmax
    return sum_of_gaussians(x, means, variances, amplitudes)


def create_jetto_config(
    ecrh_parameters: Mapping[str, Any],
    jetto_config_directory: pathlib.Path,
    simulation_config: SimulationConfig,
    ecrh_config: InputConfig,
) -> jetto_tools.config.RunConfig:
    """Create a new JETTO run configuration with the specified ECRH heating profile.

    Parameters
    ----------
    ecrh_parameters : Mapping[str, Any]
        Parameters for the ECRH profile. Keys depend on the function chosen in `ecrh_config`.
    jetto_config_directory : pathlib.Path
        Directory to store the new configuration in.
    simulation_config : SimulationConfig
        Simulation configuration object.
    ecrh_config : InputConfig
        ECRH configuration object.

    Returns
    -------
    jetto_tools.config.RunConfig
        New JETTO run configuration.
    """
    # Read exfile from template
    exfile = jetto_tools.binary.read_binary_file(
        simulation_config.template.extra_files["jetto.ex"]
    )

    # Modify the exfile with the new ECRH profile
    exfile["QECE"][0] = ecrh_config.function(exfile["XRHO"][0], **ecrh_parameters)

    # Save the exfile
    modified_exfile_path = jetto_config_directory / "jetto.ex"
    jetto_tools.binary.write_binary_exfile(exfile, modified_exfile_path)

    # Create a new config with the modified exfile
    config = jetto_tools.config.RunConfig(template)
    config.exfile = modified_exfile_path
    config.export(config_directory)

    return config
