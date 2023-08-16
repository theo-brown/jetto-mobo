# TODO: write module-level docstring
from inspect import getmembers, isclass
from typing import Literal, Optional, Union

import botorch
import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize


def generate_initial_candidates(
    bounds: torch.Tensor,
    n: int,
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate initial candidates using Sobol sampling.

    This is a simple wrapper around ``botorch.utils.sampling.draw_sobol_samples`` that reshapes the output to a tensor of shape ``(batch_size, D)``.

    Parameters
    ----------
    bounds : torch.Tensor
        A ``(2, D)`` array of lower and upper bounds for each of the ``D`` dimensions.
    n : int
        Number of initial candidates to generate.
    device : Union[str, torch.device, None] = None
        Resulting tensor will be sent to this device.
    dtype : Optional[torch.dtype] = None
        Resulting tensor will be cast to this data type.

    Returns
    -------
    torch.Tensor
        Initial candidates. Shape is ``(n, D)``.
    """
    bounds = bounds.to(device=device, dtype=dtype)
    return draw_sobol_samples(bounds, n=1, q=n).squeeze()


def generate_trial_candidates(
    observed_inputs: torch.Tensor,
    bounds: torch.Tensor,
    model: Union[GPyTorchModel, ModelListGPyTorchModel],
    acquisition_function: AcquisitionFunction,
    device: Union[torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
    n_mc_samples: int = 256,
    batch_size: int = 1,
    raw_samples: int = 512,
    n_restarts: int = 10,
    mode: Literal["sequential", "joint"] = "joint",
    batch_limit: int = 5,
    max_iterations: int = 200,
    acqf_kwargs: dict = {},
    optimisation_kwargs: dict = {},
) -> torch.Tensor:
    """
    Generate trial candidates using the given acquisition function and model.

    Parameters
    ----------
    observed_inputs : torch.Tensor
        Inputs for which the output is known.
        Shape is ``(N, D)``, where ``N`` is the number of data points and ``D`` is the number of input dimensions.
    bounds : torch.Tensor
        Bounds on the input space, shape ``(D, 2)``. ``bounds[i, 0]`` is the lower bound and ``bounds[i, 1]`` is the upper bound for the ``i``th input.
    model : Union[GPyTorchModel, ModelListGPyTorchModel]
        Trained surrogate model.
    acquisition_function : AcquisitionFunction
        Acquisition function to use for generating trial candidates. We recommend using either ``jetto_mobo.acquisition.qNoisyExpectedImprovement`` for single-objective optimisation and ``jetto_mobo.acquisition.qNoisyExpectedHypervolumeImprovement`` for multi-objective optimisation.
    device : Union[str, torch.device, None], default = None
        Torch device to use for optimising the acquisition function. If None, optimisation will be performed using the device the model is on.
    dtype : Optional[torch.dtype], default = None
        Torch data type to use for representing the candidates. If None, candidates will be cast to the same dtype as the model parameters.
    n_mc_samples : int, optional
        Number of samples to use in Monte Carlo estimation of expectations or integrals in the acquisition function (default: 256).
    batch_size : int, optional
        Number of trial candidates to generate (default: 1).
    raw_samples : int, optional
        Number of samples to use in multi-start optimisation of the acquisition function (default: 512). See `BoTorch docs on multi-start optimisation <https://botorch.org/docs/optimization#multiple-random-restarts>`_.
    n_restarts : int, optional
        Number of restarts to use in multi-start optimisation of the acquisition function (default: 10). See `BoTorch docs on multi-start optimisation <https://botorch.org/docs/optimization#multiple-random-restarts>`_.
    mode : Literal["sequential", "joint"], optional
        Whether to generate candidates using sequential or joint optimisation (default: "joint"). Joint optimisation is more accurate, but the computational cost scales rapidly with batch_size. See `BoTorch docs on joint vs sequential optimisation <https://botorch.org/docs/optimization#joint-vs-sequential-candidate-generation-for-batch-acquisition-functions>`_.
    batch_limit : int, optional
        Generate candidates in sub-batches of this size (default: 5).
    max_iterations : int, optional
        Maximum number of optimisation iterations to perform in optimising the acquisition function (passed to ``scipy.optimize.minimize`` via ``botorch.gen.gen_candidates_scipy``).
    acqf_kwargs : dict, default = {}
        Additional keyword arguments to pass to the acquisition function. Some acquisition functions have compulsory arguments (e.g. qNEHVI) - check the Botorch documentation for the specific acquisition function you are using.
    optimisation_kwargs : dict, default = {}
        Additional keyword arguments to pass to ``botorch.optim.optimize_acqf``.


    Returns
    -------
    torch.Tensor
        Trial candidates. Shape is ``(q, D)``, where ``q`` is the number of trial candidates and ``D`` is the number of input dimensions.
    """
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    # Send to correct device and data type
    observed_inputs_ = observed_inputs.to(device=device, dtype=dtype)
    bounds_ = bounds.to(device=device, dtype=dtype)
    model_ = model.to(device=device, dtype=dtype)

    # Check compatibility of model and acquisition function
    multi_objective_acquisition_functions = [
        c
        for _, c in getmembers(botorch.acquisition.multi_objective, isclass)
        if issubclass(c, botorch.acquisition.AcquisitionFunction)
    ]
    if (
        model_.num_outputs != 1
        and not acquisition_function in multi_objective_acquisition_functions
        and not ("posterior_transform" in acqf_kwargs or "objective" in acqf_kwargs)
    ):
        raise ValueError(
            f"Incompatible model and acquisition function: model has {model_.num_outputs} outputs, "
            f"but {acquisition_function.__name__} is not a BoTorch multi-objective acquisition function. "
            "If you intended to perform single-objective optimisation by transforming the outputs, specify a `posterior_transform` in acqf_kwargs. "
            "If you intended to perform single-objective optimisation based on only one of the outputs, specify an `objective` in acqf_kwargs."
            "Otherwise, swap to using a multi-obective acquisition function (we recommend qNEHVI)."
        )

    # qNEHVI requires a reference point
    if "ref_point" in acqf_kwargs:
        acqf_kwargs["ref_point"] = acqf_kwargs["ref_point"].to(
            device=device, dtype=dtype
        )

    # Set up acquisition function
    acqf = acquisition_function(
        model=model_,
        X_baseline=normalize(observed_inputs_, bounds_),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([n_mc_samples])),
        **acqf_kwargs,
    )

    # Generate trial candidates
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_,  # Setting the bounds to be the input bounds here means that we don't need to unnormalize afterwards
        q=batch_size,
        raw_samples=raw_samples,
        num_restarts=n_restarts,
        sequential=True if mode == "sequential" else False,
        options={
            "batch_limit": batch_limit,
            "maxiter": max_iterations,
        },
        **optimisation_kwargs,
    )

    return candidates.detach()
