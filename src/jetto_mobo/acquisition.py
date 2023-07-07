from typing import Literal, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize


def generate_initial_candidates(
    bounds: torch.Tensor,
    batch_size: int,
    device: Union[str, torch.device, int],
    dtype: Union[str, torch.dtype],
) -> torch.Tensor:
    """Generate initial candidates using Sobol sampling.

    This is a simple wrapper around `botorch.utils.sampling.draw_sobol_samples` that reshapes the output to a tensor of shape (batch_size, D).

    Parameters
    ----------
    bounds : torch.Tensor
        A 2xD tensor of lower and upper bounds for each of the D dimensions.
    batch_size : int
        Number of initial candidates to generate.
    device : Union[str, torch.device, int]
        Resulting tensor will be sent to this device.
    dtype : Union[str, torch.dtype]
        Resulting tensor will be cast to this data type.

    Returns
    -------
    torch.Tensor
        Initial candidates. Shape is (batch_size, D).
    """
    return (
        draw_sobol_samples(bounds, n=1, q=batch_size)
        .squeeze()
        .to(device=device, dtype=dtype)
    )


def generate_trial_candidates(
    observed_inputs: torch.Tensor,
    bounds: torch.Tensor,
    model: Union[GPyTorchModel, ModelListGPyTorchModel],
    acquisition_function: AcquisitionFunction,
    device: Union[str, torch.device, int],
    dtype: Union[str, torch.dtype],
    n_mc_samples: int = 256,
    batch_size: int = 1,
    raw_samples: int = 512,
    n_restarts: int = 10,
    mode: Literal["sequential", "joint"] = "joint",
    batch_limit: int = 5,
    max_iterations: int = 200,
    **kwargs,
) -> torch.Tensor:
    """
    Generate trial candidates using the given acquisition function and model.

    Parameters
    ----------
    observed_inputs : torch.Tensor
        Inputs for which the output is known.
        Shape is (N, D), where N is the number of data points and D is the number of input dimensions.
    bounds : torch.Tensor
        Bounds on the input space, shape (D, 2). bounds[i, 0] is the lower bound and bounds[i, 1] is the upper bound for the ith input.
    model : Union[GPyTorchModel, ModelListGPyTorchModel]
        Trained surrogate model.
    acquisition_function : AcquisitionFunction
        Acquisition function to use for generating trial candidates.
    device : Union[str, torch.device, int]
        Torch device to use for optimising the acquisition function.
    dtype : Union[str, torch.dtype]
        Torch data type to use for representing the candidates.
    n_mc_samples : int, optional
        Number of samples to use in Monte Carlo estimation of expectations or integrals in the acquisition function (default: 256).
    batch_size : int, optional
        Number of trial candidates to generate (default: 1).
    raw_samples : int, optional
        Number of samples to use in multi-start optimisation of the acquisition function (default: 512). See (https://botorch.org/docs/optimization#multiple-random-restarts).
    n_restarts : int, optional
        Number of restarts to use in multi-start optimisation of the acquisition function (default: 10). See (https://botorch.org/docs/optimization#multiple-random-restarts).
    mode : Literal["sequential", "joint"], optional
        Whether to generate candidates using sequential or joint optimisation (default: "joint"). Joint optimisation is more accurate, but the computational cost scales rapidly with batch_size. See (https://botorch.org/docs/optimization#joint-vs-sequential-candidate-generation-for-batch-acquisition-functions) for more details.
    batch_limit : int, optional
        Generate candidates in sub-batches of this size (default: 5).
    max_iterations : int, optional
        Maximum number of optimisation iterations to perform in optimising the acquisition function (passed to `scipy.optimize.minimize` via `botorch.gen.gen_candidates_scipy`).


    Returns
    -------
    torch.Tensor
        Trial candidates. Shape is (q, D), where q is the number of trial candidates and D is the number of input dimensions.
    """
    # Send to correct device and data type
    observed_inputs_ = observed_inputs.to(device=device, dtype=dtype)

    # Set up acquisition function
    acqf = acquisition_function(
        model=model,
        X_baseline=normalize(observed_inputs_, bounds),
        sampler=SobolQMCNormalSampler(num_samples=n_mc_samples),
        **kwargs,
    )

    # Generate trial candidates
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        raw_samples=raw_samples,
        num_restarts=n_restarts,
        sequential=True if mode == "sequential" else False,
        options={
            "batch_limit": batch_limit,
            "maxiter": max_iterations,
        },
        **kwargs,
    )

    return unnormalize(candidates.detach(), bounds)
