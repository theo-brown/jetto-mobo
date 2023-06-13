from typing import Union

import torch
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from jetto_mobo.configuration import AcquisitionConfig


def generate_initial_candidates(
    bounds: torch.Tensor, acquisition_config: AcquisitionConfig
) -> torch.Tensor:
    return (
        draw_sobol_samples(bounds, n=1, q=acquisition_config.batch_size)
        .squeeze()
        .to(device=acquisition_config.device, dtype=acquisition_config.dtype)
    )


def generate_trial_candidates(
    observed_inputs: torch.Tensor,
    bounds: torch.Tensor,
    acquisition_config: AcquisitionConfig,
    model: Union[GPyTorchModel, ModelListGPyTorchModel],
) -> torch.Tensor:
    """
    Generate trial candidates using the given acquisition function config.

    Parameters
    ----------
    observed_inputs : torch.Tensor
        Inputs for which the output is known.
        Shape is (N, D), where N is the number of data points and D is the number of input dimensions.
    bounds : torch.Tensor
        Bounds on the input space, shape (D, 2). bounds[i, 0] is the lower bound and bounds[i, 1] is the upper bound for the ith input.
    acquisition_config : AcquisitionConfig
        Acquisition function configuration.
    model : Union[GPyTorchModel, ModelListGPyTorchModel]
        Trained surrogate model.

    Returns
    -------
    torch.Tensor
        Trial candidates. Shape is (q, D), where q is the number of trial candidates and D is the number of input dimensions.
    """
    # Send to correct device and data type
    observed_inputs_ = observed_inputs.to(
        device=acquisition_config.device, dtype=acquisition_config.dtype
    )
    # Set up acquisition function
    acqf = acquisition_config.function(
        model=model,
        X_baseline=normalize(observed_inputs_, bounds),
        sampler=SobolQMCNormalSampler(num_samples=acquisition_config.n_sobol_samples),
        **acquisition_config.kwargs,
    )
    # Generate trial candidates
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=acquisition_config.batch_size,
        raw_samples=acquisition_config.raw_samples,
        num_restarts=acquisition_config.n_restarts,
        sequential=True if acquisition_config.mode == "sequential" else False,
        options={
            "batch_limit": acquisition_config.batch_limit,
            "maxiter": acquisition_config.max_iterations,
        },
    )

    return unnormalize(candidates.detach(), bounds)
