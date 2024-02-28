# TODO: write module-level docstring
from typing import Optional, Union

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def fit_surrogate_model(
    inputs: torch.Tensor,
    input_bounds: torch.Tensor,
    objective_values: torch.Tensor,
    constraint_values: Optional[torch.Tensor] = None,
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
    normalise: bool = True,
    standardise: bool = True,
) -> Union[SingleTaskGP, ModelListGP]:
    """
    Fit a Gaussian process surrogate model to the data.

    Parameters
    ----------
    inputs : torch.Tensor
        Input data. Shape is ``(N, D)``, where ``N`` is the number of data points and ``D`` is the number of input dimensions.
    input_bounds : torch.Tensor
        Bounds on the input space, shape ``(D, 2)``. ``bounds[i, 0]`` is the lower bound and ``bounds[i, 1]`` is the upper bound for the ``i``th input.
    objective_values : torch.Tensor
        Output data. Shape is ``(N, M)``, where ``N`` is the number of data points and ``M`` is the number of objectives.
    constraint_values : Optional[torch.Tensor], default = None
        Constraint data. Constraint functions should be of the form ``g(x) <= 0``, which means ``constraint_values`` will be negative if the constraint is satisfied.
        Shape is ``(N, C)``, where ``N`` is the number of data points and ``C`` is the number of constraints.
    device : Optional[Union[str, torch.device]], default = None
        Device to use for fitting the surrogate model. If None, the device of the input data (``X``) will be used.
    dtype : Optional[torch.dtype], default = None
        Data type to use for fitting the surrogate model. If None, the data type of the input data (``X``) will be used.
    normalise : bool, default = True
        Whether to normalise the input data before fitting the surrogate model. This normally results in improved performance.
    standardise : bool, default = True
        Whether to standardise the objective data before fitting the surrogate model. This normally results in improved performance.

    Returns
    -------
    Union[SingleTaskGP, ModelListGP]
        Fitted surrogate model.
    """
    if device is None:
        device = inputs.device
    if dtype is None:
        dtype = inputs.dtype

    # Convert to correct device and data type
    inputs_ = inputs.to(device=device, dtype=dtype)
    objective_values_ = objective_values.to(device=device, dtype=dtype)
    constraint_values_ = constraint_values.to(device=device, dtype=dtype)
    input_bounds_ = input_bounds.to(device=device, dtype=dtype)

    # Check that dimensions match
    if not inputs.shape[0] == objective_values_.shape[0]:
        raise ValueError(
            f"Shape of input and output data must match in the first dimension (got input shape {inputs.shape} and output shape {objective_values_.shape})."
        )
    if constraint_values is not None:
        if not inputs.shape[0] == constraint_values_.shape[0]:
            raise ValueError(
                f"Shape of input and constraint data must match in the first dimension (got input shape {inputs.shape} and constraint shape {constraint_values_.shape})."
            )
    if not inputs.shape[1] == input_bounds_.shape[0]:
        raise ValueError(
            f"Number of input dimensions must match the number of bounds (got input shape {inputs.shape} and bounds shape {input_bounds_.shape})."
        )

    # Transforms
    input_transform = (
        Normalize(d=inputs_.size(-1), bounds=input_bounds_) if normalise else None
    )
    # We're using a modellistgp, so each objective is treated separately
    output_transform = Standardize(m=1) if standardise else None

    # Initialise model
    if constraint_values is not None:
        objective_models = [
            SingleTaskGP(
                inputs_,
                objective_values_[:, i].unsqueeze(1),
                input_transform=input_transform,
                outcome_transform=output_transform,
            )
            for i in range(objective_values_.shape[1])
        ]
        constraint_models = [
            SingleTaskGP(
                inputs_,
                constraint_values_[:, i].unsqueeze(1),
                input_transform=input_transform,
                outcome_transform=output_transform,
            )
            for i in range(constraint_values_.shape[1])
        ]
        models = objective_models + constraint_models
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    inputs_,
                    objective_values_[:, i].unsqueeze(1),
                    input_transform=input_transform,
                    outcome_transform=output_transform,
                )
                for i in range(objective_values_.shape[1])
            ]
        )

    # Fit model
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model
