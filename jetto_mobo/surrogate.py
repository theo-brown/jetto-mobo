# TODO: write module-level docstring
from typing import Optional, Union, Callable

import torch
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel

from invariantkernels import GroupInvariantKernel


def fit_surrogate_model(
    inputs: torch.Tensor,
    input_bounds: torch.Tensor,
    objective_values: torch.Tensor,
    constraint_values: Optional[torch.Tensor] = None,
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
    normalise: bool = True,
    standardise: bool = True,
    transformation_group: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
    transformation_group : Optional[Callable[[torch.Tensor], torch.Tensor]], default = None
        A function that generates all transformed versions of an input x for a given group (i.e., the orbits of x).
        The function should take a tensor of shape (n, d) and return a tensor of shape (G, n, d) where G is the number of
        elements of the group.
        If set, the fitted surrogate model will be invariant to transformations in the group.

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
    if constraint_values is not None:
        constraint_values_ = constraint_values.to(device=device, dtype=dtype)
    input_bounds_ = input_bounds.to(device=device, dtype=dtype)

    # Check that dimensions match
    if not inputs.shape[0] == objective_values_.shape[0]:
        raise ValueError(
            f"Number of input and output points must match (got inputs.shape[0]={inputs.shape[0]} and outputs.shape[0]={objective_values_.shape[0]})."
        )
    if constraint_values is not None:
        if not inputs.shape[0] == constraint_values_.shape[0]:
            raise ValueError(
                f"Number of input and constraint_value points must match (got inputs.shape[0]={inputs.shape[0]} and constraint_values.shape[0]={constraint_values_.shape[0]})."
            )
    if not inputs.shape[1] == input_bounds_.shape[1]:
        raise ValueError(
            f"Dimensionality of input points must match the number of bounds (got input dimension {inputs.shape[1]} and {input_bounds_.shape[1]} bounds)."
        )

    # Transforms
    input_transform = (
        Normalize(d=inputs_.size(-1), bounds=input_bounds_) if normalise else None
    )
    # We're using a ModelListGP, so each objective is treated separately
    output_transform = Standardize(m=1) if standardise else None

    # Kernel
    kernel = ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=inputs_.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    if transformation_group is not None:
        kernel = GroupInvariantKernel(
            base_kernel=kernel, transformation_group=transformation_group
        )

    # Initialise model
    if constraint_values is not None:
        objective_models = [
            SingleTaskGP(
                inputs_,
                objective_values_[:, i].unsqueeze(1),
                covar_module=kernel,
                input_transform=input_transform,
                outcome_transform=output_transform,
            )
            for i in range(objective_values_.shape[1])
        ]
        constraint_models = [
            SingleTaskGP(
                inputs_,
                constraint_values_[:, i].unsqueeze(1),
                covar_module=kernel,
                input_transform=input_transform,
                outcome_transform=output_transform,
            )
            for i in range(constraint_values_.shape[1])
        ]
        models = objective_models + constraint_models
        model = ModelListGP(*models)
    else:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    inputs_,
                    objective_values_[:, i].unsqueeze(1),
                    covar_module=kernel,
                    input_transform=input_transform,
                    outcome_transform=output_transform,
                )
                for i in range(objective_values_.shape[1])
            ]
        )

    # Fit model
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll_torch(mll)
    return model
