# TODO: write module-level docstring
from typing import Literal, Optional, Union

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def fit_surrogate_model(
    X: torch.Tensor,
    X_bounds: torch.Tensor,
    Y: torch.Tensor,
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
    mode: Literal["independent", "joint"] = "joint",
    normalise: bool = True,
    standardise: bool = True,
) -> Union[SingleTaskGP, ModelListGP]:
    """
    Fit a Gaussian process surrogate model to the data.

    Parameters
    ----------
    X : torch.Tensor
        Input data. Shape is ``(N, D)``, where ``N`` is the number of data points and ``D`` is the number of input dimensions.
    X_bounds : torch.Tensor
        Bounds on the input space, shape ``(D, 2)``. ``bounds[i, 0]`` is the lower bound and ``bounds[i, 1]`` is the upper bound for the ``i``th input.
    Y : torch.Tensor
        Output data. Shape is ``(N, M)``, where ``N`` is the number of data points and ``M`` is the number of output dimensions.
    device : Optional[Union[str, torch.device]], default = None
        Device to use for fitting the surrogate model. If None, the device of the input data (``X``) will be used.
    dtype : Optional[torch.dtype], default = None
        Data type to use for fitting the surrogate model. If None, the data type of the input data (``X``) will be used.
    mode : Literal["independent", "joint"], default = "joint"
        Type of surrogate model to use. If ``"joint"``, all outputs are modelled jointly. If ``"independent"``, each output is modelled independently.
    normalise : bool, default = True
        Whether to normalise the input data before fitting the surrogate model. This normally results in improved performance.
    standardise : bool, default = True
        Whether to standardise the output data before fitting the surrogate model. This normally results in improved performance.

    Returns
    -------
    Union[SingleTaskGP, ModelListGP]
        Fitted surrogate model.
    """
    if device is None:
        device = X.device
    if dtype is None:
        dtype = X.dtype

    # Convert to correct device and data type
    X_ = X.to(device=device, dtype=dtype)
    Y_ = Y.to(device=device, dtype=dtype)
    X_bounds_ = X_bounds.to(device=device, dtype=dtype)

    # Check that dimensions match
    if not X.shape[0] == Y.shape[0]:
        raise ValueError(
            f"Shape of input and output data must match in the first dimension (got input shape {X.shape} and output shape {Y.shape})."
        )

    # Transforms
    input_transform = Normalize(d=X_.size(-1), bounds=X_bounds) if normalise else None
    output_transform = Standardize(m=Y.size(-1)) if standardise else None

    # Select model
    if mode == "joint":
        model = SingleTaskGP(
            X_,
            Y_,
            input_transform=input_transform,
            outcome_transform=output_transform,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif mode == "independent":
        model = ModelListGP(
            *[
                SingleTaskGP(
                    X_,
                    Y_[:, i].unsqueeze(1),
                    input_transform=input_transform,
                    outcome_transform=output_transform,
                )
                for i in range(Y_.shape[1])
            ]
        )
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        raise ValueError(f"Unknown mode {mode}. Must be 'joint' or 'list'.")

    # Return fitted model
    fit_gpytorch_mll(mll)
    return model
