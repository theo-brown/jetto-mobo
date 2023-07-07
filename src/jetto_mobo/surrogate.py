from typing import Literal, Union

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def fit_surrogate_model(
    X: torch.Tensor,
    X_bounds: torch.Tensor,
    Y: torch.Tensor,
    device: Union[str, torch.device, int],
    dtype: Union[str, torch.dtype],
    model: Literal["independent", "joint"] = "joint",
) -> Union[SingleTaskGP, ModelListGP]:
    """
    Fit a surrogate model to the data.

    Parameters
    ----------
    X : torch.Tensor
        Input data. Shape is (N, D), where N is the number of data points and D is the number of input dimensions.
    X_bounds : torch.Tensor
        Bounds on the input space, shape (D, 2). bounds[i, 0] is the lower bound and bounds[i, 1] is the upper bound for the ith input.
    Y : torch.Tensor
        Output data. Shape is (N, M), where N is the number of data points and M is the number of output dimensions.
    device : Union[str, torch.device, int]
        Device to use for fitting the surrogate model.
    dtype : Union[str, torch.dtype]
        Data type to use for fitting the surrogate model.
    model : Literal["independent", "joint"], default = "joint"
        Type of surrogate model to use. If "joint", all outputs are modelled jointly. If "independent", each output is modelled independently.

    Returns
    -------
    Union[SingleTaskGP, ModelListGP]
        Fitted surrogate model.
    """
    # Check that dimensions match
    if not X.shape[0] == Y.shape[0]:
        raise ValueError(
            f"Shape of input and output data must match in the first dimension (got input shape {X.shape} and output shape {Y.shape})."
        )

    # Convert to correct device and data type
    normalised_X_tensor = normalize(X, X_bounds).to(device=device, dtype=dtype)
    Y_tensor = Y.to(device=device, dtype=dtype)

    # Select model
    if model == "joint":
        model = SingleTaskGP(
            normalised_X_tensor,
            Y_tensor,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif model == "independent":
        model = ModelListGP(
            *[
                SingleTaskGP(
                    normalised_X_tensor,
                    Y_tensor[:, i].unsqueeze(1),
                )
                for i in range(Y_tensor.shape[1])
            ]
        )
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        raise ValueError(f"Unknown model type {model}. Must be 'joint' or 'list'.")

    # Return fitted model
    fit_gpytorch_mll(mll)
    return model
