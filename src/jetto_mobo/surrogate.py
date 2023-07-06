from typing import Union

import torch
from botorch import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from jetto_mobo.configuration import SurrogateConfig


def fit_surrogate_model(
    X: torch.Tensor,
    X_bounds: torch.Tensor,
    Y: torch.Tensor,
    surrogate_config: SurrogateConfig,
) -> Union[GPyTorchModel, ModelListGPyTorchModel]:
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
    surrogate_config : SurrogateConfig
        Surrogate model configuration.

    Returns
    -------
    botorch.models.model.Model
        Fitted surrogate model.
    """
    # Check that dimensions match
    if not X.shape[0] == Y.shape[0]:
        raise ValueError(
            f"Shape of input and output data must match in the first dimension (got input shape {X.shape} and output shape {Y.shape})."
        )

    # Convert to correct device and data type
    normalised_X = normalize(X, X_bounds).to(
        device=surrogate_config.device, dtype=surrogate_config.dtype
    )
    Y_ = Y.to(device=surrogate_config.device, dtype=surrogate_config.dtype)

    # Select model
    if surrogate_config.model == "SingleTaskGP":
        model = SingleTaskGP(
            normalised_X,
            Y_,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif surrogate_config.model == "ModelListGP":
        model = ModelListGP(
            *[
                SingleTaskGP(
                    normalised_X,
                    Y_[:, i].unsqueeze(1),
                )
                for i in range(Y_.shape[1])
            ]
        )
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    elif surrogate_config.model == "MultiTaskGP":
        model = MultiTaskGP(
            normalised_X,
            Y_,
            task_feature=-1,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    else:
        raise ValueError(f"Unknown model type {surrogate_config.model}.")

    # Return fitted model
    fit_gpytorch_mll(mll)
    return model
