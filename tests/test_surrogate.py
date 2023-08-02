"""
Tests for the surrogate module.

These tests just check that the functions run without error and return the correct kinds of output.
To check that the functions are actually doing what you expect, you will need to run them on some test data and check the output manually.
This is to ensure flexibility and robustness of the unit tests.

If you can think of reliable ways to test the outputs, submit a pull request!

# fit_surrogate_model
- Check model device
- Check model dtype
- Check mode
    - independent => ModelListGP
    - joint => SingleTaskGP
"""

import pytest
import torch
from torch.distributions.uniform import Uniform

from jetto_mobo.surrogate import fit_surrogate_model


@pytest.fixture
def bounds():
    # bounds has shape (n_x_dims, 2)
    # In this case, n_x_dims = 3:
    # -1 <= x[0] <= 1
    # -10 <= x[1] <= 10
    # 1 <= x[2] <= 10
    return torch.tensor([[-1, -10, 1], [1, -1, 10]], dtype=torch.float64)


@pytest.fixture
def batch_size():
    return 5


@pytest.fixture
def x(bounds):
    # x has shape (n_x_points, n_x_dims)
    n_x_points = 4
    torch.manual_seed(0)
    return Uniform(bounds[0], bounds[1]).sample([n_x_points])


@pytest.fixture
def y_1d(x):
    return x.sum(dim=1).unsqueeze(-1)


@pytest.fixture
def y_2d(x, y_1d):
    # y has shape (n_x_points, n_y_dims)
    # In this case, n_y_dims = 2:
    # y[0] = x[0] + x[1] + x[2]
    # y[1] = x[0] * x[1] * x[2]
    return torch.cat([y_1d, x.prod(dim=1).unsqueeze(-1)], dim=1)


@pytest.mark.cuda
def test_fit_surrogate_model_explicit_device(x, y_2d, bounds):
    model = fit_surrogate_model(x, bounds, y_2d, device="cuda")
    assert next(model.parameters()).is_cuda


@pytest.mark.cuda
def test_fit_surrogate_model_auto_device(x, y_2d, bounds):
    x_cuda = x.to(device="cuda")
    model = fit_surrogate_model(x_cuda, bounds, y_2d)
    assert next(model.parameters()).is_cuda


def test_fit_surrogate_model_explicit_dtype(x, y_2d, bounds):
    model = fit_surrogate_model(x, bounds, y_2d, dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float32


def test_fit_surrogate_model_auto_dtype(x, y_2d, bounds):
    x_float32 = x.to(dtype=torch.float32)
    model = fit_surrogate_model(x_float32, bounds, y_2d)
    assert next(model.parameters()).dtype == torch.float32


def test_fit_surrogate_model_independent(x, y_2d, bounds):
    model = fit_surrogate_model(x, bounds, y_2d, mode="independent")
    assert model.__class__.__name__ == "ModelListGP"


def test_fit_surrogate_model_joint(x, y_2d, bounds):
    model = fit_surrogate_model(x, bounds, y_2d, mode="joint")
    assert model.__class__.__name__ == "SingleTaskGP"
