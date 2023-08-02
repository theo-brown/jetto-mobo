"""
Tests for the acquisition module.

These tests just check that the functions run without error and return the correct kinds of output.
To check that the functions are actually doing what you expect, you will need to run them on some test data and check the output manually.
This is to ensure flexibility and robustness of the unit tests.

If you can think of reliable ways to test the outputs, submit a pull request!

# `jetto_mobo.acquisition.generate_initial_candidates`
- Check output is within bounds
- Check output shape
    - Number of candidates
    - Dimensions of candidates
- Check output dtype
- Check output device

# `jetto_mobo.acquisition.generate_trial_candidates`
- Check with different acquisition functions
- Check output shape
    - Number of candidates
    - Dimensions of candidates
- Check output is within bounds
- Device checks
- Check output dtype
"""

import pytest
import torch
from test_surrogate import bounds, x, y_1d, y_2d

from jetto_mobo.acquisition import (
    generate_initial_candidates,
    generate_trial_candidates,
    qNoisyExpectedHypervolumeImprovement,
    qNoisyExpectedImprovement,
)
from jetto_mobo.surrogate import fit_surrogate_model


@pytest.fixture
def batch_size():
    return 5


######################################################
# jetto_mobo.acquisition.generate_initial_candidates #
######################################################
def test_generate_initial_candidates_bounds(bounds, batch_size):
    candidates = generate_initial_candidates(bounds, batch_size)
    assert torch.all(candidates >= bounds[0, :])
    assert torch.all(candidates <= bounds[1, :])


def test_generate_initial_candidates_shape(bounds, batch_size):
    candidates = generate_initial_candidates(bounds, batch_size)
    assert candidates.shape == (batch_size, bounds.shape[1])


@pytest.mark.cuda
def test_generate_initial_candidates_device(bounds, batch_size):
    bounds = bounds.to(device="cpu")
    candidates = generate_initial_candidates(bounds, batch_size, device="cuda")
    assert candidates.is_cuda


def test_generate_initial_candidates_dtype(bounds, batch_size):
    bounds = bounds.to(dtype=torch.float64)
    candidates = generate_initial_candidates(bounds, batch_size, dtype=torch.float16)
    assert candidates.dtype == torch.float16


####################################################
# jetto_mobo.acquisition.generate_trial_candidates #
####################################################
@pytest.fixture
def model_2d(x, bounds, y_2d):
    return fit_surrogate_model(
        x,
        bounds,
        y_2d,
        device="cpu",
        dtype=torch.float64,
    )


@pytest.fixture
def model_1d(x, bounds, y_1d):
    return fit_surrogate_model(
        x,
        bounds,
        y_1d,
        device="cpu",
        dtype=torch.float64,
    )


def test_generate_trial_candidates_acquisition_functions(
    x, bounds, y_2d, model_2d, model_1d, batch_size
):
    candidates_1d = generate_trial_candidates(
        x,
        bounds,
        model_1d,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
    )

    candidates_2d = generate_trial_candidates(
        x,
        bounds,
        model_2d,
        acquisition_function=qNoisyExpectedHypervolumeImprovement,
        batch_size=batch_size,
        acqf_kwargs=dict(
            ref_point=torch.mean(
                y_2d, dim=0
            ),  # Compare to the average of the observed outputs
        ),
    )


@pytest.fixture
def candidates_1d(x, bounds, model_1d, batch_size):
    return generate_trial_candidates(
        x,
        bounds,
        model_1d,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
    )


@pytest.fixture
def candidates_2d(x, bounds, y_2d, model_2d, batch_size):
    return generate_trial_candidates(
        x,
        bounds,
        model_2d,
        acquisition_function=qNoisyExpectedHypervolumeImprovement,
        batch_size=batch_size,
        acqf_kwargs=dict(
            ref_point=torch.mean(
                y_2d, dim=0
            ),  # Compare to the average of the observed outputs
        ),
    )


def test_generate_trial_candidates_shape(x, batch_size, candidates_1d, candidates_2d):
    assert candidates_1d.shape == (batch_size, x.shape[1])
    assert candidates_2d.shape == (batch_size, x.shape[1])


def test_generate_trial_candidates_bounds(bounds, candidates_1d, candidates_2d):
    assert torch.all(candidates_1d >= bounds[0, :])
    assert torch.all(candidates_1d <= bounds[1, :])
    assert torch.all(candidates_2d >= bounds[0, :])
    assert torch.all(candidates_2d <= bounds[1, :])


@pytest.mark.cuda
def test_generate_trial_candidates_auto_device(x, bounds, batch_size, model_1d):
    # x on cpu, model on cuda
    model_1d_cuda = model_1d.to(device="cuda")
    candidates = generate_trial_candidates(
        x,
        bounds,
        model_1d_cuda,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
    )
    assert candidates.is_cuda


@pytest.mark.cuda
def test_generate_trial_candidates_explicit_device(x, bounds, batch_size, model_1d):
    # x on cpu, model on cuda
    candidates = generate_trial_candidates(
        x,
        bounds,
        model_1d,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
        device="cuda",
    )
    assert candidates.is_cuda


def test_generate_trial_candidates_auto_dtype(x, bounds, model_1d, batch_size):
    # x is float64, model is float32
    model_1d_float32 = model_1d.to(dtype=torch.float32)
    candidates = generate_trial_candidates(
        x,
        bounds,
        model_1d_float32,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
    )
    assert candidates.dtype == torch.float32


def test_generate_trial_candidates_explicit_dtype(x, bounds, model_1d, batch_size):
    candidates = generate_trial_candidates(
        x,
        bounds,
        model_1d,
        acquisition_function=qNoisyExpectedImprovement,
        batch_size=batch_size,
        dtype=torch.float32,
    )
    assert candidates.dtype == torch.float32
