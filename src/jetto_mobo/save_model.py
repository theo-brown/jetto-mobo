import os

import h5py
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood


def train_model_posthumously(hdf5_file, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    with h5py.File(hdf5_file, "r") as f:
        n_completed_bayesopt_steps = f["/"].attrs["n_completed_bayesopt_steps"]
        batch_size = f["/"].attrs["batch_size"]
        if f["/"].attrs["ecrh_function"] == "ga_piecewise_linear":
            n_ecrh_parameters = 12
            ecrh_parameter_bounds = torch.tensor(
                [
                    [0, 0.05, 0.01, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [1, 1, 0.09, 1, 1, 0.29, 0.9, 0.9, 1, 0.75, 0.9, 0.45],
                ],
                dtype=dtype,
                device=device,
            )
        else:
            raise NotImplementedError

        ecrh_parameters = torch.empty(
            n_completed_bayesopt_steps + 1,
            batch_size,
            n_ecrh_parameters,
            device=device,
            dtype=dtype,
        )
        value = torch.empty(
            n_completed_bayesopt_steps + 1, batch_size, device=device, dtype=dtype
        )

        ecrh_parameters[0] = torch.from_numpy(f["initialisation/ecrh_parameters"][:])
        value[0] = torch.from_numpy(f["initialisation/value"][:]).squeeze()

        for i in range(1, n_completed_bayesopt_steps + 1):
            ecrh_parameters[i - 1] = torch.from_numpy(
                f[f"bayesopt/{i}/ecrh_parameters"][:]
            )
            value[i - 1] = torch.from_numpy(f[f"bayesopt/{i}/value"][:]).squeeze()

        ecrh_parameters = ecrh_parameters.reshape(-1, n_ecrh_parameters)
        value = value.reshape(-1, 1)
        value[value.isnan()] = f["/"].attrs["jetto_fail_value"]

    model = SingleTaskGP(normalize(ecrh_parameters, ecrh_parameter_bounds), value)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll)

    checkpoint = {
        "model": model.state_dict(),
        "training_data": hdf5_file,
        "ecrh_parameter_bounds": ecrh_parameter_bounds,
        "ecrh_parameters": ecrh_parameters,
        "observed_values": value,
    }
    torch.save(checkpoint, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    train_model_posthumously(
        hdf5_file=args.hdf5_file,
        output_file=args.output_file,
    )
