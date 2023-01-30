from typing import Optional

import h5py
import torch


def save_to_hdf5(
    filename: str, dataset_name: str, data: np.ndarray, overwrite: bool = False
) -> None:
    with h5py.File(filename, "a") as f:
        if overwrite and dataset_name in f:
            f[dataset_name][...] = data
        else:
            f.create_dataset(dataset_name, data=data)


def load_from_hdf5(
    filename: str,
    dataset_name: str,
) -> np.ndarray:
    with h5py.File(filename, "r") as f:
        return np.ndarray(f[dataset_name][:])
