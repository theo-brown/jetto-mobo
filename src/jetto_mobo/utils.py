from typing import Iterable, Optional

import h5py
import numpy as np


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


def pad_1d(a: Iterable[Optional[np.ndarray]], pad_value: float = np.nan) -> np.ndarray:
    """Pad a ragged sequence of 1D arrays."""
    row_length = max([len(r) if r is not None else 0 for r in a])
    a_padded = np.full((len(a), row_length), np.nan)
    for i, r in enumerate(a):
        if r is not None:
            a_padded[i, : len(r)] = np.array(r)
    return a_padded
