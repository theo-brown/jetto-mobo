from typing import Optional

import h5py
import torch


def save_tensor(
    filename: str, dataset_name: str, data: torch.tensor, overwrite: bool = False
) -> None:
    with h5py.File(filename, "a") as f:
        if overwrite and dataset_name in f:
            f[dataset_name][...] = data.detach().cpu().numpy()
        else:
            f.create_dataset(dataset_name, data=data.detach().cpu().numpy())


def load_tensor(
    filename: str,
    dataset_name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.tensor:
    with h5py.File(filename, "r") as f:
        return torch.tensor(f[dataset_name][:], device=device, dtype=dtype)
