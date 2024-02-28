import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def write_to_file(
    output_file: Path,
    root_label: str,
    ecrh_parameters_batch: Optional[np.ndarray] = None,
    preconverged_ecrh: Optional[np.ndarray] = None,
    converged_ecrh: Optional[np.ndarray] = None,
    converged_q: Optional[np.ndarray] = None,
    objective_values: Optional[np.ndarray] = None,
    constraint_values: Optional[np.ndarray] = None,
    log_hypervolume: Optional[float] = None,
):
    """
    Utility function for writing data to hdf5 file.

    Creates ``output_file`` if it does not exist, and writes the given data to the file under the given root label.
    """
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)
    with h5py.File(output_file, "a") as f:
        if ecrh_parameters_batch is not None:
            f.create_dataset(
                f"{root_label}/ecrh_parameters", data=ecrh_parameters_batch
            )
        if preconverged_ecrh is not None:
            f.create_dataset(f"{root_label}/preconverged_ecrh", data=preconverged_ecrh)
        if converged_ecrh is not None:
            f.create_dataset(f"{root_label}/converged_ecrh", data=converged_ecrh)
        if converged_q is not None:
            f.create_dataset(f"{root_label}/converged_q", data=converged_q)
        if objective_values is not None:
            f.create_dataset(f"{root_label}/objective_values", data=objective_values)
        if constraint_values is not None:
            f.create_dataset(f"{root_label}/constraint_values", data=constraint_values)
        if log_hypervolume is not None:
            f[root_label].attrs["log_hypervolume"] = log_hypervolume
