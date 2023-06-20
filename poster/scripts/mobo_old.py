import h5py
import numpy as np
from netCDF4 import Dataset

from jetto_mobo.objective import vector_objective

with Dataset("../data/benchmark/profiles.CDF") as profiles:
    with Dataset("../data/benchmark/timetraces.CDF") as timetraces:
        benchmark_ecrh = profiles["QECE"][-1]
        benchmark_q = profiles["Q"][-1]
        benchmark_objective_value = vector_objective(profiles, timetraces)


n_steps = 5
batch_size = 5
n_objectives = 7
n_parameters = 12
n_radial_points = 150

parameters = np.zeros((n_steps * batch_size, n_parameters))
objective_values = np.zeros((n_steps * batch_size, n_objectives))
converged_ecrh = np.zeros((n_steps * batch_size, 150))
converged_q = np.zeros((n_steps * batch_size, 150))

with h5py.File("../data/mobo/ga_piecewise_linear.hdf5") as hdf5:
    for i in np.arange(n_steps):
        for j in np.arange(batch_size):
            if not np.any(np.isnan(hdf5[f"bayesopt/{i+1}/value"][j])):
                objective_values[i * batch_size + j] = hdf5[f"bayesopt/{i+1}/value"][j]
                parameters[i * batch_size + j] = hdf5[
                    f"bayesopt/{i+1}/ecrh_parameters"
                ][j]
                converged_ecrh[i * batch_size + j] = hdf5[
                    f"bayesopt/{i+1}/converged_ecrh"
                ][j]
                converged_q[i * batch_size + j] = hdf5[f"bayesopt/{i+1}/converged_q"][j]
print(
    f"Loaded results from {n_steps} optimisation steps, each with {batch_size} points (totalling {n_steps * batch_size} evaluations)."
)
print(f"Input space: {n_parameters} dimensions")
print(f"Output (objective) space: {n_objectives} dimensions")
