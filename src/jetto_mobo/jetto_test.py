import os

import ecrh
import jetto_singularity
import torch

N_INITIAL_POINTS = 5
N_PARAMETERS = 12

x_train = torch.rand((N_INITIAL_POINTS, N_PARAMETERS))
processes = []
for i in range(N_INITIAL_POINTS):
    config_directory = f"jetto/runs/initial_{i}"
    ecrh.create_config(
        "jetto/templates/spr45-v9",
        config_directory,
        lambda xrho: ecrh.piecewise_linear(xrho, x_train[i]),
    )
    processes.append(
        jetto_singularity.run("jetto/images/sim.v220922.sif", config_directory)
    )

# Wait for all processes to complete
for p in processes:
    p.wait()
