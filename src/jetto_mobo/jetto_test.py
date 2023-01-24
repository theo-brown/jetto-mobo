import asyncio

import ecrh
import jetto_subprocess
import torch

N_INITIAL_POINTS = 1
N_PARAMETERS = 12

x_train = torch.rand((N_INITIAL_POINTS, N_PARAMETERS))

run_names = [f"initial_{i}" for i in range(N_INITIAL_POINTS)]
config_directories = [f"jetto/runs/{run_name}" for run_name in run_names]
for i, config_directory in enumerate(config_directories):
    ecrh.create_config(
        "jetto/templates/spr45-v9",
        config_directory,
        lambda xrho: ecrh.piecewise_linear(xrho, x_train[i]),
    )

stdouts, stderrs, returncodes = asyncio.run(
    jetto_subprocess.run_many(
        "jetto/images/sim.v220922.sif", run_names, config_directories
    )
)

for i, (stdout, stderr, returncode) in enumerate(zip(stdouts, stderrs, returncodes)):
    print(f"Process {i}")
    print("=============")
    print("STDOUT:")
    print(stdout)
    print()
    print("STDERR:")
    print(stderr)
    print()
    print(f"RETURNCODE: {returncode}")
    print()
