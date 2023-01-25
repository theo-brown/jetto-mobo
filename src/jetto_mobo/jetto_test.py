import asyncio

import torch

from jetto_mobo import ecrh, jetto_subprocess

N_INITIAL_POINTS = 4
N_PARAMETERS = 12

ecrh_parameters = torch.rand((N_INITIAL_POINTS, N_PARAMETERS))

config = {str(i): f"jetto/runs/test/{i}" for i in range(N_INITIAL_POINTS)}
for i, config_directory in enumerate(config.values()):
    ecrh.create_config(
        "jetto/templates/spr45-v9",
        config_directory,
        lambda xrho: ecrh.piecewise_linear(xrho, ecrh_parameters[i]),
    )

results = asyncio.run(jetto_subprocess.run_many("jetto/images/sim.v220922.sif", config))

for i, (stdout, stderr, returncode) in enumerate(results):
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
