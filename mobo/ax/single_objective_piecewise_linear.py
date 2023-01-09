import subprocess
from typing import Iterable

import jetto_tools
import numpy as np
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from netCDF4 import Dataset


##########################
# Input parameterisation #
##########################
def piecewise_linear_ecrh(
    x: Iterable[float], parameters: Iterable[float]
) -> np.ndarray:
    if len(parameters) != 12:
        raise ValueError(f"Expected 12 parameters, got {len(parameters)}.")

    # On axis peak
    on_axis_peak_x = 0
    on_axis_peak_y = parameters[0]

    # On axis peak shaper
    on_axis_peak_end_x = parameters[1]
    on_axis_peak_end_y = parameters[2] * on_axis_peak_y

    # Minimum
    minimum_x = parameters[3]
    minimum_y = parameters[4]

    # Minimum shaper
    minimum_shaper_x = (minimum_x + on_axis_peak_end_x) / 2
    minimum_shaper_y = parameters[5] * minimum_y

    # Off-axis peak
    off_axis_peak_x = (minimum_x + parameters[6]) / 2
    off_axis_peak_y = parameters[7]

    # Off-axis shaper 2
    off_axis_shaper_2_x = (minimum_x + 2 * off_axis_peak_x) / 3
    off_axis_shaper_2_y = (
        parameters[8] * off_axis_peak_y + (1 - parameters[8]) * minimum_y
    )

    # Off-axis shaper 1
    off_axis_shaper_1_x = (2 * minimum_x + off_axis_peak_x) / 3
    off_axis_shaper_1_y = (
        parameters[9] * off_axis_shaper_2_y + (1 - parameters[9]) * minimum_y
    )

    # Turn-off
    turn_off_x = off_axis_peak_x + parameters[11]
    turn_off_y = 0

    # Turn-off shaper
    turn_off_shaper_x = (off_axis_peak_x + turn_off_x) / 2
    turn_off_shaper_y = parameters[10] * off_axis_peak_y

    # Collect into array
    node_xs = [
        on_axis_peak_x,
        on_axis_peak_end_x,
        minimum_shaper_x,
        minimum_x,
        off_axis_shaper_1_x,
        off_axis_shaper_2_x,
        off_axis_peak_x,
        turn_off_shaper_x,
        turn_off_x,
    ]
    node_ys = [
        on_axis_peak_y,
        on_axis_peak_end_y,
        minimum_shaper_y,
        minimum_y,
        off_axis_shaper_1_y,
        off_axis_shaper_2_y,
        off_axis_peak_y,
        turn_off_shaper_y,
        turn_off_y,
    ]

    return np.interp(x, node_xs, node_ys)


#################
# Cost function #
#################
def f1(profiles: Dataset, timetraces: Dataset):
    """q(0) - min(q)"""
    q = profiles["Q"][-1].data
    return q[0] - np.min(q)


def f2(profiles: Dataset, timetraces: Dataset, epsilon=0.3):
    """|| min(q) - (2 + epsilon) ||"""
    q = profiles["Q"][-1].data
    return np.abs(np.min(q) - (2 + epsilon))


def f3(profiles: Dataset, timetraces: Dataset):
    """argmin(q)"""
    q = profiles["Q"][-1].data
    return np.argmin(q)


def f4(profiles: Dataset, timetraces: Dataset, epsilon=0.03):
    """li - (0.25 + epsilon)"""
    li = timetraces["LI"][-1].data
    return np.abs(li - (0.25 + epsilon))


def f5(profiles: Dataset, timetraces: Dataset):
    """Fraction of locations where q is non-increasing
    Calculated by summing q[i] <= q[i-1] and dividing by len(q)"""
    q = profiles["Q"][-1].data
    return np.sum(q[1:] <= q[:-1]) / len(q)


def f6(profiles: Dataset, timetraces: Dataset):
    """Fraction of locations where dq/dr is non-increasing
    Calculated by summing dq[i] <= dq[i-1] and dividing by len(dq)"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    return np.sum(dq[1:] <= dq[:-1]) / len(q)


def f7(profiles: Dataset, timetraces: Dataset, value=3):
    """-dq at first point where q>=value and r >= argmin(q)"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    condition_1 = q >= value
    condition_2 = np.arange(len(q)) >= np.argmin(q)
    # Get index of element where both conditions are met
    i = np.where(condition_1 & condition_2)[0][0]
    return -dq[i]


def f8(profiles: Dataset, timetraces: Dataset):
    """-dq at first point where q>=4 and r >= argmin(q)"""
    return f7(profiles, timetraces, value=4)


def combined_cost_function(profiles: Dataset, timetraces: Dataset):
    return (
        0.5 * f1(profiles, timetraces)
        + 5 * f2(profiles, timetraces)
        + 6 * f3(profiles, timetraces)
        + 10 * f5(profiles, timetraces)
        + 10 * f6(profiles, timetraces)
        + 1 * f7(profiles, timetraces)
        + 2 * f8(profiles, timetraces)
    )


#######################
# Evaluation function #
#######################
JETTO_TEMPLATE = jetto_tools.template.from_directory("../../jetto/templates/spr45-v9")
JETTO_IMAGE = "../../jetto/images/sim.v220922.sif"
JETTO_OUTPUT_DIR = "../../jetto/runs"


def evaluate_piecewise_linear_ecrh(ecrh_parameters, run_name, output_dir):
    # Load the template's exfile
    exfile = jetto_tools.binary.read_binary_file(JETTO_TEMPLATE.extra_files["jetto.ex"])

    # Modify the exfile with the new ECRH profile
    new_exfile_path = f"{output_dir}/jetto.ex"
    xrho = exfile["XRHO"][0]
    exfile["QECE"][0] = piecewise_linear_ecrh(xrho, list(ecrh_parameters.values()))
    jetto_tools.binary.write_binary_exfile(exfile, new_exfile_path)

    # Create a new template with the modified exfile
    config = jetto_tools.config.RunConfig(JETTO_TEMPLATE)
    config.exfile = new_exfile_path
    config.export(output_dir)

    # Run the simulation
    process = subprocess.run(
        [
            "singularity",
            "exec",
            "--cleanenv",  # Run in a clean environment (no env variables etc)
            "--bind",
            "/tmp",
            "--bind",
            f"{output_dir}:/jetto/runs/{run_name}",  # Bind the output directory to the container's jetto run directory
            jetto_image,  # Container image
            "rjettov",
            "-x64",
            "-S",
            "-p0",
            "-n1",
            run_name,
            "build",
            "docker",  # Command to execute in container
        ]
    )

    # Get the results
    results = jetto_tools.results.JettoResults(path=output_dir)
    profiles = results.load_profiles()
    timetraces = results.load_timetraces()

    return {"cost": combined_cost_function(profiles, timetraces)}


######
# Ax #
######
ax_client = AxClient(
    generation_strategy=GenerationStrategy(
        steps=[
            # 1. Initialization step
            GenerationStep(
                model=Models.SOBOL,
                num_trials=1,  # How many trials should be produced from this generation step
            ),
            # 2. Bayesian optimization step
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
            ),
        ]
    )
)

ax_client.create_experiment(
    name="so_ecrh_piecewise_linear",
    parameters=[
        {
            "name": f"x{i}",
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for i in range(12)
    ],
    objectives={
        "cost": ObjectiveProperties(minimize=True),
    },
)

# Run experiment
for i in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=evaluate_piecewise_linear_ecrh(
            ecrh_parameters=parameters,
            run_name=f"piecewise_linear/{trial_index}",
            output_dir="../../jetto/runs/",
        ),
    )
