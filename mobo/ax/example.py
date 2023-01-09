import numpy as np
import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.plot.render import plot_config_to_html
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render

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

# Function that we're trying to optimise
def evaluation_function(parameters):
    x = np.array([parameters.get(f"x{i}") for i in range(6)])
    return {
        "hartmann6": hartmann6(x),
        "l2norm": np.sqrt((x**2).sum()),
    }


# Objectives
objectives = {
    "hartmann6": ObjectiveProperties(minimize=True),
}

# Parameters
parameters = []
for i in range(6):
    parameters.append(
        {
            "name": f"x{i}",
            "type": "range",
            "bounds": [0.0, 1.0],
        }
    )

# Constraints
parameter_constraints = [
    "x1 + x2 <= 2.0",
]
outcome_constraints = [
    "l2norm <= 1.25",
]

# Define experiment
ax_client.create_experiment(
    name="hartmann_test_experiment",
    parameters=parameters,
    objectives=objectives,
    parameter_constraints=parameter_constraints,
    outcome_constraints=outcome_constraints,
)

# Run experiment
for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=evaluation_function(parameters),
    )

best_parameters, values = ax_client.get_best_parameters()
means, covariances = values

print(f"Best parameters: {best_parameters}")
print(f"Values: {means}")
print(f"True minimum: {hartmann6.fmin}")
