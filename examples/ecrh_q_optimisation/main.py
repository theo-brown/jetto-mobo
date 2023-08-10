import argparse

import numpy as np
import torch

from jetto_mobo import acquisition, surrogate

from .ecrh_inputs import marsden_piecewise_linear, marsden_piecewise_linear_bounds
from .evaluation import evaluate, write_to_file

parser = argparse.ArgumentParser()
parser.add_argument("--jetto_template", type=str, default="../../jetto/templates/spr45")
parser.add_argument(
    "--jetto_image", type=str, default="../../jetto/images/sim.v220922.sif"
)
parser.add_argument("--jetto_timelimit", type=int, default=10400)
parser.add_argument("--jetto_fail_value", type=int, default=0)
parser.add_argument("--n_objectives", type=int, default=7)
parser.add_argument("--n_xrho_points", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--n_iterations", type=int, default=16)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--dtype", type=str, default="torch.float64")
parser.add_argument("--output_dir", type=str, default="data")
args = parser.parse_args()

# Bounds have to be tensor for fitting surrogate
parameter_bounds = torch.tensor(marsden_piecewise_linear_bounds)

# Generate initial data
print("Generating initial data...")
# Sobol sampling for initial candidates
ecrh_parameters = acquisition.generate_initial_candidates(
    bounds=parameter_bounds,
    n=args.batch_size,
    device=args.device,
    dtype=args.dtype,
)
# Save initial candidates to file
write_to_file(
    output_file=args.output_dir / "results.h5",
    root_label="initialisation",
    ecrh_parameters_batch=ecrh_parameters.detach().cpu().numpy(),
    preconverged_ecrh=np.array(
        [
            marsden_piecewise_linear(xrho=np.linspace(0, 1, args.n_xrho), parameters=p)
            for p in ecrh_parameters.detach().cpu().numpy()
        ]
    ),
)
# Evaluate initial candidates
(
    converged_ecrh,
    converged_q,
    objective_values,
) = evaluate(
    ecrh_parameters_batch=ecrh_parameters.detach().cpu().numpy(),
    batch_directory=args.output_dir / "0_initialisation",
    jetto_template=args.jetto_template,
    jetto_image=args.jetto_image,
    jetto_timelimit=args.jetto_timelimit,
    jetto_fail_value=args.jetto_fail_value,
    n_xrho=args.n_xrho_points,
    n_objectives=args.n_objectives,
)
# Save evaluated results to file
write_to_file(
    output_file=args.output_dir / "results.h5",
    root_label="initialisation",
    converged_ecrh=converged_ecrh,
    converged_q=converged_q,
    objective_values=objective_values,
)
# Train surrogate model
objective_values = torch.tensor(objective_values)
model = surrogate.fit_surrogate_model(
    X=ecrh_parameters,
    X_bounds=parameter_bounds,
    Y=objective_values,
    device=args.device,
    dtype=args.dtype,
    mode="joint",
)

for optimisation_step in range(1, args.n_iterations + 1):
    # Generate trial candidates
    print(f"Optimisation step {optimisation_step}")
    new_ecrh_parameters = acquisition.generate_trial_candidates(
        observed_inputs=ecrh_parameters,
        bounds=parameter_bounds,
        model=model,
        acquisition_function=acquisition.qNoisyExpectedHypervolumeImprovement,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        mode="sequential",
        acqf_kwargs={"ref_point": torch.zeros(objective_values.shape[1])},
    )
    write_to_file(
        f"optimisation_step_{optimisation_step}",
        ecrh_parameters_batch=new_ecrh_parameters.detach().cpu().numpy(),
        preconverged_ecrh=np.array(
            [
                marsden_piecewise_linear(
                    xrho=np.linspace(0, 1, args.n_xrho), parameters=p
                )
                for p in new_ecrh_parameters.detach().cpu().numpy()
            ]
        ),
    )

    # Evaluate candidates
    (
        converged_ecrh,
        converged_q,
        new_objective_values,
    ) = evaluate(
        ecrh_parameters_batch=new_ecrh_parameters.detach().cpu().numpy(),
        batch_directory=args.output_directory / str(optimisation_step),
        jetto_template=args.jetto_template,
        jetto_image=args.jetto_image,
        jetto_timelimit=args.jetto_timelimit,
        jetto_fail_value=args.jetto_fail_value,
        n_xrho=args.n_xrho_points,
        n_objectives=args.n_objectives,
    )
    write_to_file(
        output_file=args.output_dir / "results.h5",
        root_label=f"optimisation_step_{optimisation_step}",
        converged_ecrh=converged_ecrh,
        converged_q=converged_q,
        objective_values=new_objective_values,
    )

    # Update surrogate model
    ecrh_parameters = torch.cat([ecrh_parameters, new_ecrh_parameters])
    objective_values = torch.cat(
        [objective_values, torch.tensor(new_objective_values)]
    )  # Have to convert new_objective_values to tensor, because it is a np.ndarray output from reading JettoResults
    model = surrogate.fit_surrogate_model(
        X=ecrh_parameters,
        X_bounds=parameter_bounds,
        Y=objective_values,
        device=args.device,
        dtype=args.dtype,
        mode="joint",
    )
