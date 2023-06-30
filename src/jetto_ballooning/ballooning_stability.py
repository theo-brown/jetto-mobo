import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cvxpy as cp  # Used for constrained lower bound fit
import f90nml
import numpy as np
from pyrokinetics import Pyro
from scipy.optimize import minimize


def run_gs2_ballooning_stability(
    gs2_output_directory: Path,
    jetto_output_directory: Path,
    psi_n: float,
    gs2_template: Path,
    gs2_image: Path,
    ballstab_knobs: dict = {},
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load JETTO output
    pyro = Pyro(
        eq_file=jetto_output_directory / "jetto.eqdsk_out",
        kinetics_file=jetto_output_directory / "profiles.CDF",
        gk_code="GS2",
        gk_file=gs2_template,
    )
    # Load the flux surface
    pyro.load_local(psi_n=psi_n, local_geometry="Miller")

    # Set ballooning stability parameters
    pyro.gk_input.data["ballstab_knobs"] = {
        "beta_mul": ballstab_knobs.get("beta_mul", 2),
        "beta_div": ballstab_knobs.get("beta_div", 10),
        "shat_min": ballstab_knobs.get("shat_min", 0.1),
        "shat_max": ballstab_knobs.get("shat_max", 3.5),
        "n_beta": ballstab_knobs.get("n_beta", 40),
        "n_shat": ballstab_knobs.get("n_shat", 80),
    }

    # Generate GS2 input file
    base_filename = "ideal_ballooning_stability"
    gs2_input_file = gs2_output_directory / f"{base_filename}.gs2.in"
    pyro.write_gk_file(file_name=gs2_input_file, gk_code="GS2")

    # Run GS2
    logging.info("Launching GS2 Singularity container...")
    subprocess.run(
        [
            "singularity",
            "exec",
            "--bind",
            f"{gs2_output_directory}:/tmp/{gs2_output_directory}",
            gs2_image,
            "ideal_ball",
            gs2_input_file,
        ],
        stdout=subprocess.PIPE,
    )
    logging.info("GS2 complete.")

    # Extract results
    input_nml = f90nml.read(gs2_input_file)

    # Geometric properties
    R = input_nml["theta_grid_parameters"]["r_geo"]
    q = input_nml["theta_grid_parameters"]["qinp"]

    # Get alpha, s_hat for the JETTO-simulated design
    beta_prime = np.abs(input_nml["theta_grid_eik_knobs"]["beta_prime_input"])
    alpha = R * q**2 * beta_prime
    s_hat = input_nml["theta_grid_eik_knobs"]["s_hat_input"]

    # Get alpha, s_hat from the grid search
    s_hat_grid = np.loadtxt(gs2_output_directory / f"{base_filename}.gs2.ballstab_shat")
    beta_prime_grid = np.abs(
        np.loadtxt(gs2_output_directory / f"{base_filename}.gs2.ballstab_bp")
    )
    alpha_grid = R * q**2 * beta_prime_grid

    # Get stability from the grid search
    stability_grid = np.loadtxt(
        gs2_output_directory / f"{base_filename}.gs2.ballstab_2d"
    )

    return alpha, s_hat, alpha_grid, s_hat_grid, stability_grid


def get_stability_boundary(
    alpha_grid, s_hat_grid, stability_grid
) -> Tuple[np.ndarray, np.ndarray]:
    # Get indices of all non-zero elements
    i, j = np.nonzero(stability_grid)
    # Get unique indices for each x value
    # i are the unique x locations, mask is a boolean array to select corresponding y locatoins
    i, mask = np.unique(i, return_index=True)
    j = j[mask]

    # Get alpha, s_hat values along the boundary
    yy, xx = np.meshgrid(s_hat_grid, alpha_grid)
    alpha_boundary = xx[i, j]
    s_hat_boundary = yy[i, j]
    return alpha_boundary, s_hat_boundary


def fit_polynomial_lower_bound(
    alpha_grid, s_hat_grid, stability_grid, degree=4, margin=0
) -> float:
    x, y = get_stability_boundary(alpha_grid, s_hat_grid, stability_grid)

    # Polynomial basis expansion
    x_bar = np.vander(x, degree + 1, increasing=True)

    # Polynomial coefficients
    w = cp.Variable(x_bar.shape[1])
    # Least squares fit
    objective = cp.Minimize(cp.sum_squares(x_bar @ w - y))
    # Constrained to lie below data
    constraints = [x_bar @ w <= y - margin]
    # CVX solve
    curve_fit = cp.Problem(objective, constraints)
    result = curve_fit.solve(time_limit=5, max_iter=int(1e6))

    return w.value


def polynomial(x, w) -> np.ndarray:
    degree = w.shape[0] - 1
    return np.vander(x, degree + 1, increasing=True) @ w


def get_euclidean_distance_of_point_to_polynomial(x, y, w) -> np.ndarray:
    # Euclidean distance
    distance = lambda x_: np.sqrt((polynomial(x_, w) - y) ** 2 + (x_ - x) ** 2)
    result = minimize(distance, 0)
    return np.array(result.fun)


def get_xy_distance_of_point_to_polynomial(x, y, w) -> np.ndarray:
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Get y distance
    y_on_polynomial = polynomial(x, w)
    y_distance = np.abs(y - y_on_polynomial)

    # Get x distance
    result = minimize(lambda x_: np.abs(y - polynomial(x_, w)), 0)
    x_distance = np.abs(x - result.x)

    return np.array([x_distance, y_distance]).flatten()


def compute_ballooning_stability_margin(
    psi_n: Iterable[float],
    jetto_output_directory: Path,
    gs2_output_directory: Optional[Path] = None,
    gs2_template: Optional[Path] = None,
    gs2_image: Optional[Path] = None,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    # TODO: add args for gs2 parameters

    # Distance metric
    supported_distance_metrics = ["euclidean", "xy"]
    if distance_metric not in supported_distance_metrics:
        raise ValueError(
            f"Unrecognised distance metric: {distance_metric};"
            f" expected one of {supported_distance_metrics}."
        )

    # Directory structure
    if gs2_output_directory is None:
        gs2_output_directory = jetto_output_directory / "gs2_out"
    if gs2_template is None:
        # TODO: make this a relative path
        gs2_template = Path(
            "/home/theo/Documents/jetto-mobo/gs2/templates/gs2_template.in"
        )
    if gs2_image is None:
        # TODO: make this a relative path
        gs2_image = Path("/home/theo/Documents/jetto-mobo/gs2/images/gs2.simg")

    # Make directories
    output_directories = [
        gs2_output_directory / f"psi_n_{psi_n:.3f}" for psi_n in psi_n
    ]
    for output_directory in output_directories:
        os.makedirs(output_directory, exist_ok=True)

    # Initialise output array
    if distance_metric == "euclidean":
        margin = np.zeros(len(psi_n))
    elif distance_metric == "xy":
        margin = np.zeros((len(psi_n), 2))

    # Run GS2
    # TODO: parallelise?
    for i, (psi_n, output_directory) in enumerate(zip(psi_n, output_directories)):
        logging.info(f"Computing ballooning stability margin for psi_n = {psi_n}...")
        (
            alpha,
            s_hat,
            alpha_grid,
            s_hat_grid,
            stability_grid,
        ) = run_gs2_ballooning_stability(
            gs2_output_directory=output_directory,
            jetto_output_directory=jetto_output_directory,
            psi_n=psi_n,
            gs2_template=gs2_template,
            gs2_image=gs2_image,
        )
        w = fit_polynomial_lower_bound(
            alpha_grid, s_hat_grid, stability_grid, degree=5, margin=0.01
        )
        if distance_metric == "euclidean":
            margin[i] = get_euclidean_distance_of_point_to_polynomial(alpha, s_hat, w)
        elif distance_metric == "xy":
            margin[i] = get_xy_distance_of_point_to_polynomial(alpha, s_hat, w)

        logging.info(
            f"Finished computing ballooning stability margin for psi_n={psi_n}."
        )
    return margin


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("## Stability margin computation (Euclidean) ##")
    margin = compute_ballooning_stability_margin(
        psi_n=[0.49, 0.36, 0.35],
        jetto_output_directory=Path(
            "/home/theo/Documents/jetto-mobo/data/benchmark"
        ),  # TODO: make this a relative path
    )
    print(margin)

    print("## Stability margin computation (XY) ##")
    margin = compute_ballooning_stability_margin(
        psi_n=[0.49, 0.36, 0.35],
        jetto_output_directory=Path(
            "/home/theo/Documents/jetto-mobo/data/benchmark"
        ),  # TODO: make this a relative path
        distance_metric="xy",
    )
    print(margin)
