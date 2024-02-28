from typing import Optional

import numpy as np


def get_pareto_dominant_mask(
    objective_values: np.ndarray,
    constraint_values: Optional[np.ndarray] = None,
    lower_bounds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a mask that selects only Pareto-optimal solutions

    Parameters
    ----------
    objective_values : np.ndarray
        An n_points x n_objectives array.
    constraint_values : np.ndarray, optional
        An n_points x n_constraints array. If provided, points with objective values that violate the constraints are excluded.
    lower_bounds : np.ndarray, optional
        An n_objectives array of lower bounds for each objective. If provided, points with objective values lower than the lower bounds are excluded.

    Returns
    -------
    is_dominant
        An n_points boolean array, True where points are Pareto optimal, satisfy the constraints, and are above the lower bounds.
    """
    if lower_bounds is not None:
        if lower_bounds.shape[0] != objective_values.shape[1]:
            raise ValueError(
                f"lower_bounds has shape {lower_bounds.shape}, but expected {objective_values.shape[1]}"
            )

    is_dominant = np.zeros(objective_values.shape[0], dtype=bool)
    for i, objective_value in enumerate(objective_values):
        strictly_better_in_one_objective = (objective_values > objective_value).any(
            axis=1
        )
        at_least_as_good_in_all_objectives = (objective_values >= objective_value).all(
            axis=1
        )
        # A point is Pareto-dominated if there's a point that we could move to that
        # improves performance in one objective without losing performance in any
        # other objective
        # A point is Pareto-dominant if there are no points that dominate it
        is_dominant[i] = ~np.any(
            at_least_as_good_in_all_objectives & strictly_better_in_one_objective
        )

    if lower_bounds is not None:
        is_dominant &= np.all(objective_values > lower_bounds, axis=1)
    if constraint_values is not None:
        is_dominant &= np.all(
            constraint_values <= 0, axis=1
        )  # Constraints are negative if satisfied

    return is_dominant
