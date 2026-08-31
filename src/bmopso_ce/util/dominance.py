"""Constrained-dominance and Pareto sorting utilities.

Implements the Constrained-Dominance Principle proposed by Kalyanmoy Deb (2002)
for multiobjective optimization problems with inequality constraints.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

__all__ = ["dominates", "find_non_dominated_constrained"]


def dominates(
    f1: np.ndarray,
    f2: np.ndarray,
    cv1: float = 0.0,
    cv2: float = 0.0,
) -> bool:
    """Check if (f1, cv1) dominates (f2, cv2) by the Constrained-Dominance Principle (Deb, 2002).

    Constrained-Dominance Rules:
    1. If f1 is feasible (cv1 <= 0) and f2 is infeasible (cv2 > 0): f1 dominates f2.
    2. If f1 is infeasible (cv1 > 0) and f2 is feasible (cv2 <= 0): f1 does NOT dominate f2.
    3. If both are infeasible (cv1 > 0 and cv2 > 0): f1 dominates f2 if cv1 < cv2.
    4. If both are feasible (cv1 <= 0 and cv2 <= 0): f1 dominates f2 if f1 dominates f2
       in the standard Pareto sense (np.all(f1 <= f2) and np.any(f1 < f2)).

    Parameters
    ----------
    f1 : np.ndarray
        1D objective vector of the first solution.
    f2 : np.ndarray
        1D objective vector of the second solution.
    cv1 : float, default=0.0
        Total constraint violation of the first solution (0.0 if feasible).
    cv2 : float, default=0.0
        Total constraint violation of the second solution (0.0 if feasible).

    Returns
    -------
    bool
        True if the first solution dominates the second under constrained dominance.
    """
    v1_viol = cv1 > 0.0
    v2_viol = cv2 > 0.0

    if not v1_viol and v2_viol:
        return True
    if v1_viol and not v2_viol:
        return False
    if v1_viol and v2_viol:
        return bool(cv1 < cv2)
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def find_non_dominated_constrained(
    f: np.ndarray,
    cv: np.ndarray | None = None,
) -> np.ndarray:
    """Filter non-dominated solution indices applying the Constrained-Dominance Principle (Deb, 2002).

    Parameters
    ----------
    f : np.ndarray
        Objective matrix of shape (N, n_obj).
    cv : np.ndarray | None, default=None
        1D or 2D array of total constraint violations of shape (N,) or (N, 1).

    Returns
    -------
    np.ndarray
        1D integer array containing the indices of non-dominated solutions.
    """
    if cv is None or len(cv) == 0:
        return find_non_dominated(f)

    cv_1d = np.squeeze(cv)
    if cv_1d.ndim == 0:
        cv_1d = np.array([float(cv_1d)])

    # 1. If feasible solutions exist (cv <= 0), all infeasible solutions are dominated
    feasible_idx = np.where(cv_1d <= 0.0)[0]
    if len(feasible_idx) > 0:
        sub_idx = find_non_dominated(f[feasible_idx])
        return feasible_idx[sub_idx]

    # 2. If all solutions are infeasible, select those with minimum total violation (min CV)
    min_cv = np.min(cv_1d)
    min_cv_idx = np.where(np.isclose(cv_1d, min_cv))[0]
    sub_idx = find_non_dominated(f[min_cv_idx])
    return min_cv_idx[sub_idx]
