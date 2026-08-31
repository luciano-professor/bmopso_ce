"""Personal best (pbest) update operators following Coello Coello et al. (2004)."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from bmopso_ce.util.dominance import dominates

__all__ = ["update_personal_bests"]


def update_personal_bests(
    pbest_x: np.ndarray,
    pbest_f: np.ndarray,
    pbest_cv: np.ndarray | None,
    x: np.ndarray,
    f: np.ndarray,
    cv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update personal best positions using Constrained-Dominance and Coello Coello (2004) rules.

    Following Coello Coello et al. (2004):
    1. If current position strictly dominates pbest in memory: pbest = current position.
    2. If pbest strictly dominates current position: keep pbest.
    3. If neither of them is dominated by the other (mutually non-dominated / incomparable):
       one of them is chosen randomly with equal probability (p = 0.5).

    Parameters
    ----------
    pbest_x : np.ndarray
        Current personal best binary positions (n_particles, n_var).
    pbest_f : np.ndarray
        Current personal best objectives (n_particles, n_obj).
    pbest_cv : np.ndarray | None
        Current personal best constraint violations (n_particles,).
    x : np.ndarray
        New binary positions (n_particles, n_var).
    f : np.ndarray
        New objective values (n_particles, n_obj).
    cv : np.ndarray
        New constraint violations (n_particles,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated (pbest_x, pbest_f, pbest_cv).
    """
    n_particles = len(x)
    new_pbest_x = pbest_x.copy()
    new_pbest_f = pbest_f.copy()
    new_pbest_cv = (
        pbest_cv.copy() if pbest_cv is not None else np.zeros(n_particles, dtype=float)
    )

    for i in range(n_particles):
        f_new = f[i]
        f_old = new_pbest_f[i]
        cv_new = float(cv[i])
        cv_old = float(new_pbest_cv[i])

        if dominates(f_new, f_old, cv_new, cv_old):
            new_pbest_x[i] = x[i].copy()
            new_pbest_f[i] = f_new.copy()
            new_pbest_cv[i] = cv_new
        elif not dominates(f_old, f_new, cv_old, cv_new):
            # Incomparable: randomly choose between current position and pbest (Coello Coello et al., 2004)
            if np.random.rand() < 0.5:
                new_pbest_x[i] = x[i].copy()
                new_pbest_f[i] = f_new.copy()
                new_pbest_cv[i] = cv_new

    return new_pbest_x, new_pbest_f, new_pbest_cv
