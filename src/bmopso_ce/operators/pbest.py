"""Personal best (pbest) update operators following Coello Coello et al. (2004)."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from bmopso_ce.util.dominance import dominates_mask

__all__ = ["update_personal_bests"]


def update_personal_bests(
    pbest_x: np.ndarray,
    pbest_f: np.ndarray,
    pbest_cv: np.ndarray | None,
    x: np.ndarray,
    f: np.ndarray,
    cv: np.ndarray,
    random_state: np.random.Generator | None = None,
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
    random_state : np.random.Generator | None, default=None
        NumPy Generator used for incomparable pbest coin flips.

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
    rng = random_state if random_state is not None else np.random.default_rng()

    cv_new = np.asarray(cv, dtype=float).reshape(-1)
    cv_old = np.asarray(new_pbest_cv, dtype=float).reshape(-1)

    new_dominates_old = dominates_mask(f, new_pbest_f, cv_new, cv_old)
    old_dominates_new = dominates_mask(new_pbest_f, f, cv_old, cv_new)
    incomparable = (~new_dominates_old) & (~old_dominates_new)

    replace = new_dominates_old.copy()
    n_incomparable = int(np.count_nonzero(incomparable))
    if n_incomparable > 0:
        # Same RNG consumption order as the previous per-particle loop (index 0..N-1)
        replace[incomparable] = rng.random(n_incomparable) < 0.5

    new_pbest_x[replace] = x[replace]
    new_pbest_f[replace] = f[replace]
    new_pbest_cv[replace] = cv_new[replace]

    return new_pbest_x, new_pbest_f, new_pbest_cv
