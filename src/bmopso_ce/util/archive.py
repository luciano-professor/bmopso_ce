"""External Non-Dominated Archive management for multiobjective PSO.

Maintains Pareto-optimal solutions across generations, enforces capacity limits
via Coello Coello (2004) hypercube density pruning, and handles social leader selection
via Adaptive Grid Hypercube Roulette (Coello Coello et al., 2004; Deb, 2002).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from pymoo.core.population import Population

from bmopso_ce.util.dominance import find_non_dominated_constrained
from bmopso_ce.util.grid import AdaptiveGrid

__all__ = ["NonDominatedArchive"]


class NonDominatedArchive:
    """External archive of non-dominated Pareto solutions based on Adaptive Hypercubes.

    Parameters
    ----------
    max_size : int | None, default=200
        Maximum capacity of the external archive. If None, the archive capacity is unlimited.
    n_grid : int, default=30
        Number of grid subdivisions along each objective dimension (Coello Coello et al., 2004).
    """

    def __init__(self, max_size: int | None = 200, n_grid: int = 30) -> None:
        self.max_size: int | None = max_size
        self.n_grid: int = n_grid
        self.grid: AdaptiveGrid = AdaptiveGrid(n_grid=n_grid)
        self._x: np.ndarray | None = None
        self._f: np.ndarray | None = None
        self._cv: np.ndarray | None = None

    @property
    def x(self) -> np.ndarray | None:
        """Binary decision matrix of archive solutions (N, n_var)."""
        return self._x

    @property
    def f(self) -> np.ndarray | None:
        """Objective values matrix of archive solutions (N, n_obj)."""
        return self._f

    @property
    def cv(self) -> np.ndarray | None:
        """Total constraint violation vector of archive solutions (N,)."""
        return self._cv

    def __len__(self) -> int:
        """Return the current number of non-dominated solutions in the archive."""
        return len(self._x) if self._x is not None else 0

    def is_empty(self) -> bool:
        """Check if the archive is empty."""
        return len(self) == 0

    def update(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray | None = None,
    ) -> bool:
        """Update archive with candidate solutions, filtering by dominance and pruning by hypercube density.

        Parameters
        ----------
        x : np.ndarray
            Binary candidate solutions of shape (N, n_var).
        f : np.ndarray
            Objective values matrix of shape (N, n_obj).
        cv : np.ndarray | None, default=None
            Total constraint violations of shape (N,). If None, defaults to 0.0.

        Returns
        -------
        bool
            True if the archive contents or non-dominated front changed, False otherwise.
        """
        if cv is None:
            cv = np.zeros(len(x), dtype=float)
        cv_1d = np.squeeze(cv)
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        old_x = self._x
        old_f = self._f

        if self._x is None or self._f is None or self._cv is None:
            combined_x = x
            combined_f = f
            combined_cv = cv_1d
        else:
            combined_x = np.vstack([self._x, x])
            combined_f = np.vstack([self._f, f])
            combined_cv = np.concatenate([self._cv, cv_1d])

        # 1. Filter non-dominated solutions using Constrained-Dominance Principle (Deb, 2002)
        front_idx = find_non_dominated_constrained(combined_f, combined_cv)
        non_dom_x = combined_x[front_idx]
        non_dom_f = combined_f[front_idx]
        non_dom_cv = combined_cv[front_idx]

        # 2. Prune by Hypercube Density if exceeding max_size (Coello Coello et al., 2004)
        if self.max_size is not None and len(non_dom_x) > self.max_size:
            self._x, self._f, self._cv = self.grid.prune_archive(
                x=non_dom_x,
                f=non_dom_f,
                cv=non_dom_cv,
                max_size=self.max_size,
            )
        else:
            self._x = non_dom_x
            self._f = non_dom_f
            self._cv = non_dom_cv

        # 3. Check if archive changed
        if old_f is None:
            return True
        if len(self._f) != len(old_f):
            return True
        if not np.array_equal(self._f, old_f) or not np.array_equal(self._x, old_x):
            return True
        return False

    def select_leaders(self, n_particles: int) -> np.ndarray:
        """Select social leaders (gbest) for each particle via Adaptive Hypercube Grid Roulette.

        Parameters
        ----------
        n_particles : int
            Number of particles in the swarm.

        Returns
        -------
        np.ndarray
            Selected binary leader positions of shape (n_particles, n_var).
        """
        if self._x is None or self._f is None or len(self._x) == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")

        return self.grid.select_leaders(self._x, self._f, n_particles)

    def get_hypercube_ids(self) -> np.ndarray:
        """Compute hypercube IDs of all solutions currently in the archive."""
        if self._f is None or len(self._f) == 0:
            return np.array([])
        return self.grid.get_hypercube_ids(self._f)

    def to_population(self) -> Population:
        """Convert archive contents into a pymoo Population object."""
        if self._x is None or self._f is None:
            return Population()
        if self._cv is not None:
            return Population.new(
                X=self._x,
                F=self._f,
                CV=self._cv[:, np.newaxis],
            )
        return Population.new(X=self._x, F=self._f)
