"""Adaptive hypercube grid and leader selection utilities for multiobjective PSO.

Implements the Adaptive Grid mechanism proposed by Carlos A. Coello Coello,
Gregorio Toscano Pulido, and Maximino Salazar Lechuga (2004) in:
"Handling Multiple Objectives With Particle Swarm Optimization", IEEE Transactions
on Evolutionary Computation, Vol. 8, No. 3, pp. 256-279.
DOI: https://doi.org/10.1109/TEVC.2004.826067
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

__all__ = ["AdaptiveGrid"]


class AdaptiveGrid:
    """Adaptive hypercube grid partitioning the objective space into discrete cells.

    Following Coello Coello et al. (2004):
    1. The objective space is bounded by the minimum and maximum objective values of the
       solutions currently in the archive, expanded by an adaptive buffer.
    2. Each objective dimension is subdivided into `n_grid` equal divisions.
    3. Each solution is mapped into a discrete hypercube coordinate (k_1, k_2, ..., k_M)
       where k_j in [0, n_grid - 1].
    4. Occupied hypercubes receive fitness inversely proportional to the number of solutions
       they contain: fitness_i = 10.0 / N_i.
    5. Social leaders (gbest) are selected by performing Roulette Wheel Selection over the
       occupied hypercubes and then picking a solution uniformly at random from the selected hypercube.
    6. When the archive exceeds capacity, solutions are pruned from the most crowded hypercubes.

    Parameters
    ----------
    n_grid : int, default=30
        Number of subdivisions (grid partitions) along each objective dimension.
    """

    def __init__(self, n_grid: int = 30) -> None:
        if n_grid < 1:
            raise ValueError(f"n_grid must be at least 1, got {n_grid}.")
        self.n_grid: int = n_grid

    def compute_grid_coordinates(self, f: np.ndarray) -> np.ndarray:
        """Compute discrete grid coordinates for each solution across all objectives.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix of shape (N, n_obj).

        Returns
        -------
        np.ndarray
            Integer matrix of grid coordinates of shape (N, n_obj), where each entry
            is in [0, n_grid - 1].
        """
        n_points, n_obj = f.shape
        if n_points == 0:
            return np.empty((0, n_obj), dtype=int)

        coords = np.zeros((n_points, n_obj), dtype=int)
        f_min = np.min(f, axis=0)
        f_max = np.max(f, axis=0)

        for m in range(n_obj):
            range_m = float(f_max[m] - f_min[m])
            if range_m == 0.0:
                # If all points share the same objective value, assign to center cell
                coords[:, m] = self.n_grid // 2
            else:
                # Boundary buffer to ensure extreme points lie comfortably inside boundary cells
                buffer = range_m / (2.0 * self.n_grid)
                lower_bound = f_min[m] - buffer
                upper_bound = f_max[m] + buffer
                cell_width = (upper_bound - lower_bound) / self.n_grid

                c = np.floor((f[:, m] - lower_bound) / cell_width).astype(int)
                coords[:, m] = np.clip(c, 0, self.n_grid - 1)

        return coords

    def get_hypercube_ids(self, f: np.ndarray) -> np.ndarray:
        """Compute unique hypercube integer IDs for each solution.

        Uses multidimensional coordinate mapping via np.unique to guarantee
        exact hypercube indexing without integer overflow for any number of objectives.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix of shape (N, n_obj).

        Returns
        -------
        np.ndarray
            1D integer array of shape (N,) containing the hypercube ID of each point.
        """
        n_points, n_obj = f.shape
        if n_points == 0:
            return np.empty((0,), dtype=int)

        coords = self.compute_grid_coordinates(f)
        _, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        return inverse_indices

    def select_leaders(
        self,
        x: np.ndarray,
        f: np.ndarray,
        n_particles: int,
    ) -> np.ndarray:
        """Select social leaders (gbest) for swarm particles using Coello Coello (2004) Grid Roulette.

        Parameters
        ----------
        x : np.ndarray
            Binary solution positions in the archive of shape (N, n_var).
        f : np.ndarray
            Objective matrix in the archive of shape (N, n_obj).
        n_particles : int
            Number of particles requiring a leader.

        Returns
        -------
        np.ndarray
            Selected binary leader positions of shape (n_particles, n_var).
        """
        n_solutions = len(x)
        if n_solutions == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")
        if n_solutions == 1:
            return np.tile(x[0], (n_particles, 1))

        hypercube_ids = self.get_hypercube_ids(f)

        # Group solution indices by hypercube ID
        cubes: Dict[int, List[int]] = {}
        for idx, cube_id in enumerate(hypercube_ids):
            cubes.setdefault(int(cube_id), []).append(idx)

        unique_cubes = list(cubes.keys())
        # Fitness is inversely proportional to hypercube population: fitness_i = 10.0 / N_i
        fitnesses = np.array([10.0 / len(cubes[cid]) for cid in unique_cubes], dtype=float)
        total_fitness = float(np.sum(fitnesses))

        if total_fitness > 0.0:
            probs = fitnesses / total_fitness
        else:
            probs = np.full(len(unique_cubes), 1.0 / len(unique_cubes))

        # Select hypercubes via Roulette Wheel Selection
        selected_cube_indices = np.random.choice(len(unique_cubes), size=n_particles, p=probs)

        # For each selected hypercube, choose one solution randomly (uniform random)
        selected_leader_indices = np.zeros(n_particles, dtype=int)
        for i, cube_idx in enumerate(selected_cube_indices):
            cid = unique_cubes[cube_idx]
            members = cubes[cid]
            selected_leader_indices[i] = np.random.choice(members)

        return x[selected_leader_indices]

    def prune_archive(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray,
        max_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prune archive solutions when exceeding max capacity by targeting the most crowded hypercubes.

        Following Coello Coello et al. (2004):
        Identifies hypercubes with the highest density (most solutions) and eliminates
        a randomly selected solution from one of the most crowded hypercubes until the
        archive reaches max_size.

        Parameters
        ----------
        x : np.ndarray
            Binary solution positions (N, n_var).
        f : np.ndarray
            Objective matrix (N, n_obj).
        cv : np.ndarray
            Constraint violation array (N,).
        max_size : int
            Maximum allowable archive capacity.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Pruned (x, f, cv) arrays of length max_size.
        """
        num_to_remove = len(x) - max_size
        if num_to_remove <= 0:
            return x, f, cv

        # 1. Compute grid and hypercube grouping once
        hypercube_ids = self.get_hypercube_ids(f)
        cubes: Dict[int, List[int]] = {}
        for idx, cube_id in enumerate(hypercube_ids):
            cubes.setdefault(int(cube_id), []).append(idx)

        victims: List[int] = []

        # 2. Select solutions to eliminate by updating in-memory hypercube memberships
        for _ in range(num_to_remove):
            max_pop = max(len(members) for members in cubes.values() if len(members) > 0)
            most_crowded_cids = [
                cid for cid, members in cubes.items() if len(members) == max_pop
            ]

            chosen_cid = most_crowded_cids[np.random.randint(0, len(most_crowded_cids))]
            victim_idx = int(np.random.choice(cubes[chosen_cid]))

            # Remove from hypercube virtual membership and mark for deletion
            cubes[chosen_cid].remove(victim_idx)
            victims.append(victim_idx)

        # 3. Delete all victims in a single pass using boolean masking
        mask = np.ones(len(x), dtype=bool)
        mask[victims] = False

        return x[mask], f[mask], cv[mask]
