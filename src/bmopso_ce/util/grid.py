"""Adaptive hypercube grid and leader selection utilities for multiobjective PSO.

Implements the Adaptive Grid mechanism proposed by Carlos A. Coello Coello,
Gregorio Toscano Pulido, and Maximino Salazar Lechuga (2004) in:
"Handling Multiple Objectives With Particle Swarm Optimization", IEEE Transactions
on Evolutionary Computation, Vol. 8, No. 3, pp. 256-279.
DOI: https://doi.org/10.1109/TEVC.2004.826067
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

__all__ = ["AdaptiveGrid"]


def _unique_in_appearance_order(
    ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique IDs in first-appearance order, inverse indices, and counts.

    Parameters
    ----------
    ids : np.ndarray
        1D integer identifiers of shape (N,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        unique IDs (K,), inverse index of each input (N,), and counts (K,).
    """
    unique_sorted, first_pos, inverse_sorted, counts_sorted = np.unique(
        ids,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    appearance_order = np.argsort(first_pos)
    unique_ids = unique_sorted[appearance_order]
    counts = counts_sorted[appearance_order]

    remap = np.empty(len(appearance_order), dtype=int)
    remap[appearance_order] = np.arange(len(appearance_order))
    inverse = remap[inverse_sorted]
    return unique_ids, inverse, counts


def _grouped_member_indices(
    inverse: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack member indices by group, preserving original order within each group.

    Parameters
    ----------
    inverse : np.ndarray
        Group index of each solution, shape (N,).
    counts : np.ndarray
        Population of each group, shape (K,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Concatenated member indices (N,) and start offset of each group (K,).
    """
    order = np.argsort(inverse, kind="stable")
    starts = np.empty(len(counts), dtype=int)
    starts[0] = 0
    if len(counts) > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    return order, starts


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

        f_min = np.min(f, axis=0)
        f_max = np.max(f, axis=0)
        range_m = f_max - f_min
        zero_range = range_m == 0.0

        buffer = range_m / (2.0 * self.n_grid)
        lower_bound = f_min - buffer
        upper_bound = f_max + buffer
        cell_width = (upper_bound - lower_bound) / self.n_grid
        safe_width = np.where(zero_range, 1.0, cell_width)

        coords = np.floor((f - lower_bound) / safe_width).astype(int)
        coords = np.clip(coords, 0, self.n_grid - 1)
        coords[:, zero_range] = self.n_grid // 2
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
        random_state: np.random.Generator | None = None,
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
        random_state : np.random.Generator | None, default=None
            NumPy Generator used for hypercube roulette and member draws.

        Returns
        -------
        np.ndarray
            Selected binary leader positions of shape (n_particles, n_var).
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        n_solutions = len(x)
        if n_solutions == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")
        if n_solutions == 1:
            return np.tile(x[0], (n_particles, 1))

        hypercube_ids = self.get_hypercube_ids(f)
        unique_cubes, inverse, counts = _unique_in_appearance_order(hypercube_ids)

        # Fitness is inversely proportional to hypercube population: fitness_i = 10.0 / N_i
        fitnesses = 10.0 / counts.astype(float)
        total_fitness = float(np.sum(fitnesses))

        if total_fitness > 0.0:
            probs = fitnesses / total_fitness
        else:
            probs = np.full(len(unique_cubes), 1.0 / len(unique_cubes))

        # Select hypercubes via Roulette Wheel Selection
        selected_cube_indices = rng.choice(len(unique_cubes), size=n_particles, p=probs)

        # For each selected hypercube, choose one solution uniformly at random
        members, starts = _grouped_member_indices(inverse, counts)
        selected_counts = counts[selected_cube_indices]
        local_idx = (rng.random(n_particles) * selected_counts).astype(int)
        selected_leader_indices = members[starts[selected_cube_indices] + local_idx]

        return x[selected_leader_indices]

    def prune_archive(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray,
        max_size: int,
        random_state: np.random.Generator | None = None,
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
        random_state : np.random.Generator | None, default=None
            NumPy Generator used to break crowded-hypercube ties.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Pruned (x, f, cv) arrays of length max_size.
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        num_to_remove = len(x) - max_size
        if num_to_remove <= 0:
            return x, f, cv

        # 1. Compute grid and hypercube grouping once (first-appearance cube order)
        hypercube_ids = self.get_hypercube_ids(f)
        _, inverse, counts = _unique_in_appearance_order(hypercube_ids)
        members, starts = _grouped_member_indices(inverse, counts)
        cube_members: List[List[int]] = [
            members[starts[i] : starts[i] + counts[i]].tolist() for i in range(len(counts))
        ]
        live_counts = counts.copy()

        victims: List[int] = []

        # 2. Sequential crowded-cube removal (each deletion updates densities)
        for _ in range(num_to_remove):
            max_pop = int(np.max(live_counts))
            most_crowded = np.flatnonzero(live_counts == max_pop)
            chosen = int(most_crowded[rng.integers(0, len(most_crowded))])
            local = int(rng.integers(0, live_counts[chosen]))
            victim_idx = cube_members[chosen].pop(local)
            live_counts[chosen] -= 1
            victims.append(victim_idx)

        # 3. Delete all victims in a single pass using boolean masking
        mask = np.ones(len(x), dtype=bool)
        mask[victims] = False

        return x[mask], f[mask], cv[mask]
