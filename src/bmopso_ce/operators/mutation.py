"""Non-linear turbulence and bit-flip mutation operators."""

from __future__ import annotations

import numpy as np

__all__ = ["apply_mutation"]


def apply_mutation(
    x: np.ndarray,
    progress: float,
    mutation_rate: float | None = 0.5,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply non-linear mutation/turbulence operator to binary positions.

    Based on the formulation by Coello Coello et al. (2004):
        P_mut(t) = (1 - currentgen / totgen) ** (5 / mutation_rate)

    With mutation_rate=0.5, exponent is 10, producing high exploratory bit-flips
    in early generations and smooth convergence in later stages.

    Parameters
    ----------
    x : np.ndarray
        Binary positions matrix of shape (n_particles, n_var).
    progress : float
        Current optimization progress in [0.0, 1.0].
    mutation_rate : float | None, default=0.5
        Bit-flip mutation rate. If None, defaults to 1 / n_var.
    random_state : np.random.Generator | None, default=None
        NumPy Generator used to choose particles and bits.

    Returns
    -------
    np.ndarray
        Mutated binary positions matrix of shape (n_particles, n_var).
    """
    n_particles, n_var = x.shape
    pm: float = (1.0 / n_var) if mutation_rate is None else mutation_rate

    if pm <= 0.0 or progress >= 1.0:
        return x

    exponent: float = 5.0 / pm
    p_particle_mut: float = float((1.0 - progress) ** exponent)

    rng = random_state if random_state is not None else np.random.default_rng()
    particle_mask = rng.random(n_particles) < p_particle_mut
    bit_mask = rng.random((n_particles, n_var)) < pm
    mutate_mask = particle_mask[:, np.newaxis] & bit_mask

    return np.logical_xor(x, mutate_mask)
