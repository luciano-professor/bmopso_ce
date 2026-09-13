"""Catfish Effect Operator for Binary Multi-Objective PSO.

In the multi-objective optimization (MOO) approach of BMOPSO-CE, an External Archive (EA)
is used to store all best non-dominated solutions discovered throughout the search.

To keep the addition of the Catfish Effect as simple and computationally efficient as possible:
1. Particles from the current swarm are chosen randomly for replacement (instead of performing
   complex multi-objective comparisons to find the worst ones).
2. For each catfish particle, its initial extreme position is determined probabilistically:
   - A random number r in [0, 1] is generated.
   - If r > 0.5: the particle is placed at the upper extreme (all dimensions d = 1).
   - If r <= 0.5: the particle is placed at the lower extreme (all dimensions d = 0).
3. No re-evaluation is performed upon introducing the catfish particles; they enter the swarm
   and are naturally evaluated during the regular evolutionary search cycle.

References
----------
- Souza, L. S., Prudêncio, R. B. C., & Barros, F. A. (2014).
  "Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies",
  Anais do XV Workshop de Testes e Tolerância a Falhas (WTF 2014), SBC. DOI: 10.5753/wtf.2014.22943
- Chuang, L. Y., Tsai, S. W., & Yang, C. H. (2011).
  "Improved binary particle swarm optimization using catfish effect for feature selection",
  Expert Systems with Applications, 38(10), 12699-12707. DOI: 10.1016/j.eswa.2011.04.057
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_catfish_effect",
    "generate_extreme_binary_positions",
    "select_random_particles",
]


def select_random_particles(
    n_particles: int,
    n_to_replace: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly select particle indices from the current swarm for catfish replacement.

    Because an External Archive (EA) preserves all best non-dominated solutions,
    replacing randomly chosen particles provides high exploration diversity with minimal
    computational overhead.

    Parameters
    ----------
    n_particles : int
        Total number of particles in the swarm.
    n_to_replace : int
        Number of particles to replace.
    random_state : np.random.Generator | None, default=None
        NumPy Generator used to sample particle indices.

    Returns
    -------
    np.ndarray
        Array of selected particle indices without replacement.
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    n_to_replace = min(max(1, n_to_replace), n_particles)
    return rng.choice(n_particles, size=n_to_replace, replace=False)


def generate_extreme_binary_positions(
    n_positions: int,
    n_var: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate extreme binary positions for catfish particles.

    Following Souza et al. (WTF 2014):
    For each catfish particle, a random number r in [0, 1] is generated:
    - If r > 0.5: the particle is placed at the upper extreme (all dimensions d = 1).
    - If r <= 0.5: the particle is placed at the lower extreme (all dimensions d = 0).

    Parameters
    ----------
    n_positions : int
        Number of extreme binary vectors to generate.
    n_var : int
        Number of binary decision variables (bits).
    random_state : np.random.Generator | None, default=None
        NumPy Generator used to draw the per-particle extreme-point coin flip.

    Returns
    -------
    np.ndarray
        Array of boolean extreme binary vectors of shape (n_positions, n_var).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    r = rng.random(n_positions)

    positions = np.zeros((n_positions, n_var), dtype=bool)
    # If r > 0.5: all dimensions set to 1. If r <= 0.5: all dimensions set to 0.
    positions[r > 0.5] = True

    return positions


def apply_catfish_effect(
    x: np.ndarray,
    catfish_rate: float = 0.10,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Execute the Catfish Effect perturbation operator on a stagnated swarm.

    Randomly selects a fraction of particles (`catfish_rate`) and updates ONLY their
    positions to extreme binary points ([0, 0, ...] or [1, 1, ...]) without re-evaluating them.

    Parameters
    ----------
    x : np.ndarray
        Current binary particle positions of shape (n_particles, n_var).
    catfish_rate : float, default=0.10
        Fraction of particles to be randomly replaced by catfish particles (default: 10%).
    random_state : np.random.Generator | None, default=None
        NumPy Generator used to choose particles and extreme positions.

    Returns
    -------
    np.ndarray
        Updated particle positions matrix of shape (n_particles, n_var).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    n_particles, n_var = x.shape
    n_to_replace = max(1, int(round(catfish_rate * n_particles)))

    # 1. Randomly choose particles of current swarm to be replaced
    selected_idx = select_random_particles(
        n_particles=n_particles,
        n_to_replace=n_to_replace,
        random_state=rng,
    )

    # 2. Generate extreme binary positions for the catfish particles based on r > 0.5 vs r <= 0.5
    new_catfish_x = generate_extreme_binary_positions(
        n_positions=len(selected_idx),
        n_var=n_var,
        random_state=rng,
    )

    # 3. Update ONLY particle positions (no re-evaluation, velocities preserved)
    updated_x = x.copy()
    updated_x[selected_idx] = new_catfish_x

    return updated_x
