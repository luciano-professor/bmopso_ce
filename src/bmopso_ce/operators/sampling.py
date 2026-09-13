"""Sigmoid velocity activation and binary sampling operators."""

from __future__ import annotations

import numpy as np

__all__ = ["sigmoid", "sample_binary_positions"]


def sigmoid(v: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid activation function on continuous velocities.

    Parameters
    ----------
    v : np.ndarray
        Continuous velocity array of arbitrary shape.

    Returns
    -------
    np.ndarray
        Array of probabilities in the range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-v))


def sample_binary_positions(
    v: np.ndarray,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample boolean decision variables based on sigmoid velocity probabilities.

    Parameters
    ----------
    v : np.ndarray
        Continuous velocity matrix of shape (n_particles, n_var).
    random_state : np.random.Generator | None, default=None
        NumPy Generator used to draw uniform samples against sigmoid probabilities.

    Returns
    -------
    np.ndarray
        Boolean matrix of binary positions of shape (n_particles, n_var).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    probs = sigmoid(v)
    rand_matrix = rng.random(v.shape)
    return rand_matrix < probs
