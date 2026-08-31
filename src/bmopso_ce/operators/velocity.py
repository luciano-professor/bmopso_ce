"""Velocity update operators with inertia, cognitive, and social forces."""

from __future__ import annotations

import numpy as np

__all__ = ["update_velocity"]


def update_velocity(
    v: np.ndarray,
    x: np.ndarray,
    pbest_x: np.ndarray,
    gbest: np.ndarray,
    w: float,
    c1: float = 1.49,
    c2: float = 1.49,
    v_max: float = 4.0,
) -> np.ndarray:
    """Update particle velocities combining inertia, cognitive, and social components.

    Parameters
    ----------
    v : np.ndarray
        Current continuous velocities of shape (n_particles, n_var).
    x : np.ndarray
        Current binary positions of shape (n_particles, n_var).
    pbest_x : np.ndarray
        Personal best binary positions of shape (n_particles, n_var).
    gbest : np.ndarray
        Social leader binary positions of shape (n_particles, n_var).
    w : float
        Current inertia weight.
    c1 : float, default=1.49
        Cognitive acceleration coefficient.
    c2 : float, default=1.49
        Social acceleration coefficient.
    v_max : float, default=4.0
        Maximum allowable absolute velocity bound.

    Returns
    -------
    np.ndarray
        Clamped updated continuous velocity matrix.
    """
    n_particles, n_var = x.shape
    r1 = np.random.rand(n_particles, n_var)
    r2 = np.random.rand(n_particles, n_var)

    cognitive = c1 * r1 * (pbest_x.astype(float) - x.astype(float))
    social = c2 * r2 * (gbest.astype(float) - x.astype(float))

    new_v = w * v + cognitive + social
    return np.clip(new_v, -v_max, v_max)
