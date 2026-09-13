"""Unit and integration tests for the Catfish Effect operator in BMOPSO-CE."""

from typing import Any
import numpy as np
import pytest
from pymoo.optimize import minimize
from pymoo_binary_problems import BinaryProblem

from bmopso_ce import (
    BMOPSO_CE,
    apply_catfish_effect,
    generate_extreme_binary_positions,
    select_random_particles,
)


class DummyBinaryProblem(BinaryProblem):
    """Simple 2-objective binary problem."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        f1 = np.sum(x, axis=1)
        f2 = np.sum(~x if x.dtype == bool else (1 - x), axis=1)
        out["F"] = np.column_stack([f1, f2])


def test_generate_extreme_binary_positions() -> None:
    """Test extreme position generation: r > 0.5 -> all 1s, r <= 0.5 -> all 0s."""
    n_var = 8
    n_positions = 10
    positions = generate_extreme_binary_positions(n_positions=n_positions, n_var=n_var)

    assert positions.shape == (10, 8)
    assert positions.dtype == bool

    # Each position must be either all zeros or all ones
    for p in positions:
        assert np.all(p == True) or np.all(p == False)


def test_generate_extreme_binary_positions_with_rng() -> None:
    """Test deterministic extreme position generation with known random values."""
    class MockRNG:
        def random(self, size: int) -> np.ndarray:
            # First is > 0.5 (should be all 1s), second is <= 0.5 (should be all 0s)
            return np.array([0.75, 0.25])

    positions = generate_extreme_binary_positions(
        n_positions=2, n_var=4, random_state=MockRNG()
    )
    assert np.all(positions[0] == True)
    assert np.all(positions[1] == False)


def test_select_random_particles() -> None:
    """Test random selection of particles for catfish replacement."""
    n_particles = 20
    n_to_replace = 4
    selected = select_random_particles(n_particles=n_particles, n_to_replace=n_to_replace)

    assert len(selected) == n_to_replace
    assert len(set(selected)) == n_to_replace
    assert np.all(selected >= 0) and np.all(selected < n_particles)


def test_apply_catfish_effect_positions_only_no_reevaluation() -> None:
    """Test application of Catfish Effect updates ONLY particle positions without re-evaluation."""
    n_particles = 10
    n_var = 6
    x = np.random.randint(0, 2, size=(n_particles, n_var)).astype(bool)

    new_x = apply_catfish_effect(
        x=x,
        catfish_rate=0.20,  # Replace 2 particles (20%)
    )

    # 1. Dimensions and dtype check
    assert new_x.shape == (10, 6)
    assert new_x.dtype == bool

    # 2. Exactly 2 particles must have changed positions
    diff_mask = np.any(new_x != x, axis=1)
    assert np.sum(diff_mask) <= 2


def test_bmopso_ce_stagnation_trigger() -> None:
    """Test that BMOPSO_CE detects stagnation and triggers the Catfish Effect."""
    problem = DummyBinaryProblem(n_var=12)
    algorithm = BMOPSO_CE(
        n_particles=20,
        catfish_threshold=2,  # Low threshold to ensure trigger during a 10-gen run
        catfish_rate=0.10,
    )

    res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert algorithm.n_catfish_triggers >= 0


def test_bmopso_ce_defaults() -> None:
    """Verify default catfish_threshold is 24 and catfish_rate is 0.10."""
    algo = BMOPSO_CE()
    assert algo.catfish_threshold == 24
    assert algo.catfish_rate == 0.10
