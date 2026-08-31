"""Integration tests validating BMOPSO-CE on pymoo-binary-problems suite."""

from typing import Any
import numpy as np
import pytest
from pymoo.optimize import minimize
from pymoo_binary_problems import BinaryProblem, MKP, MOFS, MOSCP, MSTSP, MUBQP

from bmopso_ce import BMOPSO_CE


def test_bmopso_on_custom_binary_problem() -> None:
    """Verify that BMOPSO-CE solves custom BinaryProblem implementations."""
    class CustomProblem(BinaryProblem):
        def _evaluate(
            self,
            x: np.ndarray,
            out: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            out["F"] = np.column_stack([np.sum(x, axis=1), np.sum(~x, axis=1)])

    problem = CustomProblem(n_var=8, n_obj=2)
    algo = BMOPSO_CE(n_particles=15)
    res = minimize(problem, algo, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2


def test_bmopso_on_mkp() -> None:
    """Validate BMOPSO-CE execution on the Multiple Knapsack Problem (MKP)."""
    profits = np.array([12.0, 18.0, 25.0, 30.0, 42.0, 15.0])
    weights = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 6.0])
    capacities = np.array([30.0, 35.0])

    problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)
    algorithm = BMOPSO_CE(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert res.X.shape[1] == 6 * 2  # 6 items * 2 knapsacks = 12 bits
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_mubqp() -> None:
    """Validate BMOPSO-CE execution on Multiobjective Unconstrained Binary Quadratic Problem (MUBQP)."""
    problem = MUBQP.from_random(n_var=15, n_obj=2, density=0.7, seed=42)
    algorithm = BMOPSO_CE(n_particles=20, w=0.6, c1=1.4, c2=1.4)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert res.X.shape[1] == 15
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_moscp() -> None:
    """Validate BMOPSO-CE execution on Multiobjective Set Covering Problem (MOSCP)."""
    problem = MOSCP.from_random(n_elements=10, n_subsets=15, n_obj=2, density=0.3, seed=42)
    algorithm = BMOPSO_CE(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert res.X.shape[1] == 15
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_mstsp() -> None:
    """Validate BMOPSO-CE execution on Multiobjective Traveling Salesman Problem (MSTSP)."""
    dist_mat1 = np.array([
        [0.0, 2.0, 9.0, 10.0],
        [1.0, 0.0, 6.0, 4.0],
        [15.0, 7.0, 0.0, 8.0],
        [6.0, 3.0, 12.0, 0.0],
    ])
    dist_mat2 = np.array([
        [0.0, 5.0, 2.0, 7.0],
        [4.0, 0.0, 8.0, 3.0],
        [2.0, 9.0, 0.0, 5.0],
        [1.0, 6.0, 4.0, 0.0],
    ])
    problem = MSTSP(cost_matrices=[dist_mat1, dist_mat2])
    algorithm = BMOPSO_CE(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert res.X.shape[1] == 4 * 4  # 16 bits
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_mofs() -> None:
    """Validate BMOPSO-CE execution on Multiobjective Feature Selection (MOFS)."""
    x_data = np.random.RandomState(42).randn(40, 10)
    y_data = (x_data[:, 0] + x_data[:, 1] > 0).astype(int)

    problem = MOFS(x_data, y_data, min_features=1)
    algorithm = BMOPSO_CE(n_particles=15, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res.X is not None
    assert len(res.X) > 0
    assert res.X.shape[1] == 10
    assert np.all(np.isin(res.X, [True, False, 0, 1]))
    assert res.F is not None
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()
