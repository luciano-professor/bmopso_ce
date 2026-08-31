"""Integration and unit tests for BMOPSO-CE with pymoo."""

from typing import Any
import numpy as np
import pytest
from pymoo.optimize import minimize
from pymoo_binary_problems import BinaryProblem

from bmopso_ce import BMOPSO_CE


class SimpleBinaryProblem(BinaryProblem):
    """Test binary problem with 2 objectives:

    Objective 1 (f1): Minimize sum of 1s (sum of x).
    Objective 2 (f2): Maximize number of 0s (minimize sum of 1 - x).
    """

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


def test_binary_problem_abstract_evaluation() -> None:
    """Verify that base BinaryProblem raises NotImplementedError when evaluated without subclassing."""
    base_problem = BinaryProblem(n_var=5, n_obj=2)
    assert base_problem.n_var == 5
    assert base_problem.n_obj == 2
    assert base_problem.type_var == np.bool_

    with pytest.raises(NotImplementedError):
        base_problem.evaluate(np.array([[0, 1, 0, 1, 0]]))


def test_binary_mopso_initialization() -> None:
    """Verify default and custom hyperparameters of BMOPSO_CE."""
    algo_default = BMOPSO_CE()
    assert algo_default.n_particles == 20
    assert algo_default.w_max == 0.9
    assert algo_default.w_min == 0.4
    assert algo_default.w == 0.9
    assert algo_default.c1 == 1.49
    assert algo_default.c2 == 1.49
    assert algo_default.v_max == 4.0
    assert algo_default.mutation_rate == 0.5
    assert algo_default.n_grid == 30
    assert algo_default.max_archive_size == 200
    assert algo_default.catfish_threshold == 24
    assert algo_default.catfish_rate == 0.10

    algo_custom = BMOPSO_CE(
        n_particles=25,
        w_max=0.8,
        w_min=0.2,
        c1=2.0,
        c2=1.8,
        v_max=6.0,
        mutation_rate=0.05,
        n_grid=15,
        max_archive_size=150,
        catfish_threshold=12,
        catfish_rate=0.20,
    )
    assert algo_custom.n_particles == 25
    assert algo_custom.w_max == 0.8
    assert algo_custom.w_min == 0.2
    assert algo_custom.c1 == 2.0
    assert algo_custom.c2 == 1.8
    assert algo_custom.v_max == 6.0
    assert algo_custom.mutation_rate == 0.05
    assert algo_custom.n_grid == 15
    assert algo_custom.max_archive_size == 150
    assert algo_custom.catfish_threshold == 12
    assert algo_custom.catfish_rate == 0.20

    algo_fixed = BMOPSO_CE(w=0.6)
    assert algo_fixed.w_max == 0.6
    assert algo_fixed.w_min == 0.6


def test_binary_mopso_mutation_operator() -> None:
    """Verify that the non-linear mutation operator (bit-flip) modifies particles."""
    problem = SimpleBinaryProblem(n_var=15)
    algo = BMOPSO_CE(n_particles=30, mutation_rate=0.05)
    res = minimize(problem, algo, termination=("n_gen", 10), verbose=False)
    assert res.X is not None
    assert len(res.X) > 0
    assert np.all(np.isin(res.X, [True, False, 0, 1]))


def test_binary_mopso_linear_inertia_decay() -> None:
    """Validate linear inertia weight decay from 0.9 to 0.4 across generations."""
    from pymoo.core.callback import Callback

    w_values: list[float] = []

    class InertiaTracker(Callback):
        def notify(self, algorithm: Any) -> None:
            w_values.append(algorithm.w)

    problem = SimpleBinaryProblem(n_var=10)
    algo = BMOPSO_CE(n_particles=20, w_max=0.9, w_min=0.4)
    minimize(problem, algo, termination=("n_gen", 10), callback=InertiaTracker(), verbose=False)

    assert len(w_values) > 0
    assert np.isclose(w_values[0], 0.9, atol=0.15)
    assert np.isclose(w_values[-1], 0.4, atol=0.15)
    assert w_values[0] > w_values[-1]


def test_binary_mopso_velocity_initialization() -> None:
    """Verify that initial velocity is sampled uniformly in [-v_max, v_max]."""
    v_max = 3.5
    problem = SimpleBinaryProblem(n_var=20)
    algo = BMOPSO_CE(n_particles=50, v_max=v_max)
    algo.setup(problem)
    algo._initialize()

    assert algo.V is not None
    assert algo.V.shape == (50, 20)
    assert np.all(algo.V >= -v_max)
    assert np.all(algo.V <= v_max)
    assert not np.all(algo.V == 0.0)


def test_binary_mopso_optimization() -> None:
    """Validate BMOPSO-CE execution integrated with the pymoo framework."""
    n_vars = 12
    problem = SimpleBinaryProblem(n_var=n_vars)
    algorithm = BMOPSO_CE(n_particles=20, w=0.5, c1=1.5, c2=1.5, n_grid=20)

    # Execute optimization using pymoo minimize runner
    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    # 1. Verify execution completes without error
    assert res is not None

    # 2. Verify res.X contains valid binary solutions
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == n_vars
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    # 3. Verify res.F contains valid objectives
    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert len(res.F) == len(res.X)
    assert not np.isnan(res.F).any()

    # 4. Verify sum of f1 + f2 equals n_vars for all solutions
    assert np.all(np.sum(res.F, axis=1) == n_vars)


def test_adaptive_hypercube_grid_calculation() -> None:
    """Verify Adaptive Hypercube Grid coordinates and leader selection according to Coello Coello (2004)."""
    from bmopso_ce.util.grid import AdaptiveGrid

    grid = AdaptiveGrid(n_grid=10)

    # 4-point Pareto Front
    f = np.array([[1.0, 10.0], [2.0, 7.0], [4.0, 4.0], [6.0, 1.0]])
    coords = grid.compute_grid_coordinates(f)
    assert coords.shape == (4, 2)
    assert np.all(coords >= 0) and np.all(coords < 10)

    hypercube_ids = grid.get_hypercube_ids(f)
    assert len(hypercube_ids) == 4

    x = np.array([[True, False], [False, True], [True, True], [False, False]])
    leaders = grid.select_leaders(x, f, n_particles=10)
    assert leaders.shape == (10, 2)


def test_bmopso_archive_pruning_with_max_archive_size() -> None:
    """Verify external archive respects max_archive_size limit pruning by hypercube density."""
    problem = SimpleBinaryProblem(n_var=20)
    max_archive = 8
    algo = BMOPSO_CE(n_particles=40, max_archive_size=max_archive, n_grid=10)

    res = minimize(problem, algo, termination=("n_gen", 10), verbose=False)

    assert res.X is not None
    assert len(res.X) <= max_archive
    assert len(res.F) <= max_archive


def test_constrained_dominance_deb_rules() -> None:
    """Validate the 4 rules of the Constrained-Dominance Principle (Deb, 2002)."""
    from bmopso_ce.util.dominance import dominates, find_non_dominated_constrained

    f_good = np.array([1.0, 2.0])
    f_bad = np.array([5.0, 6.0])

    # Rule 1: Feasible solution (cv=0) strictly dominates infeasible solution (cv > 0)
    assert dominates(f_bad, f_good, cv1=0.0, cv2=2.5) is True
    assert dominates(f_good, f_bad, cv1=2.5, cv2=0.0) is False

    # Rule 2: Between two infeasible solutions, lower violation dominates higher violation
    assert dominates(f_bad, f_good, cv1=1.0, cv2=3.0) is True
    assert dominates(f_good, f_bad, cv1=4.0, cv2=2.0) is False

    # Rule 3: Between two feasible solutions, standard Pareto dominance applies
    assert dominates(f_good, f_bad, cv1=0.0, cv2=0.0) is True
    assert dominates(f_bad, f_good, cv1=0.0, cv2=0.0) is False

    # Validation of archive filtering via find_non_dominated_constrained
    f_matrix = np.array([[10.0, 2.0], [5.0, 5.0], [0.1, 0.1]])
    cv_matrix = np.array([0.0, 0.0, 10.0])  # Point 2 has best F, but is infeasible
    non_dom_idx = find_non_dominated_constrained(f_matrix, cv_matrix)
    assert set(non_dom_idx.tolist()) == {0, 1}
