"""Unit tests for refactored modular components: dominance, adaptive grid, operators, and archive."""

import numpy as np
import pytest
from bmopso_ce.operators import (
    apply_mutation,
    sample_binary_positions,
    sigmoid,
    update_personal_bests,
    update_velocity,
)
from bmopso_ce.util import (
    AdaptiveGrid,
    NonDominatedArchive,
    dominates,
    dominates_mask,
    find_non_dominated_constrained,
)


def test_dominance_rules_isolated() -> None:
    """Test Deb's constrained dominance rules in isolation."""
    # Feasible vs Infeasible
    assert dominates(np.array([10.0, 10.0]), np.array([1.0, 1.0]), cv1=0.0, cv2=0.5) is True
    assert dominates(np.array([1.0, 1.0]), np.array([10.0, 10.0]), cv1=0.5, cv2=0.0) is False

    # Infeasible vs Infeasible (lower violation wins)
    assert dominates(np.array([10.0, 10.0]), np.array([1.0, 1.0]), cv1=0.2, cv2=0.8) is True
    assert dominates(np.array([1.0, 1.0]), np.array([10.0, 10.0]), cv1=0.9, cv2=0.1) is False

    # Feasible vs Feasible (standard Pareto dominance)
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]), cv1=0.0, cv2=0.0) is True
    assert dominates(np.array([2.0, 3.0]), np.array([1.0, 2.0]), cv1=0.0, cv2=0.0) is False
    assert dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]), cv1=0.0, cv2=0.0) is False


def test_dominates_mask_matches_scalar_dominates() -> None:
    """Vectorized constrained-dominance must match the scalar Deb (2002) rules."""
    rng = np.random.default_rng(7)
    n = 40
    f1 = rng.uniform(0.0, 10.0, size=(n, 3))
    f2 = rng.uniform(0.0, 10.0, size=(n, 3))
    cv1 = rng.choice([0.0, 0.2, 0.8], size=n)
    cv2 = rng.choice([0.0, 0.2, 0.8], size=n)

    mask = dominates_mask(f1, f2, cv1, cv2)
    expected = np.array(
        [dominates(f1[i], f2[i], float(cv1[i]), float(cv2[i])) for i in range(n)],
        dtype=bool,
    )
    assert np.array_equal(mask, expected)


def test_compute_grid_coordinates_matches_per_objective_loop() -> None:
    """Vectorized grid mapping must match the original per-objective formula."""
    grid = AdaptiveGrid(n_grid=12)
    rng = np.random.default_rng(3)
    f = rng.uniform(0.0, 50.0, size=(25, 4))
    f[:5, 2] = 7.5  # degenerate objective column

    coords = grid.compute_grid_coordinates(f)

    expected = np.zeros_like(coords)
    f_min = np.min(f, axis=0)
    f_max = np.max(f, axis=0)
    for m in range(f.shape[1]):
        range_m = float(f_max[m] - f_min[m])
        if range_m == 0.0:
            expected[:, m] = grid.n_grid // 2
        else:
            buffer = range_m / (2.0 * grid.n_grid)
            lower_bound = f_min[m] - buffer
            upper_bound = f_max[m] + buffer
            cell_width = (upper_bound - lower_bound) / grid.n_grid
            c = np.floor((f[:, m] - lower_bound) / cell_width).astype(int)
            expected[:, m] = np.clip(c, 0, grid.n_grid - 1)

    assert np.array_equal(coords, expected)


def test_adaptive_grid_isolated() -> None:
    """Test AdaptiveGrid coordinate mapping, hypercube IDs, and leader selection."""
    grid = AdaptiveGrid(n_grid=10)

    # Empty points
    coords_empty = grid.compute_grid_coordinates(np.empty((0, 2)))
    assert coords_empty.shape == (0, 2)
    ids_empty = grid.get_hypercube_ids(np.empty((0, 2)))
    assert len(ids_empty) == 0

    # 4-point objective set
    f = np.array([[1.0, 10.0], [2.0, 7.0], [4.0, 4.0], [6.0, 1.0]])
    coords = grid.compute_grid_coordinates(f)
    assert coords.shape == (4, 2)
    assert np.all(coords >= 0) and np.all(coords < 10)

    hypercube_ids = grid.get_hypercube_ids(f)
    assert len(hypercube_ids) == 4
    assert len(set(hypercube_ids.tolist())) == 4  # All points in separate grid cells

    # Single point
    f_single = np.array([[5.0, 5.0]])
    coords_single = grid.compute_grid_coordinates(f_single)
    assert np.all(coords_single == 5)  # Center cell

    # Leader selection
    x = np.array([[True, False], [False, True], [True, True], [False, False]])
    leaders = grid.select_leaders(x, f, n_particles=8)
    assert leaders.shape == (8, 2)
    assert np.all(np.isin(leaders, [True, False]))


def test_adaptive_grid_pruning() -> None:
    """Test crowded hypercube identification and capacity pruning."""
    grid = AdaptiveGrid(n_grid=5)

    # 4 solutions, with 3 clustered together in the same region (same hypercube) and 1 isolated
    x = np.array([[True, True], [True, False], [False, True], [False, False]])
    f = np.array([[1.0, 1.0], [1.05, 1.05], [1.02, 1.03], [10.0, 10.0]])
    cv = np.zeros(4)

    pruned_x, pruned_f, pruned_cv = grid.prune_archive(x, f, cv, max_size=2)
    assert len(pruned_x) == 2
    assert len(pruned_f) == 2
    assert len(pruned_cv) == 2
    # The isolated point [10.0, 10.0] (solution 3) must be preserved because it's alone in its hypercube
    assert np.any(np.all(pruned_f == [10.0, 10.0], axis=1))

    # Large batch test: 100 solutions pruned down to 30
    x_large = np.random.randint(0, 2, size=(100, 10)).astype(bool)
    f_large = np.random.uniform(0, 100, size=(100, 2))
    cv_large = np.zeros(100)
    p_x, p_f, p_cv = grid.prune_archive(x_large, f_large, cv_large, max_size=30)
    assert len(p_x) == 30
    assert len(p_f) == 30
    assert len(p_cv) == 30


def test_operators_isolated() -> None:
    """Test sigmoid, binary sampling, velocity update, and mutation operators."""
    # Sigmoid
    v = np.array([0.0, 100.0, -100.0])
    s = sigmoid(v)
    assert np.isclose(s[0], 0.5)
    assert np.isclose(s[1], 1.0)
    assert np.isclose(s[2], 0.0)

    # Binary sampling
    positions = sample_binary_positions(np.array([[10.0, -10.0], [-10.0, 10.0]]))
    assert positions.dtype == bool
    assert positions.shape == (2, 2)

    # Velocity update with clamping
    x = np.array([[True, False]])
    pbest_x = np.array([[True, True]])
    gbest = np.array([[False, False]])
    v_init = np.array([[10.0, -10.0]])
    v_updated = update_velocity(v_init, x, pbest_x, gbest, w=0.5, v_max=4.0)
    assert np.all(v_updated <= 4.0) and np.all(v_updated >= -4.0)

    # Mutation operator
    x_mut = apply_mutation(x, progress=0.0, mutation_rate=0.5)
    assert x_mut.shape == x.shape


def test_pbest_update_coello_rules() -> None:
    """Test Coello Coello (2004) pbest update rules."""
    pbest_x = np.array([[True, False], [False, True]])
    pbest_f = np.array([[5.0, 5.0], [2.0, 8.0]])
    pbest_cv = np.array([0.0, 0.0])

    # Candidate 0 dominates pbest 0 ([1.0, 1.0] < [5.0, 5.0])
    # Candidate 1 is dominated by pbest 1 ([5.0, 10.0] > [2.0, 8.0])
    x = np.array([[False, False], [True, True]])
    f = np.array([[1.0, 1.0], [5.0, 10.0]])
    cv = np.array([0.0, 0.0])

    new_x, new_f, new_cv = update_personal_bests(pbest_x, pbest_f, pbest_cv, x, f, cv)
    # Candidate 0 must replace pbest 0
    assert np.array_equal(new_x[0], [False, False])
    assert np.array_equal(new_f[0], [1.0, 1.0])
    # Candidate 1 must NOT replace pbest 1
    assert np.array_equal(new_x[1], [False, True])
    assert np.array_equal(new_f[1], [2.0, 8.0])


def test_pbest_update_matches_scalar_loop() -> None:
    """Vectorized pbest update must match the per-particle Coello Coello (2004) loop."""
    rng = np.random.default_rng(11)
    n_particles, n_var, n_obj = 16, 5, 2
    pbest_x = rng.integers(0, 2, size=(n_particles, n_var)).astype(bool)
    pbest_f = rng.uniform(0.0, 5.0, size=(n_particles, n_obj))
    pbest_cv = rng.choice([0.0, 0.4], size=n_particles)
    x = rng.integers(0, 2, size=(n_particles, n_var)).astype(bool)
    f = rng.uniform(0.0, 5.0, size=(n_particles, n_obj))
    cv = rng.choice([0.0, 0.4], size=n_particles)

    rng_vec = np.random.default_rng(123)
    vec_x, vec_f, vec_cv = update_personal_bests(
        pbest_x, pbest_f, pbest_cv, x, f, cv, random_state=rng_vec
    )

    rng_ref = np.random.default_rng(123)
    ref_x = pbest_x.copy()
    ref_f = pbest_f.copy()
    ref_cv = pbest_cv.copy()
    for i in range(n_particles):
        if dominates(f[i], ref_f[i], float(cv[i]), float(ref_cv[i])):
            ref_x[i] = x[i]
            ref_f[i] = f[i]
            ref_cv[i] = cv[i]
        elif not dominates(ref_f[i], f[i], float(ref_cv[i]), float(cv[i])):
            if rng_ref.random() < 0.5:
                ref_x[i] = x[i]
                ref_f[i] = f[i]
                ref_cv[i] = cv[i]

    assert np.array_equal(vec_x, ref_x)
    assert np.array_equal(vec_f, ref_f)
    assert np.array_equal(vec_cv, ref_cv)


def test_archive_class_lifecycle() -> None:
    """Test NonDominatedArchive lifecycle, hypercube pruning, and leader selection."""
    archive = NonDominatedArchive(max_size=3, n_grid=10)
    assert archive.is_empty()
    assert len(archive) == 0

    # Insert initial solutions
    x1 = np.array([[True, False], [False, True], [True, True]])
    f1 = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
    archive.update(x1, f1)

    assert len(archive) == 3
    assert not archive.is_empty()
    assert archive.x is not None
    assert archive.f is not None

    # Leader selection
    leaders = archive.select_leaders(n_particles=5)
    assert leaders.shape == (5, 2)

    # Capacity pruning
    x_extra = np.array([[False, False], [True, False]])
    f_extra = np.array([[2.0, 4.0], [4.0, 2.0]])
    archive.update(x_extra, f_extra)
    assert len(archive) == 3  # Pruned to max_size=3

    # Population conversion
    pop = archive.to_population()
    assert len(pop) == 3
