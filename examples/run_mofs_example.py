"""Example script demonstrating BMOPSO on Multiobjective Feature Selection (MOFS).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Feature Selection (MOFS) benchmark instance
   with known informative, redundant, and noisy features.
2. Configure BMOPSO algorithm hyperparameters.
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting non-dominated Pareto Front, observing how BMOPSO isolates
   informative features while discarding noise to maximize classification accuracy and sparsity.

To execute:
    python examples/run_mofs_example.py
"""

import sys
from pathlib import Path
from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Ensure src/ is on PYTHONPATH for direct standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MOFS



def create_sample_mofs_instance() -> MOFS:
    """Create a sample instance of the Multiobjective Feature Selection (MOFS) problem.

    Scenario:
    - 200 patient / sample records.
    - 30 total features:
      * 8 truly informative features (indices 0 to 7)
      * 4 redundant features (indices 8 to 11)
      * 18 noisy / irrelevant features (indices 12 to 29)
    - 2 conflicting objectives:
      * Objective 1: Classification Error Rate (1 - Accuracy)
      * Objective 2: Feature Selection Ratio (||x||_1 / 30)

    Returns
    -------
    MOFS
        Configured MOFS benchmark problem instance.
    """
    problem = MOFS.from_synthetic(
        n_samples=200,
        n_features=30,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        cv=3,
        min_features=1,
        seed=42,
    )
    return problem


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MOFS,
) -> None:
    """Print detailed classification metrics and selected features for a candidate solution.

    Parameters
    ----------
    sol_idx : int
        Identifying index of the solution on the Pareto Front.
    x_flat : np.ndarray
        1D binary decision vector of length n_features.
    f_val : np.ndarray
        1D objective vector [error_rate, feature_ratio].
    problem : MOFS
        MOFS problem instance.
    """
    info = problem.decode_features(x_flat)
    err = float(f_val[0])
    acc_pct = (1.0 - err) * 100.0
    err_pct = err * 100.0
    active_indices = info["selected_indices"]
    active_str = ", ".join(map(str, active_indices[:15]))
    if len(active_indices) > 15:
        active_str += f", ... (+{len(active_indices) - 15} more)"

    print(f"\n--- Solution #{sol_idx} ---")
    print(f"  Classification Accuracy : {acc_pct:6.2f}% (Error Rate: {err_pct:5.2f}%)")
    print(f"  Selected Features Count : {info['n_selected']}/{problem.n_features} ({info['ratio'] * 100:.1f}%)")
    print(f"  Active Feature Indices  : [{active_str}]")


def main() -> None:
    """Execute complete MOFS multiobjective optimization workflow using BMOPSO_CE."""
    print("=" * 80)
    print(" MULTIOBJECTIVE FEATURE SELECTION (MOFS) WITH BMOPSO-CE")
    print("=" * 80)

    # 1. Create MOFS problem instance
    problem = create_sample_mofs_instance()
    print("\nMOFS Problem initialized successfully:")
    print(f"  - Dataset Samples (S)         : {problem.n_samples}")
    print(f"  - Total Features (D)          : {problem.n_features} bits")
    print(f"  - Objective Count (F)         : {problem.n_obj} (Error Rate vs Feature Ratio)")
    print(f"  - Cross-Validation (CV)       : {problem.cv}-fold CV")
    print(f"  - Minimum Features Required   : {problem.min_features}")

    # 2. Configure BMOPSO_CE algorithm
    algorithm = BMOPSO_CE(
        n_particles=40,       # Swarm population size
        w_max=0.9,            # Initial inertia weight (exploration)
        w_min=0.4,            # Final inertia weight (exploitation)
        c1=1.49,              # Cognitive acceleration coefficient
        c2=1.49,              # Social acceleration coefficient
        v_max=4.0,            # Velocity clamping bound [-4.0, 4.0]
        mutation_rate=0.5,    # Non-linear turbulence / bit-flip rate
        max_archive_size=100, # External Pareto archive capacity
    )

    # 3. Execute optimization using pymoo.optimize.minimize
    n_gen = 30
    print(f"\nStarting optimization with termination criterion of {n_gen} generations...")

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=42,
        verbose=False,
    )

    print("\nOptimization finished successfully!")

    # 4. Feasibility analysis across the resulting Pareto Front
    viable_indices = []
    infeasible_indices = []

    for i in range(len(res.X)):
        info = problem.decode_features(res.X[i])
        if info["is_valid"]:
            viable_indices.append(i)
        else:
            infeasible_indices.append(i)

    n_total = len(res.X)
    n_viable = len(viable_indices)
    n_infeasible = len(infeasible_indices)

    print(f"\nTotal Solutions on Pareto Front : {n_total}")
    print(f"  |-- FEASIBLE Feature Masks (Valid) : {n_viable} ({n_viable / n_total * 100:.1f}%)")
    print(f"  \\-- INFEASIBLE (Empty Masks)       : {n_infeasible} ({n_infeasible / n_total * 100:.1f}%)")

    # Sort feasible solutions by Error Rate ascending (highest accuracy first)
    viable_sorted = sorted(viable_indices, key=lambda idx: res.F[idx, 0])

    # Filter unique solutions by objective vectors
    seen_f: set[tuple[float, ...]] = set()
    unique_viable: list[int] = []
    for idx in viable_sorted:
        key = tuple(np.round(res.F[idx], 3))
        if key not in seen_f:
            seen_f.add(key)
            unique_viable.append(idx)

    # 5. Summary table of Pareto-optimal feature subsets
    print("\n" + "=" * 80)
    print(" SUMMARY OF NON-DOMINATED PARETO FEATURE SUBSETS")
    print("=" * 80)
    print(f" {'#':<3} | {'Accuracy (%)':<14} | {'Error Rate':<12} | {'Features':<10} | {'Ratio (%)':<10} | {'Selected Indices'}")
    print("-" * 80)

    for rank, idx in enumerate(unique_viable[:12], 1):
        err = res.F[idx, 0]
        acc = (1.0 - err) * 100.0
        ratio = res.F[idx, 1] * 100.0
        info = problem.decode_features(res.X[idx])
        subsets_str = str(info["selected_indices"][:6])
        if len(info["selected_indices"]) > 6:
            subsets_str = subsets_str[:-1] + ", ...]"
        print(f" {rank:<3} | {acc:<14.2f}% | {err:<12.4f} | {info['n_selected']:<10} | {ratio:<10.1f}% | {subsets_str}")

    if not unique_viable:
        print("  No feasible feature subset found in this run.")

    # 6. Detailed inspection of trade-off solutions
    print("\n" + "=" * 80)
    print(" DETAILED TRADE-OFF SUBSETS INSPECTION")
    print("=" * 80)

    if unique_viable:
        # Highest Accuracy subset
        best_acc_idx = unique_viable[0]
        print("\n[Highest Classification Accuracy Subset]:")
        print_solution_details(1, res.X[best_acc_idx], res.F[best_acc_idx], problem)

        # Most Compact / Sparse subset (Lowest Feature Ratio)
        most_sparse_idx = min(unique_viable, key=lambda idx: res.F[idx, 1])
        print("\n[Maximum Sparsity / Minimal Features Subset]:")
        print_solution_details(2, res.X[most_sparse_idx], res.F[most_sparse_idx], problem)

        # Balanced compromise subset
        compromise_idx = unique_viable[len(unique_viable) // 2]
        print("\n[Balanced Compromise Subset (High Accuracy & High Sparsity)]:")
        print_solution_details(3, res.X[compromise_idx], res.F[compromise_idx], problem)
    elif len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)

    # 7. Visualize Pareto Front using pymoo's native plotting tools
    print("\n" + "=" * 80)
    print(" PARETO FRONT VISUALIZATION (pymoo.visualization.scatter.Scatter)")
    print("=" * 80)
    if unique_viable:
        plot_F = np.column_stack([
            (1.0 - res.F[unique_viable, 0]) * 100.0,  # Accuracy (%)
            res.F[unique_viable, 1] * 100.0,          # Feature Ratio (%)
        ])
        plot = Scatter(
            title="BMOPSO on Multiobjective Feature Selection (MOFS)",
            labels=["Classification Accuracy (%)", "Feature Selection Ratio (%)"],
        )
        plot.add(plot_F, color="darkorange", label="Pareto Feature Subsets")
        print("  Scatter plot initialized successfully via pymoo.visualization.scatter.Scatter.")
        print("  (Uncomment 'plot.show()' to open the interactive graphical window).")
        # plot.show()

    print("\n" + "=" * 80)
    print(" MOFS BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

