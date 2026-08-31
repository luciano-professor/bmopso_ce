"""Example script demonstrating BMOPSO on the Multiple Knapsack Problem (MKP).

This script provides a practical walkthrough to:
1. Configure multiobjective Multiple Knapsack Problem (MKP) benchmark instances.
2. Configure BMOPSO algorithm hyperparameters.
3. Execute optimization using pymoo.optimize.minimize.
4. Analyze, decode, and verify feasibility across the resulting Pareto Front.
5. Visualize the Pareto Front using pymoo's native plotting tools.

To execute:
    python examples/run_mkp_example.py
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
from pymoo_binary_problems import MKP


def create_sample_mkp_instance() -> MKP:
    """Create a sample instance of the Multiple Knapsack Problem (MKP).

    Scenario:
    - 12 items available for transportation.
    - 3 knapsacks / cargo compartments with individual capacities.
    - Total binary decision variables: 12 items * 3 knapsacks = 36 bits.

    Returns
    -------
    MKP
        Configured MKP problem instance for multiobjective optimization.
    """
    # 1. Individual profits for each of the 12 items
    profits = np.array(
        [18.0, 26.0, 32.0, 15.0, 45.0, 22.0, 38.0, 50.0, 12.0, 29.0, 35.0, 42.0],
        dtype=float,
    )

    # 2. Individual weights for each of the 12 items
    weights = np.array(
        [6.0, 9.0, 11.0, 5.0, 16.0, 8.0, 13.0, 18.0, 4.0, 10.0, 12.0, 15.0],
        dtype=float,
    )

    # 3. Maximum capacities for the 3 knapsacks
    capacities = np.array([35.0, 40.0, 30.0], dtype=float)

    # 4. Instantiate MKP problem
    # - n_obj=2:
    #   * f1: Maximize Total Profit (pymoo minimizes -Profit)
    #   * f2: Minimize Total Weight Loaded
    problem = MKP(
        profits=profits,
        weights=weights,
        capacities=capacities,
        n_obj=2,
        maximize_profit=True,
        minimize_weight=True,
    )

    return problem


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MKP,
) -> None:
    """Print item allocation details and feasibility status for a candidate solution.

    Parameters
    ----------
    sol_idx : int
        Identifying index of the solution on the Pareto Front.
    x_flat : np.ndarray
        1D binary decision vector of the solution.
    f_val : np.ndarray
        1D objective vector [f1, f2].
    problem : MKP
        Problem instance containing item and knapsack configuration.
    """
    # Decode 1D vector into (n_items, n_knapsacks) matrix
    x_matrix = x_flat.reshape((problem.n_items, problem.n_knapsacks))

    real_profit = -f_val[0] if problem.maximize_profit else f_val[0]
    total_weight = f_val[1] if problem.minimize_weight else -f_val[1]

    print(f"\n--- Solution #{sol_idx + 1} ---")
    print(f"  Total Profit : $ {real_profit:6.2f}")
    print(f"  Total Weight :   {total_weight:6.2f} kg")

    # Details per knapsack
    is_feasible = True
    for k in range(problem.n_knapsacks):
        items_in_k = np.where(x_matrix[:, k])[0]
        weight_k = float(np.sum(problem.weights[items_in_k]))
        cap_k = float(problem.capacities[k])
        status = "OK" if weight_k <= cap_k else "VIOLATED!"
        if weight_k > cap_k:
            is_feasible = False

        items_str = ", ".join(f"Item {i} ({problem.weights[i]}kg)" for i in items_in_k) or "Empty"
        print(f"    Knapsack {k}: {weight_k:4.1f} / {cap_k:4.1f} kg [{status}] -> Items: [{items_str}]")

    # Verify single allocation constraint
    allocated_counts = np.sum(x_matrix, axis=1)
    multiple_allocations = np.where(allocated_counts > 1)[0]
    if len(multiple_allocations) > 0:
        is_feasible = False
        print(f"    Single-knapsack constraint violated for items: {multiple_allocations.tolist()}")

    status_str = "VALID (Feasible)" if is_feasible else "INVALID (Constraints Violated)"
    print(f"  Overall Status: {status_str}")


def main() -> None:
    """Execute complete MKP multiobjective optimization workflow using BMOPSO_CE."""
    print("=" * 75)
    print(" MULTIOBJECTIVE MULTIPLE KNAPSACK PROBLEM (MKP) WITH BMOPSO-CE")
    print("=" * 75)

    # 1. Create MKP problem instance
    problem = create_sample_mkp_instance()
    print("\nProblem initialized successfully:")
    print(f"  - Item Count                : {problem.n_items}")
    print(f"  - Knapsack Count            : {problem.n_knapsacks}")
    print(f"  - Decision Variables (X)    : {problem.n_var} bits")
    print(f"  - Objective Count (F)       : {problem.n_obj}")
    print(f"  - Knapsack Capacities       : {problem.capacities.tolist()}")

    # 2. Configure BMOPSO_CE algorithm
    algorithm = BMOPSO_CE(
        n_particles=60,  # Swarm population size
        w_max=0.9,       # Initial inertia weight (global exploration)
        w_min=0.4,       # Final inertia weight (local exploitation / fine-tuning)
        c1=1.49,         # Cognitive acceleration coefficient
        c2=1.49,         # Social acceleration coefficient
        v_max=4.0,       # Velocity clamping bound [-4.0, 4.0]
    )

    # 3. Execute optimization using pymoo.optimize.minimize
    n_evals = 20000
    print(f"\nStarting optimization with termination criterion of {n_evals} evaluations...")

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", n_evals),
        seed=42,
        verbose=False,
    )

    print("\nOptimization finished!")

    # 4. Feasibility analysis across the Pareto Front
    viable_indices = []
    infeasible_indices = []
    for i in range(len(res.X)):
        x_m = res.X[i].reshape((problem.n_items, problem.n_knapsacks))
        knapsack_weights = [np.dot(x_m[:, k], problem.weights) for k in range(problem.n_knapsacks)]
        weight_ok = all(w_k <= cap for w_k, cap in zip(knapsack_weights, problem.capacities))
        uniqueness_ok = np.all(np.sum(x_m, axis=1) <= 1)
        if weight_ok and uniqueness_ok:
            viable_indices.append(i)
        else:
            infeasible_indices.append(i)

    n_total = len(res.X)
    n_viable = len(viable_indices)
    n_infeasible = len(infeasible_indices)

    print(f"Total Solutions on Pareto Front : {n_total}")
    print(f"  |-- FEASIBLE Solutions (100% Valid)  : {n_viable} ({n_viable / n_total * 100:.1f}%)")
    print(f"  \\-- INFEASIBLE Solutions (Violations): {n_infeasible} ({n_infeasible / n_total * 100:.1f}%)")

    # Sort feasible solutions in descending order of profit
    viable_sorted = sorted(viable_indices, key=lambda idx: -res.F[idx, 0], reverse=True)

    # 5. Summary table of top feasible solutions
    print("\n" + "=" * 75)
    print(" SUMMARY OF TOP FEASIBLE SOLUTIONS FOUND")
    print("=" * 75)
    print(f" {'#':<3} | {'Total Profit ($)':<18} | {'Total Weight (kg)':<18} | {'Status'}")
    print("-" * 75)

    for rank, idx in enumerate(viable_sorted[:8], 1):
        profit = -res.F[idx, 0]
        weight = res.F[idx, 1]
        print(f" {rank:<3} | $ {profit:<16.2f} | {weight:<15.2f} kg | Feasible")

    if not viable_sorted:
        print("  No 100% feasible solution found in this run.")

    # 6. Detailed allocation of the best feasible solution (highest profit)
    print("\n" + "=" * 75)
    print(" DETAILED ALLOCATION OF BEST FEASIBLE SOLUTION (HIGHEST PROFIT)")
    print("=" * 75)

    if viable_sorted:
        best_viable_idx = viable_sorted[0]
        print_solution_details(best_viable_idx, res.X[best_viable_idx], res.F[best_viable_idx], problem)
    elif len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)

    # 7. Visualize Pareto Front using pymoo's native plotting tools
    print("\n" + "=" * 75)
    print(" PARETO FRONT VISUALIZATION (pymoo.visualization.scatter.Scatter)")
    print("=" * 75)
    if viable_sorted:
        plot_F = np.column_stack([-res.F[viable_sorted, 0], res.F[viable_sorted, 1]])
        plot = Scatter(
            title="BMOPSO on Multiple Knapsack Problem (MKP)",
            labels=["Total Profit ($)", "Total Weight (kg)"],
        )
        plot.add(plot_F, color="blue", label="Feasible Pareto Solutions")
        print("  Scatter plot initialized successfully via pymoo.visualization.scatter.Scatter.")
        print("  (Uncomment 'plot.show()' to open the interactive graphical window).")
        # plot.show()


if __name__ == "__main__":
    main()
