"""Example script demonstrating BMOPSO on the Multiobjective Traveling Salesman Problem (MSTSP / MOTSP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Traveling Salesman Problem (MSTSP) benchmark instance
   using binary assignment encoding.
2. Configure BMOPSO algorithm hyperparameters.
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze and decode resulting binary decision matrices into valid Hamiltonian tour sequences.
5. Inspect objective trade-offs across the non-dominated Pareto Front.

To execute:
    python examples/run_mstsp_example.py
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
from pymoo_binary_problems import MSTSP



def create_sample_mstsp_instance() -> MSTSP:
    """Create a sample instance of the Multiobjective Traveling Salesman Problem (MSTSP).

    Scenario:
    - 6 delivery locations / cities.
    - 2 conflicting criteria:
      * Objective 1: Travel Distance (km)
      * Objective 2: Toll & Transit Cost ($)
    - Binary decision variables: 6 * 6 = 36 bits (Position-City assignment matrix).

    Returns
    -------
    MSTSP
        Configured MSTSP benchmark problem instance.
    """
    # Geographic coordinates for 6 cities in Layout 1 (Distance metric)
    coords_distance = np.array([
        [10.0, 20.0],
        [25.0, 80.0],
        [45.0, 60.0],
        [70.0, 90.0],
        [85.0, 30.0],
        [60.0, 10.0],
    ], dtype=float)

    # Geographic coordinates for Layout 2 (Transit/Toll cost topology)
    coords_cost = np.array([
        [80.0, 15.0],
        [10.0, 90.0],
        [30.0, 20.0],
        [95.0, 60.0],
        [40.0, 85.0],
        [15.0, 30.0],
    ], dtype=float)

    problem = MSTSP.from_coordinates(
        coordinates_list=[coords_distance, coords_cost],
    )
    return problem



def print_tour_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MSTSP,
) -> None:
    """Print detailed route and objective metrics for a candidate solution.

    Parameters
    ----------
    sol_idx : int
        Identifying index of the solution on the Pareto Front.
    x_flat : np.ndarray
        1D binary decision vector of length (n_cities^2).
    f_val : np.ndarray
        1D objective vector [distance, cost].
    problem : MSTSP
        Problem instance containing city and cost matrices.
    """
    tour_seq, is_valid = problem.decode_tour(x_flat)
    tour_str = " -> ".join(map(str, tour_seq)) + f" -> {tour_seq[0]}"
    status_str = "VALID (Feasible Tour)" if is_valid else "INVALID (Violations Present)"

    print(f"\n--- Solution #{sol_idx} ---")
    print(f"  Distance (Obj 1)  : {f_val[0]:7.2f} km")
    print(f"  Cost (Obj 2)      : $ {f_val[1]:6.2f}")
    print(f"  Route Status      : {status_str}")
    print(f"  Decoded Tour      : {tour_str}")


def main() -> None:
    """Execute complete MSTSP multiobjective optimization workflow using BMOPSO_CE."""
    print("=" * 80)
    print(" MULTIOBJECTIVE TRAVELING SALESMAN PROBLEM (MSTSP) WITH BMOPSO-CE")
    print("=" * 80)

    # 1. Create MSTSP problem instance
    problem = create_sample_mstsp_instance()
    print("\nMSTSP Problem initialized successfully:")
    print(f"  - Number of Cities (N)        : {problem.n_cities}")
    print(f"  - Decision Variables (X)      : {problem.n_var} bits (Position-City Matrix)")
    print(f"  - Objective Count (F)         : {problem.n_obj}")
    print(f"  - Inequality Constraints (G)  : {problem.n_ieq_constr} (Row, Column, Deficit)")

    # 2. Configure BMOPSO_CE algorithm
    algorithm = BMOPSO_CE(
        n_particles=60,       # Swarm population size
        w_max=0.9,            # Initial inertia weight (exploration)
        w_min=0.4,            # Final inertia weight (exploitation)
        c1=1.49,              # Cognitive acceleration coefficient
        c2=1.49,              # Social acceleration coefficient
        v_max=4.0,            # Velocity clamping bound [-4.0, 4.0]
        mutation_rate=0.5,    # Non-linear turbulence / bit-flip rate
        max_archive_size=100, # External Pareto archive capacity
    )

    # 3. Execute optimization using pymoo.optimize.minimize
    n_evals = 25000
    print(f"\nStarting optimization with termination criterion of {n_evals} evaluations...")

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", n_evals),
        seed=42,
        verbose=False,
    )

    print("\nOptimization finished successfully!")

    # 4. Feasibility analysis across the resulting Pareto Front
    viable_indices = []
    infeasible_indices = []

    for i in range(len(res.X)):
        _, is_valid = problem.decode_tour(res.X[i])
        if is_valid:
            viable_indices.append(i)
        else:
            infeasible_indices.append(i)

    n_total = len(res.X)
    n_viable = len(viable_indices)
    n_infeasible = len(infeasible_indices)

    print(f"\nTotal Solutions on Pareto Front : {n_total}")
    print(f"  |-- FEASIBLE Tours (100% Valid) : {n_viable} ({n_viable / n_total * 100:.1f}%)")
    print(f"  \\-- INFEASIBLE (Violations)     : {n_infeasible} ({n_infeasible / n_total * 100:.1f}%)")

    # Sort feasible solutions by Objective 1 (Distance) ascending
    viable_sorted = sorted(viable_indices, key=lambda idx: res.F[idx, 0])

    # 5. Summary table of feasible Pareto tours (unique objective vectors)
    print("\n" + "=" * 80)
    print(" SUMMARY OF TOP FEASIBLE PARETO TOURS FOUND")
    print("=" * 80)
    print(f" {'#':<3} | {'Distance (km)':<16} | {'Cost ($)':<16} | {'Tour Sequence'}")
    print("-" * 80)

    seen_f: set[tuple[float, ...]] = set()
    unique_viable: list[int] = []
    for idx in viable_sorted:
        key = tuple(np.round(res.F[idx], 2))
        if key not in seen_f:
            seen_f.add(key)
            unique_viable.append(idx)

    for rank, idx in enumerate(unique_viable[:10], 1):
        dist = res.F[idx, 0]
        cost = res.F[idx, 1]
        tour_seq, _ = problem.decode_tour(res.X[idx])
        tour_str = " -> ".join(map(str, tour_seq)) + f" -> {tour_seq[0]}"
        print(f" {rank:<3} | {dist:<16.2f} | $ {cost:<14.2f} | {tour_str}")

    if not unique_viable:
        print("  No 100% feasible tour found in this run.")


    # 6. Detailed inspection of trade-off solutions
    print("\n" + "=" * 80)
    print(" DETAILED ROUTE INSPECTION FOR KEY TRADE-OFFS")
    print("=" * 80)

    if viable_sorted:
        # Best distance tour
        print("\n[Best Distance Route (Shortest Path)]:")
        best_dist_idx = viable_sorted[0]
        print_tour_details(1, res.X[best_dist_idx], res.F[best_dist_idx], problem)

        # Best cost tour
        best_cost_idx = min(viable_sorted, key=lambda idx: res.F[idx, 1])
        print("\n[Best Cost Route (Lowest Tolls)]:")
        print_tour_details(2, res.X[best_cost_idx], res.F[best_cost_idx], problem)

        # Compromise tour
        compromise_idx = viable_sorted[len(viable_sorted) // 2]
        print("\n[Balanced Compromise Route]:")
        print_tour_details(3, res.X[compromise_idx], res.F[compromise_idx], problem)
    elif len(res.X) > 0:
        print_tour_details(0, res.X[0], res.F[0], problem)

    # 7. Visualize Pareto Front using pymoo's native plotting tools
    print("\n" + "=" * 80)
    print(" PARETO FRONT VISUALIZATION (pymoo.visualization.scatter.Scatter)")
    print("=" * 80)
    display_F = res.F[viable_sorted] if viable_sorted else res.F
    label_str = "Feasible Tours" if viable_sorted else "Pareto Archive Solutions"
    plot = Scatter(
        title="BMOPSO on Multiobjective Traveling Salesman Problem (MSTSP)",
        labels=["Travel Distance (km)", "Transit / Toll Cost ($)"],
    )
    plot.add(display_F, color="forestgreen", label=label_str)
    print("  Scatter plot initialized successfully via pymoo.visualization.scatter.Scatter.")
    print("  (Uncomment 'plot.show()' to open the interactive graphical window).")
    # plot.show()


    print("\n" + "=" * 80)
    print(" MSTSP BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

