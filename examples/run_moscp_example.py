"""Example script demonstrating BMOPSO on the Multiobjective Set Covering Problem (MOSCP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Set Covering Problem (MOSCP) instance.
2. Configure BMOPSO algorithm hyperparameters.
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting Pareto Front, verifying 100% coverage feasibility across all elements.
5. Inspect trade-offs between conflicting cost criteria (e.g. Capital Installation vs Maintenance).

To execute:
    python examples/run_moscp_example.py
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
from pymoo_binary_problems import MOSCP



def create_sample_moscp_instance() -> MOSCP:
    """Create a sample instance of the Multiobjective Set Covering Problem (MOSCP).

    Scenario: Emergency Facility Location & Sensor Network Coverage
    - 25 municipal zones / regions requiring emergency coverage (m = 25).
    - 35 candidate facility locations (subsets / variables, n = 35).
    - 2 conflicting cost objectives:
      * Objective 1: Capital Installation & Infrastructure Cost ($k)
      * Objective 2: Operational Maintenance & Response Risk ($k/yr)
    - Average coverage density: 25% (each facility covers ~6-7 zones).

    Returns
    -------
    MOSCP
        Configured MOSCP benchmark problem instance.
    """
    problem = MOSCP.from_random(
        n_elements=25,
        n_subsets=35,
        n_obj=2,
        density=0.25,
        cost_range=(15.0, 95.0),
        seed=42,
    )
    return problem


def print_solution_details(
    sol_idx: int,
    x_flat: np.ndarray,
    f_val: np.ndarray,
    problem: MOSCP,
) -> None:
    """Print detailed facility selection and coverage metrics for a candidate solution.

    Parameters
    ----------
    sol_idx : int
        Identifying index of the solution on the Pareto Front.
    x_flat : np.ndarray
        1D binary decision vector of length n_subsets.
    f_val : np.ndarray
        1D objective vector [cost_1, cost_2].
    problem : MOSCP
        MOSCP problem instance.
    """
    info = problem.decode_coverage(x_flat)
    status_str = "VALID (100% Covered)" if info["is_feasible"] else f"INVALID ({len(info['uncovered_elements'])} Uncovered)"
    selected_str = ", ".join(map(str, info["selected_subsets"]))

    print(f"\n--- Solution #{sol_idx} ---")
    print(f"  Installation Cost (Obj 1) : $ {f_val[0]:6.2f} k")
    print(f"  Operational Cost (Obj 2)  : $ {f_val[1]:6.2f} k/yr")
    print(f"  Selected Facilities       : {info['n_selected']}/{problem.n_subsets} ({info['n_selected'] / problem.n_subsets * 100:.1f}%)")
    print(f"  Coverage Feasibility      : {status_str}")
    print(f"  Chosen Facility Indices   : [{selected_str}]")


def main() -> None:
    """Execute complete MOSCP multiobjective optimization workflow using BMOPSO_CE."""
    print("=" * 80)
    print(" MULTIOBJECTIVE SET COVERING PROBLEM (MOSCP) WITH BMOPSO-CE")
    print("=" * 80)

    # 1. Create MOSCP problem instance
    problem = create_sample_moscp_instance()
    print("\nMOSCP Problem initialized successfully:")
    print(f"  - Elements to Cover (m)       : {problem.n_elements} zones")
    print(f"  - Candidate Subsets (n_var)   : {problem.n_subsets} facilities (35 bits)")
    print(f"  - Objective Count (F)         : {problem.n_obj} conflicting cost criteria")
    print(f"  - Inequality Constraints (G)  : {problem.n_ieq_constr} (1 per zone)")

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
    n_evals = 20000
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
        info = problem.decode_coverage(res.X[i])
        if info["is_feasible"]:
            viable_indices.append(i)
        else:
            infeasible_indices.append(i)

    n_total = len(res.X)
    n_viable = len(viable_indices)
    n_infeasible = len(infeasible_indices)

    print(f"\nTotal Solutions on Pareto Front : {n_total}")
    print(f"  |-- FEASIBLE Sets (100% Valid)  : {n_viable} ({n_viable / n_total * 100:.1f}%)")
    print(f"  \\-- INFEASIBLE (G Violations)   : {n_infeasible} ({n_infeasible / n_total * 100:.1f}%)")

    # Sort feasible solutions by Objective 1 (Installation Cost) ascending
    viable_sorted = sorted(viable_indices, key=lambda idx: res.F[idx, 0])

    # Filter unique solutions by objective vectors
    seen_f: set[tuple[float, ...]] = set()
    unique_viable: list[int] = []
    for idx in viable_sorted:
        key = tuple(np.round(res.F[idx], 2))
        if key not in seen_f:
            seen_f.add(key)
            unique_viable.append(idx)

    # 5. Summary table of feasible Pareto solutions
    print("\n" + "=" * 80)
    print(" SUMMARY OF TOP FEASIBLE PARETO SOLUTIONS FOUND")
    print("=" * 80)
    print(f" {'#':<3} | {'Install Cost ($k)':<20} | {'Ops Cost ($k/yr)':<20} | {'Subsets':<10} | {'Selected Indices'}")
    print("-" * 80)

    for rank, idx in enumerate(unique_viable[:12], 1):
        c1 = res.F[idx, 0]
        c2 = res.F[idx, 1]
        info = problem.decode_coverage(res.X[idx])
        subsets_str = str(info["selected_subsets"][:6])
        if len(info["selected_subsets"]) > 6:
            subsets_str = subsets_str[:-1] + ", ...]"
        print(f" {rank:<3} | $ {c1:<18.2f} | $ {c2:<18.2f} | {info['n_selected']:<10} | {subsets_str}")

    if not unique_viable:
        print("  No 100% feasible solution found in this run.")

    # 6. Detailed inspection of trade-off solutions
    print("\n" + "=" * 80)
    print(" DETAILED TRADE-OFF SOLUTIONS INSPECTION")
    print("=" * 80)

    if unique_viable:
        # Best Installation Cost solution
        best_c1_idx = unique_viable[0]
        print("\n[Best Installation Cost Solution]:")
        print_solution_details(1, res.X[best_c1_idx], res.F[best_c1_idx], problem)

        # Best Operational Cost solution
        best_c2_idx = min(unique_viable, key=lambda idx: res.F[idx, 1])
        print("\n[Best Operational Cost Solution]:")
        print_solution_details(2, res.X[best_c2_idx], res.F[best_c2_idx], problem)

        # Compromise solution
        compromise_idx = unique_viable[len(unique_viable) // 2]
        print("\n[Balanced Compromise Solution]:")
        print_solution_details(3, res.X[compromise_idx], res.F[compromise_idx], problem)
    elif len(res.X) > 0:
        print_solution_details(0, res.X[0], res.F[0], problem)

    # 7. Visualize Pareto Front using pymoo's native plotting tools
    print("\n" + "=" * 80)
    print(" PARETO FRONT VISUALIZATION (pymoo.visualization.scatter.Scatter)")
    print("=" * 80)
    if unique_viable:
        plot = Scatter(
            title="BMOPSO on Multiobjective Set Covering Problem (MOSCP)",
            labels=["Capital Installation Cost ($k)", "Operational & Risk Cost ($k/yr)"],
        )
        plot.add(res.F[unique_viable], color="purple", label="Feasible Cover Subsets")
        print("  Scatter plot initialized successfully via pymoo.visualization.scatter.Scatter.")
        print("  (Uncomment 'plot.show()' to open the interactive graphical window).")
        # plot.show()

    print("\n" + "=" * 80)
    print(" MOSCP BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

