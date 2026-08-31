"""Example script demonstrating BMOPSO on the Multiobjective Unconstrained Binary Quadratic Problem (MUBQP).

This script provides a practical walkthrough to:
1. Generate and configure a Multiobjective Unconstrained Binary Quadratic Problem (MUBQP) benchmark instance.
2. Configure the BMOPSO algorithm hyperparameters.
3. Execute multiobjective optimization using pymoo.optimize.minimize.
4. Analyze the resulting non-dominated Pareto Front, inspecting objective trade-offs and decision vectors.

To execute:
    python examples/run_mubqp_example.py
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
from pymoo_binary_problems import MUBQP



def create_sample_mubqp_instance() -> MUBQP:
    """Create a sample instance of the Multiobjective Unconstrained Binary Quadratic Problem.

    Scenario:
    - 50 binary decision variables (e.g., selecting features, components, or investments).
    - 2 conflicting quadratic objectives (e.g., Maximizing Return vs. Maximizing Synergy/Reliability).
    - Density = 0.8 (80% non-zero pairwise interaction terms).
    - Interaction values sampled in range [-100.0, 100.0].

    Returns
    -------
    MUBQP
        Configured MUBQP benchmark problem instance.
    """
    n_var = 50
    n_obj = 2
    density = 0.8
    val_range = (-100.0, 100.0)

    problem = MUBQP.from_random(
        n_var=n_var,
        n_obj=n_obj,
        density=density,
        val_range=val_range,
        symmetric=True,
        maximize=True,
        seed=42,
    )
    return problem


def print_solution_details(
    sol_idx: int,
    x: np.ndarray,
    f_val: np.ndarray,
    problem: MUBQP,
) -> None:
    """Print detailed decision and objective values for a candidate Pareto solution.

    Parameters
    ----------
    sol_idx : int
        Identifying rank or index of the solution.
    x : np.ndarray
        1D binary decision vector.
    f_val : np.ndarray
        1D objective vector [f1, f2, ...].
    problem : MUBQP
        MUBQP problem instance.
    """
    real_f1 = -f_val[0] if problem.maximize else f_val[0]
    real_f2 = -f_val[1] if problem.maximize else f_val[1]
    active_bits = int(np.sum(x))
    active_indices = np.where(x)[0].tolist()

    print(f"\n--- Solution #{sol_idx} ---")
    print(f"  Objective 1 (f1)    : {real_f1:10.2f}")
    print(f"  Objective 2 (f2)    : {real_f2:10.2f}")
    print(f"  Active Bits (x_i=1) : {active_bits}/{problem.n_var} ({active_bits / problem.n_var * 100:.1f}%)")
    indices_str = ", ".join(map(str, active_indices[:15]))
    if len(active_indices) > 15:
        indices_str += f", ... (+{len(active_indices) - 15} more)"
    print(f"  Selected Variables  : [{indices_str}]")


def main() -> None:
    """Execute complete MUBQP multiobjective optimization workflow using BMOPSO_CE."""
    print("=" * 80)
    print(" MULTIOBJECTIVE UNCONSTRAINED BINARY QUADRATIC PROBLEM (MUBQP) WITH BMOPSO-CE")
    print("=" * 80)

    # 1. Create MUBQP problem instance
    problem = create_sample_mubqp_instance()
    print("\nMUBQP Problem initialized successfully:")
    print(f"  - Decision Variables (n_var) : {problem.n_var} bits")
    print(f"  - Objective Count (n_obj)    : {problem.n_obj}")
    print(f"  - Formulation                : Maximization (negated for pymoo minimization)")
    print(f"  - Constraints                : Unconstrained (n_constr = 0)")
    print(f"  - Matrix Shape per Objective : ({problem.n_var}, {problem.n_var})")

    # 2. Configure BMOPSO_CE algorithm
    algorithm = BMOPSO_CE(
        n_particles=60,       # Swarm population size
        w_max=0.9,            # Initial inertia weight (global exploration)
        w_min=0.4,            # Final inertia weight (local exploitation)
        c1=1.49,              # Cognitive acceleration coefficient
        c2=1.49,              # Social acceleration coefficient
        v_max=4.0,            # Velocity clamping limit [-4.0, 4.0]
        mutation_rate=0.5,    # Non-linear turbulence / bit-flip probability
        max_archive_size=100, # Capacity of external Pareto archive
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

    # 4. Analyze non-dominated Pareto Front
    n_solutions = len(res.X)
    print(f"\nTotal Non-Dominated Solutions on Pareto Front: {n_solutions}")

    # Convert objectives back to maximization values for reporting
    real_f1 = -res.F[:, 0] if problem.maximize else res.F[:, 0]
    real_f2 = -res.F[:, 1] if problem.maximize else res.F[:, 1]

    # Sort solutions by Objective 1 descending
    sorted_indices = np.argsort(-real_f1)

    print("\n" + "=" * 80)
    print(" SUMMARY OF NON-DOMINATED PARETO FRONT SOLUTIONS")
    print("=" * 80)
    print(f" {'#':<3} | {'Objective 1 (f1)':<18} | {'Objective 2 (f2)':<18} | {'Active Bits':<12} | {'Density (%)'}")
    print("-" * 80)

    for rank, idx in enumerate(sorted_indices[:12], 1):
        f1_val = real_f1[idx]
        f2_val = real_f2[idx]
        active = int(np.sum(res.X[idx]))
        pct = active / problem.n_var * 100
        print(f" {rank:<3} | {f1_val:<18.2f} | {f2_val:<18.2f} | {active:<12} | {pct:5.1f}%")

    if n_solutions > 12:
        print(f" ... ({n_solutions - 12} additional Pareto solutions omitted for brevity)")

    # 5. Inspect specific extreme and balanced trade-off solutions
    print("\n" + "=" * 80)
    print(" KEY TRADE-OFF SOLUTIONS INSPECTION")
    print("=" * 80)

    # Extreme solution favoring f1
    best_f1_idx = int(np.argmax(real_f1))
    print("\n[Extreme Solution - Best Objective 1]:")
    print_solution_details(1, res.X[best_f1_idx], res.F[best_f1_idx], problem)

    # Extreme solution favoring f2
    best_f2_idx = int(np.argmax(real_f2))
    print("\n[Extreme Solution - Best Objective 2]:")
    print_solution_details(2, res.X[best_f2_idx], res.F[best_f2_idx], problem)

    # Balanced / median trade-off solution
    median_idx = sorted_indices[len(sorted_indices) // 2]
    # 6. Visualize Pareto Front using pymoo's native plotting tools
    print("\n" + "=" * 80)
    print(" PARETO FRONT VISUALIZATION (pymoo.visualization.scatter.Scatter)")
    print("=" * 80)
    plot_F = np.column_stack([real_f1, real_f2])
    plot = Scatter(
        title="BMOPSO on Multiobjective UBQP",
        labels=["Objective 1 (f1 - Maximized)", "Objective 2 (f2 - Maximized)"],
    )
    plot.add(plot_F, color="crimson", label="Pareto Solutions")
    print("  Scatter plot initialized successfully via pymoo.visualization.scatter.Scatter.")
    print("  (Uncomment 'plot.show()' to open the interactive graphical window).")
    # plot.show()

    print("\n" + "=" * 80)
    print(" MUBQP BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

