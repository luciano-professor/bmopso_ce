# bmopso_ce — Binary Multi-Objective Particle Swarm Optimization with Catfish Effect for pymoo

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![pymoo](https://img.shields.io/badge/pymoo-%3E%3D0.6.0-orange.svg)](https://pymoo.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**BMOPSO-CE (`bmopso_ce`)** is an official, domain-agnostic **Binary Multi-Objective Particle Swarm Optimization with Catfish Effect** library built from the ground up for the [`pymoo`](https://pymoo.org/) multi-objective optimization framework.

> [!NOTE]
> **Nomenclature Note:**
> In the original publication (*Souza et al., WTF 2014*), this technique was originally designated as **`CatfishBMOPSO`** (or *Catfish BMOPSO*). For this library release, the name was officially updated and standardized to **`BMOPSO-CE`** (Binary Multi-Objective Particle Swarm Optimization with Catfish Effect) to align with standardized algorithmic naming conventions.

It provides a native `pymoo.core.algorithm.Algorithm` implementation of the **BMOPSO-CE** algorithm, designed to solve complex binary and combinatorial multi-objective optimization problems while avoiding premature convergence and local optima through the bio-inspired **Catfish Effect**. It seamlessly integrates with the [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-problems) benchmark suite.

---

## 🌟 Seamless `pymoo` Integration

`bmopso_ce` is engineered strictly as a first-class citizen of the `pymoo` ecosystem:

```text
+-----------------------------------------------------------------------------+
|                               pymoo Ecosystem                               |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |                        pymoo.optimize.minimize                      |   |
|   |                                                                     |   |
|   |   +----------------------+               +----------------------+   |   |
|   |   |      BMOPSO_CE       |               |    BinaryProblem     |   |   |
|   |   |  (pymoo.Algorithm)   | ------------> |   (pymoo.Problem)    |   |   |
|   |   +----------------------+               +----------------------+   |   |
|   |              |                                      |               |   |
|   |              v                                      v               |   |
|   |       Population & Archive                  out["F"] & out["G"]     |   |
|   +---------------------------------------------------------------------+   |
|                                                                             |
|   +------------------------+  +---------------------+  +----------------+   |
|   |   pymoo.indicators     |  | pymoo.visualization |  |  pymoo.core    |   |
|   |   (Hypervolume, IGD)   |  | (Scatter, Petal)    |  |  (Callback)    |   |
|   +------------------------+  +---------------------+  +----------------+   |
+-----------------------------------------------------------------------------+
```

### Key Compatibility Highlights with `pymoo`:

- **Drop-in `minimize()` Execution**: Use standard `pymoo.optimize.minimize(problem, algorithm, termination)` syntax.
- **Native Problem Architecture**: All problems inherit from `BinaryProblem`, defining `type_var=np.bool_`, `xl=0`, `xu=1`, and utilizing standard `out["F"]`, `out["G"]`, and `out["H"]` output mappings.
- **`pymoo` Termination Criteria**: Fully compatible with all `pymoo` termination formats (`("n_gen", 100)`, `("n_eval", 20000)`, `get_termination("time", "00:05:00")`, `RobustTermination`).
- **`pymoo` Callbacks & Logging**: Integrate custom `pymoo.core.callback.Callback` classes and real-time evolutionary displays.
- **`pymoo` Performance Indicators**: Calculate convergence metrics using `pymoo.indicators.hv.Hypervolume`, `IGD`, and `IGDPlus`.
- **`pymoo` Visualizations**: Instantly plot resulting non-dominated Pareto Fronts using `pymoo.visualization.scatter.Scatter`.

---

## 📚 Companion Benchmark Suite: `pymoo-binary-problems`

`bmopso_ce` pairs natively with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-problems), providing 5 standardized, vectorized binary benchmark problems:

| Problem | Class | Variables | Description | Constraints |
| :--- | :--- | :---: | :--- | :---: |
| **Multiple Knapsack** | `MKP` | N x M bits | Multi-item allocation across multiple capacity-constrained knapsacks | M + N inequalities |
| **Unconstrained Quadratic** | `MUBQP` | n bits | Multi-objective unconstrained binary quadratic interaction matrices | Unconstrained |
| **Traveling Salesman** | `MSTSP` / `MOTSP` | N^2 bits | Binary position-city assignment matrix routing over N cities | 2N + 1 inequalities |
| **Set Covering** | `MOSCP` / `MSCP` | n bits | Minimum-cost subset selection covering m universe elements | m inequalities |
| **Feature Selection** | `MOFS` / `MOBFS` | D bits | Multi-objective classification error vs. dimensionality reduction | >= k_min features |

---

## 🧬 Algorithm Background & Key Pillars

The **BMOPSO-CE** algorithm combines the foundational **BMOPSO** formulation with the **Catfish Effect** perturbation mechanism:

1. **BMOPSO-CE Formulation (originally published as `CatfishBMOPSO`)**:
   * Proposed by **Luciano S. de Souza, Ricardo B. C. Prudêncio, and Flávia de A. Barros** in:
     > **"Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies"**, published in the *XV Workshop de Testes e Tolerância a Falhas (WTF 2014)*. DOI: [10.5753/wtf.2014.22943](https://doi.org/10.5753/wtf.2014.22943). *(Note: Designated as CatfishBMOPSO in the publication and renamed to BMOPSO-CE in this software package).*
2. **Original BMOPSO Framework**:
   * Proposed by **Luciano S. de Souza, Péricles B. C. de Miranda, Ricardo B. C. Prudêncio, and Flávia de A. Barros** in:
     > **"A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on Functional Requirements Coverage and Execution Effort"**, published in the *2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2011)*. DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45).
3. **Catfish Effect in Binary PSO**:
   * Proposed by **Li-Yeh Chuang, Sheng-Wei Tsai, and Cheng-Hong Yang** in:
     > **"Improved binary particle swarm optimization using catfish effect for feature selection"**, published in *Expert Systems with Applications*, 38(10), pp. 12699-12707, 2011. DOI: [10.1016/j.eswa.2011.04.057](https://doi.org/10.1016/j.eswa.2011.04.057).
4. **MOPSO with Adaptive Hypercube Grid**:
   * Proposed by **Carlos A. Coello Coello, Gregorio Toscano Pulido, and Maximino Salazar Lechuga** in:
     > **"Handling multiple objectives with particle swarm optimization"**, published in *IEEE Transactions on Evolutionary Computation*, 8(3), pp. 256-279, 2004. DOI: [10.1109/TEVC.2004.826067](https://doi.org/10.1109/TEVC.2004.826067).
5. **Binary Particle Swarm Optimization (BPSO)**:
   * Proposed by **James Kennedy and Russell C. Eberhart** (1997) mapping continuous velocities to binary decisions via logistic sigmoid activation.
6. **Constrained-Dominance Principle**:
   * Proposed by **Kalyanmoy Deb** (2002) for constraint handling across inequality constraints without penalty tuning.

---

## 🐟 The Catfish Effect Mechanism

In standard Binary PSO, particles may rapidly cluster around a suboptimal region (local optimum), causing swarm stagnation where no new non-dominated solutions are discovered. 

Inspired by the biological phenomenon where introducing a catfish predator into a tank of sluggish sardines forces them to remain active and agile, the **Catfish Effect** introduces *catfish particles* to reinvigorate the swarm search dynamics.

```text
+-----------------------------------------------------------------------------+
|                         BMOPSO-CE EXECUTION PIPELINE                        |
|                                                                             |
|  [1. Swarm Initialization (X in {0,1}^D, V in [-Vmax, Vmax])]               |
|                         |                                                   |
|  [2. Evaluate Objectives & Constraints (F, CV)]                             |
|                         |                                                   |
|  [3. Update External Archive (EA) & Personal Bests (pbest)]                 |
|                         |                                                   |
|  [4. Check Stagnation Condition (EA changes)]                               |
|         |                                 |                                 |
|   (Archive improved)             (No improvement for c consecutive gens)    |
|         |                                 |                                 |
|   Reset stagnation               TRIGGER CATFISH EFFECT:                    |
|   counter = 0                    ---------------------------------------    |
|         |                        * Randomly select 10% particles from swarm |
|         |                        * Update ONLY positions to extreme points: |
|         |                          (e.g., [0,0...0], [1,1...1], alternating)|
|         |                        * Preserve particle velocities             |
|         |                        * No immediate re-evaluation               |
|         |                        * Reset stagnation counter = 0             |
|         +---------------------------------+                                 |
|                         |                                                   |
|  [5. Leader Selection (gbest) via Adaptive Hypercube Grid Roulette]         |
|                         |                                                   |
|  [6. Velocity Update with Clamping & Linear Inertia Decay]                  |
|                         |                                                   |
|  [7. Binary Position Sampling via Sigmoid Activation Function]              |
|                         |                                                   |
|  [8. Non-Linear Mutation / Turbulence (Coello Coello 2004)]                  |
|                         |                                                   |
|  [9. Loop until Termination Criteria Met]                                   |
+-----------------------------------------------------------------------------+
```

### Algorithmic Steps of the Catfish Effect:

1. **Stagnation Monitoring on the External Archive (EA)**:
   * In multi-objective optimization (MOO), the External Archive (EA) maintains all non-dominated solutions found during the search.
   * At each iteration, the algorithm checks whether the EA has received any newly dominating or non-dominated trade-offs.
   * If new solutions enter the archive, the stagnation counter is reset (`stagnation_counter = 0`).
   * If the archive remains unchanged, the counter increments (`stagnation_counter += 1`).

2. **Trigger Condition**:
   * When `stagnation_counter >= catfish_threshold` (default: `24` consecutive generations), the swarm is considered stagnated in a local basin of attraction, triggering the Catfish Effect.

3. **Random Particle Selection**:
   * Because all best non-dominated solutions are already preserved in the External Archive, the selection mechanism is kept lightweight and efficient by **randomly choosing particles** from the current swarm for replacement (governed by `catfish_rate`, default `0.10` or 10% of `n_particles`), avoiding computationally expensive dominance comparisons.

4. **Probabilistic Extreme Position Injection (Souza et al., WTF 2014)**:
   * **Only the particle positions are updated** to extreme boundary points in `{0, 1}^D`:
     - A random number $r \in [0, 1]$ is generated for each catfish particle.
     - **If $r > 0.5$**: The particle is placed at the upper extreme, setting all dimensions ($d$) to **1** (`[1, 1, ..., 1]`).
     - **If $r \le 0.5$**: The particle is placed at the lower extreme, setting all dimensions ($d$) to **0** (`[0, 0, ..., 0]`).

5. **Position-Only Update without Immediate Re-Evaluation**:
   * Particle velocities are preserved.
   * **No re-evaluation is performed upon introducing the catfish particles**; their positions enter the swarm and are naturally evaluated during the regular evolutionary loop of subsequent generations.
   * The stagnation counter is reset to `0`.

---

## 📐 Adaptive Hypercube Grid & Multi-Objective Components

```text
+-----------------------------------------------------------------------------+
|                     COELLO COELLO (2004) ADAPTIVE GRID                      |
|                                                                             |
|  Objective 2 ^                                                              |
|              |   [Grid 0,2]   |   [Grid 1,2]   |   [Grid 2,2] (Empty)       |
|              |                |     *  *       |                            |
|              |----------------+----------------+--------------------------  |
|              |   [Grid 0,1]   |   [Grid 1,1]   |   [Grid 2,1]               |
|              |     *          |   *  *  *  *   |     *                      |
|              |----------------+----------------+--------------------------  |
|              |   [Grid 0,0]   |   [Grid 1,0]   |   [Grid 2,0]               |
|              |     *  *       |    (Empty)     |     *  *  *                |
|              +------------------------------------------------------------> |
|             min(f1)                                                max(f1)  |
|                                                                Objective 1  |
+-----------------------------------------------------------------------------+
```

1. **Sigmoid Velocity-to-Position Mapping**:
   ```text
   P(x_id = 1) = sigmoid(v_id) = 1 / (1 + exp(-v_id))
   ```
2. **Velocity Clamping**: Bound velocities within `[-v_max, v_max]` (default `v_max = 4.0`), preventing sigmoid probability saturation.
3. **Linear Inertia Decay**: Linear transition from global exploration to local exploitation:
   ```text
   w(t) = w_max - (t / T_max) * (w_max - w_min)
   ```
4. **Non-Linear Mutation / Turbulence (Bit-Flip)**:
   ```text
   P_mut(t) = (1 - t / T_max) ** (5 / mutation_rate)
   ```
5. **Adaptive Hypercube Grid & Leader Selection**: Partition objective space into `n_grid` subdivisions per dimension. Hypercube fitness is inversely proportional to occupancy:
   ```text
   fitness_k = 10.0 / N_k
   ```
   Leaders (`gbest`) are sampled via Roulette Wheel selection over hypercubes, followed by uniform selection within the chosen hypercube.
6. **Mutually Non-Dominated `pbest` Selection**: If current position and `pbest` are mutually non-dominated, select randomly with probability `0.5`.
7. **Crowded Hypercube Pruning**: When the archive exceeds `max_archive_size`, solutions from the most crowded hypercubes are truncated.
8. **Constrained-Dominance Principle (Deb, 2002)**: Guaranteed feasibility dominance ($g(x) <= 0$) without penalty parameter tuning.

---

## ⚙️ Hyperparameters

| Parameter | Default Value | Type | Description |
| :--- | :---: | :---: | :--- |
| `n_particles` | `20` | `int` | Swarm population size |
| `w_max` | `0.9` | `float` | Initial inertia weight (global exploration) |
| `w_min` | `0.4` | `float` | Final inertia weight (local exploitation) |
| `w` | `None` | `float \| None` | If set, enforces constant inertia weight (`w_max = w_min = w`) |
| `c1` | `1.49` | `float` | Cognitive acceleration coefficient (attraction to `pbest`) |
| `c2` | `1.49` | `float` | Social acceleration coefficient (attraction to archive leader `gbest`) |
| `v_max` | `4.0` | `float` | Velocity clamping limit `[-v_max, v_max]` |
| `mutation_rate` | `0.5` | `float \| None` | Mutation / turbulence probability (*bit-flip*). Default `0.5` |
| `n_grid` | `30` | `int` | Number of subdivisions per objective for Adaptive Hypercube Grid |
| `max_archive_size`| `200` | `int \| None` | Maximum capacity of the external Pareto archive |
| `catfish_threshold` | `24` | `int` | Consecutive iterations without archive improvement before triggering Catfish Effect |
| `catfish_rate` | `0.10` | `float` | Fraction of swarm particles randomly replaced by catfish particles (e.g. 10%) |
| `return_least_infeasible`| `True` | `bool` | Return least infeasible solutions if no fully feasible solution is found |

---

## 🚀 Installation

### 1. Direct from GitHub

```bash
pip install git+https://github.com/luciano-professor/bmopso_ce.git
```

### 2. From PyPI

```bash
pip install bmopso-ce
```

### 3. Local Development / Editable Mode

```bash
git clone https://github.com/luciano-professor/bmopso_ce.git
cd bmopso_ce
pip install -e ".[dev]"
```

---

## 💡 Quick Start with `pymoo`

### 1. Custom Binary Problem Optimization & Visualization

```python
from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import BinaryProblem
from bmopso_ce import BMOPSO_CE


class CustomBinaryProblem(BinaryProblem):
    def __init__(self, n_var: int = 20) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Objective 1: Count active 1s
        f1 = np.sum(x, axis=1)
        # Objective 2: Count active 0s
        f2 = np.sum(~x if x.dtype == bool else (1 - x), axis=1)
        out["F"] = np.column_stack([f1, f2])


# Instantiate pymoo problem and BMOPSO-CE algorithm
problem = CustomBinaryProblem(n_var=20)
algorithm = BMOPSO_CE(
    n_particles=30,
    mutation_rate=0.5,
    n_grid=30,
    catfish_threshold=24,
    catfish_rate=0.10,
)

# Optimize using pymoo.optimize.minimize
res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 40),
    seed=42,
    verbose=True,
)

print(f"Found {len(res.X)} non-dominated solutions.")
print("Objectives (F):\n", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE Pareto Front on Custom Binary Problem")
plot.add(res.F, color="blue", label="Pareto Solutions")
plot.show()
```

---

### 2. Multiple Knapsack Problem (MKP)

```python
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MKP

profits = np.array([12, 18, 25, 30, 42, 15, 28, 35, 40, 50], dtype=float)
weights = np.array([4, 8, 12, 16, 20, 6, 14, 18, 22, 26], dtype=float)
capacities = np.array([35.0, 45.0, 25.0], dtype=float)  # 3 knapsacks

problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)
algorithm = BMOPSO_CE(n_particles=25, mutation_rate=0.5, n_grid=30, catfish_threshold=24)

res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Best Allocations (X):", res.X.shape)
print("Objectives [-Profit, Weight] (F):", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE on MKP", labels=["Total Profit ($)", "Total Weight (kg)"])
plot.add(np.column_stack([-res.F[:, 0], res.F[:, 1]]), color="blue", label="Pareto Solutions")
plot.show()
```

---

### 3. Multiobjective Unconstrained Binary Quadratic Problem (MUBQP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MUBQP

# Generate a synthetic MUBQP benchmark instance (50 binary variables, 2 objectives)
problem = MUBQP.from_random(
    n_var=50,
    n_obj=2,
    density=0.8,
    val_range=(-100.0, 100.0),
    symmetric=True,
    maximize=True,
    seed=42,
)

algorithm = BMOPSO_CE(n_particles=40, mutation_rate=0.5, n_grid=30, catfish_threshold=24)
res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Solutions (X):", res.X.shape)
print("Pareto Objectives [-f1, -f2] (F):", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE on MUBQP", labels=["Objective 1 (f1)", "Objective 2 (f2)"])
plot.add(-res.F, color="crimson", label="Pareto Front")
plot.show()
```

---

### 4. Multiobjective Traveling Salesman Problem (MSTSP / MOTSP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MSTSP

# Generate a synthetic MSTSP benchmark instance (6 cities, 2 objectives -> 36 binary variables)
problem = MSTSP.from_random(n_cities=6, n_obj=2, dist_range=(10.0, 100.0), seed=42)
algorithm = BMOPSO_CE(n_particles=40, mutation_rate=0.5, n_grid=30, catfish_threshold=24)

res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Tours (X):", res.X.shape)
print("Objectives [Distance, Cost] (F):", res.F)

# Decode best tour into sequence of city indices
best_tour, is_valid = problem.decode_tour(res.X[0])
print(f"Decoded Tour (Valid={is_valid}):", best_tour)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE on MSTSP", labels=["Travel Distance (km)", "Transit Cost ($)"])
plot.add(res.F, color="forestgreen", label="Feasible Tours")
plot.show()
```

---

### 5. Multiobjective Set Covering Problem (MOSCP / MSCP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MOSCP

# Generate synthetic MOSCP instance (25 zones to cover, 35 candidate facility subsets)
problem = MOSCP.from_random(
    n_elements=25,
    n_subsets=35,
    n_obj=2,
    density=0.25,
    cost_range=(15.0, 95.0),
    seed=42,
)

algorithm = BMOPSO_CE(n_particles=40, mutation_rate=0.5, n_grid=30, catfish_threshold=24)
res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Facility Subsets (X):", res.X.shape)
print("Objectives [Capital Cost, Ops Cost] (F):", res.F)

# Decode coverage details of best solution
coverage_info = problem.decode_coverage(res.X[0])
print("Selected Facilities:", coverage_info["selected_subsets"])
print(f"Coverage Feasibility (100% Valid): {coverage_info['is_valid']}")

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE on MOSCP", labels=["Capital Cost ($k)", "Operational Cost ($k/yr)"])
plot.add(res.F, color="purple", label="Feasible Cover Subsets")
plot.show()
```

---

### 6. Multiobjective Feature Selection (MOFS / MOBFS)

```python
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from bmopso_ce import BMOPSO_CE
from pymoo_binary_problems import MOFS

# Generate synthetic classification dataset (200 samples, 30 features, 8 informative)
problem = MOFS.from_synthetic(
    n_samples=200,
    n_features=30,
    n_informative=8,
    n_redundant=4,
    cv=3,
    seed=42,
)

algorithm = BMOPSO_CE(n_particles=40, mutation_rate=0.5, n_grid=30, catfish_threshold=24)
res = minimize(problem, algorithm, termination=("n_gen", 30), verbose=True)

print("Pareto Feature Masks (X):", res.X.shape)
print("Objectives [Error Rate, Feature Ratio] (F):", res.F)

# Decode feature details of best accuracy solution
feature_info = problem.decode_features(res.X[0])
print("Selected Feature Indices:", feature_info["selected_features"])
print(f"Classification Accuracy: {feature_info['accuracy'] * 100:.2f}%")

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CE on MOFS", labels=["Accuracy (%)", "Feature Ratio (%)"])
plot.add(
    np.column_stack([(1.0 - res.F[:, 0]) * 100.0, res.F[:, 1] * 100.0]),
    color="darkorange",
    label="Pareto Subsets",
)
plot.show()
```

---

## 🏃 Running Examples

Execute any of the standalone benchmark walkthrough scripts:

```bash
# 1. Multiple Knapsack Problem
python examples/run_mkp_example.py

# 2. Multiobjective Unconstrained Binary Quadratic Problem
python examples/run_mubqp_example.py

# 3. Multiobjective Traveling Salesman Problem
python examples/run_mstsp_example.py

# 4. Multiobjective Set Covering Problem
python examples/run_moscp_example.py

# 5. Multiobjective Feature Selection
python examples/run_mofs_example.py
```

---

## 🧪 Running Tests

Execute the full test suite with `pytest`:

```bash
pytest
```

---

## 📖 References

1. **BMOPSO-CE with Catfish Effect**:
   * Souza, L. S., Prudêncio, R. B. C., & Barros, F. A. (2014). *Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies*. In: Anais do XV Workshop de Testes e Tolerância a Falhas (WTF 2014), SBC, Florianópolis, SC, Brasil. DOI: [10.5753/wtf.2014.22943](https://doi.org/10.5753/wtf.2014.22943).
2. **Original BMOPSO Proposal**:
   * Souza, L. S., Miranda, P. B. C., Prudêncio, R. B. C., & Barros, F. A. (2011). *A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on Functional Requirements Coverage and Execution Effort*. In: 2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2011), IEEE, pp. 245-252. DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45).
3. **Catfish Effect in Binary Particle Swarm Optimization**:
   * Chuang, L. Y., Tsai, S. W., & Yang, C. H. (2011). *Improved binary particle swarm optimization using catfish effect for feature selection*. Expert Systems with Applications, 38(10), 12699-12707. DOI: [10.1016/j.eswa.2011.04.057](https://doi.org/10.1016/j.eswa.2011.04.057).
4. **Binary Particle Swarm Optimization (BPSO)**:
   * Kennedy, J., & Eberhart, R. C. (1997). *A discrete binary version of the particle swarm algorithm*. In: 1997 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Computational Cybernetics and Simulation, IEEE, 4, 4104-4108.
5. **Multi-Objective Particle Swarm Optimization (MOPSO) & Adaptive Hypercube Grid**:
   * Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). *Handling multiple objectives with particle swarm optimization*. IEEE Transactions on Evolutionary Computation, 8(3), 256-279. DOI: [10.1109/TEVC.2004.826067](https://doi.org/10.1109/TEVC.2004.826067).
6. **Constrained-Dominance Principle**:
   * Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
7. **pymoo Framework**:
   * Blank, J., & Deb, K. (2020). *pymoo: Multi-Objective Optimization in Python*. IEEE Access, 8, 89497-89509. DOI: [10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567).

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

