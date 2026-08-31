"""Binary Multi-Objective Particle Swarm Optimization with Catfish Effect (BMOPSO-CE) for pymoo.

Main optimization algorithm module implementing BMOPSO-CE by combining:
1. Binary PSO (Kennedy & Eberhart, 1997)
2. MOPSO with Adaptive Hypercube Grid (Coello Coello, Pulido & Lechuga, 2004)
3. Constrained-Dominance Principle (Deb, 2002)
4. Catfish Effect Operator (Chuang et al., 2011; Souza et al., WTF 2014, DOI: 10.5753/wtf.2014.22943)
"""

from __future__ import annotations

from typing import Any
import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population

from bmopso_ce.operators.catfish import apply_catfish_effect
from bmopso_ce.operators.mutation import apply_mutation
from bmopso_ce.operators.pbest import update_personal_bests
from bmopso_ce.operators.sampling import sample_binary_positions
from bmopso_ce.operators.velocity import update_velocity
from bmopso_ce.util.archive import NonDominatedArchive

__all__ = ["BMOPSO_CE"]


class BMOPSO_CE(Algorithm):
    """Binary Multiobjective Particle Swarm Optimization with Catfish Effect (BMOPSO-CE).

    Originally proposed and analyzed by Luciano S. de Souza, Ricardo B. C. Prudêncio,
    and Flávia de A. Barros in:
        "Multi-Objective Test Case Selection: A study of the influence of the Catfish effect
        on PSO based strategies", published in the XV Workshop de Testes e Tolerância a Falhas
        (WTF 2014), SBC. DOI: https://doi.org/10.5753/wtf.2014.22943

    Combining:
    1. BMOPSO Core Framework (L. S. Souza, P. B. C. Miranda, R. B. C. Prudêncio, and F. A. Barros, 2011):
       - Binary PSO with sigmoid activation mapping (J. Kennedy and R. C. Eberhart, 1997).
       - MOPSO with Adaptive Hypercube Grid, roulette leader selection, and crowded pruning
         (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004).
       - Constrained-Dominance Principle (K. Deb, 2002).
    2. Catfish Effect Operator (L. Y. Chuang, S. W. Tsai, and C. H. Yang, 2011; Souza et al., WTF 2014):
       - Stagnation detection on non-dominated Pareto archive updates.
       - Worst particle replacement at extreme points of the binary hypercube.
       - Velocity and personal best (pbest) reset to escape local optima.

    Parameters
    ----------
    n_particles : int, default=20
        Swarm population size.
    w_max : float, default=0.9
        Initial inertia weight at the beginning of optimization (high global exploration).
    w_min : float, default=0.4
        Final inertia weight at the end of optimization (local exploitation and fine-tuning).
    w : float | None, default=None
        If provided as a numerical value, fixes constant inertia weight (w_max = w_min = w).
        If None (default), inertia linearly decreases from w_max (0.9) to w_min (0.4).
    c1 : float, default=1.49
        Cognitive acceleration coefficient (attraction to personal best - pbest).
    c2 : float, default=1.49
        Social acceleration coefficient (attraction to archive leader - gbest).
    v_max : float, default=4.0
        Velocity clamping bound in [-v_max, v_max].
        Default of 4.0 bounds sigmoid probabilities to [sigmoid(-4) ≈ 0.018, sigmoid(4) ≈ 0.982].
    mutation_rate : float | None, default=0.5
        Mutation / turbulence rate (bit-flip) applied to escape local optima
        (Coello Coello et al., 2004). If None, uses adaptive 1 / n_var.
    n_grid : int, default=30
        Number of grid subdivisions along each objective dimension for the Adaptive Hypercube
        (Coello Coello et al., 2004).
    max_archive_size : int | None, default=200
        Maximum capacity of non-dominated solutions stored in the external archive.
        When exceeded, solutions from the most crowded hypercubes are pruned.
    catfish_threshold : int, default=24
        Number of consecutive generations without non-dominated archive improvement
        before triggering the Catfish Effect perturbation.
    catfish_rate : float, default=0.10
        Fraction of the swarm particles to be randomly replaced by catfish
        particles upon stagnation (default: 0.10 or 10%).
    return_least_infeasible : bool, default=True
        If True, returns least infeasible solutions when no 100% feasible solution is found.
    **kwargs : Any
        Additional keyword arguments passed to pymoo Algorithm base class.
    """

    def __init__(
        self,
        n_particles: int = 20,
        w_max: float = 0.9,
        w_min: float = 0.4,
        w: float | None = None,
        c1: float = 1.49,
        c2: float = 1.49,
        v_max: float = 4.0,
        mutation_rate: float | None = 0.5,
        n_grid: int = 30,
        max_archive_size: int | None = 200,
        catfish_threshold: int = 24,
        catfish_rate: float = 0.10,
        return_least_infeasible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(return_least_infeasible=return_least_infeasible, **kwargs)
        self.n_particles: int = n_particles
        self.w_max: float = float(w) if w is not None else float(w_max)
        self.w_min: float = float(w) if w is not None else float(w_min)
        self.w: float = self.w_max
        self.c1: float = c1
        self.c2: float = c2
        self.v_max: float = v_max
        self.mutation_rate: float | None = mutation_rate
        self.n_grid: int = n_grid
        self.max_archive_size: int | None = max_archive_size
        self.catfish_threshold: int = catfish_threshold
        self.catfish_rate: float = catfish_rate

        # External non-dominated archive manager with Adaptive Hypercube Grid
        self.archive: NonDominatedArchive = NonDominatedArchive(
            max_size=max_archive_size,
            n_grid=n_grid,
        )

        # Internal swarm state
        self.X: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.pbest_X: np.ndarray | None = None
        self.pbest_F: np.ndarray | None = None
        self.pbest_CV: np.ndarray | None = None

        # Catfish Effect state tracking
        self.stagnation_counter: int = 0
        self.n_catfish_triggers: int = 0

    def _initialize(self) -> None:
        """Initialize particle positions, velocities, personal bests, and external archive."""
        super()._initialize()
        n_var: int = self.problem.n_var

        # 1. Random initialization of binary positions (0 or 1) and continuous velocities
        self.X = np.random.randint(0, 2, size=(self.n_particles, n_var)).astype(bool)
        self.V = np.random.uniform(-self.v_max, self.v_max, size=(self.n_particles, n_var))

        # 2. Initial evaluation of objectives and constraints via pymoo evaluator
        self.pop = Population.new(X=self.X)
        self.evaluator.eval(self.problem, self.pop)
        f_eval: np.ndarray = self.pop.get("F")
        cv_eval = self.pop.get("CV")
        cv_1d: np.ndarray = (
            np.squeeze(cv_eval).astype(float)
            if cv_eval is not None
            else np.zeros(self.n_particles, dtype=float)
        )
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        # 3. Personal best (pbest) initialization
        self.pbest_X = self.X.copy()
        self.pbest_F = f_eval.copy()
        self.pbest_CV = cv_1d.copy()

        # 4. External non-dominated archive initialization
        self.archive = NonDominatedArchive(max_size=self.max_archive_size, n_grid=self.n_grid)
        self.archive.update(self.X, f_eval, cv_1d)

        # 5. Initialize Catfish Effect tracking
        self.stagnation_counter = 0
        self.n_catfish_triggers = 0

        # 6. Synchronize pymoo optimum population
        self._set_optimum()

    def _update_archive(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray | None = None,
    ) -> bool:
        """Update external archive with candidate solutions."""
        return self.archive.update(x, f, cv)

    def _set_optimum(self) -> None:
        """Set the optimal non-dominated solution set from the external archive."""
        if not self.archive.is_empty():
            self.opt = self.archive.to_population()
        elif self.pop is not None:
            self.opt = self.pop

    def _advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        """Advance one iteration in the pymoo execution flow by executing _next()."""
        self._next()

    def _next(self) -> None:
        """Execute one evolutionary step of BMOPSO-CE."""
        if (
            self.X is None
            or self.V is None
            or self.pbest_X is None
            or self.pbest_F is None
            or self.archive.is_empty()
        ):
            raise RuntimeError("The algorithm must be initialized before calling _next().")

        # 1. Select social leaders (gbest) via Adaptive Hypercube Grid Roulette (Coello Coello et al., 2004)
        gbest = self.archive.select_leaders(self.n_particles)

        # 2. Compute linear decay of inertia weight w from w_max to w_min
        progress: float = 0.0
        if (
            self.termination is not None
            and hasattr(self.termination, "perc")
            and self.termination.perc is not None
        ):
            progress = float(np.clip(self.termination.perc, 0.0, 1.0))

        current_w: float = self.w_max - progress * (self.w_max - self.w_min)
        self.w = current_w

        # 3. Update continuous velocities with clamping
        self.V = update_velocity(
            v=self.V,
            x=self.X,
            pbest_x=self.pbest_X,
            gbest=gbest,
            w=current_w,
            c1=self.c1,
            c2=self.c2,
            v_max=self.v_max,
        )

        # 4. Map velocities to binary positions via sigmoid activation (Kennedy & Eberhart, 1997)
        self.X = sample_binary_positions(self.V)

        # 5. Apply non-linear mutation / turbulence operator (Coello Coello et al., 2004)
        self.X = apply_mutation(
            x=self.X,
            progress=progress,
            mutation_rate=self.mutation_rate,
        )

        # 6. Evaluate objectives and constraints of new positions via pymoo evaluator
        self.pop = Population.new(X=self.X)
        self.evaluator.eval(self.problem, self.pop)
        f_eval: np.ndarray = self.pop.get("F")
        cv_eval = self.pop.get("CV")
        cv_1d: np.ndarray = (
            np.squeeze(cv_eval).astype(float)
            if cv_eval is not None
            else np.zeros(self.n_particles, dtype=float)
        )
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        # 7. Update personal bests (pbest) following Coello Coello et al. (2004) with constraints
        self.pbest_X, self.pbest_F, self.pbest_CV = update_personal_bests(
            pbest_x=self.pbest_X,
            pbest_f=self.pbest_F,
            pbest_cv=self.pbest_CV,
            x=self.X,
            f=f_eval,
            cv=cv_1d,
        )

        # 8. Update external archive with new positions and constraint violations
        archive_changed = self.archive.update(self.X, f_eval, cv_1d)

        # 8.1. Stagnation tracking on non-dominated Pareto archive
        if archive_changed:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # 8.2. Apply Catfish Effect to escape local optima without re-evaluation (Souza et al., WTF 2014)
        if self.stagnation_counter >= self.catfish_threshold:
            self.X = apply_catfish_effect(
                x=self.X,
                catfish_rate=self.catfish_rate,
            )
            self.stagnation_counter = 0
            self.n_catfish_triggers += 1

        # 9. Synchronize pymoo population state
        self._set_optimum()
