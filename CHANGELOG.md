# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-09-13

### Changed
- All algorithm draws now use pymoo's `self.random_state` (`np.random.Generator`) so `minimize(..., seed=s)` is reproducible.
- Vectorized personal-best updates, grid coordinate mapping, and hypercube grouping without changing Coello/Deb rules.

### Added
- `dominates_mask` for batch constrained-dominance checks.
- Reproducibility test: two independent `minimize(..., seed=42)` runs must match `X`, `F`, and `V`.

---

## [1.0.1] - 2026-09-11

### Changed
- Relicensed project from MIT License to Apache License 2.0 (`Apache-2.0`).
- Bumped package version to `1.0.1`.

---

## [1.0.0] - 2026-08-31

### Added
- **Core Algorithm (`BMOPSO_CE`)**:
  - Implemented Binary Multi-Objective Particle Swarm Optimization with Catfish Effect (originally published as `CatfishBMOPSO` in Souza et al., WTF 2014, standardized to `BMOPSO_CE`):
    - **Catfish Effect Operator** (*Chuang et al., 2011; Souza et al., WTF 2014, DOI: [10.5753/wtf.2014.22943](https://doi.org/10.5753/wtf.2014.22943)*):
      - Stagnation detection on non-dominated Pareto archive updates across consecutive iterations (`catfish_threshold`).
      - Constrained-dominance ranking to identify the worst 10% particles (`catfish_rate`).
      - Injection of catfish particles at extreme binary positions (all-0s, all-1s, alternating patterns, extreme hypercube vertices).
      - Velocity and personal best (`pbest`) re-initialization to break out of local optima and boost swarm search vitality.
    - **BMOPSO Core Foundations** (*Souza et al., ICTAI 2011, DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45)*):
      - Continuous velocity to binary position mapping via logistic sigmoid activation function (*Kennedy & Eberhart, 1997*).
      - External Pareto archive with Adaptive Hypercube Grid partitioning, hypercube fitness-based leader selection, and crowded hypercube capacity pruning (*Coello Coello et al., 2004*).
      - Personal best update with 50/50 coin-flip selection for mutually non-dominated positions (*Coello Coello et al., 2004*).
      - Constrained-Dominance Principle handling inequality constraints without penalty parameter tuning (*Deb, 2002*).
  - Direct inheritance from `pymoo.core.algorithm.Algorithm`.

- **Operators Subpackage (`bmopso_ce.operators`)**:
  - `catfish`: Catfish effect trigger, stagnation detection, worst particle identification, and extreme binary point initialization.
  - `velocity`: Clamped velocity update with dynamic linear inertia weight decay ($w_{\text{max}} \to w_{\text{min}}$).
  - `sampling`: Numerically stable sigmoid transform and boolean position sampling.
  - `mutation`: Non-linear decaying mutation / turbulence probability (*Coello Coello et al., 2004*).
  - `pbest`: Personal best replacement with Coello Coello (2004) random selection for incomparable states.

- **Utilities Subpackage (`bmopso_ce.util`)**:
  - `dominance`: Kalyanmoy Deb's constrained Pareto dominance checks and filtering.
  - `grid`: `AdaptiveGrid` hypercube objective space partitioning, hypercube fitness calculation ($10/N$), roulette leader selection, and crowded hypercube pruning.
  - `archive`: Dynamic `NonDominatedArchive` with change detection for stagnation monitoring.

- **Benchmark Integration & Examples**:
  - Seamless integration with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-poblems) benchmark suite (`MKP`, `MUBQP`, `MSTSP`, `MOSCP`, `MOFS`).
  - Standalone executable examples for all 5 benchmarks in `examples/` with native `pymoo.visualization.scatter.Scatter` plots.
  - Full pytest test suite in `tests/`.
