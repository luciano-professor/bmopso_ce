"""Package bmopso_ce: Binary Multiobjective Particle Swarm Optimization with Catfish Effect for pymoo.

Implements the BMOPSO-CE algorithm proposed by Luciano S. de Souza, Ricardo B. C. Prudêncio,
and Flávia de A. Barros in:
"Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies",
XV Workshop de Testes e Tolerância a Falhas (WTF 2014), SBC.
DOI: https://doi.org/10.5753/wtf.2014.22943

Combining:
1. BMOPSO Core Framework (L. S. Souza, P. B. C. Miranda, R. B. C. Prudêncio, and F. A. Barros, ICTAI 2011).
2. Binary PSO with Sigmoid Activation (J. Kennedy and R. C. Eberhart, 1997).
3. MOPSO with Adaptive Hypercube Grid (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004).
4. Constrained-Dominance Principle (K. Deb, 2002).
5. Catfish Effect Operator (L. Y. Chuang, S. W. Tsai, and C. H. Yang, 2011; Souza et al., WTF 2014).
"""

from .algorithms.bmopso_ce import BMOPSO_CE
from .operators.catfish import (
    apply_catfish_effect,
    generate_extreme_binary_positions,
    select_random_particles,
)
from .util.archive import NonDominatedArchive
from .util.grid import AdaptiveGrid

__version__ = "1.0.1"

__all__ = [
    "__version__",
    "AdaptiveGrid",
    "BMOPSO_CE",
    "NonDominatedArchive",
    "apply_catfish_effect",
    "generate_extreme_binary_positions",
    "select_random_particles",
]
