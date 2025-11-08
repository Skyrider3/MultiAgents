"""
Mathematical Reasoning Module
"""

from src.reasoning.symbolic import (
    SymbolicReasoner,
    ProofVerifier,
    ConjectureValidator,
    PatternAnalyzer,
    EquationSolver,
    TheoremProver
)

from src.reasoning.numerical import (
    NumericalVerifier,
    StatisticalAnalyzer,
    ComputationalExplorer
)

from src.reasoning.geometric import (
    GeometricReasoner,
    TopologicalAnalyzer
)

__all__ = [
    # Symbolic
    "SymbolicReasoner",
    "ProofVerifier",
    "ConjectureValidator",
    "PatternAnalyzer",
    "EquationSolver",
    "TheoremProver",

    # Numerical
    "NumericalVerifier",
    "StatisticalAnalyzer",
    "ComputationalExplorer",

    # Geometric
    "GeometricReasoner",
    "TopologicalAnalyzer"
]