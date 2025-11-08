"""
Multi-Agent System for Mathematical Conjecture Discovery
"""

from src.agents.base import (
    BaseAgent,
    AgentRole,
    AgentStatus,
    AgentCapabilities,
    AgentPersonality,
    AgentMetrics,
    Tool,
    ToolParameter,
    Thought,
    Memory
)

from src.agents.researcher import (
    ResearcherAgent,
    PaperAnalysis,
    MathematicalEntity
)

from src.agents.reviewer import (
    ReviewerAgent,
    ValidationResult,
    ProofVerification,
    CrossValidation
)

from src.agents.synthesizer import (
    SynthesizerAgent,
    Pattern,
    Conjecture,
    Insight,
    ConnectionGraph
)

from src.agents.challenger import (
    ChallengerAgent,
    Challenge,
    CounterExample,
    EdgeCase,
    AdversarialTest
)

from src.agents.historian import (
    HistorianAgent,
    HistoricalEvent,
    ConceptEvolution,
    Timeline,
    DevelopmentPattern,
    ImpactAnalysis
)

from src.agents.coordinator import (
    AgentCoordinator,
    WorkflowStage,
    CollaborationPattern,
    Task,
    WorkflowState
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentRole",
    "AgentStatus",
    "AgentCapabilities",
    "AgentPersonality",
    "AgentMetrics",
    "Tool",
    "ToolParameter",
    "Thought",
    "Memory",

    # Researcher
    "ResearcherAgent",
    "PaperAnalysis",
    "MathematicalEntity",

    # Reviewer
    "ReviewerAgent",
    "ValidationResult",
    "ProofVerification",
    "CrossValidation",

    # Synthesizer
    "SynthesizerAgent",
    "Pattern",
    "Conjecture",
    "Insight",
    "ConnectionGraph",

    # Challenger
    "ChallengerAgent",
    "Challenge",
    "CounterExample",
    "EdgeCase",
    "AdversarialTest",

    # Historian
    "HistorianAgent",
    "HistoricalEvent",
    "ConceptEvolution",
    "Timeline",
    "DevelopmentPattern",
    "ImpactAnalysis",

    # Coordinator
    "AgentCoordinator",
    "WorkflowStage",
    "CollaborationPattern",
    "Task",
    "WorkflowState"
]

# Version info
__version__ = "0.1.0"
__author__ = "Multi-Agent Conjecture Discovery Team"