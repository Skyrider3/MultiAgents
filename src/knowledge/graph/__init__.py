"""
Knowledge Graph Module - Neo4j Integration
"""

from src.knowledge.graph.schema import (
    NodeType,
    RelationshipType,
    NodeProperties,
    RelationshipProperties,
    PaperNode,
    AuthorNode,
    ConceptNode,
    TheoremNode,
    ConjectureNode,
    ProofNode,
    DefinitionNode,
    FormulaNode,
    DomainNode,
    GraphConstraints,
    GraphIndexes,
    get_node_schema,
    get_relationship_schema
)

from src.knowledge.graph.neo4j_manager import Neo4jManager

from src.knowledge.graph.queries import (
    MathematicalQueries,
    GraphAnalytics,
    QueryBuilder
)

__all__ = [
    # Schema
    "NodeType",
    "RelationshipType",
    "NodeProperties",
    "RelationshipProperties",
    "PaperNode",
    "AuthorNode",
    "ConceptNode",
    "TheoremNode",
    "ConjectureNode",
    "ProofNode",
    "DefinitionNode",
    "FormulaNode",
    "DomainNode",
    "GraphConstraints",
    "GraphIndexes",
    "get_node_schema",
    "get_relationship_schema",

    # Manager
    "Neo4jManager",

    # Queries
    "MathematicalQueries",
    "GraphAnalytics",
    "QueryBuilder"
]