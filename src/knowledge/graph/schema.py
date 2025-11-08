"""
Neo4j Graph Schema for Mathematical Knowledge Representation
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph"""
    PAPER = "Paper"
    AUTHOR = "Author"
    CONCEPT = "Concept"
    THEOREM = "Theorem"
    CONJECTURE = "Conjecture"
    PROOF = "Proof"
    DEFINITION = "Definition"
    FORMULA = "Formula"
    DOMAIN = "Domain"
    INSTITUTION = "Institution"
    CONFERENCE = "Conference"
    PROBLEM = "Problem"
    METHOD = "Method"
    ALGORITHM = "Algorithm"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph"""
    # Authorship and citation
    AUTHORED_BY = "AUTHORED_BY"
    CITES = "CITES"
    CITED_BY = "CITED_BY"

    # Mathematical relationships
    PROVES = "PROVES"
    DISPROVES = "DISPROVES"
    EXTENDS = "EXTENDS"
    GENERALIZES = "GENERALIZES"
    SPECIALIZES = "SPECIALIZES"
    IMPLIES = "IMPLIES"
    CONTRADICTS = "CONTRADICTS"
    EQUIVALENT_TO = "EQUIVALENT_TO"

    # Structural relationships
    DEFINES = "DEFINES"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"

    # Discovery relationships
    DISCOVERED_BY = "DISCOVERED_BY"
    VALIDATED_BY = "VALIDATED_BY"
    CHALLENGED_BY = "CHALLENGED_BY"
    SYNTHESIZED_FROM = "SYNTHESIZED_FROM"

    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    ANALOGOUS_TO = "ANALOGOUS_TO"
    ISOMORPHIC_TO = "ISOMORPHIC_TO"

    # Temporal relationships
    PRECEDED_BY = "PRECEDED_BY"
    EVOLVED_FROM = "EVOLVED_FROM"
    INFLUENCED_BY = "INFLUENCED_BY"


class NodeProperties(BaseModel):
    """Base properties for all nodes"""
    node_id: str = Field(..., description="Unique identifier for the node")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = Field(None, description="Source of the information")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PaperNode(NodeProperties):
    """Properties for Paper nodes"""
    title: str
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class AuthorNode(NodeProperties):
    """Properties for Author nodes"""
    name: str
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None
    h_index: Optional[int] = None
    research_areas: List[str] = Field(default_factory=list)


class ConceptNode(NodeProperties):
    """Properties for Concept nodes"""
    name: str
    description: str
    formal_definition: Optional[str] = None
    notation: Optional[str] = None
    domain: str
    complexity: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)


class TheoremNode(NodeProperties):
    """Properties for Theorem nodes"""
    name: str
    statement: str
    formal_statement: Optional[str] = None
    proof_status: str = "proven"  # proven, unproven, disputed
    importance: str = "medium"  # low, medium, high, fundamental
    domain: str
    assumptions: List[str] = Field(default_factory=list)
    implications: List[str] = Field(default_factory=list)
    year_proved: Optional[int] = None


class ConjectureNode(NodeProperties):
    """Properties for Conjecture nodes"""
    name: str
    statement: str
    formal_statement: Optional[str] = None
    status: str = "open"  # open, proven, disproven, partially_proven
    proposed_by: Optional[str] = None
    year_proposed: Optional[int] = None
    prize: Optional[str] = None  # e.g., "Millennium Prize"
    partial_results: List[str] = Field(default_factory=list)
    approaches_tried: List[str] = Field(default_factory=list)


class ProofNode(NodeProperties):
    """Properties for Proof nodes"""
    proof_id: str
    theorem_id: str
    proof_type: str  # direct, contradiction, induction, construction, etc.
    proof_text: Optional[str] = None
    formal_proof: Optional[str] = None
    verified: bool = False
    verification_system: Optional[str] = None  # Coq, Lean, etc.
    key_insights: List[str] = Field(default_factory=list)
    techniques_used: List[str] = Field(default_factory=list)


class DefinitionNode(NodeProperties):
    """Properties for Definition nodes"""
    term: str
    definition: str
    formal_notation: Optional[str] = None
    domain: str
    examples: List[str] = Field(default_factory=list)
    non_examples: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)


class FormulaNode(NodeProperties):
    """Properties for Formula nodes"""
    formula_id: str
    latex: str
    mathml: Optional[str] = None
    ascii: Optional[str] = None
    description: str
    variables: List[Dict[str, str]] = Field(default_factory=list)
    domain: str
    formula_type: str  # equation, inequality, identity, etc.


class DomainNode(NodeProperties):
    """Properties for Domain nodes"""
    name: str
    description: str
    parent_domain: Optional[str] = None
    subdomains: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    major_theorems: List[str] = Field(default_factory=list)
    open_problems: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)


class RelationshipProperties(BaseModel):
    """Base properties for relationships"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    validated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationRelationship(RelationshipProperties):
    """Properties for citation relationships"""
    context: Optional[str] = None
    citation_type: str = "general"  # general, method, result, criticism
    section: Optional[str] = None
    page_number: Optional[int] = None


class ProofRelationship(RelationshipProperties):
    """Properties for proof relationships"""
    proof_role: str = "main"  # main, lemma, auxiliary, alternative
    completeness: float = Field(1.0, ge=0.0, le=1.0)
    rigor_level: str = "formal"  # informal, semi-formal, formal, machine-verified


class SimilarityRelationship(RelationshipProperties):
    """Properties for similarity relationships"""
    similarity_score: float = Field(0.5, ge=0.0, le=1.0)
    similarity_type: str = "structural"  # structural, semantic, syntactic
    common_features: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)


class DependencyRelationship(RelationshipProperties):
    """Properties for dependency relationships"""
    dependency_type: str = "requires"  # requires, optional, recommended
    dependency_strength: float = Field(1.0, ge=0.0, le=1.0)
    can_substitute: List[str] = Field(default_factory=list)


class GraphConstraints:
    """Constraints for the knowledge graph"""

    # Node uniqueness constraints
    UNIQUE_CONSTRAINTS = [
        (NodeType.PAPER, "arxiv_id"),
        (NodeType.PAPER, "doi"),
        (NodeType.AUTHOR, "orcid"),
        (NodeType.THEOREM, "node_id"),
        (NodeType.CONJECTURE, "node_id"),
    ]

    # Relationship cardinality constraints
    CARDINALITY_CONSTRAINTS = {
        # A proof proves exactly one theorem
        (RelationshipType.PROVES, NodeType.PROOF, NodeType.THEOREM): (1, 1),
        # A paper can have multiple authors
        (RelationshipType.AUTHORED_BY, NodeType.PAPER, NodeType.AUTHOR): (1, None),
        # A concept belongs to at least one domain
        (RelationshipType.BELONGS_TO, NodeType.CONCEPT, NodeType.DOMAIN): (1, None),
    }

    # Valid relationship type combinations
    VALID_RELATIONSHIPS = {
        RelationshipType.AUTHORED_BY: (NodeType.PAPER, NodeType.AUTHOR),
        RelationshipType.CITES: (NodeType.PAPER, NodeType.PAPER),
        RelationshipType.PROVES: (NodeType.PROOF, NodeType.THEOREM),
        RelationshipType.DISPROVES: (NodeType.PROOF, NodeType.CONJECTURE),
        RelationshipType.EXTENDS: (NodeType.THEOREM, NodeType.THEOREM),
        RelationshipType.DEFINES: (NodeType.PAPER, NodeType.DEFINITION),
        RelationshipType.USES: (NodeType.PROOF, NodeType.THEOREM),
        RelationshipType.BELONGS_TO: (NodeType.CONCEPT, NodeType.DOMAIN),
        RelationshipType.DISCOVERED_BY: (NodeType.CONJECTURE, NodeType.AUTHOR),
    }


class GraphIndexes:
    """Indexes for efficient graph queries"""

    # Full-text search indexes
    TEXT_INDEXES = [
        (NodeType.PAPER, ["title", "abstract"]),
        (NodeType.THEOREM, ["statement"]),
        (NodeType.CONJECTURE, ["statement"]),
        (NodeType.CONCEPT, ["description"]),
    ]

    # Property indexes for fast lookup
    PROPERTY_INDEXES = [
        (NodeType.PAPER, "arxiv_id"),
        (NodeType.PAPER, "doi"),
        (NodeType.PAPER, "publication_date"),
        (NodeType.AUTHOR, "name"),
        (NodeType.AUTHOR, "orcid"),
        (NodeType.THEOREM, "domain"),
        (NodeType.CONJECTURE, "status"),
        (NodeType.CONCEPT, "domain"),
    ]

    # Composite indexes
    COMPOSITE_INDEXES = [
        (NodeType.PAPER, ["domain", "publication_date"]),
        (NodeType.THEOREM, ["domain", "importance"]),
        (NodeType.CONJECTURE, ["status", "year_proposed"]),
    ]


def get_node_schema(node_type: NodeType) -> type[NodeProperties]:
    """Get the schema class for a node type"""
    schema_map = {
        NodeType.PAPER: PaperNode,
        NodeType.AUTHOR: AuthorNode,
        NodeType.CONCEPT: ConceptNode,
        NodeType.THEOREM: TheoremNode,
        NodeType.CONJECTURE: ConjectureNode,
        NodeType.PROOF: ProofNode,
        NodeType.DEFINITION: DefinitionNode,
        NodeType.FORMULA: FormulaNode,
        NodeType.DOMAIN: DomainNode,
    }
    return schema_map.get(node_type, NodeProperties)


def get_relationship_schema(rel_type: RelationshipType) -> type[RelationshipProperties]:
    """Get the schema class for a relationship type"""
    schema_map = {
        RelationshipType.CITES: CitationRelationship,
        RelationshipType.CITED_BY: CitationRelationship,
        RelationshipType.PROVES: ProofRelationship,
        RelationshipType.DISPROVES: ProofRelationship,
        RelationshipType.SIMILAR_TO: SimilarityRelationship,
        RelationshipType.DEPENDS_ON: DependencyRelationship,
        RelationshipType.USES: DependencyRelationship,
    }
    return schema_map.get(rel_type, RelationshipProperties)


# Cypher query templates for common operations
class CypherTemplates:
    """Common Cypher query templates"""

    CREATE_NODE = """
    CREATE (n:{node_type} $properties)
    RETURN n
    """

    CREATE_RELATIONSHIP = """
    MATCH (a:{from_type} {{node_id: $from_id}})
    MATCH (b:{to_type} {{node_id: $to_id}})
    CREATE (a)-[r:{rel_type} $properties]->(b)
    RETURN r
    """

    FIND_SHORTEST_PATH = """
    MATCH (start:{node_type} {{node_id: $start_id}})
    MATCH (end:{node_type} {{node_id: $end_id}})
    MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
    RETURN path
    """

    FIND_RELATED_CONCEPTS = """
    MATCH (c:Concept {{name: $concept_name}})
    MATCH (c)-[r:RELATED_TO*1..{depth}]-(related:Concept)
    WHERE related.domain = $domain OR $domain IS NULL
    RETURN DISTINCT related, min(length(r)) as distance
    ORDER BY distance
    LIMIT $limit
    """

    FIND_PROOF_CHAIN = """
    MATCH (t:Theorem {{node_id: $theorem_id}})
    MATCH path = (t)<-[:PROVES]-(p:Proof)-[:USES*0..{max_depth}]->(dep:Theorem)
    RETURN path
    """

    FIND_CITATION_NETWORK = """
    MATCH (p:Paper {{arxiv_id: $arxiv_id}})
    MATCH (p)-[:CITES*1..{depth}]-(cited:Paper)
    RETURN DISTINCT cited, count(*) as citation_count
    ORDER BY citation_count DESC
    LIMIT $limit
    """

    FIND_CONJECTURES_BY_STATUS = """
    MATCH (c:Conjecture)
    WHERE c.status = $status
    OPTIONAL MATCH (c)<-[:PROVES|DISPROVES]-(p:Proof)
    RETURN c, collect(p) as proofs
    ORDER BY c.year_proposed DESC
    LIMIT $limit
    """

    FIND_INFLUENTIAL_PAPERS = """
    MATCH (p:Paper)
    OPTIONAL MATCH (p)<-[c:CITES]-()
    WITH p, count(c) as citation_count
    WHERE citation_count > $min_citations
    RETURN p, citation_count
    ORDER BY citation_count DESC
    LIMIT $limit
    """

    FIND_COLLABORATION_NETWORK = """
    MATCH (a1:Author {{name: $author_name}})
    MATCH (a1)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
    WHERE a1 <> a2
    WITH a2, count(DISTINCT p) as collaborations
    RETURN a2, collaborations
    ORDER BY collaborations DESC
    LIMIT $limit
    """