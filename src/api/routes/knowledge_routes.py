"""
Knowledge Graph API Routes
"""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from src.knowledge.graph.schema import (
    PaperNode, AuthorNode, ConjectureNode, TheoremNode,
    ProofNode, ConceptNode, FormulaNode
)


router = APIRouter()


class PaperQuery(BaseModel):
    """Query model for papers"""
    title: Optional[str] = None
    author: Optional[str] = None
    domain: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    limit: int = Field(default=10, le=100)


class ConjectureQuery(BaseModel):
    """Query model for conjectures"""
    status: Optional[str] = None
    domain: Optional[str] = None
    confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=10, le=100)


class GraphQuery(BaseModel):
    """General graph query model"""
    query_type: str = Field(..., description="Query type (cypher, pattern, path)")
    query: str = Field(..., description="Query string")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=100, le=1000)


class NodeCreate(BaseModel):
    """Model for creating a new node"""
    node_type: str = Field(..., description="Type of node (paper, author, conjecture, etc.)")
    properties: Dict[str, Any] = Field(..., description="Node properties")


class RelationshipCreate(BaseModel):
    """Model for creating a relationship"""
    from_node_id: str
    to_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


@router.get("/papers")
async def search_papers(
    req: Request,
    title: Optional[str] = None,
    author: Optional[str] = None,
    domain: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = Query(default=10, le=100)
):
    """
    Search papers in the knowledge graph

    Args:
        req: Request object
        title: Paper title (partial match)
        author: Author name (partial match)
        domain: Mathematical domain
        year_from: Minimum publication year
        year_to: Maximum publication year
        limit: Maximum number of results

    Returns:
        List of matching papers
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        # Build query
        query = "MATCH (p:Paper) WHERE 1=1 "
        params = {}

        if title:
            query += "AND p.title CONTAINS $title "
            params["title"] = title

        if author:
            query += "AND EXISTS((p)<-[:AUTHORED]-(a:Author {name: $author})) "
            params["author"] = author

        if domain:
            query += "AND p.domain = $domain "
            params["domain"] = domain

        if year_from:
            query += "AND p.publication_date >= $year_from "
            params["year_from"] = f"{year_from}-01-01"

        if year_to:
            query += "AND p.publication_date <= $year_to "
            params["year_to"] = f"{year_to}-12-31"

        query += "RETURN p LIMIT $limit"
        params["limit"] = limit

        result = await neo4j.execute_query(query, params)
        papers = [record["p"] for record in result]

        return papers

    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str, req: Request):
    """
    Get paper details by ID

    Args:
        paper_id: Paper ID
        req: Request object

    Returns:
        Paper details with relationships
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        # Get paper with relationships
        query = """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
        OPTIONAL MATCH (p)-[:CONTAINS_THEOREM]->(t:Theorem)
        OPTIONAL MATCH (p)-[:PROPOSES_CONJECTURE]->(c:Conjecture)
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        RETURN p,
               collect(DISTINCT a) as authors,
               collect(DISTINCT t) as theorems,
               collect(DISTINCT c) as conjectures,
               collect(DISTINCT cited.id) as citations
        """

        result = await neo4j.execute_query(query, {"paper_id": paper_id})

        if not result:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper_data = result[0]
        return {
            "paper": paper_data["p"],
            "authors": paper_data["authors"],
            "theorems": paper_data["theorems"],
            "conjectures": paper_data["conjectures"],
            "citations": paper_data["citations"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conjectures")
async def search_conjectures(
    req: Request,
    status: Optional[str] = None,
    domain: Optional[str] = None,
    confidence_min: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=10, le=100)
):
    """
    Search conjectures in the knowledge graph

    Args:
        req: Request object
        status: Conjecture status
        domain: Mathematical domain
        confidence_min: Minimum confidence score
        limit: Maximum number of results

    Returns:
        List of matching conjectures
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        query = "MATCH (c:Conjecture) WHERE c.confidence >= $confidence_min "
        params = {"confidence_min": confidence_min, "limit": limit}

        if status:
            query += "AND c.status = $status "
            params["status"] = status

        if domain:
            query += "AND c.domain = $domain "
            params["domain"] = domain

        query += "RETURN c ORDER BY c.confidence DESC LIMIT $limit"

        result = await neo4j.execute_query(query, params)
        conjectures = [record["c"] for record in result]

        return conjectures

    except Exception as e:
        logger.error(f"Error searching conjectures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/theorems")
async def search_theorems(
    req: Request,
    domain: Optional[str] = None,
    verified: Optional[bool] = None,
    limit: int = Query(default=10, le=100)
):
    """
    Search theorems in the knowledge graph

    Args:
        req: Request object
        domain: Mathematical domain
        verified: Verification status
        limit: Maximum number of results

    Returns:
        List of matching theorems
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        query = "MATCH (t:Theorem) WHERE 1=1 "
        params = {"limit": limit}

        if domain is not None:
            query += "AND t.domain = $domain "
            params["domain"] = domain

        if verified is not None:
            query += "AND t.verified = $verified "
            params["verified"] = verified

        query += "RETURN t LIMIT $limit"

        result = await neo4j.execute_query(query, params)
        theorems = [record["t"] for record in result]

        return theorems

    except Exception as e:
        logger.error(f"Error searching theorems: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes")
async def create_node(node: NodeCreate, req: Request):
    """
    Create a new node in the knowledge graph

    Args:
        node: Node creation request
        req: Request object

    Returns:
        Created node information
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        # Map node types to classes
        node_classes = {
            "paper": PaperNode,
            "author": AuthorNode,
            "conjecture": ConjectureNode,
            "theorem": TheoremNode,
            "proof": ProofNode,
            "concept": ConceptNode,
            "formula": FormulaNode,
            "dataset": DatasetNode,
            "experiment": ExperimentNode
        }

        if node.node_type not in node_classes:
            raise HTTPException(status_code=400, detail=f"Invalid node type: {node.node_type}")

        # Create node instance
        node_class = node_classes[node.node_type]
        node_instance = node_class(**node.properties)

        # Create in Neo4j
        created = await neo4j.create_node(node_instance)

        return {
            "status": "created",
            "node_id": created.get("id"),
            "node_type": node.node_type,
            "properties": created
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships")
async def create_relationship(relationship: RelationshipCreate, req: Request):
    """
    Create a relationship between nodes

    Args:
        relationship: Relationship creation request
        req: Request object

    Returns:
        Created relationship information
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        # Create relationship
        result = await neo4j.create_relationship(
            from_node_id=relationship.from_node_id,
            to_node_id=relationship.to_node_id,
            relationship_type=relationship.relationship_type,
            properties=relationship.properties
        )

        return {
            "status": "created",
            "from_node": relationship.from_node_id,
            "to_node": relationship.to_node_id,
            "relationship": relationship.relationship_type
        }

    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def execute_graph_query(query: GraphQuery, req: Request):
    """
    Execute a custom graph query

    Args:
        query: Graph query request
        req: Request object

    Returns:
        Query results
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        if query.query_type == "cypher":
            # Execute raw Cypher query (be careful with this in production!)
            result = await neo4j.execute_query(query.query, query.parameters)
        elif query.query_type == "pattern":
            # Pattern matching query
            result = await neo4j.find_pattern(query.query, limit=query.limit)
        elif query.query_type == "path":
            # Path finding query
            if "from_id" not in query.parameters or "to_id" not in query.parameters:
                raise HTTPException(status_code=400, detail="Path query requires from_id and to_id")

            result = await neo4j.find_shortest_path(
                query.parameters["from_id"],
                query.parameters["to_id"]
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid query type: {query.query_type}")

        return {
            "query_type": query.query_type,
            "results": result,
            "count": len(result) if isinstance(result, list) else 1
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_graph_statistics(req: Request):
    """
    Get knowledge graph statistics

    Args:
        req: Request object

    Returns:
        Graph statistics
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        stats = await neo4j.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains")
async def get_mathematical_domains(req: Request):
    """
    Get list of mathematical domains in the graph

    Args:
        req: Request object

    Returns:
        List of domains with counts
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        query = """
        MATCH (n)
        WHERE n.domain IS NOT NULL
        RETURN n.domain as domain, labels(n)[0] as type, count(*) as count
        ORDER BY count DESC
        """

        result = await neo4j.execute_query(query)

        domains = {}
        for record in result:
            domain = record["domain"]
            if domain not in domains:
                domains[domain] = {"total": 0, "by_type": {}}

            domains[domain]["total"] += record["count"]
            domains[domain]["by_type"][record["type"]] = record["count"]

        return domains

    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_additions(
    req: Request,
    node_type: Optional[str] = None,
    limit: int = Query(default=10, le=50)
):
    """
    Get recently added nodes

    Args:
        req: Request object
        node_type: Filter by node type
        limit: Maximum number of results

    Returns:
        List of recent nodes
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        if node_type:
            query = f"""
            MATCH (n:{node_type})
            WHERE n.created_at IS NOT NULL
            RETURN n
            ORDER BY n.created_at DESC
            LIMIT $limit
            """
        else:
            query = """
            MATCH (n)
            WHERE n.created_at IS NOT NULL
            RETURN n
            ORDER BY n.created_at DESC
            LIMIT $limit
            """

        result = await neo4j.execute_query(query, {"limit": limit})
        nodes = [record["n"] for record in result]

        return nodes

    except Exception as e:
        logger.error(f"Error getting recent additions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaborations")
async def find_collaborations(
    req: Request,
    author_name: Optional[str] = None,
    min_papers: int = Query(default=2, ge=1)
):
    """
    Find author collaborations

    Args:
        req: Request object
        author_name: Filter by specific author
        min_papers: Minimum number of co-authored papers

    Returns:
        Collaboration network
    """
    try:
        neo4j = req.app.state.neo4j
        if not neo4j:
            raise HTTPException(status_code=503, detail="Neo4j not available")

        if author_name:
            query = """
            MATCH (a1:Author {name: $author})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE a1 <> a2
            WITH a1, a2, collect(p) as papers
            WHERE size(papers) >= $min_papers
            RETURN a1.name as author1, a2.name as author2,
                   size(papers) as collaboration_count,
                   [p IN papers | p.title] as paper_titles
            ORDER BY collaboration_count DESC
            """
            params = {"author": author_name, "min_papers": min_papers}
        else:
            query = """
            MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE id(a1) < id(a2)
            WITH a1, a2, collect(p) as papers
            WHERE size(papers) >= $min_papers
            RETURN a1.name as author1, a2.name as author2,
                   size(papers) as collaboration_count
            ORDER BY collaboration_count DESC
            LIMIT 50
            """
            params = {"min_papers": min_papers}

        result = await neo4j.execute_query(query, params)

        return result

    except Exception as e:
        logger.error(f"Error finding collaborations: {e}")
        raise HTTPException(status_code=500, detail=str(e))