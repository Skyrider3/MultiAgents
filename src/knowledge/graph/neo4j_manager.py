"""
Neo4j Graph Database Manager for Mathematical Knowledge
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import Neo4jError, ConstraintError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

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
    GraphConstraints,
    GraphIndexes,
    CypherTemplates,
    get_node_schema,
    get_relationship_schema
)


class Neo4jManager:
    """
    Manager for Neo4j graph database operations
    Handles knowledge graph creation, querying, and analysis
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "multiagents123",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._initialized = False

    async def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")

            # Initialize schema if needed
            if not self._initialized:
                await self.initialize_schema()
                self._initialized = True

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def initialize_schema(self):
        """Initialize graph schema with constraints and indexes"""
        async with self.driver.session(database=self.database) as session:
            # Create uniqueness constraints
            for node_type, property_name in GraphConstraints.UNIQUE_CONSTRAINTS:
                await self._create_constraint(session, node_type.value, property_name)

            # Create property indexes
            for node_type, property_name in GraphIndexes.PROPERTY_INDEXES:
                await self._create_index(session, node_type, property_name)

            # Create composite indexes
            for node_type, properties in GraphIndexes.COMPOSITE_INDEXES:
                await self._create_composite_index(session, node_type, properties)

            # Create full-text indexes
            for node_type, properties in GraphIndexes.TEXT_INDEXES:
                await self._create_fulltext_index(session, node_type, properties)

        logger.info("Neo4j schema initialized")

    async def _create_constraint(self, session: AsyncSession, node_type: str, property_name: str):
        """Create uniqueness constraint"""
        try:
            query = f"""
            CREATE CONSTRAINT {node_type}_{property_name}_unique IF NOT EXISTS
            FOR (n:{node_type})
            REQUIRE n.{property_name} IS UNIQUE
            """
            await session.run(query)
            logger.debug(f"Created constraint: {node_type}.{property_name}")
        except Neo4jError as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create constraint: {e}")

    async def _create_index(self, session: AsyncSession, node_type: str, property_name: str):
        """Create property index"""
        try:
            query = f"""
            CREATE INDEX {node_type}_{property_name}_index IF NOT EXISTS
            FOR (n:{node_type})
            ON (n.{property_name})
            """
            await session.run(query)
            logger.debug(f"Created index: {node_type}.{property_name}")
        except Neo4jError as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create index: {e}")

    async def _create_composite_index(self, session: AsyncSession, node_type: str, properties: List[str]):
        """Create composite index"""
        try:
            props_str = ", ".join([f"n.{p}" for p in properties])
            index_name = f"{node_type}_{'_'.join(properties)}_index"
            query = f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_type})
            ON ({props_str})
            """
            await session.run(query)
            logger.debug(f"Created composite index: {index_name}")
        except Neo4jError as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create composite index: {e}")

    async def _create_fulltext_index(self, session: AsyncSession, node_type: str, properties: List[str]):
        """Create full-text search index"""
        try:
            index_name = f"{node_type}_fulltext"
            props_str = ", ".join([f"n.{p}" for p in properties])
            query = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_type})
            ON EACH [{props_str}]
            """
            await session.run(query)
            logger.debug(f"Created fulltext index: {index_name}")
        except Neo4jError as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create fulltext index: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def create_node(self, node_type: NodeType, properties: Dict[str, Any]) -> str:
        """Create a node in the graph"""
        async with self.driver.session(database=self.database) as session:
            # Validate properties against schema
            schema_class = get_node_schema(node_type)
            validated_props = schema_class(**properties).dict()

            query = f"""
            CREATE (n:{node_type.value} $properties)
            RETURN elementId(n) as node_id
            """

            result = await session.run(query, properties=validated_props)
            record = await result.single()
            node_id = record["node_id"]

            logger.debug(f"Created {node_type.value} node: {node_id}")
            return node_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        rel_type: RelationshipType,
        properties: Dict[str, Any] = None
    ) -> bool:
        """Create a relationship between nodes"""
        async with self.driver.session(database=self.database) as session:
            # Validate properties against schema
            if properties:
                schema_class = get_relationship_schema(rel_type)
                validated_props = schema_class(**properties).dict()
            else:
                validated_props = {}

            query = f"""
            MATCH (a) WHERE elementId(a) = $from_id
            MATCH (b) WHERE elementId(b) = $to_id
            CREATE (a)-[r:{rel_type.value} $properties]->(b)
            RETURN r
            """

            result = await session.run(
                query,
                from_id=from_node_id,
                to_id=to_node_id,
                properties=validated_props
            )

            record = await result.single()
            if record:
                logger.debug(f"Created relationship: {rel_type.value}")
                return True
            return False

    async def find_node(self, node_type: NodeType, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a node by properties"""
        async with self.driver.session(database=self.database) as session:
            where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (n:{node_type.value})
            WHERE {where_clause}
            RETURN n, elementId(n) as node_id
            LIMIT 1
            """

            result = await session.run(query, **properties)
            record = await result.single()

            if record:
                node_data = dict(record["n"])
                node_data["_id"] = record["node_id"]
                return node_data
            return None

    async def find_nodes(
        self,
        node_type: NodeType,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find multiple nodes with optional filters"""
        async with self.driver.session(database=self.database) as session:
            if filters:
                where_clause = " AND ".join([f"n.{k} = ${k}" for k in filters.keys()])
                query = f"""
                MATCH (n:{node_type.value})
                WHERE {where_clause}
                RETURN n, elementId(n) as node_id
                LIMIT $limit
                """
                params = {**filters, "limit": limit}
            else:
                query = f"""
                MATCH (n:{node_type.value})
                RETURN n, elementId(n) as node_id
                LIMIT $limit
                """
                params = {"limit": limit}

            result = await session.run(query, **params)
            nodes = []

            async for record in result:
                node_data = dict(record["n"])
                node_data["_id"] = record["node_id"]
                nodes.append(node_data)

            return nodes

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        async with self.driver.session(database=self.database) as session:
            # Add updated_at timestamp
            properties["updated_at"] = datetime.utcnow().isoformat()

            set_clause = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (n) WHERE elementId(n) = $node_id
            SET {set_clause}
            RETURN n
            """

            result = await session.run(query, node_id=node_id, **properties)
            record = await result.single()
            return record is not None

    async def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """Delete a node (and optionally its relationships)"""
        async with self.driver.session(database=self.database) as session:
            if cascade:
                query = """
                MATCH (n) WHERE elementId(n) = $node_id
                DETACH DELETE n
                """
            else:
                query = """
                MATCH (n) WHERE elementId(n) = $node_id
                DELETE n
                """

            result = await session.run(query, node_id=node_id)
            summary = await result.consume()
            return summary.counters.nodes_deleted > 0

    async def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two nodes"""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (start) WHERE elementId(start) = $start_id
            MATCH (end) WHERE elementId(end) = $end_id
            MATCH path = shortestPath((start)-[*..15]-(end))
            RETURN path
            """

            result = await session.run(
                query,
                start_id=start_node_id,
                end_id=end_node_id,
                max_depth=max_depth
            )
            record = await result.single()

            if record:
                path = record["path"]
                path_data = []

                for i, node in enumerate(path.nodes):
                    path_data.append({
                        "type": "node",
                        "data": dict(node),
                        "labels": list(node.labels)
                    })

                    if i < len(path.relationships):
                        rel = path.relationships[i]
                        path_data.append({
                            "type": "relationship",
                            "data": dict(rel),
                            "type_name": rel.type
                        })

                return path_data
            return None

    async def find_related_concepts(
        self,
        concept_name: str,
        depth: int = 2,
        domain: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find concepts related to a given concept"""
        async with self.driver.session(database=self.database) as session:
            query = CypherTemplates.FIND_RELATED_CONCEPTS

            result = await session.run(
                query,
                concept_name=concept_name,
                depth=depth,
                domain=domain,
                limit=limit
            )

            related_concepts = []
            async for record in result:
                concept_data = dict(record["related"])
                concept_data["distance"] = record["distance"]
                related_concepts.append(concept_data)

            return related_concepts

    async def find_proof_chain(
        self,
        theorem_id: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find the proof dependency chain for a theorem"""
        async with self.driver.session(database=self.database) as session:
            query = CypherTemplates.FIND_PROOF_CHAIN

            result = await session.run(
                query,
                theorem_id=theorem_id,
                max_depth=max_depth
            )

            proof_chain = []
            async for record in result:
                path = record["path"]
                chain_element = {
                    "theorem": dict(path.nodes[0]),
                    "proof": dict(path.nodes[1]) if len(path.nodes) > 1 else None,
                    "dependencies": [dict(n) for n in path.nodes[2:]] if len(path.nodes) > 2 else []
                }
                proof_chain.append(chain_element)

            return proof_chain

    async def find_citation_network(
        self,
        arxiv_id: str,
        depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find citation network for a paper"""
        async with self.driver.session(database=self.database) as session:
            query = CypherTemplates.FIND_CITATION_NETWORK

            result = await session.run(
                query,
                arxiv_id=arxiv_id,
                depth=depth,
                limit=limit
            )

            citations = []
            async for record in result:
                paper_data = dict(record["cited"])
                paper_data["citation_count"] = record["citation_count"]
                citations.append(paper_data)

            return citations

    async def find_conjectures_by_status(
        self,
        status: str = "open",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find conjectures by their status"""
        async with self.driver.session(database=self.database) as session:
            query = CypherTemplates.FIND_CONJECTURES_BY_STATUS

            result = await session.run(
                query,
                status=status,
                limit=limit
            )

            conjectures = []
            async for record in result:
                conjecture_data = dict(record["c"])
                conjecture_data["proofs"] = [dict(p) for p in record["proofs"]]
                conjectures.append(conjecture_data)

            return conjectures

    async def find_influential_papers(
        self,
        min_citations: int = 10,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find most influential papers by citation count"""
        async with self.driver.session(database=self.database) as session:
            query = CypherTemplates.FIND_INFLUENTIAL_PAPERS

            result = await session.run(
                query,
                min_citations=min_citations,
                limit=limit
            )

            papers = []
            async for record in result:
                paper_data = dict(record["p"])
                paper_data["citation_count"] = record["citation_count"]
                papers.append(paper_data)

            return papers

    async def calculate_centrality(
        self,
        node_type: NodeType,
        centrality_type: str = "pagerank"
    ) -> List[Tuple[str, float]]:
        """Calculate centrality measures for nodes"""
        async with self.driver.session(database=self.database) as session:
            if centrality_type == "pagerank":
                query = f"""
                CALL gds.pageRank.stream({{
                    nodeQuery: 'MATCH (n:{node_type.value}) RETURN id(n) as id',
                    relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target',
                    maxIterations: 20,
                    dampingFactor: 0.85
                }})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).node_id AS node_id, score
                ORDER BY score DESC
                LIMIT 100
                """
            elif centrality_type == "betweenness":
                query = f"""
                CALL gds.betweenness.stream({{
                    nodeQuery: 'MATCH (n:{node_type.value}) RETURN id(n) as id',
                    relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
                }})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).node_id AS node_id, score
                ORDER BY score DESC
                LIMIT 100
                """
            else:
                raise ValueError(f"Unsupported centrality type: {centrality_type}")

            result = await session.run(query)
            centrality_scores = []

            async for record in result:
                centrality_scores.append((record["node_id"], record["score"]))

            return centrality_scores

    async def detect_communities(
        self,
        algorithm: str = "louvain"
    ) -> Dict[int, List[str]]:
        """Detect communities in the graph"""
        async with self.driver.session(database=self.database) as session:
            if algorithm == "louvain":
                query = """
                CALL gds.louvain.stream({
                    nodeQuery: 'MATCH (n) RETURN id(n) as id',
                    relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
                })
                YIELD nodeId, communityId
                RETURN communityId, collect(gds.util.asNode(nodeId).node_id) as nodes
                ORDER BY size(nodes) DESC
                """
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            result = await session.run(query)
            communities = {}

            async for record in result:
                communities[record["communityId"]] = record["nodes"]

            return communities

    async def execute_custom_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query"""
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, parameters or {})
            records = []

            async for record in result:
                records.append(dict(record))

            return records

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        async with self.driver.session(database=self.database) as session:
            stats = {}

            # Count nodes by type
            for node_type in NodeType:
                query = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                result = await session.run(query)
                record = await result.single()
                stats[f"{node_type.value}_count"] = record["count"]

            # Count relationships by type
            for rel_type in RelationshipType:
                query = f"MATCH ()-[r:{rel_type.value}]->() RETURN count(r) as count"
                result = await session.run(query)
                record = await result.single()
                stats[f"{rel_type.value}_count"] = record["count"]

            # General statistics
            query = """
            MATCH (n)
            RETURN count(n) as total_nodes
            """
            result = await session.run(query)
            record = await result.single()
            stats["total_nodes"] = record["total_nodes"]

            query = """
            MATCH ()-[r]->()
            RETURN count(r) as total_relationships
            """
            result = await session.run(query)
            record = await result.single()
            stats["total_relationships"] = record["total_relationships"]

            return stats


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize manager
        manager = Neo4jManager()
        await manager.connect()

        try:
            # Create a paper node
            paper_props = {
                "node_id": "paper_001",
                "title": "A New Approach to Prime Numbers",
                "abstract": "We present a novel method...",
                "arxiv_id": "2301.12345",
                "publication_date": datetime.now()
            }
            paper_id = await manager.create_node(NodeType.PAPER, paper_props)
            print(f"Created paper: {paper_id}")

            # Create an author node
            author_props = {
                "node_id": "author_001",
                "name": "John Mathematician",
                "affiliation": "University of Mathematics"
            }
            author_id = await manager.create_node(NodeType.AUTHOR, author_props)
            print(f"Created author: {author_id}")

            # Create relationship
            await manager.create_relationship(
                paper_id,
                author_id,
                RelationshipType.AUTHORED_BY
            )
            print("Created authorship relationship")

            # Get statistics
            stats = await manager.get_graph_statistics()
            print(f"Graph statistics: {json.dumps(stats, indent=2)}")

        finally:
            await manager.close()

    # Run example
    asyncio.run(example())