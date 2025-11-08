"""
Qdrant Vector Database Manager for Mathematical Research Papers
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    Range, MatchValue, SearchRequest, UpdateStatus, HasIdCondition,
    PointIdsList, ScoredPoint, SearchParams, HnswConfig, OptimizersConfig,
    WalConfig, CollectionStatus, PayloadSchemaType
)
from qdrant_client.http import models
import uuid

from src.config import settings
from src.knowledge.vector.embeddings import EmbeddingGenerator


class QdrantManager:
    """
    Manages vector storage and retrieval for mathematical papers and concepts
    """

    def __init__(self):
        """Initialize Qdrant manager with connection settings"""
        self.host = settings.database.qdrant_host
        self.port = settings.database.qdrant_port
        self.collection_name = settings.database.qdrant_collection_name
        self.api_key = settings.database.qdrant_api_key.get_secret_value() if settings.database.qdrant_api_key else None

        # Initialize clients
        self.async_client: Optional[AsyncQdrantClient] = None
        self.sync_client: Optional[SyncQdrantClient] = None
        self.embedding_generator = EmbeddingGenerator()

        # Vector configuration
        self.vector_size = 1536  # OpenAI ada-002 embeddings size
        self.distance_metric = Distance.COSINE

        logger.info(f"Initialized QdrantManager for {self.host}:{self.port}")

    async def init(self):
        """Initialize async connection and ensure collection exists"""
        try:
            self.async_client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=30
            )

            # Check connection
            await self.async_client.get_collections()

            # Ensure collection exists
            await self.ensure_collection()

            logger.info("Qdrant async connection established successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant connection: {e}")
            raise

    def get_sync_client(self) -> SyncQdrantClient:
        """Get synchronous client for non-async operations"""
        if not self.sync_client:
            self.sync_client = SyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=30
            )
        return self.sync_client

    async def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                await self.create_collection()
            else:
                # Verify collection configuration
                collection_info = await self.async_client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    async def create_collection(self):
        """Create a new collection with optimized settings for mathematical content"""
        try:
            await self.async_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric
                ),
                hnsw_config=HnswConfig(
                    m=16,  # Number of edges per node
                    ef_construct=200,  # Size of dynamic candidate list
                    full_scan_threshold=10000
                ),
                optimizers_config=OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=4
                ),
                wal_config=WalConfig(
                    wal_capacity_mb=32,
                    wal_segments_ahead=0
                )
            )

            logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def index_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Index a research paper with its embeddings

        Args:
            paper_data: Dictionary containing paper metadata and content

        Returns:
            Point ID of the indexed paper
        """
        try:
            # Generate unique ID
            point_id = str(uuid.uuid4())

            # Prepare text for embedding
            text_content = self._prepare_text_for_embedding(paper_data)

            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(text_content)

            # Prepare payload
            payload = {
                "arxiv_id": paper_data.get("arxiv_id"),
                "title": paper_data.get("title"),
                "abstract": paper_data.get("abstract"),
                "authors": paper_data.get("authors", []),
                "categories": paper_data.get("categories", []),
                "published_date": paper_data.get("published_date"),
                "pdf_url": paper_data.get("pdf_url"),
                "domain": paper_data.get("domain", "general"),
                "concepts": paper_data.get("concepts", []),
                "theorems": paper_data.get("theorems", []),
                "conjectures": paper_data.get("conjectures", []),
                "indexed_at": datetime.utcnow().isoformat()
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload=payload
            )

            # Upsert to collection
            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Indexed paper '{paper_data.get('title')}' with ID {point_id}")
            return point_id

        except Exception as e:
            logger.error(f"Failed to index paper: {e}")
            raise

    async def search_similar_papers(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar papers using semantic search

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions

        Returns:
            List of similar papers with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)

            # Build filter if provided
            search_filter = self._build_filter(filter_conditions) if filter_conditions else None

            # Perform search
            results = await self.async_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=True,
                search_params=SearchParams(
                    hnsw_ef=128,  # Dynamic candidate list size for search
                    exact=False   # Use approximate search for speed
                )
            )

            # Format results
            papers = []
            for point in results:
                paper = point.payload.copy() if point.payload else {}
                paper["score"] = point.score
                paper["id"] = point.id
                papers.append(paper)

            logger.info(f"Found {len(papers)} similar papers for query")
            return papers

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def find_related_concepts(
        self,
        concept_embedding: List[float],
        domain: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find papers related to a specific mathematical concept

        Args:
            concept_embedding: Embedding vector of the concept
            domain: Optional domain filter
            limit: Maximum number of results

        Returns:
            List of related papers
        """
        try:
            # Build domain filter if specified
            search_filter = None
            if domain:
                search_filter = Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
                )

            # Search for related papers
            results = await self.async_client.search(
                collection_name=self.collection_name,
                query_vector=concept_embedding,
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            )

            # Extract and return papers
            papers = []
            for point in results:
                if point.payload:
                    paper = point.payload.copy()
                    paper["relevance_score"] = point.score
                    papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Failed to find related concepts: {e}")
            raise

    async def get_paper_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by its point ID

        Args:
            point_id: Qdrant point ID

        Returns:
            Paper data or None if not found
        """
        try:
            results = await self.async_client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )

            if results:
                return results[0].payload
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve paper {point_id}: {e}")
            raise

    async def update_paper_metadata(self, point_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for an existing paper

        Args:
            point_id: Qdrant point ID
            metadata: Updated metadata fields
        """
        try:
            await self.async_client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[point_id]
            )
            logger.info(f"Updated metadata for paper {point_id}")
        except Exception as e:
            logger.error(f"Failed to update paper metadata: {e}")
            raise

    async def delete_papers(self, point_ids: List[str]):
        """
        Delete papers from the collection

        Args:
            point_ids: List of point IDs to delete
        """
        try:
            await self.async_client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} papers")
        except Exception as e:
            logger.error(f"Failed to delete papers: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            info = await self.async_client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise

    def _prepare_text_for_embedding(self, paper_data: Dict[str, Any]) -> str:
        """
        Prepare paper text for embedding generation

        Args:
            paper_data: Paper metadata and content

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Add title with weight
        if paper_data.get("title"):
            parts.append(f"Title: {paper_data['title']}")

        # Add abstract
        if paper_data.get("abstract"):
            parts.append(f"Abstract: {paper_data['abstract']}")

        # Add key concepts
        if paper_data.get("concepts"):
            concepts = ", ".join(paper_data["concepts"][:10])  # Limit concepts
            parts.append(f"Key Concepts: {concepts}")

        # Add theorems summary
        if paper_data.get("theorems"):
            theorems = "; ".join(paper_data["theorems"][:5])  # Limit theorems
            parts.append(f"Theorems: {theorems}")

        # Add conjectures if present
        if paper_data.get("conjectures"):
            conjectures = "; ".join(paper_data["conjectures"][:3])
            parts.append(f"Conjectures: {conjectures}")

        return "\n\n".join(parts)

    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from conditions dictionary

        Args:
            conditions: Filter conditions

        Returns:
            Qdrant Filter object
        """
        must_conditions = []

        # Domain filter
        if conditions.get("domain"):
            must_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=conditions["domain"]))
            )

        # Category filter
        if conditions.get("category"):
            must_conditions.append(
                FieldCondition(key="categories", match=MatchValue(value=conditions["category"]))
            )

        # Date range filter
        if conditions.get("date_from"):
            must_conditions.append(
                FieldCondition(
                    key="published_date",
                    range=Range(gte=conditions["date_from"])
                )
            )

        if conditions.get("date_to"):
            must_conditions.append(
                FieldCondition(
                    key="published_date",
                    range=Range(lte=conditions["date_to"])
                )
            )

        # Author filter
        if conditions.get("author"):
            must_conditions.append(
                FieldCondition(key="authors", match=MatchValue(value=conditions["author"]))
            )

        return Filter(must=must_conditions) if must_conditions else None

    async def close(self):
        """Close connections"""
        if self.async_client:
            # Qdrant client doesn't have explicit close method
            self.async_client = None
        if self.sync_client:
            self.sync_client = None
        logger.info("Qdrant connections closed")