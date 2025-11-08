"""
Qdrant Client Wrapper for Vector Database Operations
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings


class QdrantClient:
    """
    Low-level Qdrant client wrapper for direct vector database operations
    """

    def __init__(self):
        """Initialize Qdrant client configuration"""
        self.host = settings.database.qdrant_host
        self.port = settings.database.qdrant_port
        self.api_key = settings.database.qdrant_api_key.get_secret_value() if settings.database.qdrant_api_key else None
        self.client: Optional[AsyncQdrantClient] = None
        self.is_connected = False

    async def connect(self) -> bool:
        """
        Establish connection to Qdrant server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=30
            )

            # Test connection
            collections = await self.client.get_collections()
            self.is_connected = True
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Found {len(collections.collections)} collections")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from Qdrant server"""
        if self.client:
            # Qdrant client doesn't have explicit disconnect
            self.client = None
            self.is_connected = False
            logger.info("Disconnected from Qdrant")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Qdrant server health

        Returns:
            Health status information
        """
        try:
            if not self.client:
                return {"status": "disconnected", "error": "Client not initialized"}

            # Get cluster info as health check
            collections = await self.client.get_collections()

            return {
                "status": "healthy",
                "connected": True,
                "host": self.host,
                "port": self.port,
                "collections_count": len(collections.collections)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }

    async def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant

        Returns:
            List of collection names
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            collections = await self.client.get_collections()
            return [col.name for col in collections.collections]

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a new collection

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric to use

        Returns:
            True if created successfully
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )

            logger.info(f"Created collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deleted successfully
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Collection information
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            info = await self.client.get_collection(collection_name=collection_name)

            return {
                "name": info.name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance)
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Insert vectors with payloads into a collection

        Args:
            collection_name: Target collection
            vectors: List of vector embeddings
            payloads: List of associated metadata
            ids: Optional list of IDs

        Returns:
            True if inserted successfully
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            if len(vectors) != len(payloads):
                raise ValueError("Vectors and payloads must have the same length")

            # Generate IDs if not provided
            if not ids:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # Create points
            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload
                )
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]

            # Upsert points
            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Inserted {len(points)} vectors into '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            collection_name: Collection to search in
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                for point in results
            ]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors by IDs

        Args:
            collection_name: Collection name
            ids: List of IDs to delete

        Returns:
            True if deleted successfully
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            from qdrant_client.models import PointIdsList

            await self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids)
            )

            logger.info(f"Deleted {len(ids)} vectors from '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    async def count_vectors(self, collection_name: str) -> int:
        """
        Count vectors in a collection

        Args:
            collection_name: Collection name

        Returns:
            Number of vectors
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            info = await self.client.get_collection(collection_name)
            return info.points_count

        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0

    async def optimize_collection(self, collection_name: str) -> bool:
        """
        Optimize collection for better search performance

        Args:
            collection_name: Collection to optimize

        Returns:
            True if optimized successfully
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            # Update collection optimizer
            from qdrant_client.models import OptimizersConfigDiff

            await self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                    deleted_threshold=0.2
                )
            )

            logger.info(f"Optimized collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False

    async def create_snapshot(self, collection_name: str) -> Optional[str]:
        """
        Create a snapshot of a collection

        Args:
            collection_name: Collection to snapshot

        Returns:
            Snapshot name or None if failed
        """
        try:
            if not self.client:
                raise ConnectionError("Qdrant client not connected")

            result = await self.client.create_snapshot(collection_name=collection_name)

            snapshot_name = result.name if hasattr(result, 'name') else str(result)
            logger.info(f"Created snapshot '{snapshot_name}' for '{collection_name}'")
            return snapshot_name

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()