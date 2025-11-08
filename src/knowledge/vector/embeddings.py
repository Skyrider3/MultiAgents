"""
Embedding Generation for Mathematical Text Using AWS Bedrock and Local Models
"""

import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import numpy as np
from loguru import logger
import aiohttp

from src.config import settings


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    BEDROCK_TITAN = "amazon.titan-embed-text-v1"
    BEDROCK_COHERE = "cohere.embed-english-v3"
    SENTENCE_TRANSFORMER = "sentence-transformers"
    OPENAI_ADA = "text-embedding-ada-002"


class EmbeddingGenerator:
    """
    Generate embeddings for mathematical text using various models
    """

    def __init__(self, model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMER):
        """
        Initialize embedding generator

        Args:
            model: Embedding model to use
        """
        self.model = model
        self.dimension = self._get_dimension()
        self.cache = {}  # Simple in-memory cache
        self._sentence_transformer = None

        logger.info(f"Initialized EmbeddingGenerator with model {model}")

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        dimensions = {
            EmbeddingModel.BEDROCK_TITAN: 1536,
            EmbeddingModel.BEDROCK_COHERE: 1024,
            EmbeddingModel.SENTENCE_TRANSFORMER: 768,
            EmbeddingModel.OPENAI_ADA: 1536
        }
        return dimensions.get(self.model, 768)

    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                logger.debug("Using cached embedding")
                return self.cache[cache_key]

        # Generate based on model
        if self.model == EmbeddingModel.BEDROCK_TITAN:
            embedding = await self._generate_bedrock_titan(text)
        elif self.model == EmbeddingModel.BEDROCK_COHERE:
            embedding = await self._generate_bedrock_cohere(text)
        elif self.model == EmbeddingModel.SENTENCE_TRANSFORMER:
            embedding = await self._generate_sentence_transformer(text)
        elif self.model == EmbeddingModel.OPENAI_ADA:
            embedding = await self._generate_openai(text)
        else:
            # Default to sentence transformer
            embedding = await self._generate_sentence_transformer(text)

        # Cache the result
        if use_cache:
            self.cache[cache_key] = embedding

        return embedding

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process batch in parallel
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)

            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        return embeddings

    async def _generate_bedrock_titan(self, text: str) -> np.ndarray:
        """Generate embedding using AWS Bedrock Titan model"""
        try:
            # Import boto3 only when needed
            import boto3

            # Initialize Bedrock client
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.aws.region
            )

            # Prepare request
            body = json.dumps({
                "inputText": text[:8000]  # Titan has 8k token limit
            })

            # Invoke model
            response = bedrock.invoke_model(
                body=body,
                modelId=self.model.value,
                accept='application/json',
                contentType='application/json'
            )

            # Parse response
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')

            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to generate Bedrock Titan embedding: {e}")
            # Fallback to sentence transformer
            return await self._generate_sentence_transformer(text)

    async def _generate_bedrock_cohere(self, text: str) -> np.ndarray:
        """Generate embedding using AWS Bedrock Cohere model"""
        try:
            import boto3

            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.aws.region
            )

            body = json.dumps({
                "texts": [text[:2000]],  # Cohere has smaller limit
                "input_type": "search_document"
            })

            response = bedrock.invoke_model(
                body=body,
                modelId=self.model.value,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response.get('body').read())
            embeddings = response_body.get('embeddings', [[]])

            return np.array(embeddings[0], dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to generate Bedrock Cohere embedding: {e}")
            return await self._generate_sentence_transformer(text)

    async def _generate_sentence_transformer(self, text: str) -> np.ndarray:
        """Generate embedding using local sentence transformer model"""
        try:
            # Lazy load sentence transformer
            if self._sentence_transformer is None:
                from sentence_transformers import SentenceTransformer
                # Use a good model for mathematical/scientific text
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embedding
            embedding = self._sentence_transformer.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Ensure correct dimension (pad or truncate if needed)
            if len(embedding) < self.dimension:
                # Pad with zeros
                embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
            elif len(embedding) > self.dimension:
                # Truncate
                embedding = embedding[:self.dimension]

            return embedding.astype(np.float32)

        except ImportError:
            logger.error("sentence-transformers not installed")
            # Return random embedding as last resort
            return np.random.randn(self.dimension).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate sentence transformer embedding: {e}")
            return np.random.randn(self.dimension).astype(np.float32)

    async def _generate_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        try:
            api_key = settings.external.openai_api_key
            if not api_key:
                logger.warning("OpenAI API key not configured")
                return await self._generate_sentence_transformer(text)

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key.get_secret_value()}",
                    "Content-Type": "application/json"
                }

                data = {
                    "input": text[:8000],  # OpenAI has token limits
                    "model": "text-embedding-ada-002"
                }

                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        return np.array(embedding, dtype=np.float32)
                    else:
                        error = await response.text()
                        logger.error(f"OpenAI API error: {error}")
                        return await self._generate_sentence_transformer(text)

        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            return await self._generate_sentence_transformer(text)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model.value}:{text_hash}"

    async def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric (cosine, euclidean, dot)

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        elif metric == "euclidean":
            # Euclidean distance (inverted for similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(1 / (1 + distance))

        elif metric == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    async def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 10,
        metric: str = "cosine"
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results
            metric: Similarity metric

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            sim = await self.similarity(query_embedding, candidate, metric)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")


class MathematicalEmbeddingEnhancer:
    """
    Enhance embeddings with mathematical structure understanding
    """

    def __init__(self, base_generator: EmbeddingGenerator):
        """
        Initialize enhancer

        Args:
            base_generator: Base embedding generator
        """
        self.base_generator = base_generator
        self.formula_weight = 1.5
        self.theorem_weight = 2.0
        self.proof_weight = 1.8

    async def generate_enhanced_embedding(
        self,
        text: str,
        formulas: Optional[List[str]] = None,
        theorems: Optional[List[str]] = None,
        proofs: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate embedding with mathematical structure enhancement

        Args:
            text: Main text
            formulas: List of mathematical formulas
            theorems: List of theorems
            proofs: List of proofs

        Returns:
            Enhanced embedding vector
        """
        # Get base embedding
        base_embedding = await self.base_generator.generate_embedding(text)

        # Initialize weighted embedding
        weighted_embedding = base_embedding.copy()

        # Add formula embeddings with weight
        if formulas:
            formula_text = " ".join(formulas)
            formula_embedding = await self.base_generator.generate_embedding(formula_text)
            weighted_embedding += self.formula_weight * formula_embedding

        # Add theorem embeddings with weight
        if theorems:
            theorem_text = " ".join(theorems)
            theorem_embedding = await self.base_generator.generate_embedding(theorem_text)
            weighted_embedding += self.theorem_weight * theorem_embedding

        # Add proof embeddings with weight
        if proofs:
            proof_text = " ".join(proofs[:3])  # Limit proofs
            proof_embedding = await self.base_generator.generate_embedding(proof_text)
            weighted_embedding += self.proof_weight * proof_embedding

        # Normalize the weighted embedding
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm

        return weighted_embedding.astype(np.float32)