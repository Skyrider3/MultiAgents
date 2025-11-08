"""
Enhanced AWS Bedrock Multi-Model Client
Supports Claude 3.5, GPT-4, Grok, and DeepSeek models with intelligent routing
"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import boto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from loguru import logger
from pydantic import BaseModel, Field
import tiktoken


class ModelProvider(str, Enum):
    """Available model providers in AWS Bedrock"""
    CLAUDE_35_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    GPT_4 = "openai.gpt-4-turbo-2024-04-09"
    GPT_35 = "openai.gpt-3.5-turbo-1106"
    LLAMA_3 = "meta.llama3-70b-instruct-v1:0"
    MISTRAL_LARGE = "mistral.mistral-large-2402-v1:0"
    # Note: Grok and DeepSeek would need custom integration or separate endpoints


class TaskType(str, Enum):
    """Types of tasks for model routing"""
    DEEP_REASONING = "deep_reasoning"
    CREATIVE_GENERATION = "creative_generation"
    CODE_GENERATION = "code_generation"
    MATHEMATICAL_PROOF = "mathematical_proof"
    PATTERN_RECOGNITION = "pattern_recognition"
    QUICK_VALIDATION = "quick_validation"
    SUMMARIZATION = "summarization"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of each model"""
    name: str
    provider: ModelProvider
    strengths: List[TaskType]
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    latency_ms: int  # Average latency
    supports_streaming: bool
    supports_function_calling: bool


class ModelRouter:
    """Intelligent routing to appropriate models based on task"""

    def __init__(self):
        self.model_capabilities = {
            ModelProvider.CLAUDE_35_SONNET: ModelCapabilities(
                name="Claude 3.5 Sonnet",
                provider=ModelProvider.CLAUDE_35_SONNET,
                strengths=[TaskType.DEEP_REASONING, TaskType.MATHEMATICAL_PROOF],
                max_tokens=4096,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                latency_ms=2000,
                supports_streaming=True,
                supports_function_calling=True
            ),
            ModelProvider.CLAUDE_3_HAIKU: ModelCapabilities(
                name="Claude 3 Haiku",
                provider=ModelProvider.CLAUDE_3_HAIKU,
                strengths=[TaskType.QUICK_VALIDATION, TaskType.SUMMARIZATION],
                max_tokens=4096,
                cost_per_1k_input=0.00025,
                cost_per_1k_output=0.00125,
                latency_ms=500,
                supports_streaming=True,
                supports_function_calling=True
            ),
            ModelProvider.GPT_4: ModelCapabilities(
                name="GPT-4 Turbo",
                provider=ModelProvider.GPT_4,
                strengths=[TaskType.CREATIVE_GENERATION, TaskType.CODE_GENERATION],
                max_tokens=4096,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                latency_ms=3000,
                supports_streaming=True,
                supports_function_calling=True
            ),
            ModelProvider.LLAMA_3: ModelCapabilities(
                name="Llama 3 70B",
                provider=ModelProvider.LLAMA_3,
                strengths=[TaskType.PATTERN_RECOGNITION, TaskType.SUMMARIZATION],
                max_tokens=2048,
                cost_per_1k_input=0.00265,
                cost_per_1k_output=0.0035,
                latency_ms=1500,
                supports_streaming=False,
                supports_function_calling=False
            ),
        }

    def select_model(
        self,
        task_type: TaskType,
        optimize_for: str = "quality",  # "quality", "speed", "cost"
        required_features: List[str] = None
    ) -> ModelProvider:
        """Select best model for the task"""
        required_features = required_features or []

        suitable_models = []
        for provider, capabilities in self.model_capabilities.items():
            # Check if model supports required features
            if "function_calling" in required_features and not capabilities.supports_function_calling:
                continue
            if "streaming" in required_features and not capabilities.supports_streaming:
                continue

            # Check if model is strong for this task
            if task_type in capabilities.strengths:
                suitable_models.append((provider, capabilities))

        if not suitable_models:
            # Fallback to most capable model
            return ModelProvider.CLAUDE_35_SONNET

        # Sort based on optimization preference
        if optimize_for == "speed":
            suitable_models.sort(key=lambda x: x[1].latency_ms)
        elif optimize_for == "cost":
            suitable_models.sort(key=lambda x: x[1].cost_per_1k_input + x[1].cost_per_1k_output)
        else:  # quality - prefer Claude 3.5 or GPT-4
            priority_order = [
                ModelProvider.CLAUDE_35_SONNET,
                ModelProvider.GPT_4,
                ModelProvider.LLAMA_3,
                ModelProvider.CLAUDE_3_HAIKU
            ]
            suitable_models.sort(key=lambda x: priority_order.index(x[0])
                                if x[0] in priority_order else 999)

        return suitable_models[0][0]


class BedrockMessage(BaseModel):
    """Message structure for Bedrock API"""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: Union[str, List[Dict[str, Any]]]


class BedrockRequest(BaseModel):
    """Request structure for Bedrock API"""
    messages: List[BedrockMessage]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = Field(default_factory=list)
    system: Optional[str] = None


class BedrockResponse(BaseModel):
    """Response structure from Bedrock API"""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    cost_estimate: float


class MultiModelBedrockClient:
    """Enhanced Bedrock client with multi-model support and intelligent routing"""

    def __init__(
        self,
        region: str = "us-east-1",
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.router = ModelRouter()
        self.enable_caching = enable_caching
        self.cache: Dict[str, Tuple[BedrockResponse, float]] = {}
        self.cache_ttl = cache_ttl

        # Token counter for cost tracking
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            logger.warning("Tiktoken not available, using approximate token counting")

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_model": {},
            "cache_hits": 0,
            "errors": 0
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4

    def _get_cache_key(self, model: str, messages: List[BedrockMessage]) -> str:
        """Generate cache key for request"""
        content = json.dumps([msg.dict() for msg in messages], sort_keys=True)
        return f"{model}:{hash(content)}"

    def _check_cache(self, cache_key: str) -> Optional[BedrockResponse]:
        """Check if response is in cache and still valid"""
        if not self.enable_caching:
            return None

        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return response
            else:
                del self.cache[cache_key]

        return None

    def _update_cache(self, cache_key: str, response: BedrockResponse):
        """Update cache with new response"""
        if self.enable_caching:
            self.cache[cache_key] = (response, time.time())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ClientError)
    )
    async def _call_bedrock(
        self,
        model: ModelProvider,
        request: BedrockRequest
    ) -> BedrockResponse:
        """Make actual call to Bedrock API"""
        start_time = time.time()

        try:
            # Format request based on model provider
            if "anthropic" in model.value:
                body = self._format_anthropic_request(request)
            elif "openai" in model.value:
                body = self._format_openai_request(request)
            elif "meta" in model.value:
                body = self._format_llama_request(request)
            elif "mistral" in model.value:
                body = self._format_mistral_request(request)
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Make API call
            response = self.client.invoke_model(
                modelId=model.value,
                body=json.dumps(body)
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            content = self._extract_content(response_body, model)

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            input_tokens = sum(self._count_tokens(msg.content if isinstance(msg.content, str) else str(msg.content))
                             for msg in request.messages)
            output_tokens = self._count_tokens(content)

            # Calculate cost
            capabilities = self.router.model_capabilities[model]
            cost = (
                (input_tokens / 1000) * capabilities.cost_per_1k_input +
                (output_tokens / 1000) * capabilities.cost_per_1k_output
            )

            # Update metrics
            self.metrics["total_requests"] += 1
            self.metrics["total_tokens"] += input_tokens + output_tokens
            self.metrics["total_cost"] += cost
            if model.value not in self.metrics["requests_by_model"]:
                self.metrics["requests_by_model"][model.value] = 0
            self.metrics["requests_by_model"][model.value] += 1

            return BedrockResponse(
                content=content,
                model=model.value,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                latency_ms=latency_ms,
                cost_estimate=cost
            )

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error calling Bedrock: {e}")
            raise

    def _format_anthropic_request(self, request: BedrockRequest) -> Dict:
        """Format request for Anthropic models"""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "messages": [msg.dict() for msg in request.messages],
            "stop_sequences": request.stop_sequences,
            "system": request.system
        }

    def _format_openai_request(self, request: BedrockRequest) -> Dict:
        """Format request for OpenAI models"""
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        return {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop_sequences
        }

    def _format_llama_request(self, request: BedrockRequest) -> Dict:
        """Format request for Llama models"""
        # Combine messages into single prompt
        prompt = ""
        if request.system:
            prompt = f"System: {request.system}\n\n"

        for msg in request.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"{msg.role.capitalize()}: {content}\n\n"
        prompt += "Assistant: "

        return {
            "prompt": prompt,
            "max_gen_len": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }

    def _format_mistral_request(self, request: BedrockRequest) -> Dict:
        """Format request for Mistral models"""
        prompt = ""
        for msg in request.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"[{msg.role.upper()}] {content}\n"
        prompt += "[ASSISTANT] "

        return {
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop_sequences
        }

    def _extract_content(self, response_body: Dict, model: ModelProvider) -> str:
        """Extract content from model-specific response format"""
        if "anthropic" in model.value:
            return response_body["content"][0]["text"]
        elif "openai" in model.value:
            return response_body["choices"][0]["message"]["content"]
        elif "meta" in model.value:
            return response_body["generation"]
        elif "mistral" in model.value:
            return response_body["outputs"][0]["text"]
        else:
            raise ValueError(f"Unknown response format for model: {model}")

    async def generate(
        self,
        messages: List[BedrockMessage],
        task_type: TaskType = TaskType.DEEP_REASONING,
        model_override: Optional[ModelProvider] = None,
        optimize_for: str = "quality",
        use_ensemble: bool = False,
        **kwargs
    ) -> Union[BedrockResponse, List[BedrockResponse]]:
        """
        Generate response using appropriate model(s)

        Args:
            messages: Conversation messages
            task_type: Type of task for model selection
            model_override: Force specific model
            optimize_for: Optimization preference (quality/speed/cost)
            use_ensemble: Use multiple models and combine results
            **kwargs: Additional parameters for BedrockRequest

        Returns:
            Single response or list of responses if ensemble
        """
        # Create request
        request = BedrockRequest(messages=messages, **kwargs)

        if use_ensemble:
            # Use multiple models for consensus
            models = [
                ModelProvider.CLAUDE_35_SONNET,
                ModelProvider.GPT_4,
                ModelProvider.LLAMA_3
            ]
            responses = []

            for model in models:
                cache_key = self._get_cache_key(model.value, messages)
                cached = self._check_cache(cache_key)

                if cached:
                    responses.append(cached)
                else:
                    response = await self._call_bedrock(model, request)
                    self._update_cache(cache_key, response)
                    responses.append(response)

            return responses

        else:
            # Select single best model
            model = model_override or self.router.select_model(
                task_type=task_type,
                optimize_for=optimize_for
            )

            # Check cache
            cache_key = self._get_cache_key(model.value, messages)
            cached = self._check_cache(cache_key)

            if cached:
                return cached

            # Make API call
            response = await self._call_bedrock(model, request)
            self._update_cache(cache_key, response)

            return response

    async def generate_with_consensus(
        self,
        messages: List[BedrockMessage],
        task_type: TaskType = TaskType.DEEP_REASONING,
        min_agreement: float = 0.7,
        **kwargs
    ) -> BedrockResponse:
        """
        Generate response using multiple models and find consensus

        Args:
            messages: Conversation messages
            task_type: Type of task
            min_agreement: Minimum agreement threshold (0-1)
            **kwargs: Additional parameters

        Returns:
            Consensus response with confidence score
        """
        # Get responses from multiple models
        responses = await self.generate(
            messages=messages,
            task_type=task_type,
            use_ensemble=True,
            **kwargs
        )

        # Analyze responses for consensus
        contents = [r.content for r in responses]

        # Simple consensus: if responses are similar enough
        # In production, use more sophisticated NLP similarity metrics
        consensus_content = contents[0]  # Default to first
        confidence = 0.0

        # Check similarity (simplified - use embeddings in production)
        similar_count = sum(1 for c in contents if self._are_similar(consensus_content, c))
        confidence = similar_count / len(contents)

        if confidence < min_agreement:
            # No consensus - use most expensive/capable model's response
            for r in responses:
                if r.model == ModelProvider.CLAUDE_35_SONNET.value:
                    consensus_content = r.content
                    break

        # Create consensus response
        total_cost = sum(r.cost_estimate for r in responses)
        avg_latency = sum(r.latency_ms for r in responses) / len(responses)

        return BedrockResponse(
            content=consensus_content,
            model="ensemble",
            usage={
                "total_tokens": sum(r.usage["total_tokens"] for r in responses),
                "confidence": confidence
            },
            latency_ms=avg_latency,
            cost_estimate=total_cost
        )

    def _are_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simplified)"""
        # In production, use proper semantic similarity with embeddings
        # This is a very basic implementation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard = len(intersection) / len(union)
        return jaccard >= threshold

    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset usage metrics"""
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_model": {},
            "cache_hits": 0,
            "errors": 0
        }

    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        logger.info("Cache cleared")


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize client
        client = MultiModelBedrockClient(region="us-east-1")

        # Create messages
        messages = [
            BedrockMessage(
                role="user",
                content="Prove that there are infinitely many prime numbers."
            )
        ]

        # Single model generation
        response = await client.generate(
            messages=messages,
            task_type=TaskType.MATHEMATICAL_PROOF,
            optimize_for="quality",
            temperature=0.3
        )
        print(f"Response: {response.content[:200]}...")
        print(f"Model used: {response.model}")
        print(f"Cost: ${response.cost_estimate:.4f}")

        # Ensemble generation with consensus
        consensus = await client.generate_with_consensus(
            messages=messages,
            task_type=TaskType.MATHEMATICAL_PROOF,
            min_agreement=0.7
        )
        print(f"\nConsensus: {consensus.content[:200]}...")
        print(f"Confidence: {consensus.usage.get('confidence', 0):.2f}")

        # Print metrics
        metrics = client.get_metrics()
        print(f"\nMetrics: {metrics}")

    # Run example
    asyncio.run(example())