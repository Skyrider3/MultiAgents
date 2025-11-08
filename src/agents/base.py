"""
Base Agent Framework using Pydantic for Type Safety
Implements core agent capabilities and interfaces
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from loguru import logger
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

from src.communication.protocols.a2a_protocol import (
    AgentConnection,
    AgentMessage,
    MessageBroker,
    MessageType,
    MessagePriority
)
from src.llm.bedrock_client import (
    MultiModelBedrockClient,
    BedrockMessage,
    TaskType
)


class AgentRole(str, Enum):
    """Roles that agents can play in the system"""
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"
    CHALLENGER = "challenger"
    HISTORIAN = "historian"
    COORDINATOR = "coordinator"


class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class ToolParameter(BaseModel):
    """Parameter definition for agent tools"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(BaseModel):
    """Tool definition for agents"""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    handler: Optional[Callable] = Field(exclude=True, default=None)
    requires_confirmation: bool = False

    class Config:
        arbitrary_types_allowed = True


class Thought(BaseModel):
    """Represents an agent's thought process"""
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_type: str  # deductive, inductive, abductive, analogical
    evidence: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class Memory(BaseModel):
    """Agent's memory structure"""
    short_term: List[Dict[str, Any]] = Field(default_factory=list)
    long_term: Dict[str, Any] = Field(default_factory=dict)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    episodic: List[Dict[str, Any]] = Field(default_factory=list)
    semantic: Dict[str, List[str]] = Field(default_factory=dict)

    def add_to_short_term(self, item: Dict[str, Any], max_size: int = 10):
        """Add item to short-term memory with size limit"""
        self.short_term.append(item)
        if len(self.short_term) > max_size:
            # Move oldest to long-term
            old_item = self.short_term.pop(0)
            category = old_item.get("type", "general")
            if category not in self.long_term:
                self.long_term[category] = []
            self.long_term[category].append(old_item)

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query"""
        # Simple implementation - in production, use embeddings
        relevant = []
        query_lower = query.lower()

        # Search short-term memory
        for item in self.short_term:
            if query_lower in str(item).lower():
                relevant.append(item)

        # Search long-term memory
        for category, items in self.long_term.items():
            for item in items:
                if query_lower in str(item).lower():
                    relevant.append(item)
                if len(relevant) >= limit:
                    break

        return relevant[:limit]


class AgentCapabilities(BaseModel):
    """Defines what an agent can do"""
    can_reason: bool = True
    can_learn: bool = True
    can_collaborate: bool = True
    can_challenge: bool = False
    can_synthesize: bool = False
    can_verify: bool = False
    supported_domains: List[str] = Field(default_factory=list)
    max_context_length: int = 100000
    parallel_tasks: int = 3


class AgentPersonality(BaseModel):
    """Agent's personality traits for diverse reasoning"""
    curiosity: float = Field(default=0.7, ge=0.0, le=1.0)
    skepticism: float = Field(default=0.5, ge=0.0, le=1.0)
    creativity: float = Field(default=0.6, ge=0.0, le=1.0)
    thoroughness: float = Field(default=0.8, ge=0.0, le=1.0)
    risk_tolerance: float = Field(default=0.3, ge=0.0, le=1.0)


class AgentMetrics(BaseModel):
    """Performance metrics for an agent"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_thinking_time: float = 0.0
    total_execution_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    insights_generated: int = 0
    errors_encountered: int = 0
    collaboration_score: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    Provides core functionality and interfaces
    """

    def __init__(
        self,
        agent_id: str = None,
        role: AgentRole = AgentRole.RESEARCHER,
        bedrock_client: MultiModelBedrockClient = None,
        message_broker: MessageBroker = None,
        capabilities: AgentCapabilities = None,
        personality: AgentPersonality = None,
        tools: List[Tool] = None,
        **kwargs
    ):
        self.agent_id = agent_id or f"{role.value}_{uuid.uuid4().hex[:8]}"
        self.role = role
        self.bedrock_client = bedrock_client or MultiModelBedrockClient()
        self.message_broker = message_broker
        self.capabilities = capabilities or AgentCapabilities()
        self.personality = personality or AgentPersonality()
        self.tools = tools or []
        self.status = AgentStatus.IDLE
        self.memory = Memory()
        self.metrics = AgentMetrics()
        self.current_task = None
        self.thoughts: List[Thought] = []
        self.connection = AgentConnection(self.agent_id, self._get_capability_list())

        # Register tools
        self._register_default_tools()
        self._register_custom_tools()

        # Initialize agent-specific attributes
        self._initialize(**kwargs)

        logger.info(f"Agent {self.agent_id} initialized with role {role.value}")

    @abstractmethod
    def _initialize(self, **kwargs):
        """Initialize agent-specific attributes"""
        pass

    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Thought:
        """
        Agent's thinking process - analyze context and form thoughts
        """
        pass

    @abstractmethod
    async def act(self, thought: Thought) -> Dict[str, Any]:
        """
        Agent's action based on thoughts
        """
        pass

    def _get_capability_list(self) -> List[str]:
        """Get list of agent capabilities for registration"""
        caps = []
        if self.capabilities.can_reason:
            caps.append("reasoning")
        if self.capabilities.can_learn:
            caps.append("learning")
        if self.capabilities.can_collaborate:
            caps.append("collaboration")
        if self.capabilities.can_challenge:
            caps.append("challenging")
        if self.capabilities.can_synthesize:
            caps.append("synthesis")
        if self.capabilities.can_verify:
            caps.append("verification")
        caps.extend(self.capabilities.supported_domains)
        return caps

    def _register_default_tools(self):
        """Register default tools available to all agents"""
        default_tools = [
            Tool(
                name="remember",
                description="Store information in memory",
                parameters=[
                    ToolParameter(name="key", type="str", description="Memory key"),
                    ToolParameter(name="value", type="any", description="Information to store")
                ],
                handler=self._tool_remember
            ),
            Tool(
                name="recall",
                description="Retrieve information from memory",
                parameters=[
                    ToolParameter(name="query", type="str", description="Search query")
                ],
                handler=self._tool_recall
            ),
            Tool(
                name="communicate",
                description="Send message to another agent",
                parameters=[
                    ToolParameter(name="recipient", type="str", description="Agent ID"),
                    ToolParameter(name="message", type="str", description="Message content")
                ],
                handler=self._tool_communicate
            ),
            Tool(
                name="reflect",
                description="Reflect on recent actions and learnings",
                handler=self._tool_reflect
            )
        ]
        self.tools.extend(default_tools)

    @abstractmethod
    def _register_custom_tools(self):
        """Register agent-specific tools"""
        pass

    async def _tool_remember(self, key: str, value: Any) -> Dict[str, Any]:
        """Store information in memory"""
        self.memory.working_memory[key] = value
        self.memory.add_to_short_term({"type": "fact", "key": key, "value": value})
        return {"status": "stored", "key": key}

    async def _tool_recall(self, query: str) -> Dict[str, Any]:
        """Retrieve information from memory"""
        results = self.memory.recall(query)
        return {"results": results, "count": len(results)}

    async def _tool_communicate(self, recipient: str, message: str) -> Dict[str, Any]:
        """Send message to another agent"""
        if not self.message_broker:
            return {"error": "No message broker available"}

        msg = AgentMessage(
            from_agent=self.agent_id,
            to_agent=recipient,
            type=MessageType.QUERY,
            content={"message": message}
        )
        await self.connection.send_message(msg, self.message_broker)
        self.metrics.messages_sent += 1

        return {"status": "sent", "recipient": recipient}

    async def _tool_reflect(self) -> Dict[str, Any]:
        """Reflect on recent actions and learnings"""
        recent_thoughts = self.thoughts[-5:]
        reflection = {
            "thought_count": len(recent_thoughts),
            "average_confidence": sum(t.confidence for t in recent_thoughts) / len(recent_thoughts) if recent_thoughts else 0,
            "reasoning_types": list(set(t.reasoning_type for t in recent_thoughts)),
            "insights": []
        }

        # Generate insights based on patterns
        if reflection["average_confidence"] < 0.5:
            reflection["insights"].append("Low confidence in recent reasoning - may need more evidence")
        if len(reflection["reasoning_types"]) == 1:
            reflection["insights"].append("Using single reasoning type - consider diverse approaches")

        return reflection

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main task processing pipeline
        """
        start_time = time.time()
        self.current_task = task
        self.status = AgentStatus.THINKING

        try:
            # Think about the task
            thought = await self.think(task)
            self.thoughts.append(thought)
            self.memory.add_to_short_term({
                "type": "thought",
                "content": thought.content,
                "confidence": thought.confidence
            })

            thinking_time = time.time() - start_time
            self.metrics.total_thinking_time += thinking_time

            # Act based on thoughts
            self.status = AgentStatus.EXECUTING
            action_start = time.time()
            result = await self.act(thought)

            execution_time = time.time() - action_start
            self.metrics.total_execution_time += execution_time

            # Store episode in memory
            self.memory.episodic.append({
                "task": task,
                "thought": thought.dict(),
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })

            self.metrics.tasks_completed += 1
            self.status = AgentStatus.IDLE

            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "thought": thought.dict(),
                "result": result,
                "metrics": {
                    "thinking_time": thinking_time,
                    "execution_time": execution_time,
                    "total_time": time.time() - start_time
                }
            }

        except Exception as e:
            logger.error(f"Agent {self.agent_id} encountered error: {e}")
            self.metrics.tasks_failed += 1
            self.metrics.errors_encountered += 1
            self.status = AgentStatus.ERROR

            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "error": str(e),
                "task": task
            }

    async def collaborate(self, partner_agent: 'BaseAgent', task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with another agent on a task
        """
        if not self.capabilities.can_collaborate:
            return {"error": f"Agent {self.agent_id} cannot collaborate"}

        # Send initial proposal to partner
        proposal = await self.think({
            "task": task,
            "collaboration_with": partner_agent.agent_id
        })

        if self.message_broker:
            msg = AgentMessage(
                from_agent=self.agent_id,
                to_agent=partner_agent.agent_id,
                type=MessageType.HYPOTHESIS,
                content={
                    "proposal": proposal.content,
                    "confidence": proposal.confidence,
                    "evidence": proposal.evidence
                }
            )
            await self.connection.send_message(msg, self.message_broker)

            # Wait for response
            response = await self.connection.query_and_wait(
                target_agent=partner_agent.agent_id,
                content={"awaiting": "collaboration_response"},
                broker=self.message_broker
            )

            if response:
                # Process partner's response and reach consensus
                return await self._reach_consensus(proposal, response.content)

        return {"status": "collaboration_failed", "reason": "no_response"}

    async def _reach_consensus(self, proposal: Thought, partner_response: Dict[str, Any]) -> Dict[str, Any]:
        """Reach consensus with partner agent"""
        # Simple consensus mechanism - can be made more sophisticated
        partner_confidence = partner_response.get("confidence", 0)
        combined_confidence = (proposal.confidence + partner_confidence) / 2

        if combined_confidence > 0.7:
            return {
                "status": "consensus_reached",
                "confidence": combined_confidence,
                "agreed_action": proposal.content
            }
        else:
            return {
                "status": "no_consensus",
                "confidence": combined_confidence,
                "requires": "further_discussion"
            }

    async def learn(self, feedback: Dict[str, Any]) -> bool:
        """
        Learn from feedback to improve performance
        """
        if not self.capabilities.can_learn:
            return False

        # Store feedback in semantic memory
        feedback_type = feedback.get("type", "general")
        if feedback_type not in self.memory.semantic:
            self.memory.semantic[feedback_type] = []

        self.memory.semantic[feedback_type].append(json.dumps(feedback))

        # Adjust personality based on feedback
        if feedback.get("success", False):
            # Successful outcome - slightly increase risk tolerance
            self.personality.risk_tolerance = min(1.0, self.personality.risk_tolerance + 0.01)
        else:
            # Failure - increase skepticism and thoroughness
            self.personality.skepticism = min(1.0, self.personality.skepticism + 0.02)
            self.personality.thoroughness = min(1.0, self.personality.thoroughness + 0.01)

        return True

    async def challenge(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Challenge a claim or hypothesis
        """
        if not self.capabilities.can_challenge:
            return {"error": f"Agent {self.agent_id} cannot challenge claims"}

        # Generate counter-arguments
        thought = await self.think({
            "mode": "adversarial",
            "claim": claim
        })

        challenges = []
        if thought.confidence < 0.5:
            challenges.append("Low confidence in claim validity")
        if len(thought.evidence) < 3:
            challenges.append("Insufficient evidence provided")

        return {
            "agent_id": self.agent_id,
            "challenges": challenges,
            "counter_evidence": thought.evidence,
            "alternative_hypothesis": thought.content
        }

    async def synthesize(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize multiple inputs into coherent insight
        """
        if not self.capabilities.can_synthesize:
            return {"error": f"Agent {self.agent_id} cannot synthesize"}

        # Combine inputs
        combined_context = {
            "synthesis_task": True,
            "inputs": inputs,
            "input_count": len(inputs)
        }

        thought = await self.think(combined_context)

        # Generate synthesis
        synthesis = {
            "agent_id": self.agent_id,
            "synthesized_insight": thought.content,
            "confidence": thought.confidence,
            "input_sources": len(inputs),
            "reasoning_type": thought.reasoning_type,
            "supporting_evidence": thought.evidence
        }

        self.metrics.insights_generated += 1
        return synthesis

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.status.value,
            "metrics": self.metrics.dict(),
            "personality": self.personality.dict(),
            "memory_stats": {
                "short_term": len(self.memory.short_term),
                "long_term": len(self.memory.long_term),
                "working": len(self.memory.working_memory),
                "episodic": len(self.memory.episodic),
                "semantic": len(self.memory.semantic)
            }
        }

    async def shutdown(self):
        """Graceful shutdown"""
        self.status = AgentStatus.TERMINATED

        # Save memory to persistent storage if needed
        if self.memory.episodic or self.memory.semantic:
            # TODO: Implement persistent storage
            logger.info(f"Agent {self.agent_id} saving memory before shutdown")

        # Unregister from message broker
        if self.message_broker:
            await self.message_broker.unregister_agent(self.agent_id)

        logger.info(f"Agent {self.agent_id} shutdown complete")