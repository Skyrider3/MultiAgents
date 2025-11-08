"""
A2A-Style Agent-to-Agent Communication Protocol
Inspired by: https://github.com/a2aproject/A2A

This module implements a robust communication protocol for multi-agent systems
with message tracing, verification, and replay capabilities.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import msgpack
import zmq.asyncio
from pydantic import BaseModel, Field, validator
from loguru import logger


class MessageType(str, Enum):
    """Types of messages in the A2A protocol"""

    # Core message types
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"

    # Agent collaboration types
    HYPOTHESIS = "HYPOTHESIS"
    VALIDATION = "VALIDATION"
    CHALLENGE = "CHALLENGE"
    CONSENSUS = "CONSENSUS"

    # Control messages
    HEARTBEAT = "HEARTBEAT"
    REGISTER = "REGISTER"
    UNREGISTER = "UNREGISTER"

    # Research-specific types
    DISCOVERY = "DISCOVERY"
    CITATION = "CITATION"
    PROOF = "PROOF"
    CONJECTURE = "CONJECTURE"


class MessagePriority(int, Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class AgentMessage(BaseModel):
    """Structure for agent-to-agent messages"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str
    to_agent: Optional[str] = None  # None for broadcasts
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    signature: Optional[str] = None
    parent_message_id: Optional[str] = None  # For conversation threading
    trace_id: Optional[str] = None  # For distributed tracing

    class Config:
        use_enum_values = True

    @validator('content')
    def validate_content(cls, v, values):
        """Validate content based on message type"""
        msg_type = values.get('type')

        if msg_type == MessageType.HYPOTHESIS:
            required = {'statement', 'confidence', 'evidence'}
            if not required.issubset(v.keys()):
                raise ValueError(f"HYPOTHESIS messages require {required}")

        elif msg_type == MessageType.PROOF:
            required = {'theorem', 'steps', 'assumptions'}
            if not required.issubset(v.keys()):
                raise ValueError(f"PROOF messages require {required}")

        return v


class ConversationTrace:
    """Tracks complete conversation history between agents"""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.messages: List[AgentMessage] = []
        self.participants: Set[str] = set()
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def add_message(self, message: AgentMessage):
        """Add a message to the trace"""
        self.messages.append(message)
        self.participants.add(message.from_agent)
        if message.to_agent:
            self.participants.add(message.to_agent)

    def get_conversation_graph(self) -> Dict[str, Any]:
        """Generate conversation flow graph"""
        graph = {
            'nodes': list(self.participants),
            'edges': [],
            'messages': []
        }

        for msg in self.messages:
            if msg.to_agent:
                graph['edges'].append({
                    'from': msg.from_agent,
                    'to': msg.to_agent,
                    'message_id': msg.id,
                    'type': msg.type
                })

            graph['messages'].append({
                'id': msg.id,
                'type': msg.type,
                'timestamp': msg.timestamp,
                'from': msg.from_agent,
                'to': msg.to_agent
            })

        return graph

    def replay(self) -> List[AgentMessage]:
        """Get messages in chronological order for replay"""
        return sorted(self.messages, key=lambda m: m.timestamp)


class MessageBroker:
    """Central message broker for agent communication"""

    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

        self.agents: Dict[str, 'AgentConnection'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.traces: Dict[str, ConversationTrace] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}

        # Message history for replay
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 10000

        # Crypto keys for message signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    async def register_agent(self, agent_id: str, connection: 'AgentConnection'):
        """Register an agent with the broker"""
        self.agents[agent_id] = connection
        logger.info(f"Agent {agent_id} registered")

        # Broadcast registration to other agents
        msg = AgentMessage(
            from_agent="broker",
            type=MessageType.REGISTER,
            content={"agent_id": agent_id, "capabilities": connection.capabilities}
        )
        await self.broadcast(msg)

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")

            # Broadcast unregistration
            msg = AgentMessage(
                from_agent="broker",
                type=MessageType.UNREGISTER,
                content={"agent_id": agent_id}
            )
            await self.broadcast(msg)

    async def route_message(self, message: AgentMessage):
        """Route message to appropriate agent(s)"""
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

        # Track in conversation trace
        if message.trace_id:
            if message.trace_id not in self.traces:
                self.traces[message.trace_id] = ConversationTrace(message.trace_id)
            self.traces[message.trace_id].add_message(message)

        # Sign message
        message.signature = self._sign_message(message)

        # Route based on recipient
        if message.to_agent:
            # Direct message
            if message.to_agent in self.agents:
                await self.agents[message.to_agent].receive_message(message)
            else:
                logger.warning(f"Agent {message.to_agent} not found")
        else:
            # Broadcast message
            await self.broadcast(message)

        # Call registered handlers
        if message.type in self.message_handlers:
            for handler in self.message_handlers[message.type]:
                await handler(message)

    async def broadcast(self, message: AgentMessage):
        """Broadcast message to all agents"""
        # Serialize with msgpack for efficiency
        data = msgpack.packb(message.dict())
        await self.socket.send(data)

        # Also send to connected agents
        for agent_id, connection in self.agents.items():
            if agent_id != message.from_agent:
                await connection.receive_message(message)

    def _sign_message(self, message: AgentMessage) -> str:
        """Sign message for verification"""
        message_bytes = json.dumps(message.dict(exclude={'signature'})).encode()
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()

    def verify_message(self, message: AgentMessage) -> bool:
        """Verify message signature"""
        if not message.signature:
            return False

        try:
            message_bytes = json.dumps(message.dict(exclude={'signature'})).encode()
            signature_bytes = bytes.fromhex(message.signature)

            self.public_key.verify(
                signature_bytes,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for specific message types"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def get_trace(self, trace_id: str) -> Optional[ConversationTrace]:
        """Get conversation trace by ID"""
        return self.traces.get(trace_id)

    def get_agent_history(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Get message history for a specific agent"""
        agent_messages = [
            msg for msg in self.message_history
            if msg.from_agent == agent_id or msg.to_agent == agent_id
        ]
        return agent_messages[-limit:]

    async def replay_conversation(self, trace_id: str, speed: float = 1.0):
        """Replay a conversation for debugging/analysis"""
        trace = self.get_trace(trace_id)
        if not trace:
            return

        messages = trace.replay()
        base_time = messages[0].timestamp if messages else 0

        for i, msg in enumerate(messages):
            if i > 0:
                delay = (msg.timestamp - messages[i-1].timestamp) / speed
                await asyncio.sleep(delay)

            # Re-route the message
            await self.route_message(msg)

    def export_trace(self, trace_id: str, format: str = "json") -> Union[str, bytes]:
        """Export conversation trace in various formats"""
        trace = self.get_trace(trace_id)
        if not trace:
            return "{}" if format == "json" else b""

        if format == "json":
            return json.dumps({
                'trace_id': trace.trace_id,
                'participants': list(trace.participants),
                'messages': [msg.dict() for msg in trace.messages],
                'graph': trace.get_conversation_graph()
            }, indent=2)

        elif format == "msgpack":
            return msgpack.packb({
                'trace_id': trace.trace_id,
                'participants': list(trace.participants),
                'messages': [msg.dict() for msg in trace.messages]
            })

        else:
            raise ValueError(f"Unsupported format: {format}")


class AgentConnection:
    """Represents a connection to an agent"""

    def __init__(self, agent_id: str, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.capabilities = capabilities or []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: Dict[str, asyncio.Future] = {}

    async def send_message(self, message: AgentMessage, broker: MessageBroker):
        """Send a message through the broker"""
        await broker.route_message(message)

    async def receive_message(self, message: AgentMessage):
        """Receive a message from the broker"""
        await self.message_queue.put(message)

        # If this is a response to a query, resolve the future
        if message.parent_message_id in self.response_futures:
            self.response_futures[message.parent_message_id].set_result(message)

    async def query_and_wait(
        self,
        target_agent: str,
        content: Dict[str, Any],
        timeout: float = 30.0,
        broker: MessageBroker = None
    ) -> Optional[AgentMessage]:
        """Send a query and wait for response"""
        query_msg = AgentMessage(
            from_agent=self.agent_id,
            to_agent=target_agent,
            type=MessageType.QUERY,
            content=content
        )

        # Create future for response
        future = asyncio.Future()
        self.response_futures[query_msg.id] = future

        # Send query
        if broker:
            await self.send_message(query_msg, broker)

        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Query {query_msg.id} timed out")
            return None
        finally:
            # Clean up future
            if query_msg.id in self.response_futures:
                del self.response_futures[query_msg.id]

    async def get_next_message(self, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Get next message from queue"""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=timeout
                )
            else:
                return await self.message_queue.get()
        except asyncio.TimeoutError:
            return None


class ProtocolMonitor:
    """Monitor and analyze agent communication patterns"""

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.metrics: Dict[str, Any] = {
            'total_messages': 0,
            'messages_by_type': {},
            'messages_by_agent': {},
            'average_response_time': 0,
            'conversation_lengths': []
        }

    async def start_monitoring(self):
        """Start monitoring agent communications"""
        # Register handlers for all message types
        for msg_type in MessageType:
            self.broker.register_handler(msg_type, self._handle_message)

    async def _handle_message(self, message: AgentMessage):
        """Process message for metrics"""
        self.metrics['total_messages'] += 1

        # Count by type
        msg_type = message.type
        if msg_type not in self.metrics['messages_by_type']:
            self.metrics['messages_by_type'][msg_type] = 0
        self.metrics['messages_by_type'][msg_type] += 1

        # Count by agent
        agent = message.from_agent
        if agent not in self.metrics['messages_by_agent']:
            self.metrics['messages_by_agent'][agent] = 0
        self.metrics['messages_by_agent'][agent] += 1

        # Track response times for queries
        if message.type == MessageType.RESPONSE and message.parent_message_id:
            # Find original query in history
            for hist_msg in self.broker.message_history:
                if hist_msg.id == message.parent_message_id:
                    response_time = message.timestamp - hist_msg.timestamp
                    self._update_average_response_time(response_time)
                    break

    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time"""
        n = self.metrics['total_messages']
        old_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (old_avg * (n - 1) + new_time) / n

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

    def get_communication_graph(self) -> Dict[str, Any]:
        """Generate communication graph between agents"""
        graph = {'nodes': [], 'edges': {}}

        for agent in self.broker.agents.keys():
            graph['nodes'].append(agent)

        # Count messages between each pair of agents
        for msg in self.broker.message_history:
            if msg.to_agent:
                key = f"{msg.from_agent}->{msg.to_agent}"
                if key not in graph['edges']:
                    graph['edges'][key] = 0
                graph['edges'][key] += 1

        return graph


# Example usage
if __name__ == "__main__":
    async def example():
        # Create broker
        broker = MessageBroker()

        # Create agent connections
        researcher = AgentConnection("researcher_01", ["analyze", "extract"])
        reviewer = AgentConnection("reviewer_01", ["validate", "critique"])

        # Register agents
        await broker.register_agent("researcher_01", researcher)
        await broker.register_agent("reviewer_01", reviewer)

        # Send a hypothesis message
        hypothesis_msg = AgentMessage(
            from_agent="researcher_01",
            to_agent="reviewer_01",
            type=MessageType.HYPOTHESIS,
            content={
                "statement": "All Mersenne primes are of the form 2^p - 1 where p is prime",
                "confidence": 0.95,
                "evidence": ["paper_1", "theorem_2"]
            },
            trace_id="trace_001"
        )

        await researcher.send_message(hypothesis_msg, broker)

        # Reviewer receives and responds
        received = await reviewer.get_next_message(timeout=1.0)
        if received:
            response = AgentMessage(
                from_agent="reviewer_01",
                to_agent="researcher_01",
                type=MessageType.VALIDATION,
                parent_message_id=received.id,
                content={
                    "status": "confirmed",
                    "notes": "This is a well-known property of Mersenne primes"
                },
                trace_id="trace_001"
            )
            await reviewer.send_message(response, broker)

        # Get conversation trace
        trace = broker.get_trace("trace_001")
        if trace:
            print(f"Conversation participants: {trace.participants}")
            print(f"Total messages: {len(trace.messages)}")
            print(f"Conversation graph: {trace.get_conversation_graph()}")

    # Run example
    asyncio.run(example())