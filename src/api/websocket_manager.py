"""
WebSocket Manager for Real-time Communication
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Optional, Any
import json
import asyncio
from datetime import datetime
from loguru import logger
from enum import Enum


class MessageType(str, Enum):
    AGENT_STATUS = "agent_status"
    AGENT_MESSAGE = "agent_message"
    TASK_UPDATE = "task_update"
    DISCOVERY = "discovery"
    ERROR = "error"
    SYSTEM = "system"
    PROGRESS = "progress"


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasting
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.logger = logger.bind(module="websocket_manager")

    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Accept a new WebSocket connection

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            metadata: Optional client metadata
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        self.connection_metadata[client_id]["connected_at"] = datetime.now().isoformat()

        self.logger.info(f"Client {client_id} connected")

        # Send welcome message
        await self.send_personal_message(
            client_id,
            {
                "type": MessageType.SYSTEM,
                "message": "Connected to Multi-Agent System",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
        )

    def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection

        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]

            # Remove from all subscriptions
            for topic in self.subscriptions:
                self.subscriptions[topic].discard(client_id)

            self.logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, client_id: str, message: Dict[str, Any]):
        """
        Send message to a specific client

        Args:
            client_id: Target client ID
            message: Message to send
        """
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None):
        """
        Broadcast message to all connected clients

        Args:
            message: Message to broadcast
            exclude: Set of client IDs to exclude
        """
        exclude = exclude or set()
        disconnected = []

        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """
        Broadcast message to clients subscribed to a topic

        Args:
            topic: Topic name
            message: Message to broadcast
        """
        if topic not in self.subscriptions:
            return

        message["topic"] = topic
        disconnected = []

        for client_id in self.subscriptions[topic]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    self.logger.error(f"Error sending to {client_id}: {e}")
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    def subscribe(self, client_id: str, topic: str):
        """
        Subscribe a client to a topic

        Args:
            client_id: Client ID
            topic: Topic to subscribe to
        """
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()

        self.subscriptions[topic].add(client_id)
        self.logger.debug(f"Client {client_id} subscribed to {topic}")

    def unsubscribe(self, client_id: str, topic: str):
        """
        Unsubscribe a client from a topic

        Args:
            client_id: Client ID
            topic: Topic to unsubscribe from
        """
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            self.logger.debug(f"Client {client_id} unsubscribed from {topic}")

    async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """
        Handle incoming message from a client

        Args:
            client_id: Sender client ID
            message: Received message
        """
        message_type = message.get("type")

        if message_type == "subscribe":
            topic = message.get("topic")
            if topic:
                self.subscribe(client_id, topic)
                await self.send_personal_message(
                    client_id,
                    {
                        "type": MessageType.SYSTEM,
                        "message": f"Subscribed to {topic}",
                        "topic": topic
                    }
                )

        elif message_type == "unsubscribe":
            topic = message.get("topic")
            if topic:
                self.unsubscribe(client_id, topic)
                await self.send_personal_message(
                    client_id,
                    {
                        "type": MessageType.SYSTEM,
                        "message": f"Unsubscribed from {topic}",
                        "topic": topic
                    }
                )

        elif message_type == "ping":
            await self.send_personal_message(
                client_id,
                {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }
            )

        else:
            # Forward to other handlers or broadcast
            self.logger.debug(f"Received message from {client_id}: {message_type}")

    # Agent-specific broadcasting methods

    async def broadcast_agent_status(self, agent_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """
        Broadcast agent status update

        Args:
            agent_id: Agent identifier
            status: Agent status
            details: Additional details
        """
        message = {
            "type": MessageType.AGENT_STATUS,
            "agent_id": agent_id,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_topic(f"agent_{agent_id}", message)
        await self.broadcast_to_topic("agents", message)

    async def broadcast_agent_message(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Broadcast agent message

        Args:
            agent_id: Agent identifier
            content: Message content
            metadata: Message metadata
        """
        message = {
            "type": MessageType.AGENT_MESSAGE,
            "agent_id": agent_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_topic(f"agent_{agent_id}", message)
        await self.broadcast_to_topic("agents", message)

    async def broadcast_task_update(self, task_id: str, status: str, progress: float, details: Optional[Dict[str, Any]] = None):
        """
        Broadcast task progress update

        Args:
            task_id: Task identifier
            status: Task status
            progress: Progress percentage (0-100)
            details: Additional details
        """
        message = {
            "type": MessageType.TASK_UPDATE,
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_topic(f"task_{task_id}", message)
        await self.broadcast_to_topic("tasks", message)

    async def broadcast_discovery(self, discovery_type: str, content: Dict[str, Any]):
        """
        Broadcast a new discovery

        Args:
            discovery_type: Type of discovery (conjecture, pattern, proof, etc.)
            content: Discovery content
        """
        message = {
            "type": MessageType.DISCOVERY,
            "discovery_type": discovery_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_topic("discoveries", message)
        await self.broadcast(message)  # Broadcast to all

    async def broadcast_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Broadcast error message

        Args:
            error_type: Type of error
            message: Error message
            details: Error details
        """
        error_message = {
            "type": MessageType.ERROR,
            "error_type": error_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast(error_message)

    async def broadcast_progress(self, operation: str, current: int, total: int, details: Optional[Dict[str, Any]] = None):
        """
        Broadcast progress update

        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            details: Additional details
        """
        message = {
            "type": MessageType.PROGRESS,
            "operation": operation,
            "current": current,
            "total": total,
            "percentage": (current / total * 100) if total > 0 else 0,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_topic("progress", message)

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about active connections

        Returns:
            Connection statistics
        """
        return {
            "total_connections": len(self.active_connections),
            "clients": list(self.active_connections.keys()),
            "topics": {
                topic: len(subscribers)
                for topic, subscribers in self.subscriptions.items()
            },
            "metadata": self.connection_metadata
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint handler

    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await websocket_manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await websocket_manager.handle_client_message(client_id, data)

    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)
        await websocket.close()


# Example usage for testing
if __name__ == "__main__":
    import asyncio

    async def test_websocket_manager():
        manager = WebSocketManager()

        # Simulate broadcasting
        await manager.broadcast_agent_status("agent_1", "active", {"task": "analyzing"})
        await manager.broadcast_task_update("task_1", "in_progress", 50.0)
        await manager.broadcast_discovery(
            "conjecture",
            {
                "title": "New Prime Pattern",
                "description": "Discovered pattern in prime gaps",
                "confidence": 0.85
            }
        )

        print("WebSocket manager test complete")

    asyncio.run(test_websocket_manager())