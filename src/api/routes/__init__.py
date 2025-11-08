"""
API Routes Module
"""

from src.api.routes import agent_routes
from src.api.routes import knowledge_routes
from src.api.routes import ingestion_routes
from src.api.routes import reasoning_routes

__all__ = [
    "agent_routes",
    "knowledge_routes",
    "ingestion_routes",
    "reasoning_routes"
]