"""
FastAPI Application Setup
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import time
from typing import Any, Dict

from src.config import settings, setup_logging
from src.api.routes import agent_routes, knowledge_routes, ingestion_routes, reasoning_routes
from src.api.websocket_manager import WebSocketManager
from src.knowledge.graph.neo4j_manager import Neo4jManager
from src.knowledge.vector.qdrant_client import QdrantClient


# Global instances
neo4j_manager = None
qdrant_client = None
websocket_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    global neo4j_manager, qdrant_client, websocket_manager

    # Startup
    logger.info("Starting Multi-Agent Mathematical Discovery System...")

    # Setup logging
    setup_logging()

    # Initialize Neo4j
    try:
        neo4j_manager = Neo4jManager()
        await neo4j_manager.connect()
        logger.info("Neo4j connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")

    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient()
        await qdrant_client.connect()
        logger.info("Qdrant connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")

    # Initialize WebSocket manager
    websocket_manager = WebSocketManager()

    # Store in app state
    app.state.neo4j = neo4j_manager
    app.state.qdrant = qdrant_client
    app.state.websocket = websocket_manager

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    if neo4j_manager:
        await neo4j_manager.close()

    if qdrant_client:
        await qdrant_client.disconnect()

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    """
    app = FastAPI(
        title="Multi-Agent Mathematical Discovery System",
        description="AI-powered system for mathematical conjecture discovery using multi-agent collaboration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    # Include routers
    app.include_router(agent_routes.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(knowledge_routes.router, prefix="/api/v1/knowledge", tags=["knowledge"])
    app.include_router(ingestion_routes.router, prefix="/api/v1/ingestion", tags=["ingestion"])
    app.include_router(reasoning_routes.router, prefix="/api/v1/reasoning", tags=["reasoning"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Multi-Agent Mathematical Discovery System",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "agents": "/api/v1/agents",
                "knowledge": "/api/v1/knowledge",
                "ingestion": "/api/v1/ingestion",
                "reasoning": "/api/v1/reasoning",
                "websocket": "/ws"
            }
        }

    # Health check
    @app.get("/health")
    async def health_check():
        health_status = {
            "status": "healthy",
            "services": {
                "neo4j": "connected" if app.state.neo4j else "disconnected",
                "qdrant": "connected" if app.state.qdrant else "disconnected",
                "websocket": "active" if app.state.websocket else "inactive"
            }
        }
        return health_status

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Get system metrics"""
        metrics_data = {
            "agents": {
                "total_tasks": 0,
                "completed_tasks": 0,
                "active_agents": 0
            },
            "knowledge": {
                "total_papers": 0,
                "total_conjectures": 0,
                "total_theorems": 0
            },
            "websocket": {
                "active_connections": len(app.state.websocket.active_connections) if app.state.websocket else 0
            }
        }

        # Get actual metrics from Neo4j if available
        if app.state.neo4j:
            try:
                stats = await app.state.neo4j.get_statistics()
                metrics_data["knowledge"].update(stats)
            except:
                pass

        return metrics_data

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
    """
    app = create_app()

    uvicorn.run(
        app if not reload else "src.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run with default settings
    run_server(
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.environment == "development"
    )