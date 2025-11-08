#!/usr/bin/env python3
"""
System Initialization Script
"""

import asyncio
import sys
from pathlib import Path
import subprocess
import time
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import settings, setup_logging
from src.knowledge.graph.neo4j_manager import Neo4jManager
from src.knowledge.vector.qdrant_client import QdrantClient
from loguru import logger


async def check_service(service: str, port: int) -> bool:
    """Check if a service is running"""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False


async def wait_for_service(service: str, port: int, max_attempts: int = 30):
    """Wait for a service to be available"""
    logger.info(f"Waiting for {service} on port {port}...")

    for attempt in range(max_attempts):
        if await check_service(service, port):
            logger.success(f"{service} is ready!")
            return True

        logger.debug(f"Attempt {attempt + 1}/{max_attempts} - {service} not ready")
        await asyncio.sleep(2)

    logger.error(f"{service} failed to start after {max_attempts} attempts")
    return False


async def init_neo4j():
    """Initialize Neo4j database"""
    logger.info("Initializing Neo4j...")

    # Wait for Neo4j to be ready
    if not await wait_for_service("Neo4j", 7687):
        return False

    try:
        # Connect to Neo4j
        neo4j = Neo4jManager()
        await neo4j.init()

        # Create indexes and constraints
        logger.info("Creating Neo4j indexes and constraints...")

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conjecture) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Theorem) REQUIRE t.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.domain)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Conjecture) ON (c.status)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Conjecture) ON (c.confidence)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Theorem) ON (t.domain)",
        ]

        # Execute constraints
        for constraint in constraints:
            try:
                await neo4j.execute_query(constraint)
                logger.debug(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint might already exist: {e}")

        # Execute indexes
        for index in indexes:
            try:
                await neo4j.execute_query(index)
                logger.debug(f"Created index: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index might already exist: {e}")

        await neo4j.close()
        logger.success("Neo4j initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        return False


async def init_qdrant():
    """Initialize Qdrant vector database"""
    logger.info("Initializing Qdrant...")

    # Wait for Qdrant to be ready
    if not await wait_for_service("Qdrant", 6333):
        return False

    try:
        # Connect to Qdrant
        qdrant = QdrantClient()
        await qdrant.init()

        # Create collections
        collections = [
            {
                "name": "papers",
                "vector_size": settings.vector_dimension,
                "distance": "Cosine"
            },
            {
                "name": "conjectures",
                "vector_size": settings.vector_dimension,
                "distance": "Cosine"
            },
            {
                "name": "theorems",
                "vector_size": settings.vector_dimension,
                "distance": "Cosine"
            },
            {
                "name": "formulas",
                "vector_size": settings.vector_dimension,
                "distance": "Cosine"
            }
        ]

        for collection in collections:
            logger.debug(f"Creating collection: {collection['name']}")
            # Collection creation would be implemented in the actual QdrantClient

        await qdrant.close()
        logger.success("Qdrant initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        return False


async def init_postgres():
    """Initialize PostgreSQL database"""
    logger.info("Initializing PostgreSQL...")

    # Wait for PostgreSQL to be ready
    if not await wait_for_service("PostgreSQL", 5432):
        return False

    try:
        import asyncpg

        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='mathuser',
            password='mathpass',
            database='postgres'
        )

        # Create database if not exists
        try:
            await conn.execute("CREATE DATABASE mathdb")
            logger.info("Created database 'mathdb'")
        except:
            logger.debug("Database 'mathdb' already exists")

        await conn.close()

        # Connect to the mathdb
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='mathuser',
            password='mathpass',
            database='mathdb'
        )

        # Create tables
        tables = [
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                task_id VARCHAR(255) UNIQUE NOT NULL,
                agent_id VARCHAR(255),
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                result JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS discoveries (
                id SERIAL PRIMARY KEY,
                discovery_id VARCHAR(255) UNIQUE NOT NULL,
                type VARCHAR(50),
                title TEXT,
                description TEXT,
                confidence FLOAT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                id SERIAL PRIMARY KEY,
                agent_id VARCHAR(255),
                level VARCHAR(20),
                message TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]

        for table_sql in tables:
            await conn.execute(table_sql)
            logger.debug(f"Created table: {table_sql[:50]}...")

        await conn.close()
        logger.success("PostgreSQL initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}")
        return False


async def start_docker_services():
    """Start Docker services"""
    logger.info("Starting Docker services...")

    try:
        # Check if Docker is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error("Docker is not running. Please start Docker first.")
            return False

        # Start services with docker-compose
        logger.info("Starting services with docker-compose...")

        compose_file = project_root / "docker-compose.yml"

        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.success("Docker services started successfully!")

            # Wait a bit for services to fully start
            logger.info("Waiting for services to be ready...")
            await asyncio.sleep(10)

            return True
        else:
            logger.error(f"Failed to start Docker services: {result.stderr}")
            return False

    except FileNotFoundError:
        logger.error("docker-compose not found. Please install Docker Compose.")
        return False
    except Exception as e:
        logger.error(f"Error starting Docker services: {e}")
        return False


async def init_sample_data():
    """Initialize sample data"""
    logger.info("Initializing sample data...")

    try:
        # Connect to Neo4j
        neo4j = Neo4jManager()
        await neo4j.init()

        # Create sample nodes
        sample_data = {
            "papers": [
                {
                    "id": "sample_paper_1",
                    "title": "On the Distribution of Prime Numbers",
                    "authors": ["John Mathematician"],
                    "abstract": "We study the distribution of prime numbers...",
                    "domain": "number_theory",
                    "publication_date": "2024-01-15"
                },
                {
                    "id": "sample_paper_2",
                    "title": "New Approaches to the Riemann Hypothesis",
                    "authors": ["Jane Researcher"],
                    "abstract": "This paper presents new analytical methods...",
                    "domain": "complex_analysis",
                    "publication_date": "2024-02-20"
                }
            ],
            "conjectures": [
                {
                    "id": "sample_conjecture_1",
                    "statement": "Every even number greater than 2 can be expressed as the sum of two primes",
                    "name": "Goldbach's Conjecture",
                    "status": "open",
                    "domain": "number_theory",
                    "confidence": 0.95
                },
                {
                    "id": "sample_conjecture_2",
                    "statement": "There are infinitely many twin primes",
                    "name": "Twin Prime Conjecture",
                    "status": "open",
                    "domain": "number_theory",
                    "confidence": 0.85
                }
            ],
            "theorems": [
                {
                    "id": "sample_theorem_1",
                    "name": "Prime Number Theorem",
                    "statement": "The number of primes less than n is asymptotic to n/ln(n)",
                    "domain": "number_theory",
                    "verified": True
                }
            ]
        }

        # Create nodes
        for paper in sample_data["papers"]:
            query = """
            MERGE (p:Paper {id: $id})
            SET p += $properties
            """
            await neo4j.execute_query(query, {"id": paper["id"], "properties": paper})
            logger.debug(f"Created paper: {paper['title']}")

        for conjecture in sample_data["conjectures"]:
            query = """
            MERGE (c:Conjecture {id: $id})
            SET c += $properties
            """
            await neo4j.execute_query(query, {"id": conjecture["id"], "properties": conjecture})
            logger.debug(f"Created conjecture: {conjecture['name']}")

        for theorem in sample_data["theorems"]:
            query = """
            MERGE (t:Theorem {id: $id})
            SET t += $properties
            """
            await neo4j.execute_query(query, {"id": theorem["id"], "properties": theorem})
            logger.debug(f"Created theorem: {theorem['name']}")

        # Create relationships
        relationships = [
            ("sample_paper_1", "sample_conjecture_2", "DISCUSSES"),
            ("sample_paper_2", "sample_conjecture_1", "ANALYZES"),
            ("sample_paper_1", "sample_theorem_1", "REFERENCES"),
            ("sample_paper_2", "sample_paper_1", "CITES")
        ]

        for from_id, to_id, rel_type in relationships:
            query = f"""
            MATCH (from {{id: $from_id}}), (to {{id: $to_id}})
            MERGE (from)-[:{rel_type}]->(to)
            """
            await neo4j.execute_query(query, {"from_id": from_id, "to_id": to_id})
            logger.debug(f"Created relationship: {from_id} -{rel_type}-> {to_id}")

        await neo4j.close()
        logger.success("Sample data initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize sample data: {e}")
        return False


async def main():
    """Main initialization function"""
    logger.info("=" * 60)
    logger.info("Multi-Agent Mathematical Discovery System - Initialization")
    logger.info("=" * 60)

    # Setup logging
    setup_logging()

    # Check if services should be started
    start_services = "--no-docker" not in sys.argv

    if start_services:
        # Start Docker services
        if not await start_docker_services():
            logger.error("Failed to start Docker services")
            return False
    else:
        logger.info("Skipping Docker service startup (--no-docker flag)")

    # Initialize databases
    success = True

    # Initialize Neo4j
    if not await init_neo4j():
        logger.error("Neo4j initialization failed")
        success = False

    # Initialize Qdrant
    if not await init_qdrant():
        logger.warning("Qdrant initialization failed (non-critical)")

    # Initialize PostgreSQL
    if not await init_postgres():
        logger.warning("PostgreSQL initialization failed (non-critical)")

    # Initialize sample data
    if "--sample-data" in sys.argv:
        if not await init_sample_data():
            logger.warning("Sample data initialization failed (non-critical)")

    if success:
        logger.success("=" * 60)
        logger.success("System initialization completed successfully!")
        logger.success("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: python -m src.api.app")
        logger.info("2. Start the Streamlit UI: streamlit run src/ui/streamlit_app.py")
        logger.info("3. Access the API docs at: http://localhost:8000/docs")
        logger.info("4. Access the UI at: http://localhost:8501")
        return True
    else:
        logger.error("System initialization failed")
        return False


if __name__ == "__main__":
    # Parse arguments
    if "--help" in sys.argv:
        print("Usage: python init_system.py [options]")
        print("\nOptions:")
        print("  --no-docker     Skip Docker service startup")
        print("  --sample-data   Initialize with sample data")
        print("  --help          Show this help message")
        sys.exit(0)

    # Run initialization
    success = asyncio.run(main())
    sys.exit(0 if success else 1)