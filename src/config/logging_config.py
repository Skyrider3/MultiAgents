"""
Logging Configuration using Loguru
"""

import sys
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.config.settings import settings, LogLevel


def serialize_record(record: dict) -> str:
    """Serialize log record to JSON"""
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra fields if present
    if record.get("extra"):
        subset["extra"] = record["extra"]

    # Add exception info if present
    if record.get("exception"):
        subset["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback,
        }

    return json.dumps(subset, default=str) + "\n"


def setup_logging():
    """Configure logging for the application"""

    # Remove default logger
    logger.remove()

    # Console logging - always use human-readable format for better developer experience
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=settings.monitoring.log_level.value,
        enqueue=True,
        colorize=True,
        backtrace=False,
        diagnose=False
    )

    # File logging - always JSON
    log_file = settings.logs_dir / f"multiagents_{datetime.now():%Y%m%d}.log"
    logger.add(
        log_file,
        serialize=True,
        format="{message}",
        level=settings.monitoring.log_level.value,
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=settings.debug
    )

    # Error logging to separate file
    error_log_file = settings.logs_dir / f"errors_{datetime.now():%Y%m%d}.log"
    logger.add(
        error_log_file,
        serialize=True,
        format="{message}",
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    # Agent-specific logging
    agent_log_file = settings.logs_dir / f"agents_{datetime.now():%Y%m%d}.log"
    logger.add(
        agent_log_file,
        serialize=True,
        format="{message}",
        level="DEBUG" if settings.agents.enable_agent_tracing else "INFO",
        filter=lambda record: "agent" in record["name"].lower(),
        rotation="100 MB",
        retention="14 days",
        compression="gz",
        enqueue=True
    )

    # Performance logging
    if settings.monitoring.enable_metrics:
        perf_log_file = settings.logs_dir / f"performance_{datetime.now():%Y%m%d}.log"
        logger.add(
            perf_log_file,
            serialize=True,
            format="{message}",
            level="INFO",
            filter=lambda record: "performance" in record.get("extra", {}).get("type", ""),
            rotation="50 MB",
            retention="7 days",
            compression="gz",
            enqueue=True
        )

    logger.info(f"Logging configured for {settings.environment} environment")
    logger.info(f"Log level: {settings.monitoring.log_level}")
    logger.info(f"Log directory: {settings.logs_dir}")


# Custom logger instances for different components
def get_agent_logger(agent_id: str):
    """Get logger for specific agent"""
    return logger.bind(agent_id=agent_id, component="agent")


def get_graph_logger():
    """Get logger for graph operations"""
    return logger.bind(component="graph")


def get_api_logger():
    """Get logger for API"""
    return logger.bind(component="api")


def get_ingestion_logger():
    """Get logger for document ingestion"""
    return logger.bind(component="ingestion")


# Performance logging helpers
def log_performance(operation: str, duration: float, metadata: dict = None):
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation}",
        extra={
            "type": "performance",
            "operation": operation,
            "duration_ms": duration * 1000,
            "metadata": metadata or {}
        }
    )


def log_agent_action(agent_id: str, action: str, details: dict = None):
    """Log agent actions"""
    logger.info(
        f"Agent action: {action}",
        extra={
            "type": "agent_action",
            "agent_id": agent_id,
            "action": action,
            "details": details or {}
        }
    )


def log_graph_operation(operation: str, node_type: str = None, relationship_type: str = None, count: int = None):
    """Log graph database operations"""
    logger.debug(
        f"Graph operation: {operation}",
        extra={
            "type": "graph_operation",
            "operation": operation,
            "node_type": node_type,
            "relationship_type": relationship_type,
            "count": count
        }
    )


# Context managers for timing operations
from contextlib import contextmanager
import time


@contextmanager
def log_timing(operation: str, logger_instance=logger):
    """Context manager for timing operations"""
    start_time = time.time()
    logger_instance.debug(f"Starting: {operation}")

    try:
        yield
    finally:
        duration = time.time() - start_time
        logger_instance.info(
            f"Completed: {operation}",
            extra={
                "type": "timing",
                "operation": operation,
                "duration_seconds": duration
            }
        )


@contextmanager
def log_agent_task(agent_id: str, task_type: str):
    """Context manager for agent tasks"""
    agent_logger = get_agent_logger(agent_id)
    start_time = time.time()
    agent_logger.info(f"Starting task: {task_type}")

    try:
        yield agent_logger
    except Exception as e:
        agent_logger.error(f"Task failed: {task_type}", exception=e)
        raise
    finally:
        duration = time.time() - start_time
        agent_logger.info(
            f"Task completed: {task_type}",
            extra={
                "duration_seconds": duration,
                "task_type": task_type
            }
        )


# Initialize logging on import
setup_logging()