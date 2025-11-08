"""
Agent API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, WebSocket
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from loguru import logger

from src.agents.coordinator import AgentCoordinator
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.challenger import ChallengerAgent
from src.agents.historian import HistorianAgent
from src.api.websocket_manager import websocket_manager, websocket_endpoint


router = APIRouter()


class TaskRequest(BaseModel):
    """Request model for creating a new task"""
    task_type: str = Field(..., description="Type of task (research, review, synthesis, challenge)")
    description: str = Field(..., description="Task description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: str
    status: str
    created_at: datetime
    agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AgentStatus(BaseModel):
    """Agent status information"""
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    tasks_completed: int
    uptime: float


class WorkflowRequest(BaseModel):
    """Request to start a complete workflow"""
    papers: List[str] = Field(..., description="List of paper IDs or paths")
    workflow_type: str = Field(default="standard", description="Workflow type")
    options: Dict[str, Any] = Field(default_factory=dict, description="Workflow options")


# Agent instances (would be managed by a proper service in production)
agents = {}


def get_or_create_agent(agent_type: str, agent_id: Optional[str] = None) -> Any:
    """Get or create an agent instance"""
    if agent_id and agent_id in agents:
        return agents[agent_id]

    agent_map = {
        "coordinator": AgentCoordinator,
        "researcher": ResearcherAgent,
        "reviewer": ReviewerAgent,
        "synthesizer": SynthesizerAgent,
        "challenger": ChallengerAgent,
        "historian": HistorianAgent
    }

    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_class = agent_map[agent_type]
    agent = agent_class()

    if agent_id:
        agents[agent_id] = agent
    else:
        agents[agent.id] = agent

    return agent


@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Create a new task for an agent

    Args:
        request: Task request details
        background_tasks: FastAPI background tasks
        req: Request object for accessing app state

    Returns:
        Task response with ID and status
    """
    try:
        # Determine which agent should handle this task
        agent_type_map = {
            "research": "researcher",
            "review": "reviewer",
            "synthesis": "synthesizer",
            "challenge": "challenger",
            "history": "historian",
            "coordinate": "coordinator"
        }

        agent_type = agent_type_map.get(request.task_type, "coordinator")
        agent = get_or_create_agent(agent_type)

        # Create task ID
        task_id = f"task_{datetime.now().timestamp()}"

        # Broadcast task creation
        await websocket_manager.broadcast_task_update(
            task_id,
            "created",
            0,
            {
                "description": request.description,
                "agent_id": agent.id,
                "priority": request.priority
            }
        )

        # Run task in background
        async def run_task():
            try:
                # Update status
                await websocket_manager.broadcast_task_update(task_id, "running", 10)

                # Execute task
                result = await agent.process({
                    "task": request.description,
                    "context": request.context
                })

                # Update completion
                await websocket_manager.broadcast_task_update(
                    task_id,
                    "completed",
                    100,
                    {"result": result}
                )

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                await websocket_manager.broadcast_task_update(
                    task_id,
                    "failed",
                    0,
                    {"error": str(e)}
                )

        background_tasks.add_task(run_task)

        return TaskResponse(
            task_id=task_id,
            status="created",
            created_at=datetime.now(),
            agent_id=agent.id
        )

    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get task status

    Args:
        task_id: Task identifier

    Returns:
        Task status information
    """
    # In production, this would query a task database
    return TaskResponse(
        task_id=task_id,
        status="unknown",
        created_at=datetime.now()
    )


@router.get("/agents", response_model=List[AgentStatus])
async def list_agents():
    """
    List all active agents

    Returns:
        List of agent status information
    """
    agent_list = []

    for agent_id, agent in agents.items():
        agent_list.append(AgentStatus(
            agent_id=agent_id,
            agent_type=agent.__class__.__name__.replace("Agent", "").lower(),
            status="active" if hasattr(agent, "is_active") and agent.is_active else "idle",
            current_task=None,
            tasks_completed=0,
            uptime=0
        ))

    return agent_list


@router.get("/agents/{agent_id}", response_model=AgentStatus)
async def get_agent_status(agent_id: str):
    """
    Get specific agent status

    Args:
        agent_id: Agent identifier

    Returns:
        Agent status information
    """
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]

    return AgentStatus(
        agent_id=agent_id,
        agent_type=agent.__class__.__name__.replace("Agent", "").lower(),
        status="active",
        current_task=None,
        tasks_completed=0,
        uptime=0
    )


@router.post("/agents/{agent_id}/message")
async def send_message_to_agent(agent_id: str, message: Dict[str, Any]):
    """
    Send a message to a specific agent

    Args:
        agent_id: Agent identifier
        message: Message to send

    Returns:
        Response from agent
    """
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]

    try:
        # Process message
        response = await agent.process(message)

        # Broadcast agent message
        await websocket_manager.broadcast_agent_message(
            agent_id,
            str(response),
            {"type": "response"}
        )

        return {"agent_id": agent_id, "response": response}

    except Exception as e:
        logger.error(f"Error processing message for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/start", response_model=TaskResponse)
async def start_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Start a complete multi-agent workflow

    Args:
        request: Workflow request
        background_tasks: Background tasks
        req: Request object

    Returns:
        Workflow task response
    """
    try:
        # Create coordinator agent
        coordinator = get_or_create_agent("coordinator")

        # Create workflow task ID
        task_id = f"workflow_{datetime.now().timestamp()}"

        # Broadcast workflow start
        await websocket_manager.broadcast_task_update(
            task_id,
            "started",
            0,
            {
                "workflow_type": request.workflow_type,
                "papers_count": len(request.papers)
            }
        )

        # Run workflow in background
        async def run_workflow():
            try:
                # Execute workflow
                result = await coordinator.execute_workflow(
                    papers=request.papers,
                    options=request.options
                )

                # Broadcast completion
                await websocket_manager.broadcast_task_update(
                    task_id,
                    "completed",
                    100,
                    {"result": result}
                )

                # Broadcast discoveries
                if "discoveries" in result:
                    for discovery in result["discoveries"]:
                        await websocket_manager.broadcast_discovery(
                            discovery.get("type", "unknown"),
                            discovery
                        )

            except Exception as e:
                logger.error(f"Workflow {task_id} failed: {e}")
                await websocket_manager.broadcast_task_update(
                    task_id,
                    "failed",
                    0,
                    {"error": str(e)}
                )

        background_tasks.add_task(run_workflow)

        return TaskResponse(
            task_id=task_id,
            status="started",
            created_at=datetime.now(),
            agent_id=coordinator.id
        )

    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/reset")
async def reset_agents():
    """
    Reset all agents

    Returns:
        Reset confirmation
    """
    global agents
    agents = {}

    await websocket_manager.broadcast_agent_status(
        "system",
        "reset",
        {"message": "All agents have been reset"}
    )

    return {"status": "success", "message": "All agents reset"}


# WebSocket endpoint
@router.websocket("/ws/{client_id}")
async def websocket_route(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time communication

    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await websocket_endpoint(websocket, client_id)


# Agent communication endpoints
@router.post("/agents/broadcast")
async def broadcast_to_agents(message: Dict[str, Any]):
    """
    Broadcast message to all agents

    Args:
        message: Message to broadcast

    Returns:
        Broadcast confirmation
    """
    responses = {}

    for agent_id, agent in agents.items():
        try:
            response = await agent.process(message)
            responses[agent_id] = response
        except Exception as e:
            responses[agent_id] = {"error": str(e)}

    return {
        "status": "broadcast_complete",
        "recipients": len(agents),
        "responses": responses
    }


@router.get("/agents/{agent_id}/memory")
async def get_agent_memory(agent_id: str, memory_type: str = "short_term"):
    """
    Get agent memory contents

    Args:
        agent_id: Agent identifier
        memory_type: Type of memory (short_term, long_term, working)

    Returns:
        Memory contents
    """
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents[agent_id]

    if memory_type == "short_term":
        memory = agent.short_term_memory
    elif memory_type == "long_term":
        memory = agent.long_term_memory
    elif memory_type == "working":
        memory = agent.working_memory
    else:
        raise HTTPException(status_code=400, detail="Invalid memory type")

    return {
        "agent_id": agent_id,
        "memory_type": memory_type,
        "contents": memory.retrieve_all() if hasattr(memory, "retrieve_all") else []
    }


@router.delete("/agents/{agent_id}")
async def remove_agent(agent_id: str):
    """
    Remove an agent

    Args:
        agent_id: Agent identifier

    Returns:
        Removal confirmation
    """
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    del agents[agent_id]

    await websocket_manager.broadcast_agent_status(
        agent_id,
        "removed",
        {"message": f"Agent {agent_id} has been removed"}
    )

    return {"status": "success", "message": f"Agent {agent_id} removed"}