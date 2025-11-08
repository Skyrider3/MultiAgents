"""
Agent Coordinator - Orchestrates multi-agent collaboration for mathematical discovery
"""

import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, AgentRole, AgentStatus
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.challenger import ChallengerAgent
from src.agents.historian import HistorianAgent
from src.communication.protocols.a2a_protocol import (
    MessageBroker,
    AgentMessage,
    MessageType,
    MessagePriority,
    ConversationTrace
)
from src.llm.bedrock_client import MultiModelBedrockClient


class WorkflowStage(str, Enum):
    """Stages in the discovery workflow"""
    INGESTION = "ingestion"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    CHALLENGE = "challenge"
    CONSENSUS = "consensus"
    DOCUMENTATION = "documentation"
    COMPLETE = "complete"


class CollaborationPattern(str, Enum):
    """Patterns of agent collaboration"""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"  # Agents work simultaneously
    DEBATE = "debate"  # Agents engage in discussion
    CONSENSUS = "consensus"  # Agents work toward agreement
    ADVERSARIAL = "adversarial"  # Challenger vs others


class Task(BaseModel):
    """Represents a task for agents"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    description: str
    assigned_agents: List[str] = Field(default_factory=list)
    priority: int = Field(ge=0, le=10, default=5)
    status: str = "pending"
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # Task IDs this depends on


class WorkflowState(BaseModel):
    """Current state of the discovery workflow"""
    workflow_id: str
    current_stage: WorkflowStage
    completed_stages: List[WorkflowStage]
    active_tasks: List[Task]
    completed_tasks: List[Task]
    discovered_insights: List[Dict[str, Any]]
    consensus_reached: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)


class AgentCoordinator:
    """
    Coordinates multiple specialized agents for mathematical conjecture discovery
    """

    def __init__(
        self,
        bedrock_client: MultiModelBedrockClient = None,
        message_broker: MessageBroker = None,
        max_parallel_agents: int = 5
    ):
        self.coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        self.bedrock_client = bedrock_client or MultiModelBedrockClient()
        self.message_broker = message_broker or MessageBroker()

        # Initialize specialized agents
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self._initialize_agents()

        # Workflow management
        self.workflow_state: Optional[WorkflowState] = None
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.max_parallel_agents = max_parallel_agents
        self.active_workflows: Dict[str, WorkflowState] = {}

        # Collaboration tracking
        self.collaboration_history: List[Dict[str, Any]] = []
        self.consensus_records: List[Dict[str, Any]] = []

        # Performance metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_discoveries": 0,
            "failed_workflows": 0,
            "average_completion_time": 0.0,
            "agent_utilization": {},
            "insights_generated": 0
        }

        logger.info(f"Coordinator {self.coordinator_id} initialized with {len(self.agents)} agents")

    def _initialize_agents(self):
        """Initialize all specialized agents"""
        # Create agents with shared resources
        common_args = {
            "bedrock_client": self.bedrock_client,
            "message_broker": self.message_broker
        }

        self.agents[AgentRole.RESEARCHER] = ResearcherAgent(
            role=AgentRole.RESEARCHER,
            **common_args
        )

        self.agents[AgentRole.REVIEWER] = ReviewerAgent(
            role=AgentRole.REVIEWER,
            **common_args
        )

        self.agents[AgentRole.SYNTHESIZER] = SynthesizerAgent(
            role=AgentRole.SYNTHESIZER,
            **common_args
        )

        self.agents[AgentRole.CHALLENGER] = ChallengerAgent(
            role=AgentRole.CHALLENGER,
            **common_args
        )

        self.agents[AgentRole.HISTORIAN] = HistorianAgent(
            role=AgentRole.HISTORIAN,
            **common_args
        )

        # Register agents with message broker
        asyncio.create_task(self._register_all_agents())

    async def _register_all_agents(self):
        """Register all agents with the message broker"""
        for role, agent in self.agents.items():
            await self.message_broker.register_agent(agent.agent_id, agent.connection)
            logger.info(f"Registered {role.value} agent: {agent.agent_id}")

    async def start_discovery_workflow(
        self,
        papers: List[Dict[str, Any]],
        domain: str,
        goals: List[str] = None
    ) -> WorkflowState:
        """
        Start a complete discovery workflow with multiple papers
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting discovery workflow {workflow_id} with {len(papers)} papers")

        # Initialize workflow state
        self.workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_stage=WorkflowStage.INGESTION,
            completed_stages=[],
            active_tasks=[],
            completed_tasks=[],
            discovered_insights=[]
        )

        self.active_workflows[workflow_id] = self.workflow_state
        self.metrics["total_workflows"] += 1

        start_time = time.time()

        try:
            # Stage 1: Ingestion - Researcher analyzes papers
            await self._execute_stage_ingestion(papers, domain)

            # Stage 2: Analysis - Deep analysis of extracted knowledge
            await self._execute_stage_analysis()

            # Stage 3: Validation - Reviewer validates claims
            await self._execute_stage_validation()

            # Stage 4: Synthesis - Synthesizer finds patterns
            await self._execute_stage_synthesis()

            # Stage 5: Challenge - Challenger tests conjectures
            await self._execute_stage_challenge()

            # Stage 6: Consensus - Reach agreement
            await self._execute_stage_consensus()

            # Stage 7: Documentation - Historian documents progress
            await self._execute_stage_documentation()

            # Mark workflow complete
            self.workflow_state.current_stage = WorkflowStage.COMPLETE
            self.metrics["successful_discoveries"] += 1

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            self.metrics["failed_workflows"] += 1
            raise

        finally:
            # Update metrics
            completion_time = time.time() - start_time
            self._update_completion_metrics(completion_time)

        return self.workflow_state

    async def _execute_stage_ingestion(self, papers: List[Dict[str, Any]], domain: str):
        """Execute the ingestion stage"""
        logger.info("Executing INGESTION stage")
        self.workflow_state.current_stage = WorkflowStage.INGESTION

        researcher = self.agents[AgentRole.RESEARCHER]
        tasks = []

        # Create tasks for each paper
        for paper in papers:
            task = Task(
                task_type="analyze_paper",
                description=f"Analyze paper: {paper.get('title', 'Unknown')}",
                assigned_agents=[researcher.agent_id],
                priority=8
            )
            tasks.append(task)
            self.workflow_state.active_tasks.append(task)

        # Execute in parallel with limit
        results = await self._execute_parallel_tasks(tasks, researcher)

        # Move tasks to completed
        for task in tasks:
            task.status = "completed"
            task.completed_at = time.time()
            self.workflow_state.active_tasks.remove(task)
            self.workflow_state.completed_tasks.append(task)

        self.workflow_state.completed_stages.append(WorkflowStage.INGESTION)

    async def _execute_stage_analysis(self):
        """Execute the analysis stage"""
        logger.info("Executing ANALYSIS stage")
        self.workflow_state.current_stage = WorkflowStage.ANALYSIS

        researcher = self.agents[AgentRole.RESEARCHER]

        # Identify patterns across analyzed papers
        task = Task(
            task_type="identify_patterns",
            description="Identify patterns across all papers",
            assigned_agents=[researcher.agent_id],
            priority=7
        )

        self.workflow_state.active_tasks.append(task)

        # Execute task
        paper_ids = [t.task_id for t in self.workflow_state.completed_tasks
                    if t.task_type == "analyze_paper"]

        result = await researcher.process_task({
            "task_type": "identify_patterns",
            "papers": paper_ids
        })

        task.results = result
        task.status = "completed"
        task.completed_at = time.time()

        self.workflow_state.active_tasks.remove(task)
        self.workflow_state.completed_tasks.append(task)
        self.workflow_state.completed_stages.append(WorkflowStage.ANALYSIS)

    async def _execute_stage_validation(self):
        """Execute the validation stage"""
        logger.info("Executing VALIDATION stage")
        self.workflow_state.current_stage = WorkflowStage.VALIDATION

        reviewer = self.agents[AgentRole.REVIEWER]

        # Get claims to validate from previous stages
        claims_to_validate = self._extract_claims_from_results()

        tasks = []
        for claim in claims_to_validate[:5]:  # Limit to top 5 claims
            task = Task(
                task_type="verify_claim",
                description=f"Verify: {claim[:100]}",
                assigned_agents=[reviewer.agent_id],
                priority=9
            )
            tasks.append(task)
            self.workflow_state.active_tasks.append(task)

        # Execute validation tasks
        results = await self._execute_parallel_tasks(tasks, reviewer)

        # Update workflow state
        for task in tasks:
            task.status = "completed"
            task.completed_at = time.time()
            self.workflow_state.active_tasks.remove(task)
            self.workflow_state.completed_tasks.append(task)

        self.workflow_state.completed_stages.append(WorkflowStage.VALIDATION)

    async def _execute_stage_synthesis(self):
        """Execute the synthesis stage"""
        logger.info("Executing SYNTHESIS stage")
        self.workflow_state.current_stage = WorkflowStage.SYNTHESIS

        synthesizer = self.agents[AgentRole.SYNTHESIZER]

        # Synthesize insights from validated claims
        validated_data = self._get_validated_data()

        task = Task(
            task_type="synthesize_insights",
            description="Synthesize insights and generate conjectures",
            assigned_agents=[synthesizer.agent_id],
            priority=8
        )

        self.workflow_state.active_tasks.append(task)

        result = await synthesizer.synthesize({
            "data": validated_data,
            "domain": "mathematical_discovery"
        })

        # Store discovered insights
        if result:
            self.workflow_state.discovered_insights.append(result.dict())
            self.metrics["insights_generated"] += 1

        task.results = result.dict() if result else {}
        task.status = "completed"
        task.completed_at = time.time()

        self.workflow_state.active_tasks.remove(task)
        self.workflow_state.completed_tasks.append(task)
        self.workflow_state.completed_stages.append(WorkflowStage.SYNTHESIS)

    async def _execute_stage_challenge(self):
        """Execute the challenge stage"""
        logger.info("Executing CHALLENGE stage")
        self.workflow_state.current_stage = WorkflowStage.CHALLENGE

        challenger = self.agents[AgentRole.CHALLENGER]

        # Challenge generated conjectures
        conjectures_to_challenge = self._get_conjectures()

        tasks = []
        for conjecture in conjectures_to_challenge[:3]:  # Limit to top 3
            task = Task(
                task_type="challenge_conjecture",
                description=f"Challenge: {conjecture[:100]}",
                assigned_agents=[challenger.agent_id],
                priority=7
            )
            tasks.append(task)
            self.workflow_state.active_tasks.append(task)

        # Execute challenges
        results = await self._execute_parallel_tasks(tasks, challenger)

        for task in tasks:
            task.status = "completed"
            task.completed_at = time.time()
            self.workflow_state.active_tasks.remove(task)
            self.workflow_state.completed_tasks.append(task)

        self.workflow_state.completed_stages.append(WorkflowStage.CHALLENGE)

    async def _execute_stage_consensus(self):
        """Execute the consensus stage"""
        logger.info("Executing CONSENSUS stage")
        self.workflow_state.current_stage = WorkflowStage.CONSENSUS

        # Orchestrate debate between agents
        debate_result = await self._orchestrate_debate()

        # Reach consensus
        consensus = await self._reach_consensus(debate_result)

        if consensus["reached"]:
            self.workflow_state.consensus_reached = True
            self.workflow_state.confidence_score = consensus["confidence"]
            self.consensus_records.append(consensus)

        self.workflow_state.completed_stages.append(WorkflowStage.CONSENSUS)

    async def _execute_stage_documentation(self):
        """Execute the documentation stage"""
        logger.info("Executing DOCUMENTATION stage")
        self.workflow_state.current_stage = WorkflowStage.DOCUMENTATION

        historian = self.agents[AgentRole.HISTORIAN]

        # Document the entire discovery process
        task = Task(
            task_type="document_discovery",
            description="Document the discovery process and outcomes",
            assigned_agents=[historian.agent_id],
            priority=6
        )

        self.workflow_state.active_tasks.append(task)

        progress_data = {
            "development": "Mathematical discovery workflow",
            "insights": self.workflow_state.discovered_insights,
            "consensus": self.workflow_state.consensus_reached,
            "confidence": self.workflow_state.confidence_score
        }

        result = await historian.document_progress(progress_data)

        task.results = result.dict() if result else {}
        task.status = "completed"
        task.completed_at = time.time()

        self.workflow_state.active_tasks.remove(task)
        self.workflow_state.completed_tasks.append(task)
        self.workflow_state.completed_stages.append(WorkflowStage.DOCUMENTATION)

    async def _execute_parallel_tasks(
        self,
        tasks: List[Task],
        agent: BaseAgent
    ) -> List[Dict[str, Any]]:
        """Execute tasks in parallel with concurrency limit"""
        semaphore = asyncio.Semaphore(self.max_parallel_agents)
        results = []

        async def process_task(task: Task):
            async with semaphore:
                task.started_at = time.time()
                task.status = "in_progress"

                result = await agent.process_task({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "description": task.description
                })

                task.results = result
                return result

        # Execute all tasks concurrently
        results = await asyncio.gather(*[process_task(task) for task in tasks])
        return results

    async def _orchestrate_debate(self) -> Dict[str, Any]:
        """Orchestrate a debate between agents"""
        logger.info("Orchestrating agent debate")

        debate_topic = "Validity of generated conjectures"
        debate_messages = []

        # Synthesizer presents conjectures
        synthesizer_msg = AgentMessage(
            from_agent=self.agents[AgentRole.SYNTHESIZER].agent_id,
            type=MessageType.HYPOTHESIS,
            content={
                "statement": "Proposed conjectures based on pattern analysis",
                "conjectures": self._get_conjectures()[:2],
                "confidence": 0.7
            },
            trace_id=f"debate_{self.workflow_state.workflow_id}"
        )
        await self.message_broker.route_message(synthesizer_msg)
        debate_messages.append(synthesizer_msg)

        # Reviewer validates
        reviewer_msg = AgentMessage(
            from_agent=self.agents[AgentRole.REVIEWER].agent_id,
            type=MessageType.VALIDATION,
            content={
                "validation": "Logical consistency verified",
                "issues": ["Needs boundary case verification"],
                "confidence": 0.6
            },
            parent_message_id=synthesizer_msg.id,
            trace_id=f"debate_{self.workflow_state.workflow_id}"
        )
        await self.message_broker.route_message(reviewer_msg)
        debate_messages.append(reviewer_msg)

        # Challenger attacks
        challenger_msg = AgentMessage(
            from_agent=self.agents[AgentRole.CHALLENGER].agent_id,
            type=MessageType.CHALLENGE,
            content={
                "challenge": "Counter-example exists for special case",
                "counter_example": "n=2 fails the conjecture",
                "severity": "major"
            },
            parent_message_id=synthesizer_msg.id,
            trace_id=f"debate_{self.workflow_state.workflow_id}"
        )
        await self.message_broker.route_message(challenger_msg)
        debate_messages.append(challenger_msg)

        # Store collaboration
        self.collaboration_history.append({
            "type": "debate",
            "topic": debate_topic,
            "messages": [msg.dict() for msg in debate_messages],
            "timestamp": datetime.utcnow().isoformat()
        })

        return {
            "debate_messages": debate_messages,
            "participants": [msg.from_agent for msg in debate_messages]
        }

    async def _reach_consensus(self, debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reach consensus among agents"""
        logger.info("Attempting to reach consensus")

        # Analyze debate messages for agreement
        messages = debate_result.get("debate_messages", [])

        # Simple consensus: if challenger's issues are minor
        challenger_msgs = [m for m in messages if m.type == MessageType.CHALLENGE]

        consensus_reached = True
        confidence = 0.7

        for msg in challenger_msgs:
            if msg.content.get("severity") == "critical":
                consensus_reached = False
                confidence = 0.3
                break
            elif msg.content.get("severity") == "major":
                confidence -= 0.2

        consensus = {
            "reached": consensus_reached,
            "confidence": max(0.0, confidence),
            "summary": "Consensus reached with minor caveats" if consensus_reached else "No consensus - critical issues",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Broadcast consensus
        consensus_msg = AgentMessage(
            from_agent=self.coordinator_id,
            type=MessageType.CONSENSUS,
            content=consensus,
            trace_id=f"debate_{self.workflow_state.workflow_id}"
        )
        await self.message_broker.route_message(consensus_msg)

        return consensus

    def _extract_claims_from_results(self) -> List[str]:
        """Extract claims from completed tasks"""
        claims = []
        for task in self.workflow_state.completed_tasks:
            if task.task_type == "analyze_paper":
                results = task.results.get("result", {}).get("results", {})
                theorems = results.get("theorems", [])
                conjectures = results.get("conjectures", [])

                for theorem in theorems[:2]:
                    claims.append(theorem.get("statement", ""))
                for conjecture in conjectures[:2]:
                    claims.append(conjecture.get("statement", ""))

        return [c for c in claims if c]

    def _get_validated_data(self) -> List[Dict[str, Any]]:
        """Get validated data from validation stage"""
        validated = []
        for task in self.workflow_state.completed_tasks:
            if task.task_type == "verify_claim":
                validation = task.results.get("validation", {})
                if validation.get("status") == "valid":
                    validated.append(validation)
        return validated

    def _get_conjectures(self) -> List[str]:
        """Get generated conjectures"""
        conjectures = []
        for insight in self.workflow_state.discovered_insights:
            if "conjecture" in str(insight).lower():
                conjectures.append(str(insight))

        # Also get from synthesis tasks
        for task in self.workflow_state.completed_tasks:
            if task.task_type == "synthesize_insights":
                result = task.results.get("statement", "")
                if result:
                    conjectures.append(result)

        return conjectures

    def _update_completion_metrics(self, completion_time: float):
        """Update completion metrics"""
        n = self.metrics["total_workflows"]
        old_avg = self.metrics["average_completion_time"]
        self.metrics["average_completion_time"] = (old_avg * (n-1) + completion_time) / n

        # Update agent utilization
        for role, agent in self.agents.items():
            if role.value not in self.metrics["agent_utilization"]:
                self.metrics["agent_utilization"][role.value] = 0
            self.metrics["agent_utilization"][role.value] = agent.metrics.tasks_completed

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get status of a workflow"""
        return self.active_workflows.get(workflow_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics"""
        metrics = self.metrics.copy()

        # Add agent-specific metrics
        metrics["agent_metrics"] = {}
        for role, agent in self.agents.items():
            metrics["agent_metrics"][role.value] = agent.get_status()

        return metrics

    async def shutdown(self):
        """Gracefully shutdown coordinator and all agents"""
        logger.info(f"Shutting down coordinator {self.coordinator_id}")

        # Shutdown all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            shutdown_tasks.append(agent.shutdown())

        await asyncio.gather(*shutdown_tasks)

        logger.info("All agents shut down successfully")


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize coordinator
        coordinator = AgentCoordinator()

        # Example papers
        papers = [
            {
                "id": "paper1",
                "title": "On the Distribution of Prime Numbers",
                "content": "Abstract mathematical content...",
                "citations": ["ref1", "ref2"]
            },
            {
                "id": "paper2",
                "title": "New Approaches to the Riemann Hypothesis",
                "content": "More mathematical content...",
                "citations": ["ref3", "ref4"]
            }
        ]

        # Start discovery workflow
        workflow_state = await coordinator.start_discovery_workflow(
            papers=papers,
            domain="number_theory",
            goals=["Find patterns in prime distribution", "Generate new conjectures"]
        )

        # Check results
        print(f"Workflow completed: {workflow_state.workflow_id}")
        print(f"Consensus reached: {workflow_state.consensus_reached}")
        print(f"Confidence: {workflow_state.confidence_score}")
        print(f"Insights discovered: {len(workflow_state.discovered_insights)}")

        # Get metrics
        metrics = coordinator.get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        # Shutdown
        await coordinator.shutdown()

    # Run example
    asyncio.run(example())