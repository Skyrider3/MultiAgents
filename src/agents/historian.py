"""
Historian Agent - Specialized in tracking evolution of mathematical concepts over time
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base import (
    BaseAgent,
    AgentRole,
    AgentCapabilities,
    AgentPersonality,
    Tool,
    ToolParameter,
    Thought,
    TaskType
)
from src.llm.bedrock_client import BedrockMessage


class HistoricalEvent(BaseModel):
    """Represents a historical mathematical event"""
    event_id: str
    timestamp: str
    event_type: str  # discovery, proof, refutation, generalization, application
    description: str
    actors: List[str]  # Mathematicians or agents involved
    impact_score: float = Field(ge=0.0, le=1.0)
    related_concepts: List[str]
    sources: List[str]


class ConceptEvolution(BaseModel):
    """Tracks the evolution of a mathematical concept"""
    concept_name: str
    origin_date: Optional[str]
    origin_context: str
    evolution_stages: List[Dict[str, Any]]
    current_understanding: str
    key_contributors: List[str]
    major_breakthroughs: List[str]
    open_questions: List[str]


class Timeline(BaseModel):
    """Represents a timeline of mathematical developments"""
    timeline_id: str
    domain: str
    start_date: str
    end_date: str
    events: List[HistoricalEvent]
    key_periods: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]


class DevelopmentPattern(BaseModel):
    """Patterns in mathematical development"""
    pattern_id: str
    pattern_type: str  # cyclic, progressive, branching, convergent
    description: str
    examples: List[Dict[str, Any]]
    predictive_value: float = Field(ge=0.0, le=1.0)
    recurring_interval: Optional[str] = None


class ImpactAnalysis(BaseModel):
    """Analysis of a development's impact"""
    development: str
    immediate_impact: List[str]
    long_term_impact: List[str]
    influenced_fields: List[str]
    citation_growth: Dict[str, int]
    adoption_rate: float = Field(ge=0.0, le=1.0)


class HistorianAgent(BaseAgent):
    """
    Historian Agent specialized in:
    - Tracking evolution of mathematical concepts
    - Documenting discovery timelines
    - Analyzing historical patterns
    - Predicting future developments
    - Maintaining institutional memory
    """

    def _initialize(self, **kwargs):
        """Initialize Historian-specific attributes"""
        self.historical_events: List[HistoricalEvent] = []
        self.concept_evolutions: Dict[str, ConceptEvolution] = {}
        self.timelines: Dict[str, Timeline] = {}
        self.development_patterns: List[DevelopmentPattern] = []
        self.impact_analyses: List[ImpactAnalysis] = []
        self.knowledge_genealogy: Dict[str, List[str]] = {}  # Concept -> Prerequisites

        # Specialized capabilities for historian
        self.capabilities = AgentCapabilities(
            can_reason=True,
            can_learn=True,
            can_collaborate=True,
            can_synthesize=True,
            supported_domains=[
                "history_of_mathematics",
                "concept_evolution",
                "trend_analysis",
                "impact_assessment",
                "predictive_modeling"
            ],
            max_context_length=150000,
            parallel_tasks=3
        )

        # Historian personality - methodical and analytical
        self.personality = AgentPersonality(
            curiosity=0.8,
            skepticism=0.5,
            creativity=0.4,
            thoroughness=0.95,  # Very thorough in documentation
            risk_tolerance=0.2  # Conservative in predictions
        )

    def _register_custom_tools(self):
        """Register historian-specific tools"""
        historian_tools = [
            Tool(
                name="track_concept_evolution",
                description="Track how a concept evolved over time",
                parameters=[
                    ToolParameter(name="concept", type="str", description="Mathematical concept"),
                    ToolParameter(name="time_range", type="dict", description="Time range to analyze")
                ],
                handler=self._tool_track_concept_evolution
            ),
            Tool(
                name="create_timeline",
                description="Create a timeline of developments",
                parameters=[
                    ToolParameter(name="domain", type="str", description="Mathematical domain"),
                    ToolParameter(name="events", type="list", description="List of events")
                ],
                handler=self._tool_create_timeline
            ),
            Tool(
                name="analyze_impact",
                description="Analyze the impact of a development",
                parameters=[
                    ToolParameter(name="development", type="str", description="Development to analyze"),
                    ToolParameter(name="metrics", type="list", description="Impact metrics to use")
                ],
                handler=self._tool_analyze_impact
            ),
            Tool(
                name="find_historical_patterns",
                description="Find patterns in mathematical development",
                parameters=[
                    ToolParameter(name="data", type="list", description="Historical data"),
                    ToolParameter(name="pattern_type", type="str", description="Type of pattern to find")
                ],
                handler=self._tool_find_historical_patterns
            ),
            Tool(
                name="predict_next_breakthrough",
                description="Predict likely next breakthroughs",
                parameters=[
                    ToolParameter(name="domain", type="str", description="Mathematical domain"),
                    ToolParameter(name="current_state", type="dict", description="Current state of field")
                ],
                handler=self._tool_predict_next_breakthrough
            ),
            Tool(
                name="document_discovery",
                description="Document a new discovery",
                parameters=[
                    ToolParameter(name="discovery", type="dict", description="Discovery details"),
                    ToolParameter(name="context", type="dict", description="Discovery context")
                ],
                handler=self._tool_document_discovery
            ),
            Tool(
                name="trace_influence_network",
                description="Trace influence networks between concepts",
                parameters=[
                    ToolParameter(name="root_concept", type="str", description="Starting concept"),
                    ToolParameter(name="depth", type="int", description="Network depth", default=3)
                ],
                handler=self._tool_trace_influence_network
            )
        ]
        self.tools.extend(historian_tools)

    async def think(self, context: Dict[str, Any]) -> Thought:
        """
        Historian's thinking process - analytical and temporal
        """
        task_type = context.get("task_type", "track_evolution")

        if task_type == "track_evolution":
            concept = context.get("concept", "")
            history = context.get("history", [])

            prompt = f"""As a mathematical historian, trace the evolution of this concept:

Concept: {concept}

Historical Data:
{json.dumps(history[:20], indent=2)}

Analyze:
1. Origin and initial formulation
2. Key developmental stages
3. Major breakthroughs and contributors
4. Paradigm shifts or reformulations
5. Current state of understanding
6. Predicted future developments
7. Connections to other mathematical concepts

Provide a comprehensive historical analysis with temporal context.
"""

        elif task_type == "analyze_timeline":
            events = context.get("events", [])
            domain = context.get("domain", "general")

            prompt = f"""Analyze this timeline of mathematical developments:

Domain: {domain}
Events: {json.dumps(events[:30], indent=2)}

Identify:
1. Key periods of rapid development
2. Stagnation periods and their causes
3. Breakthrough moments
4. Influence of external factors (wars, technology, etc.)
5. Patterns in discovery timing
6. Collaboration networks over time

Create a narrative of mathematical progress.
"""

        elif task_type == "predict_development":
            current_state = context.get("current_state", {})
            trends = context.get("trends", [])

            prompt = f"""Based on historical patterns, predict future developments:

Current State:
{json.dumps(current_state, indent=2)}

Recent Trends:
{json.dumps(trends[:10], indent=2)}

Consider:
1. Historical patterns of similar developments
2. Current rate of progress
3. Available tools and techniques
4. Open problems likely to be solved
5. Potential paradigm shifts
6. Time estimates for major breakthroughs

Make data-driven predictions based on historical precedent.
"""

        elif task_type == "document_impact":
            development = context.get("development", "")
            initial_context = context.get("initial_context", {})

            prompt = f"""Document and analyze the impact of this development:

Development: {development}
Initial Context: {json.dumps(initial_context, indent=2)}

Assess:
1. Immediate impact on the field
2. Long-term ramifications
3. Influence on other mathematical areas
4. Practical applications that emerged
5. New research directions opened
6. Citation and adoption metrics

Provide a comprehensive impact assessment.
"""
        else:
            prompt = f"Analyze historical context of: {json.dumps(context)}"

        # Get LLM response with historical analysis
        messages = [
            BedrockMessage(role="user", content=prompt)
        ]

        response = await self.bedrock_client.generate(
            messages=messages,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.3,  # Lower temperature for factual historical analysis
            max_tokens=4096
        )

        # Create analytical thought
        confidence = self._calculate_historical_confidence(response.content, context)

        thought = Thought(
            content=response.content,
            confidence=confidence,
            reasoning_type="temporal",  # Historians use temporal reasoning
            evidence=self._extract_historical_evidence(response.content)
        )

        return thought

    async def act(self, thought: Thought) -> Dict[str, Any]:
        """
        Execute historical documentation and analysis actions
        """
        actions_taken = []
        results = {}

        # Parse thought for historical insights
        if "evolution" in thought.content.lower() or "evolved" in thought.content.lower():
            evolution = await self._extract_evolution_data(thought)
            if evolution:
                actions_taken.append("tracked_evolution")
                results["evolution"] = evolution.dict()
                self.concept_evolutions[evolution.concept_name] = evolution

        if "timeline" in thought.content.lower() or "chronolog" in thought.content.lower():
            timeline = await self._create_timeline_from_thought(thought)
            if timeline:
                actions_taken.append("created_timeline")
                results["timeline"] = timeline.dict()
                self.timelines[timeline.domain] = timeline

        if "pattern" in thought.content.lower():
            patterns = await self._extract_patterns_from_thought(thought)
            if patterns:
                actions_taken.append("identified_patterns")
                results["patterns"] = [p.dict() for p in patterns]
                self.development_patterns.extend(patterns)

        if "impact" in thought.content.lower() or "influence" in thought.content.lower():
            impact = await self._analyze_impact_from_thought(thought)
            if impact:
                actions_taken.append("analyzed_impact")
                results["impact"] = impact.dict()
                self.impact_analyses.append(impact)

        # Document as historical event
        event = self._create_historical_event(thought)
        self.historical_events.append(event)

        # Store in long-term memory
        self.memory.long_term["historical_record"] = self.memory.long_term.get("historical_record", [])
        self.memory.long_term["historical_record"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "thought": thought.content,
            "results": results
        })

        return {
            "actions_taken": actions_taken,
            "results": results,
            "confidence": thought.confidence,
            "historical_significance": self._assess_historical_significance(results)
        }

    async def _tool_track_concept_evolution(
        self,
        concept: str,
        time_range: Dict[str, Any]
    ) -> ConceptEvolution:
        """Track how a concept evolved over time"""

        evolution = ConceptEvolution(
            concept_name=concept,
            origin_date=time_range.get("start", "ancient"),
            origin_context=f"Early formulation of {concept}",
            evolution_stages=[],
            current_understanding=f"Modern understanding of {concept}",
            key_contributors=[],
            major_breakthroughs=[],
            open_questions=[]
        )

        # Add evolution stages based on historical patterns
        stages = [
            {"period": "initial", "description": "First formalization", "year": "early"},
            {"period": "development", "description": "Refinement and extension", "year": "middle"},
            {"period": "modern", "description": "Current formulation", "year": "recent"}
        ]

        evolution.evolution_stages = stages

        # Identify key contributors (would use knowledge base in production)
        if "prime" in concept.lower():
            evolution.key_contributors = ["Euclid", "Euler", "Gauss", "Riemann"]
            evolution.major_breakthroughs = ["Prime Number Theorem", "Riemann Hypothesis"]

        return evolution

    async def _tool_create_timeline(
        self,
        domain: str,
        events: List[Dict[str, Any]]
    ) -> Timeline:
        """Create a timeline of developments"""

        # Sort events by date if available
        sorted_events = sorted(events, key=lambda x: x.get("date", "0"))

        historical_events = []
        for event in sorted_events:
            hist_event = HistoricalEvent(
                event_id=f"event_{len(self.historical_events)+len(historical_events)+1}",
                timestamp=event.get("date", "unknown"),
                event_type=event.get("type", "discovery"),
                description=event.get("description", ""),
                actors=event.get("actors", []),
                impact_score=event.get("impact", 0.5),
                related_concepts=event.get("concepts", []),
                sources=event.get("sources", [])
            )
            historical_events.append(hist_event)

        timeline = Timeline(
            timeline_id=f"timeline_{len(self.timelines)+1}",
            domain=domain,
            start_date=sorted_events[0].get("date", "unknown") if sorted_events else "unknown",
            end_date=sorted_events[-1].get("date", "unknown") if sorted_events else "unknown",
            events=historical_events,
            key_periods=self._identify_key_periods(historical_events),
            trend_analysis=self._analyze_trends(historical_events)
        )

        return timeline

    async def _tool_analyze_impact(
        self,
        development: str,
        metrics: List[str]
    ) -> ImpactAnalysis:
        """Analyze the impact of a development"""

        impact = ImpactAnalysis(
            development=development,
            immediate_impact=[],
            long_term_impact=[],
            influenced_fields=[],
            citation_growth={},
            adoption_rate=0.0
        )

        # Analyze based on metrics
        if "citations" in metrics:
            # Simulated citation growth
            impact.citation_growth = {
                "year_1": 10,
                "year_5": 100,
                "year_10": 500
            }

        if "applications" in metrics:
            impact.influenced_fields = ["physics", "computer_science", "engineering"]

        if "theoretical" in metrics:
            impact.immediate_impact = ["New proof techniques", "Simplified existing proofs"]
            impact.long_term_impact = ["Opened new research areas", "Influenced education"]

        # Calculate adoption rate based on impact
        if impact.citation_growth:
            max_citations = max(impact.citation_growth.values())
            impact.adoption_rate = min(1.0, max_citations / 1000)

        return impact

    async def _tool_find_historical_patterns(
        self,
        data: List[Dict[str, Any]],
        pattern_type: str
    ) -> List[DevelopmentPattern]:
        """Find patterns in mathematical development"""
        patterns = []

        if pattern_type == "cyclic":
            # Look for repeating patterns
            pattern = DevelopmentPattern(
                pattern_id=f"pattern_{len(self.development_patterns)+1}",
                pattern_type="cyclic",
                description="Periods of intense activity followed by consolidation",
                examples=[{"period": "1900-1930", "activity": "foundations crisis"}],
                predictive_value=0.7,
                recurring_interval="30-40 years"
            )
            patterns.append(pattern)

        elif pattern_type == "progressive":
            # Look for steady progress
            pattern = DevelopmentPattern(
                pattern_id=f"pattern_{len(self.development_patterns)+2}",
                pattern_type="progressive",
                description="Steady accumulation of results",
                examples=[{"field": "number_theory", "rate": "linear"}],
                predictive_value=0.8
            )
            patterns.append(pattern)

        elif pattern_type == "branching":
            # Look for field divergence
            pattern = DevelopmentPattern(
                pattern_id=f"pattern_{len(self.development_patterns)+3}",
                pattern_type="branching",
                description="Single concept spawning multiple subfields",
                examples=[{"origin": "calculus", "branches": ["analysis", "differential_geometry"]}],
                predictive_value=0.6
            )
            patterns.append(pattern)

        return patterns

    async def _tool_predict_next_breakthrough(
        self,
        domain: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict likely next breakthroughs"""

        predictions = {
            "domain": domain,
            "current_maturity": self._assess_field_maturity(current_state),
            "likely_breakthroughs": [],
            "time_estimates": {},
            "confidence": 0.6,
            "based_on_patterns": []
        }

        # Make predictions based on historical patterns
        open_problems = current_state.get("open_problems", [])

        if open_problems:
            # Estimate breakthrough likelihood
            for problem in open_problems[:3]:
                if "long_standing" in str(problem).lower():
                    predictions["likely_breakthroughs"].append({
                        "problem": problem,
                        "likelihood": 0.3,
                        "timeframe": "10-20 years"
                    })
                else:
                    predictions["likely_breakthroughs"].append({
                        "problem": problem,
                        "likelihood": 0.6,
                        "timeframe": "5-10 years"
                    })

        # Base on historical patterns
        if domain in ["number_theory", "algebra"]:
            predictions["based_on_patterns"].append("Similar problems solved every 15-20 years")

        return predictions

    async def _tool_document_discovery(
        self,
        discovery: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HistoricalEvent:
        """Document a new discovery"""

        event = HistoricalEvent(
            event_id=f"event_{len(self.historical_events)+1}",
            timestamp=datetime.utcnow().isoformat(),
            event_type="discovery",
            description=discovery.get("description", "New discovery"),
            actors=discovery.get("discoverers", ["agent_system"]),
            impact_score=self._assess_discovery_impact(discovery),
            related_concepts=discovery.get("concepts", []),
            sources=context.get("sources", [])
        )

        # Add to historical record
        self.historical_events.append(event)

        # Update knowledge genealogy
        for concept in event.related_concepts:
            if concept not in self.knowledge_genealogy:
                self.knowledge_genealogy[concept] = []
            self.knowledge_genealogy[concept].append(event.event_id)

        return event

    async def _tool_trace_influence_network(
        self,
        root_concept: str,
        depth: int = 3
    ) -> Dict[str, Any]:
        """Trace influence networks between concepts"""

        network = {
            "root": root_concept,
            "influenced_concepts": [],
            "prerequisite_concepts": [],
            "parallel_developments": [],
            "network_depth": depth
        }

        # Trace influences (would use knowledge graph in production)
        if "theorem" in root_concept.lower():
            network["influenced_concepts"] = ["Corollary A", "Generalization B"]
            network["prerequisite_concepts"] = ["Lemma X", "Definition Y"]

        # Find parallel developments
        network["parallel_developments"] = self._find_parallel_developments(root_concept)

        return network

    def _identify_key_periods(self, events: List[HistoricalEvent]) -> List[Dict[str, Any]]:
        """Identify key periods in timeline"""
        periods = []

        if len(events) > 5:
            # Group events by impact
            high_impact_events = [e for e in events if e.impact_score > 0.7]

            if high_impact_events:
                periods.append({
                    "name": "Golden Age",
                    "events": len(high_impact_events),
                    "description": "Period of major breakthroughs"
                })

        return periods

    def _analyze_trends(self, events: List[HistoricalEvent]) -> Dict[str, Any]:
        """Analyze trends in events"""
        trends = {
            "activity_level": "moderate",
            "acceleration": False,
            "dominant_type": "discovery"
        }

        if events:
            # Count event types
            type_counts = {}
            for event in events:
                type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1

            if type_counts:
                trends["dominant_type"] = max(type_counts, key=type_counts.get)

        return trends

    def _assess_field_maturity(self, state: Dict[str, Any]) -> str:
        """Assess the maturity of a mathematical field"""
        open_problems = len(state.get("open_problems", []))
        recent_breakthroughs = len(state.get("recent_breakthroughs", []))

        if open_problems > 10 and recent_breakthroughs < 2:
            return "stagnant"
        elif recent_breakthroughs > 5:
            return "rapidly_developing"
        else:
            return "mature"

    def _assess_discovery_impact(self, discovery: Dict[str, Any]) -> float:
        """Assess the impact of a discovery"""
        impact = 0.5  # Base impact

        if "breakthrough" in str(discovery).lower():
            impact += 0.3
        if "solves" in str(discovery).lower():
            impact += 0.2
        if "generalizes" in str(discovery).lower():
            impact += 0.1

        return min(1.0, impact)

    def _find_parallel_developments(self, concept: str) -> List[str]:
        """Find parallel developments to a concept"""
        # Would use knowledge graph in production
        parallels = []

        if "geometry" in concept.lower():
            parallels.append("Parallel development in topology")
        if "algebra" in concept.lower():
            parallels.append("Similar work in category theory")

        return parallels

    async def _extract_evolution_data(self, thought: Thought) -> Optional[ConceptEvolution]:
        """Extract evolution data from thought"""
        if thought.confidence > 0.5:
            return ConceptEvolution(
                concept_name="Extracted concept",
                origin_date="historical",
                origin_context=thought.content[:200],
                evolution_stages=[],
                current_understanding="Modern formulation",
                key_contributors=[],
                major_breakthroughs=[],
                open_questions=[]
            )
        return None

    async def _create_timeline_from_thought(self, thought: Thought) -> Optional[Timeline]:
        """Create timeline from thought content"""
        if "timeline" in thought.content.lower():
            return Timeline(
                timeline_id=f"timeline_{len(self.timelines)+1}",
                domain="general",
                start_date="historical",
                end_date="present",
                events=[],
                key_periods=[],
                trend_analysis={}
            )
        return None

    async def _extract_patterns_from_thought(self, thought: Thought) -> List[DevelopmentPattern]:
        """Extract patterns from thought"""
        patterns = []

        if "pattern" in thought.content.lower():
            patterns.append(DevelopmentPattern(
                pattern_id=f"pattern_{len(self.development_patterns)+1}",
                pattern_type="discovered",
                description=thought.content[:200],
                examples=[],
                predictive_value=thought.confidence
            ))

        return patterns

    async def _analyze_impact_from_thought(self, thought: Thought) -> Optional[ImpactAnalysis]:
        """Analyze impact from thought"""
        if "impact" in thought.content.lower():
            return ImpactAnalysis(
                development="Recent development",
                immediate_impact=["Direct consequences"],
                long_term_impact=["Future implications"],
                influenced_fields=[],
                citation_growth={},
                adoption_rate=0.5
            )
        return None

    def _create_historical_event(self, thought: Thought) -> HistoricalEvent:
        """Create historical event from thought"""
        return HistoricalEvent(
            event_id=f"event_{len(self.historical_events)+1}",
            timestamp=datetime.utcnow().isoformat(),
            event_type="analysis",
            description=thought.content[:200],
            actors=[self.agent_id],
            impact_score=thought.confidence,
            related_concepts=[],
            sources=[]
        )

    def _assess_historical_significance(self, results: Dict[str, Any]) -> float:
        """Assess historical significance of results"""
        significance = 0.5

        if "evolution" in results:
            significance += 0.2
        if "timeline" in results:
            significance += 0.1
        if "patterns" in results:
            significance += 0.15
        if "impact" in results:
            significance += 0.15

        return min(1.0, significance)

    def _calculate_historical_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate confidence for historical analysis"""
        confidence = 0.6  # Base confidence

        # Increase for comprehensive analysis
        if len(response) > 1000:
            confidence += 0.1

        # Increase for temporal references
        temporal_terms = ["evolved", "developed", "historical", "timeline", "period"]
        for term in temporal_terms:
            if term in response.lower():
                confidence += 0.05

        # Decrease for uncertainty
        if "unknown" in response.lower() or "unclear" in response.lower():
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _extract_historical_evidence(self, text: str) -> List[str]:
        """Extract historical evidence from text"""
        evidence = []

        import re
        patterns = [
            r"in (\d{4})",  # Years
            r"([A-Z][a-z]+ \d{4})",  # Month Year
            r"(\d+th century)",  # Centuries
            r"([A-Z][a-z]+ era)",  # Eras
            r"historically\s+([^.]+)"  # Historical references
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                evidence.append(match.group(0))

        return evidence[:10]

    async def document_progress(self, progress_data: Dict[str, Any]) -> HistoricalEvent:
        """
        Main method to document mathematical progress
        """
        logger.info(f"Historian {self.agent_id} documenting progress")

        context = {
            "task_type": "document_impact",
            "development": progress_data.get("development", ""),
            "initial_context": progress_data
        }

        result = await self.process_task(context)

        # Return the created event
        if self.historical_events:
            return self.historical_events[-1]

        # Create default event
        return HistoricalEvent(
            event_id=f"event_{len(self.historical_events)+1}",
            timestamp=datetime.utcnow().isoformat(),
            event_type="documentation",
            description="Progress documented",
            actors=[self.agent_id],
            impact_score=0.5,
            related_concepts=[],
            sources=[]
        )