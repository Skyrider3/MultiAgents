"""
Researcher Agent - Specialized in analyzing mathematical papers and extracting knowledge
"""

import asyncio
import json
import re
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
from src.communication.protocols.a2a_protocol import MessageType
from src.llm.bedrock_client import BedrockMessage


class PaperAnalysis(BaseModel):
    """Structure for paper analysis results"""
    paper_id: str
    title: str
    abstract: str
    key_concepts: List[str]
    theorems: List[Dict[str, Any]]
    conjectures: List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]
    proofs: List[Dict[str, Any]]
    open_problems: List[str]
    citations: List[str]
    mathematical_notation: Dict[str, str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class MathematicalEntity(BaseModel):
    """Represents a mathematical entity extracted from papers"""
    entity_type: str  # theorem, conjecture, lemma, corollary, definition
    name: str
    statement: str
    formal_notation: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    implications: List[str] = Field(default_factory=list)
    source_paper: str
    page_number: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent specialized in:
    - Extracting mathematical knowledge from papers
    - Identifying theorems, conjectures, and proofs
    - Building knowledge graphs
    - Finding research gaps
    """

    def _initialize(self, **kwargs):
        """Initialize Researcher-specific attributes"""
        self.analyzed_papers: Dict[str, PaperAnalysis] = {}
        self.extracted_entities: List[MathematicalEntity] = []
        self.knowledge_domains: Set[str] = set()
        self.citation_graph: Dict[str, List[str]] = {}

        # Specialized capabilities for researcher
        self.capabilities = AgentCapabilities(
            can_reason=True,
            can_learn=True,
            can_collaborate=True,
            can_synthesize=True,
            supported_domains=[
                "number_theory",
                "graph_theory",
                "algebraic_geometry",
                "combinatorics",
                "topology",
                "analysis"
            ],
            max_context_length=150000,
            parallel_tasks=5
        )

        # Researcher personality - curious and thorough
        self.personality = AgentPersonality(
            curiosity=0.9,
            skepticism=0.4,
            creativity=0.7,
            thoroughness=0.95,
            risk_tolerance=0.3
        )

    def _register_custom_tools(self):
        """Register researcher-specific tools"""
        researcher_tools = [
            Tool(
                name="extract_theorems",
                description="Extract theorems from mathematical text",
                parameters=[
                    ToolParameter(name="text", type="str", description="Mathematical text"),
                    ToolParameter(name="domain", type="str", description="Mathematical domain")
                ],
                handler=self._tool_extract_theorems
            ),
            Tool(
                name="extract_conjectures",
                description="Extract conjectures and open problems",
                parameters=[
                    ToolParameter(name="text", type="str", description="Mathematical text")
                ],
                handler=self._tool_extract_conjectures
            ),
            Tool(
                name="identify_patterns",
                description="Identify patterns across multiple papers",
                parameters=[
                    ToolParameter(name="papers", type="list", description="List of paper IDs")
                ],
                handler=self._tool_identify_patterns
            ),
            Tool(
                name="trace_proof_structure",
                description="Analyze and trace proof structure",
                parameters=[
                    ToolParameter(name="proof_text", type="str", description="Proof text")
                ],
                handler=self._tool_trace_proof_structure
            ),
            Tool(
                name="extract_definitions",
                description="Extract mathematical definitions",
                parameters=[
                    ToolParameter(name="text", type="str", description="Mathematical text")
                ],
                handler=self._tool_extract_definitions
            ),
            Tool(
                name="build_citation_network",
                description="Build citation network from papers",
                parameters=[
                    ToolParameter(name="papers", type="list", description="List of papers")
                ],
                handler=self._tool_build_citation_network
            ),
            Tool(
                name="identify_research_gaps",
                description="Identify gaps in current research",
                parameters=[
                    ToolParameter(name="domain", type="str", description="Research domain")
                ],
                handler=self._tool_identify_research_gaps
            )
        ]
        self.tools.extend(researcher_tools)

    async def think(self, context: Dict[str, Any]) -> Thought:
        """
        Researcher's thinking process - analyze papers and extract knowledge
        """
        task_type = context.get("task_type", "analyze_paper")

        # Build prompt for analysis
        if task_type == "analyze_paper":
            paper_content = context.get("paper_content", "")
            paper_title = context.get("paper_title", "Unknown")

            prompt = f"""As a mathematical researcher, analyze this paper thoroughly:

Title: {paper_title}

Content: {paper_content[:50000]}  # Truncate for context limit

Please extract:
1. Key theorems and their statements
2. Conjectures and open problems
3. Mathematical definitions
4. Proof techniques used
5. Citations to other important work
6. Potential research gaps or opportunities

Focus on formal mathematical content and be precise in your extraction.
"""

        elif task_type == "identify_patterns":
            papers = context.get("papers", [])
            prompt = f"""Analyze these {len(papers)} papers to identify common patterns:

Papers: {json.dumps([p.get('title', '') for p in papers[:10]], indent=2)}

Look for:
1. Recurring mathematical structures
2. Similar proof techniques
3. Common conjectures or problems
4. Methodological patterns
5. Potential connections between different domains
"""

        elif task_type == "research_gaps":
            domain = context.get("domain", "general")
            current_knowledge = context.get("current_knowledge", [])

            prompt = f"""Identify research gaps in {domain} mathematics:

Current knowledge base:
{json.dumps(current_knowledge[:20], indent=2)}

Identify:
1. Unproven conjectures that could be approached
2. Missing connections between theorems
3. Potential generalizations
4. Unexplored problem variants
5. Methodological improvements needed
"""
        else:
            prompt = f"Analyze the mathematical content: {json.dumps(context)}"

        # Get LLM response
        messages = [
            BedrockMessage(role="user", content=prompt)
        ]

        response = await self.bedrock_client.generate(
            messages=messages,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.3,  # Lower temperature for factual extraction
            max_tokens=4096
        )

        # Parse response and create thought
        confidence = self._calculate_confidence(response.content, context)

        thought = Thought(
            content=response.content,
            confidence=confidence,
            reasoning_type="analytical" if task_type == "analyze_paper" else "inductive",
            evidence=self._extract_evidence(response.content)
        )

        return thought

    async def act(self, thought: Thought) -> Dict[str, Any]:
        """
        Execute research actions based on thoughts
        """
        # Parse the thought to determine actions
        actions_taken = []
        results = {}

        # Extract theorems if mentioned
        if "theorem" in thought.content.lower():
            theorems = await self._tool_extract_theorems(
                thought.content,
                "general"
            )
            actions_taken.append("extracted_theorems")
            results["theorems"] = theorems

        # Extract conjectures if mentioned
        if any(word in thought.content.lower() for word in ["conjecture", "hypothesis", "open problem"]):
            conjectures = await self._tool_extract_conjectures(thought.content)
            actions_taken.append("extracted_conjectures")
            results["conjectures"] = conjectures

        # Extract definitions if mentioned
        if "definition" in thought.content.lower() or "define" in thought.content.lower():
            definitions = await self._tool_extract_definitions(thought.content)
            actions_taken.append("extracted_definitions")
            results["definitions"] = definitions

        # Store in memory
        self.memory.add_to_short_term({
            "type": "research_analysis",
            "thought": thought.content,
            "results": results,
            "confidence": thought.confidence
        })

        return {
            "actions_taken": actions_taken,
            "results": results,
            "confidence": thought.confidence,
            "entities_extracted": len(self.extracted_entities)
        }

    async def _tool_extract_theorems(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Extract theorems from text"""
        theorems = []

        # Pattern matching for theorem-like structures
        theorem_patterns = [
            r"Theorem\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"Lemma\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"Proposition\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"Corollary\s*(\d+\.?\d*)?:?\s*([^.]+\.)"
        ]

        for pattern in theorem_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                theorem_text = match.group(0)

                # Create mathematical entity
                entity = MathematicalEntity(
                    entity_type="theorem",
                    name=f"Theorem_{len(theorems)+1}",
                    statement=theorem_text,
                    source_paper=self.current_task.get("paper_id", "unknown"),
                    confidence=0.8
                )

                self.extracted_entities.append(entity)
                theorems.append(entity.dict())

        # Use LLM for more sophisticated extraction
        if len(theorems) < 2 and len(text) > 100:
            prompt = f"Extract all mathematical theorems from this text:\n{text[:2000]}\nFormat: List each theorem clearly."

            messages = [BedrockMessage(role="user", content=prompt)]
            response = await self.bedrock_client.generate(
                messages=messages,
                task_type=TaskType.DEEP_REASONING,
                temperature=0.2
            )

            # Parse LLM response for theorems
            llm_theorems = self._parse_llm_theorems(response.content)
            theorems.extend(llm_theorems)

        return theorems

    async def _tool_extract_conjectures(self, text: str) -> List[Dict[str, Any]]:
        """Extract conjectures and open problems"""
        conjectures = []

        # Pattern matching
        conjecture_patterns = [
            r"Conjecture\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"Open [Pp]roblem\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"Hypothesis\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"We conjecture that\s*([^.]+\.)",
            r"It remains open whether\s*([^.]+\.)"
        ]

        for pattern in conjecture_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                conjecture_text = match.group(0)

                entity = MathematicalEntity(
                    entity_type="conjecture",
                    name=f"Conjecture_{len(conjectures)+1}",
                    statement=conjecture_text,
                    source_paper=self.current_task.get("paper_id", "unknown"),
                    confidence=0.85
                )

                self.extracted_entities.append(entity)
                conjectures.append(entity.dict())

        return conjectures

    async def _tool_extract_definitions(self, text: str) -> List[Dict[str, Any]]:
        """Extract mathematical definitions"""
        definitions = []

        # Pattern matching for definitions
        definition_patterns = [
            r"Definition\s*(\d+\.?\d*)?:?\s*([^.]+\.)",
            r"We define\s+([^.]+\.)",
            r"([A-Z][a-z]+)\s+is defined as\s+([^.]+\.)",
            r"Let\s+([^.]+)\s+be\s+([^.]+\.)"
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                def_text = match.group(0)

                entity = MathematicalEntity(
                    entity_type="definition",
                    name=f"Definition_{len(definitions)+1}",
                    statement=def_text,
                    source_paper=self.current_task.get("paper_id", "unknown"),
                    confidence=0.9
                )

                self.extracted_entities.append(entity)
                definitions.append(entity.dict())

        return definitions

    async def _tool_identify_patterns(self, papers: List[str]) -> Dict[str, Any]:
        """Identify patterns across multiple papers"""
        patterns = {
            "common_concepts": [],
            "recurring_techniques": [],
            "connected_theorems": [],
            "similar_conjectures": []
        }

        # Analyze stored paper analyses
        relevant_analyses = [
            self.analyzed_papers[pid]
            for pid in papers
            if pid in self.analyzed_papers
        ]

        if len(relevant_analyses) < 2:
            return {"error": "Need at least 2 analyzed papers for pattern detection"}

        # Find common concepts
        all_concepts = [set(pa.key_concepts) for pa in relevant_analyses]
        if all_concepts:
            common = set.intersection(*all_concepts)
            patterns["common_concepts"] = list(common)

        # Find similar conjectures
        all_conjectures = []
        for pa in relevant_analyses:
            all_conjectures.extend(pa.conjectures)

        # Group similar conjectures (simplified - use embeddings in production)
        for i, conj1 in enumerate(all_conjectures):
            for conj2 in all_conjectures[i+1:]:
                similarity = self._calculate_similarity(
                    conj1.get("statement", ""),
                    conj2.get("statement", "")
                )
                if similarity > 0.7:
                    patterns["similar_conjectures"].append({
                        "conjecture_1": conj1,
                        "conjecture_2": conj2,
                        "similarity": similarity
                    })

        return patterns

    async def _tool_trace_proof_structure(self, proof_text: str) -> Dict[str, Any]:
        """Analyze and trace the structure of a proof"""
        structure = {
            "proof_type": "",
            "steps": [],
            "techniques": [],
            "assumptions": [],
            "conclusion": ""
        }

        # Identify proof type
        if "contradiction" in proof_text.lower():
            structure["proof_type"] = "proof_by_contradiction"
        elif "induction" in proof_text.lower():
            structure["proof_type"] = "proof_by_induction"
        elif "direct" in proof_text.lower() or "assume" in proof_text.lower():
            structure["proof_type"] = "direct_proof"
        else:
            structure["proof_type"] = "unknown"

        # Extract proof steps
        step_patterns = [
            r"Step\s+(\d+):?\s*([^.]+\.)",
            r"(\d+)\.\s*([^.]+\.)",
            r"First,\s*([^.]+\.)",
            r"Next,\s*([^.]+\.)",
            r"Finally,\s*([^.]+\.)",
            r"Therefore,\s*([^.]+\.)"
        ]

        for pattern in step_patterns:
            matches = re.finditer(pattern, proof_text, re.MULTILINE)
            for match in matches:
                structure["steps"].append(match.group(0))

        # Extract assumptions
        if "assume" in proof_text.lower():
            assumption_pattern = r"[Aa]ssume\s+([^.]+\.)"
            matches = re.finditer(assumption_pattern, proof_text)
            structure["assumptions"] = [match.group(1) for match in matches]

        # Extract conclusion
        conclusion_patterns = [
            r"[Tt]herefore,?\s*([^.]+\.)",
            r"[Tt]hus,?\s*([^.]+\.)",
            r"[Hh]ence,?\s*([^.]+\.)",
            r"QED\.?\s*([^.]*)"
        ]

        for pattern in conclusion_patterns:
            match = re.search(pattern, proof_text)
            if match:
                structure["conclusion"] = match.group(0)
                break

        return structure

    async def _tool_build_citation_network(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build citation network from papers"""
        network = {
            "nodes": [],
            "edges": [],
            "clusters": []
        }

        for paper in papers:
            paper_id = paper.get("id", str(len(network["nodes"])))

            # Add node
            network["nodes"].append({
                "id": paper_id,
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "authors": paper.get("authors", [])
            })

            # Add edges for citations
            citations = paper.get("citations", [])
            for cited_id in citations:
                network["edges"].append({
                    "from": paper_id,
                    "to": cited_id,
                    "type": "cites"
                })

            # Update internal citation graph
            self.citation_graph[paper_id] = citations

        # Identify clusters (simplified - use graph algorithms in production)
        # This would use community detection algorithms

        return network

    async def _tool_identify_research_gaps(self, domain: str) -> List[Dict[str, Any]]:
        """Identify gaps in current research"""
        gaps = []

        # Analyze extracted entities for the domain
        domain_entities = [
            e for e in self.extracted_entities
            if domain.lower() in e.statement.lower()
        ]

        # Find unproven conjectures
        conjectures = [e for e in domain_entities if e.entity_type == "conjecture"]
        for conjecture in conjectures:
            gaps.append({
                "type": "unproven_conjecture",
                "description": conjecture.statement,
                "importance": "high" if "fundamental" in conjecture.statement.lower() else "medium",
                "suggested_approach": "Explore using techniques from related proven theorems"
            })

        # Find missing connections
        theorems = [e for e in domain_entities if e.entity_type == "theorem"]
        if len(theorems) > 1:
            # Check for potential generalizations
            gaps.append({
                "type": "potential_generalization",
                "description": f"Possible generalization connecting {len(theorems)} theorems in {domain}",
                "importance": "medium",
                "suggested_approach": "Look for common structural patterns"
            })

        # Find methodological gaps
        if len(self.analyzed_papers) > 5:
            techniques_used = set()
            for paper_analysis in self.analyzed_papers.values():
                # This would extract proof techniques used
                pass

            gaps.append({
                "type": "methodological",
                "description": f"Limited diversity in proof techniques for {domain}",
                "importance": "low",
                "suggested_approach": "Apply techniques from adjacent fields"
            })

        return gaps

    def _calculate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on evidence quality
        if len(response) > 1000:
            confidence += 0.1

        if any(word in response.lower() for word in ["theorem", "proof", "lemma"]):
            confidence += 0.15

        if "references" in context or "citations" in response.lower():
            confidence += 0.1

        # Decrease confidence for uncertainty markers
        uncertainty_markers = ["might", "possibly", "perhaps", "unclear", "unknown"]
        for marker in uncertainty_markers:
            if marker in response.lower():
                confidence -= 0.05

        return min(max(confidence, 0.0), 1.0)

    def _extract_evidence(self, text: str) -> List[str]:
        """Extract evidence statements from text"""
        evidence = []

        # Look for evidence patterns
        evidence_patterns = [
            r"[Aa]ccording to\s+([^,]+),",
            r"[Aa]s shown in\s+([^,]+),",
            r"[Bb]ased on\s+([^,]+),",
            r"\[(\d+)\]",  # Citation markers
            r"([A-Z][a-z]+\s+et al\.\s+\(\d{4}\))"  # Author citations
        ]

        for pattern in evidence_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                evidence.append(match.group(0))

        return evidence[:10]  # Limit to top 10 evidence items

    def _parse_llm_theorems(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse theorems from LLM response"""
        theorems = []

        # Split response into potential theorems
        lines = llm_response.split('\n')
        current_theorem = []

        for line in lines:
            if any(word in line.lower() for word in ["theorem", "lemma", "proposition", "corollary"]):
                if current_theorem:
                    # Save previous theorem
                    theorem_text = ' '.join(current_theorem)
                    entity = MathematicalEntity(
                        entity_type="theorem",
                        name=f"Theorem_{len(theorems)+1}",
                        statement=theorem_text,
                        source_paper="llm_extraction",
                        confidence=0.7
                    )
                    theorems.append(entity.dict())
                current_theorem = [line]
            elif current_theorem:
                current_theorem.append(line)

        # Don't forget the last theorem
        if current_theorem:
            theorem_text = ' '.join(current_theorem)
            entity = MathematicalEntity(
                entity_type="theorem",
                name=f"Theorem_{len(theorems)+1}",
                statement=theorem_text,
                source_paper="llm_extraction",
                confidence=0.7
            )
            theorems.append(entity.dict())

        return theorems

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple Jaccard similarity - use embeddings in production
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def analyze_paper(self, paper_content: str, paper_metadata: Dict[str, Any]) -> PaperAnalysis:
        """
        Main method to analyze a research paper
        """
        logger.info(f"Researcher {self.agent_id} analyzing paper: {paper_metadata.get('title', 'Unknown')}")

        # Process the paper
        context = {
            "task_type": "analyze_paper",
            "paper_content": paper_content,
            "paper_title": paper_metadata.get("title", ""),
            "paper_id": paper_metadata.get("id", str(len(self.analyzed_papers)))
        }

        result = await self.process_task(context)

        # Create paper analysis
        analysis = PaperAnalysis(
            paper_id=context["paper_id"],
            title=paper_metadata.get("title", ""),
            abstract=paper_metadata.get("abstract", ""),
            key_concepts=[],
            theorems=result.get("result", {}).get("results", {}).get("theorems", []),
            conjectures=result.get("result", {}).get("results", {}).get("conjectures", []),
            definitions=result.get("result", {}).get("results", {}).get("definitions", []),
            proofs=[],
            open_problems=[],
            citations=paper_metadata.get("citations", []),
            mathematical_notation={},
            confidence_score=result.get("result", {}).get("confidence", 0.5)
        )

        # Store analysis
        self.analyzed_papers[context["paper_id"]] = analysis

        # Update metrics
        self.metrics.tasks_completed += 1

        return analysis