"""
Challenger Agent - Specialized in adversarial testing and disproving hypotheses
"""

import asyncio
import json
import random
from typing import Any, Dict, List, Optional, Tuple
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


class Challenge(BaseModel):
    """Represents a challenge to a claim or conjecture"""
    challenge_id: str
    target_claim: str
    challenge_type: str  # counterexample, logical_flaw, edge_case, assumption_violation
    challenge_description: str
    evidence: List[str] = Field(default_factory=list)
    severity: str  # critical, major, minor
    confidence: float = Field(ge=0.0, le=1.0)
    requires_response: bool = True


class CounterExample(BaseModel):
    """A specific counter-example to a claim"""
    claim: str
    counter_example: str
    mathematical_form: Optional[str] = None
    verified: bool = False
    impact: str  # refutes_completely, limits_scope, requires_modification
    domain_restriction: Optional[str] = None


class EdgeCase(BaseModel):
    """An edge case that challenges a theorem or conjecture"""
    case_id: str
    description: str
    input_values: Dict[str, Any]
    expected_behavior: str
    actual_behavior: str
    breaks_assumption: Optional[str] = None
    severity: str


class AdversarialTest(BaseModel):
    """An adversarial test case"""
    test_id: str
    test_type: str  # boundary, stress, contradiction, pathological
    target: str
    test_input: Any
    expected_failure: str
    actual_result: Optional[str] = None
    passed: Optional[bool] = None


class ChallengerAgent(BaseAgent):
    """
    Challenger Agent specialized in:
    - Adversarial testing of conjectures
    - Finding counter-examples
    - Identifying edge cases
    - Stress-testing mathematical claims
    - Playing devil's advocate
    """

    def _initialize(self, **kwargs):
        """Initialize Challenger-specific attributes"""
        self.challenges_issued: List[Challenge] = []
        self.counter_examples: List[CounterExample] = []
        self.edge_cases: List[EdgeCase] = []
        self.adversarial_tests: List[AdversarialTest] = []
        self.refuted_claims: List[str] = []

        # Specialized capabilities for challenger
        self.capabilities = AgentCapabilities(
            can_reason=True,
            can_learn=True,
            can_collaborate=True,
            can_challenge=True,  # Primary capability
            can_verify=True,
            supported_domains=[
                "adversarial_testing",
                "counterexample_generation",
                "edge_case_analysis",
                "stress_testing",
                "logical_analysis"
            ],
            max_context_length=100000,
            parallel_tasks=5
        )

        # Challenger personality - highly skeptical and adversarial
        self.personality = AgentPersonality(
            curiosity=0.7,
            skepticism=0.99,  # Extremely skeptical
            creativity=0.8,  # Creative in finding counter-examples
            thoroughness=0.9,
            risk_tolerance=0.8  # Willing to challenge everything
        )

    def _register_custom_tools(self):
        """Register challenger-specific tools"""
        challenger_tools = [
            Tool(
                name="generate_counterexample",
                description="Generate counter-example to a claim",
                parameters=[
                    ToolParameter(name="claim", type="str", description="Claim to challenge"),
                    ToolParameter(name="domain", type="str", description="Mathematical domain")
                ],
                handler=self._tool_generate_counterexample
            ),
            Tool(
                name="find_edge_cases",
                description="Find edge cases for a theorem",
                parameters=[
                    ToolParameter(name="theorem", type="str", description="Theorem to test"),
                    ToolParameter(name="assumptions", type="list", description="Theorem assumptions")
                ],
                handler=self._tool_find_edge_cases
            ),
            Tool(
                name="adversarial_test",
                description="Create adversarial test cases",
                parameters=[
                    ToolParameter(name="target", type="str", description="Target to test"),
                    ToolParameter(name="test_type", type="str", description="Type of test")
                ],
                handler=self._tool_adversarial_test
            ),
            Tool(
                name="challenge_proof",
                description="Challenge a mathematical proof",
                parameters=[
                    ToolParameter(name="proof", type="str", description="Proof to challenge"),
                    ToolParameter(name="focus", type="str", description="Focus area for challenge")
                ],
                handler=self._tool_challenge_proof
            ),
            Tool(
                name="stress_test_conjecture",
                description="Stress test a conjecture",
                parameters=[
                    ToolParameter(name="conjecture", type="str", description="Conjecture to test"),
                    ToolParameter(name="test_cases", type="list", description="Test cases to use")
                ],
                handler=self._tool_stress_test_conjecture
            ),
            Tool(
                name="find_assumption_violations",
                description="Find violations of assumptions",
                parameters=[
                    ToolParameter(name="assumptions", type="list", description="List of assumptions"),
                    ToolParameter(name="context", type="dict", description="Context to check")
                ],
                handler=self._tool_find_assumption_violations
            ),
            Tool(
                name="generate_pathological_case",
                description="Generate pathological cases",
                parameters=[
                    ToolParameter(name="structure", type="str", description="Mathematical structure"),
                    ToolParameter(name="properties", type="list", description="Expected properties")
                ],
                handler=self._tool_generate_pathological_case
            )
        ]
        self.tools.extend(challenger_tools)

    async def think(self, context: Dict[str, Any]) -> Thought:
        """
        Challenger's thinking process - adversarial and skeptical
        """
        task_type = context.get("task_type", "challenge_claim")

        if task_type == "challenge_claim":
            claim = context.get("claim", "")
            evidence = context.get("evidence", [])

            prompt = f"""As an adversarial challenger, rigorously attack this claim:

Claim: {claim}

Supporting Evidence:
{json.dumps(evidence, indent=2) if evidence else "No evidence provided"}

Your mission:
1. Find counter-examples that refute the claim
2. Identify edge cases where the claim fails
3. Discover hidden assumptions that may be violated
4. Create pathological cases that break the claim
5. Find logical flaws or circular reasoning
6. Test boundary conditions
7. Question every aspect ruthlessly

Be creative and thorough in your attack. Leave no stone unturned.
"""

        elif task_type == "generate_counterexample":
            statement = context.get("statement", "")
            domain = context.get("domain", "general")

            prompt = f"""Generate devastating counter-examples for this statement:

Statement: {statement}
Domain: {domain}

Find:
1. Direct counter-examples that completely refute
2. Edge cases that limit the statement's scope
3. Pathological constructions that violate expectations
4. Boundary cases where the statement breaks down
5. Special cases not covered by the statement

Be inventive - think of cases others would miss.
"""

        elif task_type == "stress_test":
            conjecture = context.get("conjecture", "")
            known_results = context.get("known_results", [])

            prompt = f"""Stress test this conjecture to destruction:

Conjecture: {conjecture}

Known Results:
{json.dumps(known_results, indent=2)}

Apply:
1. Extreme value testing
2. Randomized property testing
3. Adversarial input generation
4. Consistency checking across domains
5. Computational verification for small cases
6. Asymptotic behavior analysis

Find where and how it breaks.
"""

        elif task_type == "adversarial_proof_review":
            proof = context.get("proof", "")

            prompt = f"""Adversarially review this proof:

Proof: {proof}

Attack vectors:
1. Hidden assumptions not explicitly stated
2. Logical gaps or unjustified leaps
3. Circular dependencies
4. Incorrect generalizations
5. Edge cases not considered
6. Mathematical errors or typos
7. Scope limitations not acknowledged

Be ruthless in your critique.
"""
        else:
            prompt = f"Challenge everything about: {json.dumps(context)}"

        # Get LLM response with adversarial mindset
        messages = [
            BedrockMessage(role="user", content=prompt)
        ]

        response = await self.bedrock_client.generate(
            messages=messages,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.7,  # Some creativity for finding counter-examples
            max_tokens=4096
        )

        # Create adversarial thought
        confidence = self._calculate_adversarial_confidence(response.content, context)

        thought = Thought(
            content=response.content,
            confidence=confidence,
            reasoning_type="adversarial",
            evidence=self._extract_adversarial_evidence(response.content)
        )

        return thought

    async def act(self, thought: Thought) -> Dict[str, Any]:
        """
        Execute adversarial actions based on challenging thoughts
        """
        actions_taken = []
        results = {}

        # Parse thought for specific challenges
        if "counter" in thought.content.lower() or "counterexample" in thought.content.lower():
            counter_examples = await self._extract_counterexamples(thought)
            if counter_examples:
                actions_taken.append("generated_counterexamples")
                results["counter_examples"] = [ce.dict() for ce in counter_examples]
                self.counter_examples.extend(counter_examples)

        if "edge case" in thought.content.lower() or "boundary" in thought.content.lower():
            edge_cases = await self._extract_edge_cases(thought)
            if edge_cases:
                actions_taken.append("found_edge_cases")
                results["edge_cases"] = [ec.dict() for ec in edge_cases]
                self.edge_cases.extend(edge_cases)

        if "flaw" in thought.content.lower() or "error" in thought.content.lower():
            challenges = await self._extract_challenges(thought)
            if challenges:
                actions_taken.append("issued_challenges")
                results["challenges"] = [c.dict() for c in challenges]
                self.challenges_issued.extend(challenges)

        if "refute" in thought.content.lower() or "disprove" in thought.content.lower():
            claim = self.current_task.get("claim", "")
            if claim and claim not in self.refuted_claims:
                actions_taken.append("refuted_claim")
                self.refuted_claims.append(claim)
                results["refuted"] = claim

        # Store in memory
        self.memory.add_to_short_term({
            "type": "challenge",
            "thought": thought.content,
            "results": results,
            "confidence": thought.confidence
        })

        return {
            "actions_taken": actions_taken,
            "results": results,
            "confidence": thought.confidence,
            "impact": self._assess_challenge_impact(results)
        }

    async def _tool_generate_counterexample(self, claim: str, domain: str) -> Optional[CounterExample]:
        """Generate a counter-example to a claim"""

        # Try different strategies for counter-example generation
        strategies = [
            self._try_extreme_values,
            self._try_special_cases,
            self._try_symmetry_breaking,
            self._try_contradiction
        ]

        for strategy in strategies:
            counter = await strategy(claim, domain)
            if counter:
                counter_example = CounterExample(
                    claim=claim,
                    counter_example=counter,
                    verified=False,  # Would need symbolic verification
                    impact="refutes_completely" if "all" in claim.lower() else "limits_scope"
                )
                return counter_example

        return None

    async def _tool_find_edge_cases(
        self,
        theorem: str,
        assumptions: List[str]
    ) -> List[EdgeCase]:
        """Find edge cases for a theorem"""
        edge_cases = []

        # Check boundary conditions
        if "positive" in theorem.lower():
            edge_cases.append(EdgeCase(
                case_id=f"edge_{len(self.edge_cases)+1}",
                description="Zero boundary case",
                input_values={"n": 0},
                expected_behavior="Theorem should handle n=0",
                actual_behavior="Potential division by zero or undefined",
                breaks_assumption="n > 0",
                severity="major"
            ))

        # Check infinity cases
        if "finite" in theorem.lower() or "bounded" in theorem.lower():
            edge_cases.append(EdgeCase(
                case_id=f"edge_{len(self.edge_cases)+2}",
                description="Infinite input case",
                input_values={"x": "infinity"},
                expected_behavior="Theorem assumes finite values",
                actual_behavior="Breaks under infinite input",
                breaks_assumption="finite domain",
                severity="critical"
            ))

        # Check empty set cases
        if "set" in theorem.lower() or "collection" in theorem.lower():
            edge_cases.append(EdgeCase(
                case_id=f"edge_{len(self.edge_cases)+3}",
                description="Empty set case",
                input_values={"S": "{}"},
                expected_behavior="Should handle empty set",
                actual_behavior="May produce unexpected result",
                severity="minor"
            ))

        return edge_cases

    async def _tool_adversarial_test(self, target: str, test_type: str) -> AdversarialTest:
        """Create adversarial test case"""

        test = AdversarialTest(
            test_id=f"test_{len(self.adversarial_tests)+1}",
            test_type=test_type,
            target=target,
            test_input=None,
            expected_failure=""
        )

        if test_type == "boundary":
            test.test_input = self._generate_boundary_input(target)
            test.expected_failure = "Fails at boundary"
        elif test_type == "stress":
            test.test_input = self._generate_stress_input(target)
            test.expected_failure = "Performance degradation or failure"
        elif test_type == "contradiction":
            test.test_input = self._generate_contradictory_input(target)
            test.expected_failure = "Logical contradiction"
        elif test_type == "pathological":
            test.test_input = self._generate_pathological_input(target)
            test.expected_failure = "Unexpected behavior"

        return test

    async def _tool_challenge_proof(self, proof: str, focus: str) -> List[Challenge]:
        """Challenge a mathematical proof"""
        challenges = []

        # Focus-specific challenges
        if focus == "assumptions":
            hidden_assumptions = self._find_hidden_assumptions(proof)
            for assumption in hidden_assumptions:
                challenges.append(Challenge(
                    challenge_id=f"chal_{len(self.challenges_issued)+len(challenges)+1}",
                    target_claim=proof[:100],
                    challenge_type="assumption_violation",
                    challenge_description=f"Hidden assumption: {assumption}",
                    severity="major",
                    confidence=0.7,
                    requires_response=True
                ))

        elif focus == "logic":
            logical_flaws = self._find_logical_flaws(proof)
            for flaw in logical_flaws:
                challenges.append(Challenge(
                    challenge_id=f"chal_{len(self.challenges_issued)+len(challenges)+1}",
                    target_claim=proof[:100],
                    challenge_type="logical_flaw",
                    challenge_description=flaw,
                    severity="critical",
                    confidence=0.8,
                    requires_response=True
                ))

        elif focus == "completeness":
            gaps = self._find_completeness_gaps(proof)
            for gap in gaps:
                challenges.append(Challenge(
                    challenge_id=f"chal_{len(self.challenges_issued)+len(challenges)+1}",
                    target_claim=proof[:100],
                    challenge_type="incompleteness",
                    challenge_description=f"Missing: {gap}",
                    severity="major",
                    confidence=0.6,
                    requires_response=True
                ))

        return challenges

    async def _tool_stress_test_conjecture(
        self,
        conjecture: str,
        test_cases: List[Any]
    ) -> Dict[str, Any]:
        """Stress test a conjecture with multiple test cases"""
        results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "breaking_cases": []
        }

        for test_case in test_cases:
            # Simulate testing (would use symbolic math in production)
            if random.random() < 0.3:  # 30% failure rate for demo
                results["failed"] += 1
                results["breaking_cases"].append({
                    "input": test_case,
                    "reason": "Conjecture fails for this input"
                })
            else:
                results["passed"] += 1

        results["stress_score"] = results["passed"] / len(test_cases) if test_cases else 0

        return results

    async def _tool_find_assumption_violations(
        self,
        assumptions: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """Find violations of assumptions in given context"""
        violations = []

        for assumption in assumptions:
            # Check each assumption against context
            if "positive" in assumption.lower() and context.get("allows_negative", False):
                violations.append(f"Assumption '{assumption}' violated: negative values possible")

            if "finite" in assumption.lower() and context.get("includes_infinity", False):
                violations.append(f"Assumption '{assumption}' violated: infinite values present")

            if "continuous" in assumption.lower() and context.get("discrete", False):
                violations.append(f"Assumption '{assumption}' violated: discrete structure")

        return violations

    async def _tool_generate_pathological_case(
        self,
        structure: str,
        properties: List[str]
    ) -> Dict[str, Any]:
        """Generate pathological cases that violate expected properties"""
        pathological = {
            "structure": structure,
            "case": "",
            "violates": [],
            "description": ""
        }

        # Generate based on structure type
        if "function" in structure.lower():
            pathological["case"] = "Dirichlet function (discontinuous everywhere)"
            pathological["violates"] = ["continuity", "differentiability"]
            pathological["description"] = "Function that is 1 on rationals, 0 on irrationals"

        elif "sequence" in structure.lower():
            pathological["case"] = "Sequence that converges but sum diverges"
            pathological["violates"] = ["absolute convergence"]
            pathological["description"] = "e.g., alternating harmonic series"

        elif "set" in structure.lower():
            pathological["case"] = "Cantor set (uncountable but measure zero)"
            pathological["violates"] = ["intuitive size notion"]
            pathological["description"] = "Uncountable set with zero Lebesgue measure"

        elif "graph" in structure.lower():
            pathological["case"] = "Graph with chromatic number > clique number"
            pathological["violates"] = ["naive coloring bounds"]
            pathological["description"] = "Mycielski construction"

        return pathological

    async def _try_extreme_values(self, claim: str, domain: str) -> Optional[str]:
        """Try extreme values as counter-examples"""
        if "all" in claim.lower() and "positive" in claim.lower():
            return "Counter-example: n = 0 (boundary case)"
        if "continuous" in claim.lower():
            return "Counter-example: Dirichlet function (discontinuous everywhere)"
        return None

    async def _try_special_cases(self, claim: str, domain: str) -> Optional[str]:
        """Try special cases as counter-examples"""
        if "prime" in claim.lower():
            return "Counter-example: n = 2 (only even prime)"
        if "odd" in claim.lower():
            return "Counter-example: n = 1 (neither prime nor composite)"
        return None

    async def _try_symmetry_breaking(self, claim: str, domain: str) -> Optional[str]:
        """Try symmetry-breaking cases"""
        if "symmetric" in claim.lower():
            return "Counter-example: Non-symmetric matrix with real eigenvalues"
        if "commutative" in claim.lower():
            return "Counter-example: Matrix multiplication (non-commutative)"
        return None

    async def _try_contradiction(self, claim: str, domain: str) -> Optional[str]:
        """Try proof by contradiction"""
        if "unique" in claim.lower():
            return "Counter-example: Multiple solutions exist for certain parameters"
        if "always" in claim.lower():
            return "Counter-example: Fails for carefully constructed edge case"
        return None

    def _generate_boundary_input(self, target: str) -> Any:
        """Generate boundary test input"""
        return {"type": "boundary", "values": [0, -1, 1, "inf", "-inf", "null"]}

    def _generate_stress_input(self, target: str) -> Any:
        """Generate stress test input"""
        return {"type": "stress", "size": 10000, "complexity": "exponential"}

    def _generate_contradictory_input(self, target: str) -> Any:
        """Generate contradictory input"""
        return {"type": "contradiction", "property_A": True, "property_not_A": True}

    def _generate_pathological_input(self, target: str) -> Any:
        """Generate pathological input"""
        return {"type": "pathological", "structure": "Cantor_set"}

    def _find_hidden_assumptions(self, proof: str) -> List[str]:
        """Find hidden assumptions in a proof"""
        hidden = []

        if "therefore" in proof.lower() and "assume" not in proof.lower():
            hidden.append("Implicit assumptions not stated")

        if "obvious" in proof.lower() or "clearly" in proof.lower():
            hidden.append("Claims obviousness without justification")

        if "wlog" in proof.lower() or "without loss" in proof.lower():
            hidden.append("WLOG may hide important cases")

        return hidden

    def _find_logical_flaws(self, proof: str) -> List[str]:
        """Find logical flaws in a proof"""
        flaws = []

        if proof.count("therefore") > proof.count("because") + proof.count("since"):
            flaws.append("More conclusions than justifications")

        if "similar" in proof.lower() and "identical" not in proof.lower():
            flaws.append("Treats similar as identical")

        if "must" in proof.lower() and "only" not in proof.lower():
            flaws.append("Claims necessity without proving uniqueness")

        return flaws

    def _find_completeness_gaps(self, proof: str) -> List[str]:
        """Find completeness gaps in a proof"""
        gaps = []

        if "case 1" in proof.lower() and "case 2" not in proof.lower():
            gaps.append("Incomplete case analysis")

        if "base case" in proof.lower() and "inductive step" not in proof.lower():
            gaps.append("Incomplete induction")

        if "exists" in proof.lower() and "construct" not in proof.lower():
            gaps.append("Claims existence without construction")

        return gaps

    async def _extract_counterexamples(self, thought: Thought) -> List[CounterExample]:
        """Extract counter-examples from thought"""
        counter_examples = []

        # Parse thought for counter-examples
        import re
        pattern = r"counter-?example:?\s*([^.]+)"
        matches = re.finditer(pattern, thought.content.lower())

        for match in matches:
            counter_examples.append(CounterExample(
                claim=self.current_task.get("claim", ""),
                counter_example=match.group(1),
                verified=False,
                impact="refutes_completely"
            ))

        return counter_examples

    async def _extract_edge_cases(self, thought: Thought) -> List[EdgeCase]:
        """Extract edge cases from thought"""
        edge_cases = []

        if "edge case" in thought.content.lower() or "boundary" in thought.content.lower():
            edge_cases.append(EdgeCase(
                case_id=f"edge_{len(self.edge_cases)+1}",
                description=thought.content[:200],
                input_values={},
                expected_behavior="Normal operation",
                actual_behavior="Failure or unexpected result",
                severity="major"
            ))

        return edge_cases

    async def _extract_challenges(self, thought: Thought) -> List[Challenge]:
        """Extract challenges from thought"""
        challenges = []

        if thought.confidence > 0.7:
            challenges.append(Challenge(
                challenge_id=f"chal_{len(self.challenges_issued)+1}",
                target_claim=self.current_task.get("claim", ""),
                challenge_type="comprehensive",
                challenge_description=thought.content[:500],
                evidence=thought.evidence,
                severity="major",
                confidence=thought.confidence,
                requires_response=True
            ))

        return challenges

    def _assess_challenge_impact(self, results: Dict[str, Any]) -> str:
        """Assess the impact of challenges"""
        if "refuted" in results:
            return "critical"
        elif "counter_examples" in results and len(results["counter_examples"]) > 2:
            return "high"
        elif "edge_cases" in results and len(results["edge_cases"]) > 3:
            return "medium"
        else:
            return "low"

    def _calculate_adversarial_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate confidence for adversarial challenge"""
        confidence = 0.5

        # Increase for counter-examples found
        if "counter" in response.lower():
            confidence += 0.2

        # Increase for logical flaws found
        if "flaw" in response.lower() or "error" in response.lower():
            confidence += 0.15

        # Increase for edge cases found
        if "edge case" in response.lower():
            confidence += 0.1

        # Increase for refutation
        if "refute" in response.lower() or "disprove" in response.lower():
            confidence += 0.2

        return min(1.0, confidence)

    def _extract_adversarial_evidence(self, text: str) -> List[str]:
        """Extract adversarial evidence from text"""
        evidence = []

        import re
        patterns = [
            r"fails when\s+([^.]+)",
            r"breaks for\s+([^.]+)",
            r"counter-?example:?\s*([^.]+)",
            r"contradiction:?\s*([^.]+)"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                evidence.append(match.group(0))

        return evidence[:10]

    async def challenge_claim(self, claim: str, context: Dict[str, Any] = None) -> Challenge:
        """
        Main method to challenge a claim
        """
        logger.info(f"Challenger {self.agent_id} attacking claim: {claim[:100]}...")

        task_context = {
            "task_type": "challenge_claim",
            "claim": claim,
            "evidence": context.get("evidence", []) if context else []
        }

        result = await self.process_task(task_context)

        # Return the most significant challenge
        if self.challenges_issued:
            return self.challenges_issued[-1]

        # Create default challenge
        return Challenge(
            challenge_id=f"chal_{len(self.challenges_issued)+1}",
            target_claim=claim,
            challenge_type="general",
            challenge_description="Requires further investigation",
            severity="minor",
            confidence=0.5,
            requires_response=True
        )