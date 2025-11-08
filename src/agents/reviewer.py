"""
Reviewer Agent - Specialized in critical validation and verification of mathematical claims
"""

import asyncio
import json
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


class ValidationResult(BaseModel):
    """Result of validating a mathematical claim"""
    claim: str
    status: str  # valid, invalid, uncertain, requires_verification
    confidence: float = Field(ge=0.0, le=1.0)
    issues_found: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    counter_examples: List[str] = Field(default_factory=list)
    verification_steps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ProofVerification(BaseModel):
    """Result of proof verification"""
    proof_id: str
    original_claim: str
    proof_valid: bool
    logical_errors: List[str] = Field(default_factory=list)
    missing_steps: List[str] = Field(default_factory=list)
    assumptions_verified: Dict[str, bool] = Field(default_factory=dict)
    rigor_score: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)


class CrossValidation(BaseModel):
    """Cross-validation results across multiple sources"""
    claim_id: str
    sources_checked: List[str]
    consistency_score: float = Field(ge=0.0, le=1.0)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    agreements: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent specialized in:
    - Critical evaluation of mathematical claims
    - Proof verification and validation
    - Cross-referencing across sources
    - Identifying contradictions and gaps
    - Logical consistency checking
    """

    def _initialize(self, **kwargs):
        """Initialize Reviewer-specific attributes"""
        self.validation_history: List[ValidationResult] = []
        self.verified_proofs: Dict[str, ProofVerification] = {}
        self.known_contradictions: List[Dict[str, Any]] = []
        self.verification_cache: Dict[str, ValidationResult] = {}

        # Specialized capabilities for reviewer
        self.capabilities = AgentCapabilities(
            can_reason=True,
            can_learn=True,
            can_collaborate=True,
            can_challenge=True,  # Key capability for reviewer
            can_verify=True,  # Key capability for reviewer
            supported_domains=[
                "logic",
                "proof_theory",
                "formal_verification",
                "mathematical_logic",
                "consistency_checking"
            ],
            max_context_length=100000,
            parallel_tasks=3
        )

        # Reviewer personality - skeptical and thorough
        self.personality = AgentPersonality(
            curiosity=0.6,
            skepticism=0.95,  # Very skeptical
            creativity=0.4,
            thoroughness=0.98,  # Extremely thorough
            risk_tolerance=0.1  # Very low risk tolerance
        )

    def _register_custom_tools(self):
        """Register reviewer-specific tools"""
        reviewer_tools = [
            Tool(
                name="verify_claim",
                description="Verify a mathematical claim",
                parameters=[
                    ToolParameter(name="claim", type="str", description="The claim to verify"),
                    ToolParameter(name="context", type="dict", description="Supporting context")
                ],
                handler=self._tool_verify_claim
            ),
            Tool(
                name="check_proof",
                description="Check the validity of a proof",
                parameters=[
                    ToolParameter(name="proof", type="str", description="The proof to check"),
                    ToolParameter(name="theorem", type="str", description="The theorem being proved")
                ],
                handler=self._tool_check_proof
            ),
            Tool(
                name="find_contradictions",
                description="Find contradictions in claims",
                parameters=[
                    ToolParameter(name="claims", type="list", description="List of claims to check")
                ],
                handler=self._tool_find_contradictions
            ),
            Tool(
                name="cross_reference",
                description="Cross-reference claim across sources",
                parameters=[
                    ToolParameter(name="claim", type="str", description="Claim to cross-reference"),
                    ToolParameter(name="sources", type="list", description="Sources to check")
                ],
                handler=self._tool_cross_reference
            ),
            Tool(
                name="check_logical_consistency",
                description="Check logical consistency of arguments",
                parameters=[
                    ToolParameter(name="arguments", type="list", description="Arguments to check")
                ],
                handler=self._tool_check_logical_consistency
            ),
            Tool(
                name="verify_assumptions",
                description="Verify assumptions in a proof",
                parameters=[
                    ToolParameter(name="assumptions", type="list", description="List of assumptions"),
                    ToolParameter(name="context", type="dict", description="Context for verification")
                ],
                handler=self._tool_verify_assumptions
            ),
            Tool(
                name="generate_counter_example",
                description="Try to generate counter-examples",
                parameters=[
                    ToolParameter(name="claim", type="str", description="Claim to counter"),
                    ToolParameter(name="domain", type="str", description="Mathematical domain")
                ],
                handler=self._tool_generate_counter_example
            )
        ]
        self.tools.extend(reviewer_tools)

    async def think(self, context: Dict[str, Any]) -> Thought:
        """
        Reviewer's thinking process - critical analysis and verification
        """
        task_type = context.get("task_type", "verify_claim")

        if task_type == "verify_claim":
            claim = context.get("claim", "")
            evidence = context.get("evidence", [])

            prompt = f"""As a rigorous mathematical reviewer, critically evaluate this claim:

Claim: {claim}

Provided Evidence:
{json.dumps(evidence, indent=2) if evidence else "No evidence provided"}

Please:
1. Check the logical validity of the claim
2. Identify any potential issues or gaps
3. Verify consistency with known mathematical facts
4. Look for potential counter-examples
5. Assess the strength of the evidence
6. Rate confidence in the claim's validity (0-1)

Be extremely thorough and skeptical. Question every assumption.
"""

        elif task_type == "verify_proof":
            proof = context.get("proof", "")
            theorem = context.get("theorem", "")

            prompt = f"""Rigorously verify this mathematical proof:

Theorem: {theorem}

Proof: {proof}

Check for:
1. Logical errors or fallacies
2. Missing steps or unjustified leaps
3. Unverified assumptions
4. Circular reasoning
5. Correctness of each step
6. Completeness of the argument

Provide detailed feedback on any issues found.
"""

        elif task_type == "find_contradictions":
            claims = context.get("claims", [])

            prompt = f"""Analyze these claims for contradictions:

Claims:
{json.dumps(claims, indent=2)}

Identify:
1. Direct contradictions between claims
2. Logical inconsistencies
3. Mutually exclusive statements
4. Subtle conflicts in implications
5. Temporal or contextual contradictions

Be thorough in your analysis.
"""

        elif task_type == "cross_validate":
            claim = context.get("claim", "")
            sources = context.get("sources", [])

            prompt = f"""Cross-validate this claim across multiple sources:

Claim: {claim}

Sources to check:
{json.dumps(sources, indent=2)}

Examine:
1. Consistency across sources
2. Variations in statements
3. Supporting evidence in each source
4. Contradictory information
5. Reliability of sources

Provide a consistency score and detailed analysis.
"""
        else:
            prompt = f"Critically review: {json.dumps(context)}"

        # Get LLM response with high skepticism
        messages = [
            BedrockMessage(role="user", content=prompt)
        ]

        response = await self.bedrock_client.generate(
            messages=messages,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.1,  # Very low temperature for consistency
            max_tokens=4096
        )

        # Create skeptical thought
        confidence = self._calculate_skeptical_confidence(response.content, context)

        thought = Thought(
            content=response.content,
            confidence=confidence,
            reasoning_type="deductive",  # Reviewers use deductive reasoning
            evidence=self._extract_verification_evidence(response.content)
        )

        return thought

    async def act(self, thought: Thought) -> Dict[str, Any]:
        """
        Execute review actions based on critical thinking
        """
        actions_taken = []
        results = {}

        # Parse thought for specific verification needs
        if "invalid" in thought.content.lower() or "error" in thought.content.lower():
            actions_taken.append("identified_issues")
            results["issues"] = self._extract_issues(thought.content)

        if "counter" in thought.content.lower() or "counterexample" in thought.content.lower():
            actions_taken.append("found_counterexamples")
            results["counterexamples"] = self._extract_counterexamples(thought.content)

        if "missing" in thought.content.lower() or "gap" in thought.content.lower():
            actions_taken.append("identified_gaps")
            results["gaps"] = self._extract_gaps(thought.content)

        if "assumption" in thought.content.lower():
            actions_taken.append("checked_assumptions")
            results["assumptions"] = self._extract_assumptions(thought.content)

        # Create validation result
        validation = ValidationResult(
            claim=self.current_task.get("claim", ""),
            status=self._determine_validation_status(thought.content),
            confidence=thought.confidence,
            issues_found=results.get("issues", []),
            counter_examples=results.get("counterexamples", []),
            recommendations=self._generate_recommendations(thought.content)
        )

        self.validation_history.append(validation)

        # Store in memory
        self.memory.add_to_short_term({
            "type": "validation",
            "thought": thought.content,
            "validation": validation.dict(),
            "confidence": thought.confidence
        })

        return {
            "actions_taken": actions_taken,
            "validation": validation.dict(),
            "confidence": thought.confidence,
            "severity": self._assess_severity(validation)
        }

    async def _tool_verify_claim(self, claim: str, context: Dict[str, Any]) -> ValidationResult:
        """Verify a mathematical claim"""
        # Check cache first
        cache_key = f"{claim}_{json.dumps(context, sort_keys=True)}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]

        # Perform verification
        issues = []
        supporting_evidence = []

        # Check for common logical fallacies
        fallacies = self._check_logical_fallacies(claim)
        issues.extend(fallacies)

        # Check mathematical consistency
        consistency_issues = await self._check_mathematical_consistency(claim, context)
        issues.extend(consistency_issues)

        # Look for supporting evidence
        if "evidence" in context:
            for evidence in context["evidence"]:
                if self._supports_claim(evidence, claim):
                    supporting_evidence.append(evidence)

        # Determine validation status
        if len(issues) > 3:
            status = "invalid"
            confidence = 0.2
        elif len(issues) > 0:
            status = "uncertain"
            confidence = 0.5
        elif len(supporting_evidence) > 2:
            status = "valid"
            confidence = 0.8
        else:
            status = "requires_verification"
            confidence = 0.4

        result = ValidationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            issues_found=issues,
            supporting_evidence=supporting_evidence,
            verification_steps=["Checked logical consistency", "Verified mathematical properties", "Reviewed evidence"],
            recommendations=["Seek additional verification" if status == "uncertain" else "Claim appears sound"]
        )

        # Cache result
        self.verification_cache[cache_key] = result
        return result

    async def _tool_check_proof(self, proof: str, theorem: str) -> ProofVerification:
        """Check the validity of a proof"""
        logical_errors = []
        missing_steps = []
        assumptions = {}

        # Analyze proof structure
        proof_lines = proof.split('\n')

        # Check for common proof errors
        if "therefore" in proof.lower() and "because" not in proof.lower():
            logical_errors.append("Conclusion without justification")

        if "assume" in proof.lower():
            # Extract and verify assumptions
            import re
            assumption_pattern = r"assume\s+(.+?)(?:\.|,)"
            matches = re.finditer(assumption_pattern, proof.lower())
            for match in matches:
                assumption = match.group(1)
                # Simplified verification - in production, use symbolic math
                assumptions[assumption] = True  # Default to verified

        # Check for circular reasoning
        if theorem.lower() in proof.lower():
            logical_errors.append("Potential circular reasoning detected")

        # Check proof completeness
        if not any(end in proof.lower() for end in ["qed", "therefore", "thus", "hence"]):
            missing_steps.append("No clear conclusion statement")

        # Calculate rigor score
        rigor_score = 1.0
        rigor_score -= len(logical_errors) * 0.2
        rigor_score -= len(missing_steps) * 0.15
        rigor_score = max(0.0, min(1.0, rigor_score))

        verification = ProofVerification(
            proof_id=f"proof_{len(self.verified_proofs)}",
            original_claim=theorem,
            proof_valid=len(logical_errors) == 0 and len(missing_steps) == 0,
            logical_errors=logical_errors,
            missing_steps=missing_steps,
            assumptions_verified=assumptions,
            rigor_score=rigor_score,
            suggestions=self._generate_proof_suggestions(logical_errors, missing_steps)
        )

        self.verified_proofs[verification.proof_id] = verification
        return verification

    async def _tool_find_contradictions(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Find contradictions between claims"""
        contradictions = []

        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], start=i+1):
                # Check for direct contradiction
                if self._are_contradictory(claim1, claim2):
                    contradictions.append({
                        "claim1": claim1,
                        "claim2": claim2,
                        "type": "direct_contradiction",
                        "confidence": 0.9
                    })

                # Check for logical incompatibility
                incompatibility = self._check_logical_compatibility(claim1, claim2)
                if incompatibility:
                    contradictions.append({
                        "claim1": claim1,
                        "claim2": claim2,
                        "type": "logical_incompatibility",
                        "reason": incompatibility,
                        "confidence": 0.7
                    })

        self.known_contradictions.extend(contradictions)
        return contradictions

    async def _tool_cross_reference(self, claim: str, sources: List[Dict[str, Any]]) -> CrossValidation:
        """Cross-reference a claim across sources"""
        sources_checked = []
        agreements = []
        contradictions = []

        for source in sources:
            source_id = source.get("id", f"source_{len(sources_checked)}")
            sources_checked.append(source_id)

            source_content = source.get("content", "")

            # Check if source supports claim
            if self._source_supports_claim(source_content, claim):
                agreements.append({
                    "source": source_id,
                    "support_type": "direct",
                    "excerpt": source_content[:200]
                })
            # Check for contradictions
            elif self._source_contradicts_claim(source_content, claim):
                contradictions.append({
                    "source": source_id,
                    "contradiction_type": "direct",
                    "excerpt": source_content[:200]
                })

        # Calculate consistency score
        total_sources = len(sources_checked)
        if total_sources == 0:
            consistency_score = 0.0
        else:
            consistency_score = len(agreements) / total_sources

        validation = CrossValidation(
            claim_id=f"claim_{len(self.validation_history)}",
            sources_checked=sources_checked,
            consistency_score=consistency_score,
            contradictions=contradictions,
            agreements=agreements,
            confidence=0.8 if consistency_score > 0.7 else 0.4
        )

        return validation

    async def _tool_check_logical_consistency(self, arguments: List[str]) -> Dict[str, Any]:
        """Check logical consistency of arguments"""
        consistency_report = {
            "consistent": True,
            "issues": [],
            "logical_flow": [],
            "recommendations": []
        }

        # Check each argument
        for i, arg in enumerate(arguments):
            # Check for logical fallacies
            fallacies = self._check_logical_fallacies(arg)
            if fallacies:
                consistency_report["consistent"] = False
                consistency_report["issues"].extend([f"Argument {i+1}: {f}" for f in fallacies])

            # Check logical flow between arguments
            if i > 0:
                if not self._follows_logically(arguments[i-1], arg):
                    consistency_report["consistent"] = False
                    consistency_report["issues"].append(f"Logical gap between arguments {i} and {i+1}")

        return consistency_report

    async def _tool_verify_assumptions(self, assumptions: List[str], context: Dict[str, Any]) -> Dict[str, bool]:
        """Verify assumptions in a proof"""
        verified_assumptions = {}

        for assumption in assumptions:
            # Check if assumption is valid in context
            is_valid = await self._is_assumption_valid(assumption, context)
            verified_assumptions[assumption] = is_valid

            if not is_valid:
                self.known_contradictions.append({
                    "type": "invalid_assumption",
                    "assumption": assumption,
                    "context": context
                })

        return verified_assumptions

    async def _tool_generate_counter_example(self, claim: str, domain: str) -> Optional[str]:
        """Try to generate counter-examples"""
        # Use LLM to generate potential counter-examples
        prompt = f"""Generate a counter-example for this claim if possible:

Claim: {claim}
Domain: {domain}

Try to find a specific case where this claim fails.
If no counter-example exists, explain why the claim is likely true.
"""

        messages = [BedrockMessage(role="user", content=prompt)]
        response = await self.bedrock_client.generate(
            messages=messages,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.7  # Some creativity needed for counter-examples
        )

        if "counter-example" in response.content.lower() or "fails when" in response.content.lower():
            return response.content
        return None

    def _check_logical_fallacies(self, text: str) -> List[str]:
        """Check for common logical fallacies"""
        fallacies = []
        text_lower = text.lower()

        # Ad hominem
        if any(word in text_lower for word in ["stupid", "ignorant", "foolish"]):
            fallacies.append("Potential ad hominem attack")

        # Circular reasoning
        if "because" in text_lower and text_lower.count("because") > 2:
            fallacies.append("Possible circular reasoning")

        # False dichotomy
        if "either" in text_lower and "or" in text_lower and "only" in text_lower:
            fallacies.append("Potential false dichotomy")

        # Hasty generalization
        if "all" in text_lower and "always" in text_lower:
            fallacies.append("Possible hasty generalization")

        return fallacies

    async def _check_mathematical_consistency(self, claim: str, context: Dict[str, Any]) -> List[str]:
        """Check mathematical consistency"""
        issues = []

        # Check for mathematical impossibilities
        impossible_patterns = [
            (r"divide[d]?\s+by\s+zero", "Division by zero"),
            (r"square root of\s+(?:a\s+)?negative", "Square root of negative number in real domain"),
            (r"infinity\s+equals\s+\d+", "Infinity cannot equal a finite number")
        ]

        import re
        for pattern, issue in impossible_patterns:
            if re.search(pattern, claim.lower()):
                issues.append(issue)

        return issues

    def _supports_claim(self, evidence: str, claim: str) -> bool:
        """Check if evidence supports claim"""
        # Simplified - use NLP similarity in production
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        overlap = claim_words.intersection(evidence_words)
        return len(overlap) > len(claim_words) * 0.3

    def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are contradictory"""
        # Look for negation patterns
        if "not" in claim1.lower() and "not" not in claim2.lower():
            # Check if claims are about the same subject
            if len(set(claim1.split()).intersection(set(claim2.split()))) > 3:
                return True
        return False

    def _check_logical_compatibility(self, claim1: str, claim2: str) -> Optional[str]:
        """Check logical compatibility between claims"""
        # Simplified logic checking
        if "all" in claim1.lower() and "none" in claim2.lower():
            return "Universal quantifier conflict"
        if "if" in claim1.lower() and "then" in claim1.lower():
            # Check if claim2 violates the implication
            # This would need more sophisticated logic parsing
            pass
        return None

    def _source_supports_claim(self, source: str, claim: str) -> bool:
        """Check if source supports claim"""
        return self._supports_claim(source, claim)

    def _source_contradicts_claim(self, source: str, claim: str) -> bool:
        """Check if source contradicts claim"""
        return "not" in source.lower() and any(word in source.lower() for word in claim.lower().split())

    def _follows_logically(self, premise: str, conclusion: str) -> bool:
        """Check if conclusion follows from premise"""
        # Simplified - would use formal logic in production
        return len(set(premise.split()).intersection(set(conclusion.split()))) > 2

    async def _is_assumption_valid(self, assumption: str, context: Dict[str, Any]) -> bool:
        """Check if assumption is valid in context"""
        # Simplified validation
        return "false" not in assumption.lower() and "invalid" not in assumption.lower()

    def _calculate_skeptical_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate confidence with skeptical bias"""
        base_confidence = 0.3  # Start with low confidence

        # Increase slightly for strong evidence
        if "proven" in response.lower():
            base_confidence += 0.3
        if "verified" in response.lower():
            base_confidence += 0.2

        # Decrease for uncertainty
        uncertainty_terms = ["might", "possibly", "unclear", "unknown", "assumption"]
        for term in uncertainty_terms:
            if term in response.lower():
                base_confidence -= 0.1

        return max(0.0, min(1.0, base_confidence))

    def _extract_verification_evidence(self, text: str) -> List[str]:
        """Extract verification evidence"""
        evidence = []

        # Look for verification markers
        import re
        patterns = [
            r"verified by\s+([^.]+)",
            r"confirmed in\s+([^.]+)",
            r"proven in\s+([^.]+)",
            r"shown by\s+([^.]+)"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                evidence.append(match.group(0))

        return evidence

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issues from review text"""
        issues = []
        issue_keywords = ["error", "mistake", "wrong", "incorrect", "flaw", "problem", "issue"]

        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in issue_keywords):
                issues.append(line.strip())

        return issues

    def _extract_counterexamples(self, text: str) -> List[str]:
        """Extract counter-examples from text"""
        counterexamples = []

        import re
        patterns = [
            r"counter-?example:?\s*([^.]+)",
            r"fails when\s+([^.]+)",
            r"not true for\s+([^.]+)"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                counterexamples.append(match.group(1))

        return counterexamples

    def _extract_gaps(self, text: str) -> List[str]:
        """Extract identified gaps"""
        gaps = []
        gap_keywords = ["missing", "gap", "omitted", "skipped", "unclear"]

        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in gap_keywords):
                gaps.append(line.strip())

        return gaps

    def _extract_assumptions(self, text: str) -> List[str]:
        """Extract assumptions from text"""
        assumptions = []

        import re
        patterns = [
            r"assumes?\s+([^.]+)",
            r"assumption:?\s*([^.]+)",
            r"presumes?\s+([^.]+)"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                assumptions.append(match.group(1))

        return assumptions

    def _determine_validation_status(self, text: str) -> str:
        """Determine validation status from review text"""
        text_lower = text.lower()

        if "invalid" in text_lower or "false" in text_lower:
            return "invalid"
        elif "valid" in text_lower or "correct" in text_lower:
            return "valid"
        elif "uncertain" in text_lower or "unclear" in text_lower:
            return "uncertain"
        else:
            return "requires_verification"

    def _generate_recommendations(self, text: str) -> List[str]:
        """Generate recommendations based on review"""
        recommendations = []

        if "proof" in text.lower() and "missing" in text.lower():
            recommendations.append("Add missing proof steps")
        if "assumption" in text.lower():
            recommendations.append("Verify all assumptions explicitly")
        if "unclear" in text.lower():
            recommendations.append("Clarify ambiguous statements")
        if "counter" in text.lower():
            recommendations.append("Address potential counter-examples")

        return recommendations

    def _generate_proof_suggestions(self, errors: List[str], missing: List[str]) -> List[str]:
        """Generate suggestions for proof improvement"""
        suggestions = []

        if errors:
            suggestions.append(f"Fix {len(errors)} logical errors identified")
        if missing:
            suggestions.append(f"Add {len(missing)} missing steps")

        suggestions.append("Consider adding more detailed justifications")
        suggestions.append("Verify all implicit assumptions")

        return suggestions

    def _assess_severity(self, validation: ValidationResult) -> str:
        """Assess severity of validation issues"""
        if validation.status == "invalid":
            return "critical"
        elif len(validation.issues_found) > 3:
            return "high"
        elif len(validation.issues_found) > 0:
            return "medium"
        else:
            return "low"

    async def review_claim(self, claim: str, evidence: List[str] = None) -> ValidationResult:
        """
        Main method to review a mathematical claim
        """
        logger.info(f"Reviewer {self.agent_id} reviewing claim: {claim[:100]}...")

        context = {
            "task_type": "verify_claim",
            "claim": claim,
            "evidence": evidence or []
        }

        result = await self.process_task(context)
        return result.get("validation", ValidationResult(claim=claim, status="error", confidence=0.0))