"""
CrewAI-based Mathematical Proof System
This implements the workflow:
User Query → Number Theory Agent → Conjectures → Reduce to Lemmas →
Experimenter → Verifier (5 iterations) → Result
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import existing retrieval functions
from agents import semantic_retrieve, symbol_retrieve

load_dotenv()

# Initialize LLM (using GPT-4 for better reasoning)
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Can upgrade to gpt-4 for better performance
    temperature=0.7
)

# Custom tools for agents
@tool("Retrieve Evidence")
def retrieve_evidence(query: str) -> str:
    """
    Retrieve relevant evidence from the paper database
    using both semantic and symbol-based search.
    """
    # Expand query for number theory
    extras = ["multiplicative", "Dirichlet", "mod p", "L-function", "elliptic", "congruence"]
    expanded_query = query + " " + " ".join(extras)

    # Get semantic hits
    sem_hits = semantic_retrieve(expanded_query, k=8)

    # Extract symbols from query
    import re
    symbols = re.findall(r"\\[A-Za-z]+\{[^}]*\}", query)
    symbols += re.findall(r"[a-zA-Z]_\{?[0-9nkp]+\}?", query)

    # Get symbol hits
    sym_hits = symbol_retrieve(symbols) if symbols else []

    # Merge evidence
    evidence = sem_hits + sym_hits

    # Format evidence
    evidence_text = []
    for e in evidence[:10]:  # Limit to top 10
        evidence_text.append(f"Source: {e['meta'].get('source', 'unknown')}\n{e['text'][:500]}")

    return "\n\n---\n\n".join(evidence_text)


@tool("Reduce to Known Lemmas")
def reduce_to_lemmas(conjecture: str, evidence: str) -> str:
    """
    Attempt to reduce a conjecture to known lemmas and theorems.
    Returns a structured reduction with references.
    """
    # This would ideally use a symbolic math engine
    # For now, we'll use LLM-based reduction
    reduction_prompt = f"""
    Given the following conjecture and evidence, identify:
    1. Known lemmas/theorems that could be applied
    2. Required intermediate steps
    3. Missing pieces that need to be proven

    Conjecture: {conjecture}

    Evidence:
    {evidence[:2000]}

    Provide a structured reduction in JSON format.
    """

    # In a real implementation, this would call a symbolic reasoning system
    return json.dumps({
        "known_lemmas": ["Placeholder for lemma detection"],
        "reduction_steps": ["Step 1", "Step 2"],
        "missing_pieces": ["Gap 1", "Gap 2"]
    })


# Define Agents
class MathAgents:

    @staticmethod
    def create_number_theory_agent():
        return Agent(
            role="Expert Number Theory Mathematician",
            goal="Analyze user queries and generate precise mathematical conjectures",
            backstory="""You are a world-renowned number theorist with expertise in:
            - Analytic number theory
            - Algebraic number theory
            - Ramanujan tau functions
            - L-functions and zeta functions
            - Modular forms

            You carefully analyze queries and generate well-formed conjectures that are:
            1. Mathematically precise (in LaTeX)
            2. Testable
            3. Grounded in existing theory
            4. Novel yet plausible""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[retrieve_evidence, reduce_to_lemmas]
        )

    @staticmethod
    def create_experimenter_agent():
        return Agent(
            role="Mathematical Experimenter",
            goal="Design and execute computational experiments to test conjectures",
            backstory="""You are an expert in computational number theory with skills in:
            - SageMath programming
            - PARI/GP
            - Python numerical computing
            - Efficient algorithm design
            - Statistical analysis of numerical patterns

            You create rigorous test plans that:
            1. Start with small test cases
            2. Scale to large numbers efficiently
            3. Include edge cases
            4. Generate counterexamples when found
            5. Provide statistical confidence measures""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    @staticmethod
    def create_verifier_agent():
        return Agent(
            role="Mathematical Proof Verifier",
            goal="Rigorously verify proof attempts and identify logical gaps",
            backstory="""You are a mathematical logician specializing in:
            - Formal proof verification
            - Lean/Coq/Isabelle theorem provers
            - Identifying logical fallacies
            - Proof by contradiction
            - Mathematical induction

            You critically examine proofs by:
            1. Checking each logical step
            2. Identifying unstated assumptions
            3. Verifying citations and lemma applications
            4. Suggesting corrections or alternative approaches
            5. Providing clear verdicts: VALID, INVALID, or INCOMPLETE""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    @staticmethod
    def create_coordinator_agent():
        return Agent(
            role="Research Coordinator",
            goal="Orchestrate the proof process and synthesize results",
            backstory="""You coordinate mathematical research by:
            - Managing the proof workflow
            - Tracking iteration progress
            - Synthesizing agent outputs
            - Making go/no-go decisions
            - Producing final research reports""",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )


class ProofSystem:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.agents = self._create_agents()

    def _create_agents(self):
        return {
            "number_theorist": MathAgents.create_number_theory_agent(),
            "experimenter": MathAgents.create_experimenter_agent(),
            "verifier": MathAgents.create_verifier_agent(),
            "coordinator": MathAgents.create_coordinator_agent()
        }

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point for processing a mathematical query.
        """
        results = {
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "conjectures": [],
            "proofs": [],
            "status": "in_progress"
        }

        # Task 1: Generate Conjectures
        conjecture_task = Task(
            description=f"""
            Analyze the following query and generate up to 3 precise mathematical conjectures:

            Query: {user_query}

            For each conjecture provide:
            1. LaTeX statement
            2. Intuitive explanation
            3. Connection to known results
            4. Suggested proof strategy

            Use the retrieve_evidence tool to gather relevant information.
            Format output as JSON.
            """,
            agent=self.agents["number_theorist"],
            expected_output="JSON formatted conjectures with evidence"
        )

        # Task 2: Reduce to Lemmas
        reduction_task = Task(
            description="""
            Take the conjectures from the previous task and:
            1. Identify known lemmas that could be applied
            2. Break down into smaller subproblems
            3. Highlight gaps that need to be proven

            Use the reduce_to_lemmas tool for formal reduction.
            """,
            agent=self.agents["number_theorist"], # TODO: make reduction agent
            expected_output="Structured reduction to known lemmas",
            context=[conjecture_task]
        )

        # Task 3: Initial Proof Attempt
        proof_task = Task(
            description="""
            Based on the conjecture and its reduction to lemmas:
            1. Design computational experiments to test the conjecture
            2. Generate test code in SageMath/Python
            3. Run tests on small cases
            4. Attempt to construct a proof or find counterexamples

            Provide detailed reasoning for each step.
            """,
            agent=self.agents["experimenter"],
            expected_output="Proof attempt with experimental evidence",
            context=[reduction_task]
        )

        # Task 4: Iterative Verification
        verification_task = Task(
            description=f"""
            Engage in iterative proof refinement:

            1. Verify the proof attempt from the experimenter
            2. Identify logical gaps or errors
            3. Suggest corrections
            4. Iterate up to {self.max_iterations} times

            For each iteration:
            - Verifier checks the proof
            - If invalid, provide specific feedback
            - Experimenter revises based on feedback
            - Continue until proof is valid or max iterations reached

            Final output should indicate:
            - PROVEN: Valid proof found
            - CONJECTURE: No valid proof found after {self.max_iterations} iterations
            - DISPROVEN: Counterexample found
            """,
            agent=self.agents["verifier"],
            expected_output="Verification result with detailed feedback",
            context=[proof_task]
        )

        # Task 5: Final Synthesis
        synthesis_task = Task(
            description="""
            Synthesize all results into a comprehensive report:

            1. Summary of original query
            2. Generated conjectures
            3. Proof attempts and verification results
            4. Final status for each conjecture
            5. Recommendations for future research

            Format as a structured JSON report.
            """,
            agent=self.agents["coordinator"],
            expected_output="Final JSON report",
            context=[conjecture_task, reduction_task, proof_task, verification_task]
        )

        # Create and run the crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[
                conjecture_task,
                reduction_task,
                proof_task,
                verification_task,
                synthesis_task
            ],
            process=Process.sequential,  # Tasks run in order
            verbose=True
        )

        # Execute the crew
        result = crew.kickoff()

        # Parse and return results
        results["final_output"] = result
        results["status"] = "completed"

        # Save to file
        filename = f"proof_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filename}")

        return results


# Specialized version for iterative proof verification
class IterativeProofVerifier:
    """
    Handles the iterative conversation between experimenter and verifier.
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.experimenter = MathAgents.create_experimenter_agent()
        self.verifier = MathAgents.create_verifier_agent()

    def verify_proof(self, conjecture: str, initial_proof: str) -> Dict[str, Any]:
        """
        Iteratively verify and refine a proof.
        """
        proof_history = []
        current_proof = initial_proof

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Verifier checks the proof
            verification_task = Task(
                description=f"""
                Verify the following proof for the conjecture:

                Conjecture: {conjecture}

                Proof Attempt:
                {current_proof}

                Provide:
                1. Verdict: VALID, INVALID, or INCOMPLETE
                2. Specific issues found (if any)
                3. Suggestions for improvement

                Be extremely rigorous in your verification.
                """,
                agent=self.verifier,
                expected_output="Verification verdict with detailed feedback"
            )

            verification_result = verification_task.execute()

            proof_history.append({
                "iteration": iteration + 1,
                "proof": current_proof,
                "verification": verification_result
            })

            # Check if proof is valid
            if "VALID" in verification_result.upper():
                return {
                    "status": "PROVEN",
                    "iterations": iteration + 1,
                    "history": proof_history,
                    "final_proof": current_proof
                }

            # If not valid and not last iteration, refine
            if iteration < self.max_iterations - 1:
                refinement_task = Task(
                    description=f"""
                    Refine your proof based on the verifier's feedback:

                    Verifier Feedback:
                    {verification_result}

                    Original Conjecture: {conjecture}

                    Provide an improved proof that addresses all issues raised.
                    """,
                    agent=self.experimenter,
                    expected_output="Refined proof attempt"
                )

                current_proof = refinement_task.execute()

        # Max iterations reached without valid proof
        return {
            "status": "CONJECTURE",
            "iterations": self.max_iterations,
            "history": proof_history,
            "final_proof": current_proof
        }


def main():
    """
    Example usage of the proof system.
    """
    # Initialize the system
    proof_system = ProofSystem(max_iterations=5)

    # Example queries
    example_queries = [
        "distribution of zeros of Ramanujan tau(n) modulo small primes",
        "density of primes p where tau(p) ≡ 0 (mod p)",
        "asymptotic behavior of sum of tau(n) for n up to X"
    ]

    # Process the first query
    query = example_queries[0]
    print(f"Processing query: {query}")

    results = proof_system.process_query(query)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
