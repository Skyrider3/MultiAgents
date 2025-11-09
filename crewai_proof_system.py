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
from pydantic import BaseModel, Field
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

    Args:
        query: The search query string
    """
    # Ensure query is a string
    if not isinstance(query, str):
        # Handle case where query might be a dict (CrewAI sometimes passes structured data)
        if isinstance(query, dict):
            # Extract the actual query string from the dict
            if 'description' in query:
                query = query['description']
            elif 'query' in query:
                query = query['query']
            elif 'value' in query:
                query = query['value']
            else:
                # Try to find any string value in the dict
                for key, value in query.items():
                    if isinstance(value, str):
                        query = value
                        break
                else:
                    query = str(query)  # Last resort: convert to string
    else :
        pass
    # Ensure query is not empty
    if not query or query.strip() == "":
        query = "number theory mathematical conjecture"  # Default fallback

    # Expand query for number theory
    extras = ["multiplicative", "Dirichlet", "mod p", "L-function", "elliptic", "congruence"]
    expanded_query = query + " " + " ".join(extras)

    try:
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

        result = "\n\n---\n\n".join(evidence_text)

        # If no evidence found, return a default message
        if not result:
            return "No specific evidence found in the database. Please proceed with general mathematical knowledge."

        return result

    except Exception as e:
        print(f"Error in retrieve_evidence: {str(e)}")
        return f"Error retrieving evidence: {str(e)}. Proceeding with general knowledge."


@tool("Reduce to Known Lemmas")
def reduce_to_lemmas(conjecture: str, evidence: str = "") -> str:
    """
    Attempt to reduce a conjecture to known lemmas and theorems.
    Returns a structured reduction with references.

    Args:
        conjecture: The mathematical conjecture to reduce
        evidence: Optional evidence to use for reduction
    """
    # Handle case where inputs might be dicts
    if isinstance(conjecture, dict):
        conjecture = conjecture.get('description', conjecture.get('conjecture', str(conjecture)))
    if isinstance(evidence, dict):
        evidence = evidence.get('description', evidence.get('evidence', str(evidence)))
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

        # Task 4: Iterative Verification - USING THE ACTUAL ITERATIVE VERIFIER
        # We need to extract results from previous tasks first
        print("\n🔄 Starting Iterative Verification Process...")

        # Create the iterative verifier with proper back-and-forth
        iterative_verifier = IterativeProofVerifier(max_iterations=self.max_iterations)

        # We'll need to execute the previous tasks first to get their outputs
        # Then use those outputs in the iterative verification
        # For now, create a placeholder task that will trigger the iteration
        verification_task = Task(
            description=f"""
            PLACEHOLDER TASK - Actual iteration happens via IterativeProofVerifier

            This task triggers the iterative verification process between
            Experimenter and Verifier for up to {self.max_iterations} iterations.
            """,
            agent=self.agents["coordinator"],  # Use coordinator as placeholder
            expected_output="Iterative verification results",
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

        # Create and run the crew for initial tasks (conjecture, reduction, initial proof)
        initial_crew = Crew(
            agents=[
                self.agents["number_theorist"],
                self.agents["experimenter"]
            ],
            tasks=[
                conjecture_task,
                reduction_task,
                proof_task
            ],
            process=Process.sequential,  # Tasks run in order
            verbose=True
        )

        # Execute initial tasks
        print("\n" + "="*60)
        print("PHASE 1: Conjecture Generation and Initial Proof")
        print("="*60 + "\n")

        initial_results = initial_crew.kickoff()
        results["initial_phase"] = initial_results

        # Extract conjecture and initial proof from results
        # In production, you'd parse these from the actual outputs
        conjecture_text = "The conjecture generated from the query"  # This should be extracted
        initial_proof = str(initial_results)  # The proof attempt output

        # PHASE 2: Run the actual iterative verification
        print("\n" + "="*60)
        print(f"PHASE 2: Iterative Verification (Max {self.max_iterations} iterations)")
        print("="*60 + "\n")

        iterative_verifier = IterativeProofVerifier(max_iterations=self.max_iterations)
        verification_results = iterative_verifier.verify_proof(
            conjecture=conjecture_text,
            initial_proof=initial_proof
        )

        results["verification"] = verification_results

        # PHASE 3: Final synthesis with all results
        print("\n" + "="*60)
        print("PHASE 3: Final Synthesis")
        print("="*60 + "\n")

        # Update synthesis task with actual results
        synthesis_task.description = f"""
        Synthesize all results into a comprehensive report:

        1. Summary of original query: {user_query}
        2. Generated conjectures from initial phase
        3. Proof attempts and verification results
        4. Final status: {verification_results['status']}
        5. Iterations used: {verification_results['iterations']}
        6. Recommendations for future research

        Verification History:
        {json.dumps(verification_results['history'], indent=2)}

        Format as a structured JSON report.
        """

        final_crew = Crew(
            agents=[self.agents["coordinator"]],
            tasks=[synthesis_task],
            process=Process.sequential,
            verbose=True
        )

        final_result = final_crew.kickoff()

        # Parse and return results
        results["final_output"] = final_result
        results["status"] = verification_results["status"]

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
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} of {self.max_iterations}")
            print(f"{'='*60}\n")

            # STEP 1: Verifier checks the proof
            print(f"📝 VERIFIER: Checking proof attempt...")
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

            # Execute verification using a mini-crew
            verification_crew = Crew(
                agents=[self.verifier],
                tasks=[verification_task],
                process=Process.sequential,
                verbose=False
            )
            verification_result = verification_crew.kickoff()

            print(f"\n🔍 VERIFIER VERDICT: {verification_result[:200]}...")  # Show first 200 chars

            proof_history.append({
                "iteration": iteration + 1,
                "proof": current_proof,
                "verification": verification_result
            })

            # Check if proof is valid
            if "VALID" in str(verification_result).upper():
                print(f"\n✅ PROOF VALIDATED! Stopping after {iteration + 1} iterations.")
                return {
                    "status": "PROVEN",
                    "iterations": iteration + 1,
                    "history": proof_history,
                    "final_proof": current_proof
                }

            # If not valid and not last iteration, refine
            if iteration < self.max_iterations - 1:
                print(f"\n🔧 EXPERIMENTER: Refining proof based on feedback...")
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

                # Execute refinement using a mini-crew
                refinement_crew = Crew(
                    agents=[self.experimenter],
                    tasks=[refinement_task],
                    process=Process.sequential,
                    verbose=False
                )
                current_proof = refinement_crew.kickoff()

                print(f"\n📝 EXPERIMENTER OUTPUT: {str(current_proof)[:200]}...")  # Show first 200 chars

        # Max iterations reached without valid proof
        print(f"\n⚠️ Maximum iterations ({self.max_iterations}) reached without valid proof.")
        print("Status: REMAINS CONJECTURE")
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