"""
Simplified CrewAI Demo for Mathematical Proof Workflow
Demonstrates the iterative conversation between Experimenter and Verifier
"""

import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Simple example without external dependencies on existing code
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def create_simple_proof_crew(query: str, max_iterations: int = 5):
    """
    Create a crew that implements the workflow:
    Query → Number Theorist → Experimenter ↔ Verifier → Result
    """

    # 1. Number Theory Agent
    number_theorist = Agent(
        role="Number Theory Expert",
        goal="Generate mathematical conjectures from queries",
        backstory="You are an expert in number theory who creates precise conjectures.",
        verbose=True,
        llm=llm
    )

    # 2. Experimenter Agent
    experimenter = Agent(
        role="Mathematical Experimenter",
        goal="Create proofs and test conjectures computationally",
        backstory="""You design experiments and attempt proofs.
        You respond to verifier feedback by improving your proofs.""",
        verbose=True,
        llm=llm
    )

    # 3. Verifier Agent
    verifier = Agent(
        role="Proof Verifier",
        goal="Rigorously verify mathematical proofs",
        backstory="""You are a strict verifier who checks every logical step.
        You provide clear verdicts: VALID, INVALID, or INCOMPLETE.""",
        verbose=True,
        llm=llm
    )

    # Task 1: Generate Conjectures
    conjecture_task = Task(
        description=f"""
        Based on this query: "{query}"

        Generate 2 mathematical conjectures. For each:
        1. State the conjecture clearly in mathematical notation
        2. Explain why it's interesting
        3. Suggest how to approach proving it

        Format as numbered list.
        """,
        agent=number_theorist,
        expected_output="List of 2 conjectures"
    )

    # Task 2: Reduce to Known Lemmas
    reduction_task = Task(
        description="""
        Take the first conjecture from above and:
        1. Identify what known theorems might help
        2. Break it into smaller steps
        3. List what needs to be proven

        Be specific about the mathematical structure.
        """,
        agent=number_theorist,
        expected_output="Reduction to known lemmas",
        context=[conjecture_task]
    )

    # Task 3: Initial Proof Attempt
    initial_proof_task = Task(
        description="""
        Based on the conjecture and its reduction:
        1. Write a proof attempt
        2. Include computational tests if applicable
        3. Be explicit about each logical step

        Your proof will be verified, so be thorough.
        """,
        agent=experimenter,
        expected_output="Initial proof attempt",
        context=[reduction_task]
    )

    # Task 4: Iterative Verification (simulated with one comprehensive task)
    iterative_verification_task = Task(
        description=f"""
        ITERATIVE PROOF VERIFICATION PROCESS:

        You will now simulate {max_iterations} iterations of verification:

        For each iteration:
        1. VERIFIER: Check the current proof for logical errors
        2. VERIFIER: Give verdict (VALID/INVALID/INCOMPLETE) with specific feedback
        3. If INVALID/INCOMPLETE:
           - EXPERIMENTER: Revise the proof based on feedback
        4. If VALID or max iterations reached: Stop

        Show the conversation as:

        ITERATION 1:
        Verifier: [feedback and verdict]
        Experimenter: [revised proof if needed]

        ITERATION 2:
        ...

        Final Status: PROVEN or REMAINS CONJECTURE
        """,
        agent=verifier,  # Verifier leads the iterative process
        expected_output=f"Record of up to {max_iterations} verification iterations",
        context=[initial_proof_task]
    )

    # Task 5: Final Report
    final_report_task = Task(
        description="""
        Create a final summary:
        1. Original query
        2. Generated conjectures
        3. Proof verification outcome
        4. Status: PROVEN, DISPROVEN, or REMAINS CONJECTURE
        5. Key insights learned

        Format as a clear, structured report.
        """,
        agent=number_theorist,  # Number theorist synthesizes
        expected_output="Final summary report",
        context=[conjecture_task, iterative_verification_task]
    )

    # Create the crew
    crew = Crew(
        agents=[number_theorist, experimenter, verifier],
        tasks=[
            conjecture_task,
            reduction_task,
            initial_proof_task,
            iterative_verification_task,
            final_report_task
        ],
        process=Process.sequential,
        verbose=True
    )

    return crew


def run_proof_workflow(query: str):
    """
    Run the complete proof workflow for a given query.
    """
    print(f"\n{'='*60}")
    print(f"STARTING PROOF WORKFLOW")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Create and run the crew
    crew = create_simple_proof_crew(query, max_iterations=3)
    result = crew.kickoff()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "query": query,
        "timestamp": timestamp,
        "result": result
    }

    filename = f"crewai_proof_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"WORKFLOW COMPLETE")
    print(f"Results saved to: {filename}")
    print(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    # Example queries to test
    test_queries = [
        "Prove that there are infinitely many primes p where p ≡ 1 (mod 4)",
        "Distribution of gaps between consecutive prime numbers",
        "Density of integers n where tau(n) is prime"
    ]

    # Run the first query
    result = run_proof_workflow(test_queries[0])
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))