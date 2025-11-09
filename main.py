"""
CrewAI-based Mathematical Proof System
This implements the workflow:
User Query → Number Theory Agent → Conjectures → Reduce to Lemmas →
Experimenter → Verifier (5 iterations) → Result
"""

import os
import json
import re
from typing import List, Dict, Any
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import existing retrieval functions
from retriever import semantic_retrieve, symbol_retrieve

load_dotenv()

# Initialize LLM (using GPT-4 for better reasoning)
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Can upgrade to gpt-4 for better performance
    temperature=0.7
)

verifier_llm = ChatOpenAI(
    model="gpt-4o",  # Can upgrade to gpt-4 for better performance
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
    def create_algebraic_topology_agent():
        return Agent(
            role="Expert Algebraic Topology Mathematician",
            goal="Analyze user queries and generate precise mathematical conjectures in algebraic topology",
            backstory="""You are a world-renowned mathematician specializing in algebraic topology with expertise in:
            - Homology and cohomology theory
            - Homotopy theory and fundamental groups
            - Spectral sequences
            - K-theory (topological and algebraic)
            - Characteristic classes
            - Fiber bundles and fibrations
            - Manifold topology
            - Category theory in topology
            - Homological algebra
            - Topological groups and Lie groups
            - Differential topology
            - Cobordism theory

            You carefully analyze queries and generate well-formed conjectures that are:
            1. Mathematically precise (in LaTeX)
            2. Testable through computational or theoretical methods
            3. Grounded in existing algebraic topology theory
            4. Novel yet plausible
            5. Suitable for proof by topological or homological techniques""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[retrieve_evidence, reduce_to_lemmas]
        )

    @staticmethod
    def create_number_theory_experimenter():
        return Agent(
            role="Number Theory Experimenter",
            goal="Design and execute computational experiments to test number theory conjectures",
            backstory="""You are an expert in computational number theory with skills in:
            - SageMath programming for number theory
            - PARI/GP for arithmetic computations
            - Python numerical computing
            - Efficient algorithm design for large numbers
            - Statistical analysis of numerical patterns
            - Prime number computations
            - Modular arithmetic and congruences
            - L-function computations
            - Distribution analysis

            You create rigorous test plans for number theory that:
            1. Start with small test cases (small primes, small n)
            2. Scale to large numbers efficiently
            3. Include edge cases (primes, powers, special numbers)
            4. Generate counterexamples when found
            5. Provide statistical confidence measures
            6. Analyze asymptotic behavior
            7. Test modular properties and distributions
            8. Verify patterns across different number ranges""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    @staticmethod
    def create_algebraic_topology_experimenter():
        return Agent(
            role="Algebraic Topology Experimenter",
            goal="Design and execute computational experiments to test algebraic topology conjectures",
            backstory="""You are an expert in computational algebraic topology with skills in:
            - SageMath for topological computations
            - Computational topology libraries (GUDHI, Dionysus, Ripser)
            - Persistent homology computations
            - Homology and cohomology calculations
            - Python for topological data analysis
            - Simplicial complex computations
            - Homotopy group calculations
            - Spectral sequence computations
            - Characteristic class calculations
            - Manifold topology verification
            - K-theory computations

            You create rigorous test plans for algebraic topology that:
            1. Start with simple spaces (spheres, tori, projective spaces)
            2. Test on canonical examples and counterexamples
            3. Include edge cases (contractible spaces, discrete spaces)
            4. Generate counterexamples when found
            5. Compute homology and cohomology groups
            6. Verify homotopy invariants
            7. Check topological properties (connectedness, compactness, etc.)
            8. Test homological properties (exact sequences, long exact sequences)
            9. Verify characteristic classes and invariants
            10. Compute fundamental groups and higher homotopy groups""",
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
            llm=verifier_llm
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
    def __init__(self, max_iterations: int = 5, agent_type: str = "number_theory"):
        """
        Initialize the proof system.
        
        Args:
            max_iterations: Maximum number of verification iterations
            agent_type: Type of agent to use - "number_theory" or "algebraic_topology"
        """
        self.max_iterations = max_iterations
        self.agent_type = agent_type
        self.agents = self._create_agents()

    def _create_agents(self):
        """Create agents based on the selected agent type."""
        if self.agent_type == "algebraic_topology":
            conjecture_agent = MathAgents.create_algebraic_topology_agent()
            experimenter_agent = MathAgents.create_algebraic_topology_experimenter()
        else:
            conjecture_agent = MathAgents.create_number_theory_agent()
            experimenter_agent = MathAgents.create_number_theory_experimenter()
        
        return {
            "conjecture_agent": conjecture_agent,
            "number_theorist": MathAgents.create_number_theory_agent(),  # Keep for compatibility
            "algebraic_topology": MathAgents.create_algebraic_topology_agent(),  # Keep for reference
            "experimenter": experimenter_agent,
            "number_theory_experimenter": MathAgents.create_number_theory_experimenter(),  # Keep for reference
            "algebraic_topology_experimenter": MathAgents.create_algebraic_topology_experimenter(),  # Keep for reference
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

        # Step 1: Generate list of conjectures
        conjecture_task = Task(
            description=f"""
            Analyze the following query and generate a list of 2-5 precise mathematical conjectures:

            Query: {user_query}

            For each conjecture, provide in JSON format:
            1. "LaTeX": The mathematical statement in LaTeX format
            2. "explanation": Intuitive explanation
            3. "connection": Connection to known results

            Use the retrieve_evidence tool to gather relevant information.
            
            IMPORTANT: Return your response as a valid JSON array of objects, where each object represents one conjecture.
            Example format:
            [
              {{
                "LaTeX": "Conjecture statement in LaTeX",
                "explanation": "Explanation text",
                "connection": "Connection to known results"
              }},
              ...
            ]
            """,
            agent=self.agents["conjecture_agent"],
            expected_output="JSON array of conjectures, each with LaTeX, explanation, connection, and proof_strategy fields"
        )

        # Generate conjectures
        conjecture_crew = Crew(
            agents=[self.agents["conjecture_agent"]],
            tasks=[conjecture_task],
            process=Process.sequential,
            verbose=True
        )
        conjecture_crew.kickoff()
        
        # Parse conjectures from output
        conjectures_list = self._parse_conjectures(conjecture_task.output)
        results["conjectures"] = conjectures_list
        
        if not conjectures_list:
            print("Warning: No conjectures were generated. Ending process.")
            results["status"] = "failed"
            results["error"] = "No conjectures generated"
            return results
        
        print(f"\n{'='*60}")
        print(f"Generated {len(conjectures_list)} conjectures")
        print(f"{'='*60}\n")
        
        # Step 2: Process each conjecture individually
        conjecture_results = []
        theorems = []
        remaining_conjectures = []
        
        for idx, conj_data in enumerate(conjectures_list, 1):
            print(f"\n{'='*60}")
            print(f"Processing Conjecture {idx}/{len(conjectures_list)}")
            print(f"{'='*60}")
            
            # Extract conjecture text
            conjecture_text = conj_data.get("LaTeX", conj_data.get("statement", ""))
            if not conjecture_text:
                print(f"Warning: Conjecture {idx} has no text, skipping...")
                continue
            
            print(f"Conjecture: {conjecture_text[:100]}...")
            
            # Create reduction task for this specific conjecture
            reduction_task = Task(
                description=f"""
                For the following conjecture, identify:
                1. Known lemmas/theorems that could be applied
                2. Break down into smaller subproblems
                3. Highlight gaps that need to be proven
                
                Conjecture: {conjecture_text}
                
                Use the reduce_to_lemmas tool for formal reduction.
                """,
                agent=self.agents["conjecture_agent"],
                expected_output="Structured reduction to known lemmas"
            )
            
            # Create proof task for this specific conjecture
            proof_task = Task(
                description=f"""
                For the following conjecture, attempt to construct a rigorous mathematical proof:
                
                Conjecture: {conjecture_text}
                
                1. Design computational experiments to test the conjecture
                2. Generate test code in SageMath/Python
                3. Run tests on small cases
                4. Attempt to construct a rigorous mathematical proof
                
                Provide detailed reasoning for each step.
                """,
                agent=self.agents["experimenter"],
                expected_output="Proof attempt with experimental evidence",
                context=[reduction_task]
            )
            
            # Run reduction and proof for this conjecture
            proof_crew = Crew(
                agents=[self.agents["conjecture_agent"], self.agents["experimenter"]],
                tasks=[reduction_task, proof_task],
                process=Process.sequential,
                verbose=True
            )
            proof_crew.kickoff()
            
            # Get the proof attempt
            initial_proof = str(proof_task.output) if proof_task.output else "No proof generated"
            
            # Verify the proof iteratively
            verifier_loop = IterativeProofVerifier(max_iterations=self.max_iterations, agent_type=self.agent_type)
            verification_result = verifier_loop.verify_proof(conjecture_text, initial_proof)
            
            # Determine status: THEOREM if proven, CONJECTURE if not
            if verification_result["status"] == "PROVEN":
                conj_data["status"] = "THEOREM"
                conj_data["proof"] = verification_result["final_proof"]
                conj_data["verification_iterations"] = verification_result["iterations"]
                conj_data["verification_result"] = verification_result.get("verification_result", "")
                theorems.append(conj_data)
                print(f"\n✓ Conjecture {idx} PROVEN → Now a THEOREM")
            else:
                conj_data["status"] = "CONJECTURE"
                conj_data["verification_history"] = verification_result["history"]
                conj_data["verification_iterations"] = verification_result["iterations"]
                conj_data["final_proof_attempt"] = verification_result["final_proof"]
                remaining_conjectures.append(conj_data)
                print(f"\n✗ Conjecture {idx} remains a CONJECTURE (not proven after {verification_result['iterations']} iterations)")
            
            conjecture_results.append({
                "conjecture_number": idx,
                "conjecture": conj_data,
                "verification": verification_result
            })
        
        results["conjecture_results"] = conjecture_results
        results["theorems"] = theorems
        results["remaining_conjectures"] = remaining_conjectures
        
        # Step 3: Generate final synthesis report
        synthesis_task = Task(
            description=f"""
            Create a comprehensive final report summarizing:
            
            1. Original query: {user_query}
            2. Total conjectures generated: {len(conjectures_list)}
            3. THEOREMS (proven): {len(theorems)}
               {self._format_conjecture_list(theorems)}
            4. CONJECTURES (unproven): {len(remaining_conjectures)}
               {self._format_conjecture_list(remaining_conjectures)}
            5. For each theorem: provide the proof and verification details
            6. For each conjecture: explain why it remains unproven and suggest future research directions
            
            Format as a structured JSON report with clear distinction between theorems and conjectures.
            The report should clearly identify which statements are proven (theorems) vs unproven (conjectures).
            """,
            agent=self.agents["coordinator"],
            expected_output="Final JSON report with theorems and conjectures clearly identified"
        )
        
        synthesis_crew = Crew(
            agents=[self.agents["coordinator"]],
            tasks=[synthesis_task],
            process=Process.sequential,
            verbose=True
        )
        final_report = synthesis_crew.kickoff()
        
        results["final_output"] = str(final_report) if final_report else "No report generated"
        results["status"] = "completed"
        results["summary"] = {
            "total_conjectures": len(conjectures_list),
            "theorems_proven": len(theorems),
            "conjectures_remaining": len(remaining_conjectures)
        }

        
        # Save to file
        filename = f"proof_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filename}")

        return results
    
    def _parse_conjectures(self, conjecture_output) -> List[Dict]:
        """
        Parse conjectures from the task output.
        Handles both JSON and text formats.
        """
        if not conjecture_output:
            return []
        
        conjecture_text = str(conjecture_output)
        conjectures = []
        
        # Try to parse as JSON first
        try:
            # Look for JSON array or object in the output
            json_match = re.search(r'\[.*\]', conjecture_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    # Validate structure
                    for conj in parsed:
                        if isinstance(conj, dict):
                            conjectures.append(conj)
                    if conjectures:
                        return conjectures
            
            # Try JSON object with "conjectures" key
            json_match = re.search(r'\{.*\}', conjecture_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "conjectures" in parsed:
                    return parsed["conjectures"]
                elif isinstance(parsed, list):
                    return parsed
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"JSON parsing failed: {e}, trying text parsing...")
        
        # Fallback: parse from text format
        # Look for numbered conjectures or LaTeX statements
        pattern = r'(?:Conjecture\s+)?(\d+)\.\s*(.*?)(?=\n\s*(?:\d+\.|Conjecture|$))'
        matches = re.findall(pattern, conjecture_text, re.DOTALL | re.IGNORECASE)
        
        for num, text in matches:
            # Extract LaTeX if present
            latex_match = re.search(r'\\[a-zA-Z]+\([^)]*\)[^.]*\.', text)
            latex = latex_match.group(0) if latex_match else text.strip()[:200]
            
            conjectures.append({
                "number": int(num),
                "LaTeX": latex,
                "statement": text.strip()[:500],
                "explanation": text.strip(),
                "status": "UNKNOWN"
            })
        
        # If no numbered format, try to extract from JSON-like structures in text
        if not conjectures:
            # Look for "LaTeX" fields
            latex_pattern = r'"LaTeX":\s*"([^"]+)"'
            latex_matches = re.findall(latex_pattern, conjecture_text)
            for idx, latex in enumerate(latex_matches, 1):
                conjectures.append({
                    "number": idx,
                    "LaTeX": latex,
                    "statement": latex,
                    "explanation": f"Conjecture {idx}",
                    "status": "UNKNOWN"
                })
        
        # Last resort: treat entire output as single conjecture
        if not conjectures:
            conjectures.append({
                "number": 1,
                "LaTeX": conjecture_text[:200],
                "statement": conjecture_text[:500],
                "explanation": conjecture_text,
                "status": "UNKNOWN"
            })
        
        return conjectures
    
    def _format_conjecture_list(self, conj_list: List[Dict]) -> str:
        """Format a list of conjectures for the synthesis task description."""
        if not conj_list:
            return "   None"
        
        formatted = []
        for idx, conj in enumerate(conj_list, 1):
            latex = conj.get("LaTeX", conj.get("statement", f"Conjecture {idx}"))
            # Truncate if too long
            if len(latex) > 150:
                latex = latex[:150] + "..."
            formatted.append(f"   {idx}. {latex}")
        
        return "\n".join(formatted)


# Specialized version for iterative proof verification
class IterativeProofVerifier:
    """
    Handles the iterative conversation between experimenter and verifier.
    """

    def __init__(self, max_iterations: int = 5, agent_type: str = "number_theory"):
        self.max_iterations = max_iterations
        self.agent_type = agent_type
        if agent_type == "algebraic_topology":
            self.experimenter = MathAgents.create_algebraic_topology_experimenter()
        else:
            self.experimenter = MathAgents.create_number_theory_experimenter()
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

            # Create a crew to execute the verification task
            verification_crew = Crew(
                agents=[self.verifier],
                tasks=[verification_task],
                process=Process.sequential,
                verbose=True
            )
            verification_result_obj = verification_crew.kickoff()
            verification_result = str(verification_result_obj) if verification_result_obj else ""

            proof_history.append({
                "iteration": iteration + 1,
                "proof": current_proof,
                "verification": verification_result
            })

            # Check if proof is valid (more robust check)
            verification_upper = verification_result.upper()
            is_valid = False
            
            # Check for explicit "VALID" verdict, but not "INVALID" or "INCOMPLETE"
            if "VERDICT: VALID" in verification_upper or "VERDICT IS VALID" in verification_upper:
                is_valid = True
            elif "VALID" in verification_upper:
                # Make sure it's not part of "INVALID" or "INCOMPLETE"
                if "INVALID" not in verification_upper and "INCOMPLETE" not in verification_upper:
                    # Check if "VALID" appears as a standalone word
                    if re.search(r'\bVALID\b', verification_upper):
                        is_valid = True
            
            if is_valid:
                print(f"✓ Proof verified as VALID in iteration {iteration + 1}")
                return {
                    "status": "PROVEN",
                    "iterations": iteration + 1,
                    "history": proof_history,
                    "final_proof": current_proof,
                    "verification_result": verification_result
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

                # Create a crew to execute the refinement task
                refinement_crew = Crew(
                    agents=[self.experimenter],
                    tasks=[refinement_task],
                    process=Process.sequential,
                    verbose=True
                )
                refinement_result = refinement_crew.kickoff()
                current_proof = str(refinement_result) if refinement_result else current_proof

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
    import sys
    
    # Determine agent type from command line or use default
    agent_type = "number_theory"  # default
    if len(sys.argv) > 1:
        agent_type = sys.argv[1]  # "number_theory" or "algebraic_topology"
    
    # Initialize the system with selected agent type
    proof_system = ProofSystem(max_iterations=5, agent_type=agent_type)
    
    print(f"Using agent type: {agent_type}")
    print("="*60)

    # Example queries based on agent type
    if agent_type == "algebraic_topology":
        example_queries = [
            "homology groups of product spaces",
            "homotopy groups of spheres",
            "characteristic classes of vector bundles",
            "K-theory of topological spaces",
            "fundamental groups of manifolds"
        ]
    else:
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
