# Conjecture Generation System

A crewAI-based system for generating scientific conjectures about a given topic using RAG (Retrieval-Augmented Generation).

## Features

- **RAG-based Retrieval**: Uses FAISS index to retrieve relevant documents from research papers
- **CrewAI-based Multi-Agent System**: Multiple specialized agents working together
- **Specialized Agents**: 
  - Number Theory Agent: For number theory and arithmetic problems
  - Algebraic Topology Agent: For algebraic topology and topological mathematics
- **Individual Conjecture Processing**: Each conjecture is processed and verified separately
- **Theorem vs Conjecture Classification**: Automatically distinguishes proven theorems from unproven conjectures
- **Interactive CLI**: Simple command-line interface for querying topics
- **Programmatic API**: Use the system programmatically in your own code

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the `MultiAgents` directory with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Ensure the FAISS index files are present in the `index/` directory:
   - `faiss.index`
   - `meta.pkl`
   - `symbol_index.json`

## Usage

### Using Number Theory Agent (Default)

Run the main script with number theory agent:
```bash
python main.py number_theory
```

Or simply:
```bash
python main.py
```

### Using Algebraic Topology Agent

Run the main script with algebraic topology agent:
```bash
python main.py algebraic_topology
```

Or use the example script:
```bash
python example_algebraic_topology.py
```

### What the System Does

For each query, the system will:
1. Generate a list of conjectures (2-5 conjectures)
2. Process each conjecture individually:
   - Reduce to known lemmas
   - Attempt to construct a proof
   - Verify the proof iteratively (up to 5 iterations)
3. Classify results:
   - **THEOREMS**: Conjectures with valid proofs
   - **CONJECTURES**: Conjectures that remain unproven
4. Generate a comprehensive final report

### Programmatic Usage

```python
from main import ProofSystem

# Initialize with number theory agent
proof_system = ProofSystem(max_iterations=5, agent_type="number_theory")

# Or with algebraic topology agent
proof_system = ProofSystem(max_iterations=5, agent_type="algebraic_topology")

# Process a query
results = proof_system.process_query("your query here")

# Access results
print(f"Total conjectures: {results['summary']['total_conjectures']}")
print(f"Theorems proven: {results['summary']['theorems_proven']}")
print(f"Conjectures remaining: {results['summary']['conjectures_remaining']}")

# Access theorems
for theorem in results['theorems']:
    print(f"Theorem: {theorem['LaTeX']}")
    print(f"Proof: {theorem['proof']}")

# Access remaining conjectures
for conj in results['remaining_conjectures']:
    print(f"Conjecture: {conj['LaTeX']}")
```

See `example_algebraic_topology.py` for a complete example with the algebraic topology agent.

## Example

```
Topic: quantum computing
Generating conjectures about: quantum computing
--------------------------------------------------

Retrieved Documents:
  1. 2511.03458v1.pdf (page 183)
  2. 2511.03987v1.pdf (page 6)
  3. 2511.04240v1.pdf (page 5)

Generated Conjectures:
--------------------------------------------------
1. [First conjecture about quantum computing]
2. [Second conjecture about quantum computing]
3. [Third conjecture about quantum computing]
```

## Architecture

The system consists of:

- **retriever.py**: Handles FAISS index loading and document retrieval
- **main.py**: Main proof system with multi-agent workflow
- **example_algebra_logic.py**: Example script for algebra/logic agent
- **example.py**: Example script for number theory agent

### Agent Types

1. **Number Theory Agent**: Specialized in:
   - Analytic number theory
   - Algebraic number theory
   - L-functions and zeta functions
   - Modular forms
   
   **Number Theory Experimenter**: Specialized in:
   - Prime number computations
   - Modular arithmetic and congruences
   - L-function computations
   - Distribution analysis
   - Asymptotic behavior analysis
   - Numerical pattern analysis

2. **Algebraic Topology Agent**: Specialized in:
   - Homology and cohomology theory
   - Homotopy theory and fundamental groups
   - Spectral sequences
   - K-theory (topological and algebraic)
   - Characteristic classes
   - Fiber bundles and fibrations
   - Manifold topology
   - Category theory in topology
   
   **Algebraic Topology Experimenter**: Specialized in:
   - Computational topology libraries (GUDHI, Dionysus, Ripser)
   - Persistent homology computations
   - Homology and cohomology calculations
   - Simplicial complex computations
   - Homotopy group calculations
   - Spectral sequence computations
   - Characteristic class calculations
   - Manifold topology verification
   - K-theory computations

## Workflow Structure

The multi-agent system follows this workflow:

1. **Generate Conjectures**: Specialized agent generates 2-5 conjectures
2. **For Each Conjecture**:
   - **Reduce to Lemmas**: Identify known theorems/lemmas that apply
   - **Attempt Proof**: Experimenter agent constructs proof attempts
   - **Verify Iteratively**: Verifier checks proof, refines if needed (up to 5 iterations)
   - **Classify**: Mark as THEOREM if proven, CONJECTURE if not
3. **Synthesize Results**: Coordinator generates final report

### Workflow Diagram

```
[User Query] 
    ↓
[Generate List of Conjectures]
    ↓
[For Each Conjecture]
    ├── [Reduce to Lemmas]
    ├── [Attempt Proof]
    ├── [Verify (iterative)]
    └── [Classify: THEOREM or CONJECTURE]
    ↓
[Generate Final Report]
    ├── Theorems (proven)
    └── Conjectures (unproven)
```

## Configuration

You can customize the flow by adjusting:

- **Model**: Change the OpenAI model (default: "gpt-4")
- **Temperature**: Adjust creativity (default: 0.7)
- **Retrieval Count**: Number of documents to retrieve (default: 5)
- **Embedding Model**: Use OpenAI or HuggingFace embeddings

## Notes

- The embedding model used must match the one used to create the FAISS index
- If OpenAI API key is not available, the system will fall back to HuggingFace embeddings
- The system requires the FAISS index to be built from the same papers in the `papers/` directory

