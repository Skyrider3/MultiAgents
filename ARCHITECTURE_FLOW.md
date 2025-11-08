# Multi-Agent Number Theory Research System - Architecture & Flow

## System Overview

This is an intelligent multi-agent system designed to assist with number theory research by combining semantic search, symbolic reasoning, and experimental validation through specialized AI agents.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT RESEARCH SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│   Phase 1    │      │   Phase 2    │      │     Phase 3      │
│   INGEST     │─────>│   RETRIEVE   │─────>│  AGENT WORKFLOW  │
│              │      │              │      │                  │
└──────────────┘      └──────────────┘      └──────────────────┘
```

---

## 📊 Detailed Component Flow

### Phase 1: Data Ingestion (`ingest.py`)

```
                    PDF PAPERS (papers/*.pdf)
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │    PyPDFLoader: Extract Text          │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  RecursiveCharacterTextSplitter       │
        │  - chunk_size: 800                    │
        │  - chunk_overlap: 120                 │
        │  - Preserves math blocks              │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Extract LaTeX Symbols & Formulas     │
        │  - \tau(n), a_n, \zeta(s)             │
        │  - \pmod{p}, mod operations           │
        │  - Function notation like L(s)        │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Generate Embeddings                  │
        │  Option A: OpenAI Embeddings          │
        │  Option B: SBERT (all-mpnet-base-v2)  │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Build Indices (index/ directory)     │
        │  1. faiss.index - Vector DB           │
        │  2. meta.pkl - Texts & Metadata       │
        │  3. symbol_index.json - Symbol Map    │
        └───────────────────────────────────────┘
```

**Key Functions:**
- `load_and_chunk(pdf_path)` - Loads PDF and creates overlapping chunks
- `extract_latex_symbols(text)` - Regex extraction of mathematical notation
- `build_indices()` - Main orchestrator creating all indices

---

### Phase 2: Retrieval System (`agents.py`)

```
                    USER QUERY
                        │
                        ▼
        ┌───────────────────────────────┐
        │   DUAL RETRIEVAL STRATEGY     │
        └───────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    ┌──────────────┐        ┌──────────────┐
    │  SEMANTIC    │        │   SYMBOL     │
    │  RETRIEVAL   │        │  RETRIEVAL   │
    └──────────────┘        └──────────────┘
            │                       │
            │                       │
            ▼                       ▼
    • Embed query           • Extract LaTeX symbols
    • FAISS search            from query (e.g., \tau)
    • Return top-k          • Lookup in symbol_index
      similar chunks        • Return matching chunks
            │                       │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │  MERGED EVIDENCE SET  │
            └───────────────────────┘
```

**Key Functions:**
- `semantic_retrieve(query, k=6)` - Vector similarity search using FAISS
- `symbol_retrieve(symbols, max_chunks_per_symbol=6)` - Precise symbol matching

---

### Phase 3: Multi-Agent Workflow (`run_session.py` + `agents.py`)

```
                         USER QUERY
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Query Expansion              │
              │  Add domain keywords:         │
              │  "multiplicative", "Dirichlet"│
              │  "L-function", "congruence"   │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Dual Retrieval               │
              │  (Semantic + Symbol)          │
              └───────────────────────────────┘
                              │
                              ▼
        ╔═══════════════════════════════════════════╗
        ║         AGENT 1: NUMBER THEORIST          ║
        ╠═══════════════════════════════════════════╣
        ║  • Analyzes evidence chunks               ║
        ║  • Proposes up to 5 conjectures (LaTeX)   ║
        ║  • Provides intuition & references        ║
        ║  • Suggests proof approaches              ║
        ║  • Includes numeric check summaries       ║
        ╚═══════════════════════════════════════════╝
                              │
                              ▼
                    [Extract Statements]
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        ╔═══════════════════╗       ╔═══════════════════╗
        ║  AGENT 2:         ║       ║  AGENT 3:         ║
        ║  EXPERIMENTER     ║       ║  SYMBOLIC         ║
        ╠═══════════════════╣       ╠═══════════════════╣
        ║ • Generate test   ║       ║ • Symbolic        ║
        ║   code (Sage/     ║       ║   simplification  ║
        ║   PARI/Python)    ║       ║ • Reduction to    ║
        ║ • Create test     ║       ║   known lemmas    ║
        ║   plans           ║       ║ • Algorithmic     ║
        ║ • Resource        ║       ║   proof steps     ║
        ║   estimates       ║       ║ • Formalization   ║
        ║ • Find counter-   ║       ║   in Sage/Lean    ║
        ║   examples        ║       ║                   ║
        ╚═══════════════════╝       ╚═══════════════════╝
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    [Results Aggregation]
                              │
                              ▼
              ╔═══════════════════════════════════════╗
              ║    AGENT 4: COORDINATOR               ║
              ╠═══════════════════════════════════════╣
              ║  • Aggregates all proposals           ║
              ║  • Scores for novelty & interest      ║
              ║  • Prioritizes experiments            ║
              ║  • Recommends best candidate          ║
              ║  • Creates execution plan             ║
              ╚═══════════════════════════════════════╝
                              │
                              ▼
                ┌───────────────────────────┐
                │  OUTPUT: session_{ts}.json│
                │  • All proposals          │
                │  • Agent outputs          │
                │  • Coordinator summary    │
                └───────────────────────────┘
```

---

## 🔄 Complete End-to-End Flow Diagram

```mermaid
graph TB
    Start([User Query]) --> QueryExp[Query Expansion<br/>Add domain keywords]
    
    QueryExp --> Retrieve{Dual Retrieval}
    
    Retrieve --> SemRet[Semantic Retrieval<br/>FAISS Vector Search]
    Retrieve --> SymRet[Symbol Retrieval<br/>LaTeX Pattern Match]
    
    SemRet --> Merge[Merge Evidence]
    SymRet --> Merge
    
    Merge --> NT[🤖 Number Theorist Agent<br/>Generate Conjectures]
    
    NT --> Parse[Parse Statements<br/>Extract LaTeX formulas]
    
    Parse --> Loop{For each<br/>statement}
    
    Loop --> Exp[🧪 Experimenter Agent<br/>Test Code Generation]
    Loop --> Sym[🔣 Symbolic Agent<br/>Proof Attempts]
    
    Exp --> Collect[Collect Results]
    Sym --> Collect
    
    Collect --> Loop2{More<br/>statements?}
    Loop2 -->|Yes| Loop
    Loop2 -->|No| Coord
    
    Coord[🎯 Coordinator Agent<br/>Prioritize & Aggregate]
    
    Coord --> Save[Save to JSON<br/>session_{timestamp}.json]
    Save --> End([Complete])
    
    style NT fill:#e1f5ff
    style Exp fill:#fff4e1
    style Sym fill:#ffe1f5
    style Coord fill:#e1ffe1
```

---

## 📦 Data Flow & File Structure

```
MultiAgents/
│
├── papers/                    # INPUT: Research PDFs
│   └── *.pdf
│
├── index/                     # GENERATED: Search indices
│   ├── faiss.index           # Vector database
│   ├── meta.pkl              # Chunk texts & metadata
│   └── symbol_index.json     # LaTeX symbol → chunk mapping
│
├── session_*.json            # OUTPUT: Research sessions
│
├── ingest.py                 # Phase 1: Indexing pipeline
├── agents.py                 # Phase 2 & 3: Agents & retrieval
├── run_session.py            # Phase 3: Session orchestration
└── pyproject.toml            # Dependencies & configuration
```

---

## 🎯 Agent Responsibilities Matrix

| Agent | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Number Theorist** | Query + Evidence | Up to 5 conjectures in LaTeX with intuition, references, and proof approaches | Generate novel hypotheses grounded in literature |
| **Experimenter** | Statement + Evidence | Test code (Sage/PARI/Python), test plan, resource estimates, counterexamples | Validate conjectures empirically |
| **Symbolic** | Statement + Evidence | Reduction steps, known lemmas, algorithmic proof outline | Formal mathematical reasoning |
| **Coordinator** | All agent outputs | Prioritized recommendations, best candidate, execution plan | Synthesize and rank proposals |

---

## 🔧 Key Technical Decisions

### 1. **Dual Retrieval Strategy**
   - **Semantic**: Catches conceptual similarities (e.g., "distribution" → related theorems)
   - **Symbol**: Precise matching for mathematical notation (e.g., `\tau(n)` → Ramanujan tau)

### 2. **Chunking Strategy**
   - Size: 800 characters with 120 overlap
   - Preserves mathematical blocks by respecting `\n\n` separators
   - Maintains context across chunk boundaries

### 3. **LLM Integration**
   - Flexible: Supports OpenAI (GPT-4) or local models
   - Each agent has specialized prompts
   - Fallback to FakeListLLM for testing without API key

### 4. **Symbol Extraction**
   - Regex patterns for LaTeX: `\tau`, `a_n`, `\pmod{p}`
   - Enables formula-specific retrieval
   - Builds inverted index: symbol → chunk_ids

---

## 🚀 Execution Flow Example

**Input Query:**
```
"distribution of zeros of Ramanujan tau(n) modulo small primes"
```

**Execution Steps:**

1. **Query Expansion**
   - Add: "multiplicative", "Dirichlet", "mod p", "L-function", "elliptic", "congruence", "density"

2. **Retrieval** (10 semantic + symbol matches for `\tau`)
   - Returns ~10-20 evidence chunks from indexed papers

3. **Number Theorist**
   - Proposes conjectures like: "$\tau(n) \equiv 0 \pmod{p}$ has density..."
   - Cites chunk IDs and suggests proof techniques

4. **Experimenter** (for each conjecture)
   - Generates Sage code to test modulo small primes
   - Suggests test range: p < 1000

5. **Symbolic** (for each conjecture)
   - Attempts reduction to Deligne's theorem
   - Suggests formal verification steps

6. **Coordinator**
   - Ranks proposals by feasibility
   - Recommends: "Test Conjecture #2 first - most tractable"

7. **Output**
   - Saves complete session to `session_1699123456.json`

---

## 🧩 Dependencies & Technologies

- **LangChain**: Agent orchestration, prompts, LLM integration
- **FAISS**: High-performance vector similarity search
- **Sentence Transformers**: Local embeddings (all-mpnet-base-v2)
- **PyPDF**: PDF parsing
- **NumPy**: Numerical operations for embeddings
- **OpenAI API** (optional): GPT-4 for agent reasoning

---

## 💡 Design Patterns

1. **RAG Pattern**: Retrieval-Augmented Generation for grounding AI in literature
2. **Multi-Agent System**: Specialized agents with distinct responsibilities
3. **Pipeline Architecture**: Ingest → Retrieve → Reason → Coordinate
4. **Hybrid Search**: Combining vector search with structured symbol lookup
5. **Evidence Provenance**: All outputs trace back to source chunks

---

## 🔮 System Capabilities

✅ **Semantic paper search** across number theory literature  
✅ **Formula-aware retrieval** using LaTeX symbol extraction  
✅ **Automated conjecture generation** grounded in evidence  
✅ **Experimental validation** with generated test code  
✅ **Symbolic reasoning** to connect to known theorems  
✅ **Intelligent prioritization** of research directions  

---

## 📝 Usage Workflow

```bash
# Step 1: Ingest papers (one-time setup)
python ingest.py

# Step 2: Run research session
python run_session.py "your number theory question"

# Example:
python run_session.py "distribution of zeros of Ramanujan tau(n) modulo small primes"

# Step 3: Review output
cat session_*.json
```

---

## 🎓 System Philosophy

This system embodies a **collaborative research assistant** paradigm:

- **Number Theorist** = Creative hypothesis generation
- **Experimenter** = Empirical validation
- **Symbolic** = Formal mathematical rigor
- **Coordinator** = Research strategy

Together, they mirror the cognitive processes of a research team, augmented by access to the entire literature corpus through semantic search.

---

**Last Updated:** 2025-11-08  
**Version:** 0.1.0

