# 🔬 Streamlit Proof System Visualizers

This repository includes two Streamlit applications for visualizing the CrewAI Mathematical Proof System with **verifiable reasoning** and **visual graphs** of agent interactions.

---

## 📦 Installation

```bash
# Install Streamlit-specific requirements
pip install -r requirements_streamlit.txt

# Or using Poetry
poetry add streamlit plotly networkx pandas streamlit-autorefresh
```

---

## 🚀 Applications

### 1. **Proof System Visualizer** (`streamlit_proof_visualizer.py`)

A comprehensive visualization tool that shows:
- 📊 **Metrics Dashboard**: Final status, iterations used, execution time
- 🔗 **Interactive Conversation Graph**: Visual flow between agents using Plotly
- 📚 **Evidence Attribution**: Each insight linked to source papers
- 🔄 **Iteration History**: Step-by-step verification process

**Run:**
```bash
streamlit run streamlit_proof_visualizer.py
```

**Features:**
- Load existing proof session JSON files
- Demo mode with sample data
- Interactive network graph showing agent communications
- Source attribution for all evidence
- Phase-by-phase execution breakdown

### 2. **Real-time Proof Monitor** (`streamlit_realtime_proof.py`)

Live monitoring of proof system execution with:
- ⏱️ **Real-time Updates**: See agents working in real-time
- 📜 **Timeline View**: Chronological message flow
- 🔗 **Sankey Diagram**: Flow of information between agents
- 📝 **Evidence Log**: Track all citations and sources
- 📊 **Progress Tracking**: Live iteration counter

**Run:**
```bash
streamlit run streamlit_realtime_proof.py
```

**Features:**
- Live agent message interception
- Real-time progress updates
- Evidence tracking with confidence scores
- Automatic verdict detection
- Thread-safe execution

---

## 🎯 Key Features: Verifiable Reasoning

Both apps implement **verifiable reasoning** by:

### 1. **Evidence Attribution**
```python
# Every agent action includes evidence
{
    "agent": "Number Theorist",
    "action": "Generate Conjecture",
    "evidence": [
        {
            "source": "Deligne 1974",
            "text": "By the Weil conjectures...",
            "confidence": 0.95
        }
    ]
}
```

### 2. **Source Tracking**
- Papers referenced: `papers/deligne_1974.pdf`
- Theorems cited: `Deligne's bound: |τ(p)| ≤ 2p^(11/2)`
- Chunk IDs: `[Paper_A__12]`

### 3. **Reasoning Chain**
Each iteration shows:
- **Verifier Feedback**: Specific issues identified
- **Experimenter Response**: How issues were addressed
- **Evidence Added**: New citations or calculations

---

## 📊 Visual Graphs

### 1. **Interactive Network Graph** (Plotly)
Shows connections between:
- User → Number Theorist (Query)
- Number Theorist → Experimenter (Conjectures)
- Experimenter ↔ Verifier (Iterative refinement)
- Verifier → Coordinator (Final status)

### 2. **Sankey Diagram**
Visualizes message flow volume between agents:
- Width represents number of messages
- Color-coded by agent type
- Interactive hover details

### 3. **Timeline View**
Chronological display with:
- Timestamps for each action
- Color-coded agent messages
- Expandable evidence sections
- Iteration markers

---

## 🖥️ User Interface

### Main Dashboard
```
┌─────────────────────────────────────────┐
│  🔬 Mathematical Proof System           │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐            │
│  │ Metrics  │  │  Graph   │            │
│  └──────────┘  └──────────┘            │
│                                         │
│  Phase 1: Setup                         │
│  ├── 🧠 Number Theorist                 │
│  │   └── Evidence: 3 papers             │
│  └── 🔬 Experimenter                    │
│      └── Initial proof                  │
│                                         │
│  Phase 2: Iterations                    │
│  ├── Iteration 1                        │
│  │   ├── 🔍 Verifier: INVALID           │
│  │   └── 🔧 Experimenter: Refining      │
│  ├── Iteration 2                        │
│  │   ├── 🔍 Verifier: INCOMPLETE        │
│  │   └── 🔧 Experimenter: Adding calc   │
│  └── Iteration 3                        │
│      └── ✅ Verifier: VALID             │
│                                         │
│  Phase 3: Synthesis                     │
│  └── 📊 Coordinator: Final report       │
└─────────────────────────────────────────┘
```

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
STREAMLIT_THEME=light
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2196F3"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
headless = true
```

---

## 📊 Sample Outputs

### Evidence Display
```
📚 Source: Deligne 1974
Evidence: By the Weil conjectures proof, we have
the bound |τ(p)| ≤ 2p^(11/2) for all primes p...
Confidence: 95%
```

### Iteration Progress
```
🔄 Iteration 2 of 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40%

Verifier: ❌ INVALID - Missing justification
Experimenter: Adding citation to Lehmer 1947...
```

### Final Status
```
✅ PROOF VALIDATED
Iterations Used: 3/5
Time Elapsed: 45 seconds
Evidence Sources: 6 unique papers
```

---

## 🎨 Customization

### Add Custom Agent Colors
```python
agent_colors = {
    "Number Theorist": "#2196F3",  # Blue
    "Experimenter": "#FF9800",     # Orange
    "Verifier": "#E91E63",         # Pink
    "Coordinator": "#4CAF50"       # Green
}
```

### Modify Evidence Extraction
```python
def extract_evidence(text):
    # Add custom citation patterns
    patterns = [
        r'\[([^\]]+)\]',     # [Citation]
        r'\((\w+ \d{4})\)',  # (Author Year)
        r'\\cite\{([^}]+)\}' # \cite{key}
    ]
```

---

## 🚦 Running Both Apps

### Development Mode
```bash
# Terminal 1: Run proof visualizer
streamlit run streamlit_proof_visualizer.py --server.port 8501

# Terminal 2: Run real-time monitor
streamlit run streamlit_realtime_proof.py --server.port 8502
```

### Production Deployment
```bash
# Using Docker
docker build -t proof-visualizer .
docker run -p 8501:8501 proof-visualizer

# Using Streamlit Cloud
# Push to GitHub and deploy via streamlit.io
```

---

## 📝 Usage Examples

### 1. Load Previous Session
```python
# Upload proof_session_20241108_143022.json
# View iterations, evidence, and final status
```

### 2. Run New Proof
```python
# Enter query: "Distribution of tau(n) modulo primes"
# Set max iterations: 5
# Click "Run Proof System"
# Watch real-time updates
```

### 3. Export Results
```python
# Download JSON with full proof history
# Export evidence log as CSV
# Save interaction graph as image
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements_streamlit.txt
```

### Issue: "FAISS index not found"
```bash
# Run the ingestion first
python ingest.py
```

### Issue: "OpenAI API error"
```bash
# Check .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## 🎯 Key Benefits

1. **Transparency**: See exactly how agents reason
2. **Verifiability**: Every claim linked to evidence
3. **Interactivity**: Explore the proof process
4. **Real-time**: Watch agents work live
5. **Visual**: Understand flow through graphs

---

## 📚 Further Reading

- [CrewAI Documentation](https://docs.crewai.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Graph Objects](https://plotly.com/python/)

---

**Ready to visualize mathematical proofs with verifiable reasoning! 🚀**