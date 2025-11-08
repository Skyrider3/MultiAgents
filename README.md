# 🧮 Multi-Agent Mathematical Conjecture Discovery System

A sophisticated multi-agent AI system for discovering mathematical patterns and generating conjectures through collaborative reasoning, powered by AWS Bedrock and advanced graph networks. This system demonstrates emergent intelligence through agent collaboration to advance mathematical research.

## 🎯 Overview

This system demonstrates emergent intelligence through multi-agent collaboration, where specialized AI agents work together to:
- Analyze mathematical research papers
- Cross-validate theorems and proofs
- Discover hidden patterns across domains
- Generate novel mathematical conjectures
- Provide verifiable reasoning traces

## 🏗️ Architecture

### Multi-Agent Framework
- **Researcher Agent**: Deep analysis of mathematical papers (Skepticism: 0.2)
- **Reviewer Agent**: Critical validation and verification (Skepticism: 0.95)
- **Synthesizer Agent**: Pattern discovery and conjecture generation
- **Challenger Agent**: Adversarial testing and counterexample generation
- **Historian Agent**: Temporal evolution and trend tracking
- **Coordinator Agent**: Orchestrates workflow and consensus building

### Core Technologies
- **LLM Providers**: AWS Bedrock (Claude 3.5, GPT-4, Grok, DeepSeek)
- **Knowledge Graph**: Neo4j for relationship modeling
- **Vector Search**: Qdrant for semantic similarity
- **Mathematical Engine**: SymPy & SageMath
- **Document Processing**: DeepSeek-OCR for formula extraction
- **Communication**: A2A-style protocol for agent coordination

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- AWS Account with Bedrock access
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MultiAgents.git
cd MultiAgents
```

2. **Set up environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your AWS credentials and API keys
nano .env
```

3. **Install dependencies**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

4. **Start infrastructure**
```bash
# Start Neo4j, Qdrant, PostgreSQL, Redis
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# Initialize databases
python scripts/setup_databases.py
```

5. **Launch the system**
```bash
# Start FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# In a new terminal, start Streamlit dashboard
streamlit run src/ui/app.py
```

## 📊 Usage

### Web Dashboard
Access the Streamlit dashboard at `http://localhost:8501`

Features:
- Upload research papers (PDF/arXiv)
- Monitor agent reasoning in real-time
- Explore knowledge graph visualizations
- View generated insights and conjectures

### API Endpoints
API documentation available at `http://localhost:8000/docs`

Key endpoints:
- `POST /papers/upload` - Upload research papers
- `GET /agents/status` - Monitor agent activity
- `POST /conjectures/generate` - Trigger conjecture discovery
- `GET /insights/latest` - Retrieve generated insights

### Command Line Interface
```bash
# Ingest papers from arXiv
python scripts/ingest_papers.py --topic "number theory" --count 10

# Run agent analysis
python scripts/run_analysis.py --agents all --papers recent

# Export insights
python scripts/export_insights.py --format latex --output results.tex
```

## 🧠 Agent Communication Protocol (A2A-Style)

The system implements a sophisticated agent-to-agent communication protocol:

```python
# Example agent message
{
    "id": "msg_123",
    "from_agent": "researcher_01",
    "to_agent": "reviewer_01",
    "type": "HYPOTHESIS",
    "content": {
        "statement": "All even numbers > 2 can be expressed as sum of two primes",
        "confidence": 0.85,
        "evidence": ["paper_arxiv_2301_12345", "theorem_goldbach_1742"]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "signature": "cryptographic_signature_here"
}
```

## 📁 Project Structure

```
MultiAgents/
├── src/
│   ├── agents/         # Multi-agent implementations
│   ├── knowledge/      # Knowledge processing & storage
│   ├── reasoning/      # Mathematical reasoning engine
│   ├── communication/  # A2A protocol implementation
│   ├── llm/           # AWS Bedrock integrations
│   ├── api/           # FastAPI backend
│   └── ui/            # Streamlit dashboard
├── tests/             # Test suites
├── scripts/           # Utility scripts
├── data/              # Local data storage
└── docs/              # Documentation
```

## 🔬 Mathematical Domains

Current focus areas:
- **Number Theory**: Prime distributions, Riemann Hypothesis
- **Graph Theory**: Coloring problems, connectivity
- **Algebraic Geometry**: Variety classifications
- **Combinatorics**: Ramsey theory, partition functions

## 🎮 Demo Scenarios

### Scenario 1: Twin Prime Discovery
```python
# The system analyzes papers on twin primes
# Discovers patterns in prime gaps
# Generates testable hypotheses about prime distributions
```

### Scenario 2: Graph Coloring Patterns
```python
# Ingests graph theory papers
# Identifies structural similarities
# Proposes new coloring algorithms
```

## 📈 Performance Metrics

- **Paper Processing**: ~3 minutes per paper
- **Conjecture Generation**: 5-10 candidates per session
- **Accuracy**: >85% formula extraction accuracy
- **Cost**: <$0.50 per paper with AWS Bedrock

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- Additional agent types
- New mathematical domains
- Visualization improvements
- Performance optimizations

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [Agent Design](docs/agent_design.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)

## 🔒 Security

- All agent communications are cryptographically signed
- AWS credentials managed via IAM roles
- Data encryption at rest and in transit
- Regular security audits with Bandit

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- AWS Bedrock for LLM infrastructure
- DeepSeek for OCR technology
- Neo4j for graph database
- Mathematical research community



**Built with ❤️ for advancing mathematical discovery through AI collaboration**