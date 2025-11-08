#!/bin/bash

# Multi-Agent Mathematical Discovery System - Run Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Multi-Agent Mathematical Discovery System - Startup Script"
echo "======================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check if Docker is running
echo -e "${YELLOW}Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check if docker-compose is installed
echo -e "${YELLOW}Checking docker-compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ docker-compose is installed${NC}"

# Parse command line arguments
INIT_SYSTEM=true
START_API=true
START_UI=true
SAMPLE_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-init)
            INIT_SYSTEM=false
            shift
            ;;
        --no-api)
            START_API=false
            shift
            ;;
        --no-ui)
            START_UI=false
            shift
            ;;
        --sample-data)
            SAMPLE_DATA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-init      Skip system initialization"
            echo "  --no-api       Don't start the API server"
            echo "  --no-ui        Don't start the Streamlit UI"
            echo "  --sample-data  Initialize with sample data"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install poetry
    poetry install
else
    source .venv/bin/activate
fi

# Initialize system
if [ "$INIT_SYSTEM" = true ]; then
    echo -e "${YELLOW}Initializing system...${NC}"

    # Start Docker services
    echo -e "${YELLOW}Starting Docker services...${NC}"
    docker-compose up -d

    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10

    # Run initialization script
    INIT_ARGS=""
    if [ "$SAMPLE_DATA" = true ]; then
        INIT_ARGS="--sample-data"
    fi

    python scripts/init_system.py --no-docker $INIT_ARGS

    if [ $? -ne 0 ]; then
        echo -e "${RED}System initialization failed${NC}"
        exit 1
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"

    # Kill API server
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi

    # Kill Streamlit
    if [ ! -z "$UI_PID" ]; then
        kill $UI_PID 2>/dev/null || true
    fi

    # Ask about Docker services
    read -p "Stop Docker services? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down
    fi

    echo -e "${GREEN}Shutdown complete${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start API server
if [ "$START_API" = true ]; then
    echo -e "${YELLOW}Starting API server...${NC}"
    python -m src.api.app &
    API_PID=$!
    echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"

    # Wait for API to be ready
    echo -e "${YELLOW}Waiting for API to be ready...${NC}"
    MAX_ATTEMPTS=30
    ATTEMPT=0

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API is ready${NC}"
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        sleep 1
    done

    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo -e "${RED}API failed to start${NC}"
        cleanup
        exit 1
    fi
fi

# Start Streamlit UI
if [ "$START_UI" = true ]; then
    echo -e "${YELLOW}Starting Streamlit UI...${NC}"
    streamlit run src/ui/streamlit_app.py --server.port 8501 &
    UI_PID=$!
    echo -e "${GREEN}✓ Streamlit UI started (PID: $UI_PID)${NC}"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}System is running!${NC}"
echo "======================================================================"
echo ""
echo "Access points:"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • API Health Check:  http://localhost:8000/health"
echo "  • Streamlit UI:      http://localhost:8501"
echo "  • Neo4j Browser:     http://localhost:7474"
echo "  • Qdrant Dashboard:  http://localhost:6333/dashboard"
echo "  • Prometheus:        http://localhost:9090"
echo "  • Grafana:          http://localhost:3000"
echo ""
echo "Press Ctrl+C to shutdown the system"
echo ""

# Keep script running
while true; do
    sleep 1
done