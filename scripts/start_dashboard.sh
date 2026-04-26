#!/bin/bash
# ===========================================
# Start Dashboard Script
# ===========================================

set -e

echo "🖥️  Starting Hospital FL Dashboard"
echo "==================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Start backend in background
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000...${NC}"
cd dashboard/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 2

# Check if frontend dependencies are installed
if [ ! -d "dashboard/frontend/node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    cd dashboard/frontend
    npm install
    cd ../..
fi

# Start frontend
echo -e "${GREEN}Starting React frontend on http://localhost:3000...${NC}"
cd dashboard/frontend
npm start &
FRONTEND_PID=$!
cd ../..

echo ""
echo -e "${GREEN}Dashboard started!${NC}"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

