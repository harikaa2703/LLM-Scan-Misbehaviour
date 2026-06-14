#!/usr/bin/env bash
# Linux/macOS startup script for LLMSCAN

echo "Starting LLMSCAN (Backend + Frontend)..."
echo

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install/upgrade backend dependencies
echo "Installing backend dependencies..."
pip install -q -r requirements.txt

# Start backend in background
echo "Starting backend server..."
python backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Navigate to frontend and start it
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
echo "Starting frontend development server..."
npm run dev

# Clean up backend when frontend is terminated
trap "kill $BACKEND_PID" EXIT
