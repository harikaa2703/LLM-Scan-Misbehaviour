@echo off
echo Starting LLMSCAN (Backend + Frontend)...
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/upgrade backend dependencies
echo Installing backend dependencies...
pip install -q -r requirements.txt

REM Start backend in a new window
echo Starting backend server...
start cmd /k python backend.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Navigate to frontend and start it
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

REM Start frontend
echo Starting frontend development server...
npm run dev

pause
