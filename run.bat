@echo off
echo Starting Guardian AI Accident Detection System...

:: Step 1: Create venv if missing
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Step 2: Activate venv
call .venv\Scripts\activate

:: Step 3: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Step 4: Install requirements
echo Installing backend requirements...
pip install -r backend\requirements.txt

:: Step 5: Start backend in new terminal
echo Launching FastAPI backend on http://127.0.0.1:8000 ...
start cmd /k "cd backend && ..\.venv\Scripts\activate && uvicorn main:app --reload --host 127.0.0.1 --port 8000"

:: Step 6: Start frontend in new terminal
echo Launching frontend on http://localhost:8080 ...
start cmd /k "cd frontend && ..\.venv\Scripts\activate && python -m http.server 8080"

:: Step 7: Open frontend in browser
start http://localhost:8080

echo Guardian AI is running. Close windows to stop.
pause
