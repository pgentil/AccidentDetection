# run.ps1 - Guardian AI startup script

# Stop on errors
$ErrorActionPreference = "Stop"

Write-Host "Starting Guardian AI Accident Detection System..."

# Activate virtual environment
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
.venv\Scripts\Activate

# Install dependencies
Write-Host "Installing backend requirements..."
pip install -r backend\requirements.txt

# Start backend
Write-Host "Launching FastAPI backend on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; uvicorn main:app --reload --host 127.0.0.1 --port 8000"

# Start frontend
Write-Host "Launching frontend on http://localhost:8080 ..."
cd frontend
python -m http.server 8080
