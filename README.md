# Accident Detection System

This project includes a FastAPI backend and a simple HTML/JS frontend.


## 1) Create and activate a virtual environment

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate


## 2) Install requirements


pip install -r backend/requirements.txt


## 3) Run the backend

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will be available at `http://127.0.0.1:8000`.

## 4) Open the frontend

Option A: Open `frontend/index.html` directly in your browser.

Option B: Serve it locally:
```bash
cd ../frontend
python -m http.server 8080
# Open http://localhost:8080
```

## Using the app

- Upload a video file. The backend processes it and returns frames with detections.

## Notes

- Ensure the backend is running before opening the frontend.
- If you use an external AI service (e.g., Roboflow), configure the API key as required by `backend/main.py`.
- Supported video formats typically include MP4, AVI, and MOV.

