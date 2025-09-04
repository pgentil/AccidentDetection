from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from inference.models.utils import get_roboflow_model
import cv2
import tempfile
import os

# === CONFIGURATION ===
model_name = "amazon-accident-detection-o3juo"
model_version = "3"
api_key = "ktSFVMakkE69oahKbqtv"

# Load Roboflow model
model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=api_key
)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store last uploaded video path
LAST_UPLOADED_VIDEO = None

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)
        for prediction in results[0].predictions:
            x_center = int(prediction.x)
            y_center = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)

            x0 = x_center - w // 2
            y0 = y_center - h // 2
            x1 = x_center + w // 2
            y1 = y_center + h // 2

            label = prediction.class_name
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    cap.release()

# ---------------- Upload video and prepare live feed ----------------
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    global LAST_UPLOADED_VIDEO
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(await file.read())
    temp_file.close()

    LAST_UPLOADED_VIDEO = temp_file.name  # store for live feed

    return {"message": f"Video uploaded successfully. Access live feed at /video_feed/"}

# ---------------- Live feed in browser ----------------
@app.get("/video_feed/")
def video_feed():
    if not LAST_UPLOADED_VIDEO or not os.path.exists(LAST_UPLOADED_VIDEO):
        return {"error": "No video uploaded yet."}

    return StreamingResponse(
        process_video(LAST_UPLOADED_VIDEO),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ---------------- HTML page to display live feed ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
        <head>
            <title>Accident Detection Live Feed</title>
        </head>
        <body>
            <h1>Live Accident Detection Feed</h1>
            <p>Upload a video via /upload_video/ to see live feed below:</p>
            <img src="/video_feed/" width="800" />
            <p>Use /upload_video/ endpoint to also download processed MJPEG via Python script.</p>
        </body>
    </html>
    """
    return html_content
