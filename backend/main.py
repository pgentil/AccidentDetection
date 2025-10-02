from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from inference.models.utils import get_roboflow_model
import cv2
import tempfile
import os
import threading
import subprocess
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# === CONFIGURATION ===
model_name = "amazon-accident-detection-o3juo"
model_version = "3"
api_key = "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"  
sendgrid_api = os.getenv("SENDGRID_API_KEY")  # Load from environment variable for security

# Load Roboflow model
model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=api_key
)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominio frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Video Upload (lo que ya teníamos antes)
# ===========================
LAST_UPLOADED_VIDEO = None
LAST_EMAIL = None


def process_video(video_path: str, email: str = None):
    cap = cv2.VideoCapture(video_path)
    accident_flag = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
        for prediction in results[0].predictions:
            if prediction.class_name == "accident" and not accident_flag and email is not None:
                accident_flag = True
                # Send email notification
                message = Mail(
                    from_email='pgentil@ucm.es',
                    to_emails=email,
                    subject='Accident Detected',
                    html_content='An accident has been detected in the uploaded video. Please check the live feed.'
                )
                try:
                    sg = SendGridAPIClient(sendgrid_api)
                    response = sg.send(message)
                    print(response.status_code)
                    print(response.body)
                    print(response.headers)
                except Exception as e:
                    print(str(e))

            # Dibujar bounding box
            x_center = int(prediction.x)
            y_center = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)
            x0, y0 = x_center - w // 2, y_center - h // 2
            x1, y1 = x_center + w // 2, y_center + h // 2
            label = prediction.class_name
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(frame, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    cap.release()


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), email: str = Form(...)):
    global LAST_UPLOADED_VIDEO, LAST_EMAIL
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(await file.read())
    temp_file.close()

    LAST_UPLOADED_VIDEO = temp_file.name
    LAST_EMAIL = email
    print(f"Received email: {email}")

    return {"message": "Video uploaded successfully. Access live feed at /video_feed/", "email": email}


@app.get("/video_feed/")
def video_feed():
    if not LAST_UPLOADED_VIDEO or not os.path.exists(LAST_UPLOADED_VIDEO):
        return {"error": "No video uploaded yet."}

    return StreamingResponse(
        process_video(LAST_UPLOADED_VIDEO, LAST_EMAIL),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ===========================
# NUEVO: Live Cameras
# ===========================
import subprocess

class Camera:
    def __init__(self, source: str):
        self.source = self.get_stream_url(source)
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame = None

    def get_stream_url(self, source: str):
        # If it's a YouTube link, extract real stream using yt-dlp
        if "youtube.com" in source or "youtu.be" in source:
            try:
                result = subprocess.run(
                    ["yt-dlp", "-f", "best", "-g", source],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            except Exception as e:
                raise ValueError(f"Failed to extract YouTube stream: {e}")
        return source

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera source: {self.source}")
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Run inference
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            for pred in results[0].predictions:
                x_center, y_center = int(pred.x), int(pred.y)
                w, h = int(pred.width), int(pred.height)
                x0, y0 = x_center - w // 2, y_center - h // 2
                x1, y1 = x_center + w // 2, y_center + h // 2
                label = pred.class_name
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
                cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.last_frame = frame

    def get_frame(self):
        return self.last_frame

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


camera: Camera | None = None


@app.post("/start_camera/")
def start_camera(source: str = Form(...)):
    global camera
    if camera:
        camera.stop()
    try:
        camera = Camera(source)
        camera.start()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Camera started from source {source}"}


@app.get("/stop_camera/")
def stop_camera():
    global camera
    if not camera:
        return {"error": "No camera running"}
    camera.stop()
    camera = None
    return {"message": "Camera stopped"}


def generate_frames():
    global camera
    while camera and camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.get("/camera_feed/")
def camera_feed():
    if not camera or not camera.running:
        return {"error": "Camera not running"}
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------- HTML page ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
        <head><title>Accident Detection Live Feed</title></head>
        <body>
            <h1>Guardian AI</h1>
            <p>Upload a video via /upload_video/ or add cameras via /add_camera/</p>
            <p>Then access: /video_feed/ or /camera/{id}/feed</p>
        </body>
    </html>"""
    return html_content
