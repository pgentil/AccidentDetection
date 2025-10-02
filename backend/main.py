from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from inference.models.utils import get_roboflow_model
import cv2
import tempfile
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import time
import shutil

# === CONFIGURATION ===
model_name = "amazon-accident-detection-o3juo"
model_version = "3"
api_key = "ktSFVMakkE69oahKbqtv"
sendgrid_api = None #os.getenv("SENDGRID_api_key")  # Load from environment variable for security
# Load Roboflow model
model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=api_key
)
TARGET_FPS = 5  # Desired FPS for processing

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
LAST_EMAIL = None

def reduce_fps(video_path: str, output_path: str, target_fps: int = 5, cut_video: bool = False, video_len : int = 10):
    """Reduce the FPS of the input video to target_fps. Optionally cut the video to a certain length in seconds. 
    Returns the list of frames extracted if needed. Saves the reduced FPS video to output_path. 
    Motivation: reduce latency in inference and processing time."""
    #capturamos el original
    frames = []
    cap = cv2.VideoCapture(video_path)
    #agarramos los fps del video original
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if target_fps >= original_fps:
        print("Target FPS is greater than or equal to original FPS. No reduction applied.")
        # Just copy the original file to output path
        
        shutil.copy2(video_path, output_path)
        
    else:
        print(f"Original FPS: {original_fps}")
        #caluclamos el intervalo de frames para reducir a target_fps
        frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1
        #preparamos el video de salida codec mp4v
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (int(cap.get(3)), int(cap.get(4))))
        if (cut_video):
            max_frames = target_fps * video_len
        
        i = 0
        effective_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret or (cut_video and effective_frames >= max_frames): # Si no hay más frames o hemos alcanzado el límite
                break
            if i % frame_interval == 0:
                #escribimos el frame en el video de salida
                out.write(frame)
                frames.append(frame)
                effective_frames += 1
            i += 1
        out.release()
    cap.release()
    return frames

def send_email_notification(email: str):
    """
    Send an email notification about the detected accident.
    """
    message = Mail(
    from_email='pgentil@ucm.es',
    to_emails=email,
    subject='Accident Detected',
    html_content='An accident has been detected in the uploaded video. Please check the live feed.')
    try:
        if (sendgrid_api is None):
            print("SendGrid API key not configured.")
            return
        sg = SendGridAPIClient(sendgrid_api)
        # sg.set_sendgrid_data_residency("eu")
        # uncomment the above line if you are sending mail using a regional EU subuser
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
        LAST_EMAIL = None  # Reset after sending
    except Exception as e:
        print(str(e))
    return

def draw_bounding_box(frame, prediction):
    """Draw bounding box and label on the frame for a given prediction."""
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
    return 

def process_video(video_path: str, email: str = None):
    """Process the video frame by frame, perform inference, and yield frames with bounding boxes."""
    # Create unique temp file for this video processing
    temp_processed_path = video_path.replace('.mp4', '_processed.mp4')
    
    try:
        reduce_fps(video_path, temp_processed_path, target_fps=TARGET_FPS, cut_video=False)
        start = time.time()
        cap = cv2.VideoCapture(temp_processed_path)
        accident_flag = False  # Flag to track if an accident has been detected
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Inference
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            for prediction in results[0].predictions:
                if prediction.class_name == "accident" and not accident_flag and email is not None:
                    accident_flag = True
                    send_email_notification(email)
                draw_bounding_box(frame, prediction)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        end = time.time()
        accident_flag = False  # Reset flag for next video
        print(f"Processing time: {end - start} seconds")
        cap.release()
    finally:
        # Clean up temp processed file
        if os.path.exists(temp_processed_path):
            try:
                os.remove(temp_processed_path)
            except:
                pass

# ---------------- Upload video and prepare live feed ----------------
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), email: str = Form(...)):
    global LAST_UPLOADED_VIDEO, LAST_EMAIL
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(await file.read())
    temp_file.close()
    # Store the path of the uploaded video
    # call queueing system if needed
    LAST_UPLOADED_VIDEO = temp_file.name  # store for live feed
    LAST_EMAIL = email  # store email for notifications
    # For now, just print/log the email address (extend as needed)
    print(f"Received email: {email}")
    
    
    return {"message": f"Video uploaded successfully. Access live feed at /video_feed/", "email": email}

# ---------------- Live feed in browser ----------------
@app.get("/video_feed/")
def video_feed(t: str = None):  # Accept timestamp parameter to prevent caching
    print(f"Accessing video feed... LAST_UPLOADED_VIDEO: {LAST_UPLOADED_VIDEO}")
    if not LAST_UPLOADED_VIDEO or not os.path.exists(LAST_UPLOADED_VIDEO):
        print("Error: No video uploaded or file doesn't exist")
        return {"error": "No video uploaded yet."}

    print(f"Starting video processing for: {LAST_UPLOADED_VIDEO}")
    
    # Add headers to prevent caching and ensure fresh connections
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "close"
    }
    
    return StreamingResponse(
        process_video(LAST_UPLOADED_VIDEO, LAST_EMAIL),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers
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
