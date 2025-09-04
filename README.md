# 🚨 Accident Detection System

A real-time accident detection system using AI and computer vision, built with FastAPI backend and modern HTML/JavaScript frontend.

## 🏗️ Architecture

- **Backend**: FastAPI server with Roboflow AI model for accident detection
- **Frontend**: Stunning multi-section website with hero, about, upload, and team sections
- **AI Model**: Amazon accident detection model via Roboflow

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the Frontend
Open `frontend/index.html` in your web browser, or serve it using a local server:

```bash
cd frontend
python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

## 📱 How to Use

1. **Upload Video**: Drag and drop a video file or click to browse
2. **Real-time Analysis**: The system will process the video and display it with accident detection bounding boxes
3. **Live Feed**: Watch the processed video with real-time accident detection overlays

## 🔧 Features

- **Multi-Section Design**: Hero section, detailed about section, video upload, and team showcase
- **Drag & Drop**: Intuitive video file upload with visual feedback
- **Real-time Processing**: Live accident detection with bounding boxes
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Modern UI/UX**: Beautiful gradients, animations, and smooth interactions
- **Progress Tracking**: Visual feedback during video processing
- **Error Handling**: User-friendly error messages and status updates
- **Smooth Navigation**: Fixed navbar with smooth scrolling between sections

## 🌐 API Endpoints

- `POST /upload_video/` - Upload video for analysis
- `GET /video_feed/` - Stream processed video with accident detection
- `GET /` - HTML interface

## ⚠️ Important Notes

- Ensure your backend is running on `localhost:8000`
- The system temporarily stores uploaded videos for processing
- Supported video formats: MP4, AVI, MOV, etc.
- The AI model requires an internet connection for Roboflow API calls

## 🐛 Troubleshooting

- **CORS Errors**: Make sure the backend is running and CORS is enabled
- **Video Not Loading**: Check that the video file is valid and the backend is processing it
- **Model Errors**: Verify your Roboflow API key is valid and the model is accessible

## 🔒 Security Note

In production, update the CORS settings in `backend/main.py` to only allow your specific frontend domain instead of `"*"`. 