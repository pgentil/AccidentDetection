from inference.models.utils import get_roboflow_model
import cv2
import time
import os


model_name = "amazon-accident-detection-o3juo"
model_version = "3"
api_key = "ktSFVMakkE69oahKbqtv"
temp_vid = "./temp_video.mp4"

model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=api_key
)

def reduce_fps(video_path: str, target_fps: int = 5, cut_video: bool = False, video_len : int = 10):
    #capturamos el original
    cap = cv2.VideoCapture(video_path)
    #agarramos los fps del video original
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    #caluclamos el intervalo de frames para reducir a target_fps
    frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1
    #preparamos el video de salida codec mp4v
    out = cv2.VideoWriter(temp_vid, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (int(cap.get(3)), int(cap.get(4))))
    if (cut_video):
        max_frames = target_fps * video_len
    frames = []
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
    cap.release()
    out.release()
    return frames


def process_video(video_path: str | None = None, frames: list = None):
    i = 0
    if frames is not None:
        for frame in frames:
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            print(results[0].predictions if results[0].predictions else "No predictions")
            i = i + 1
    else:
        if video_path is None:
            raise ValueError("Either video_path or frames must be provided")
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            # print(results[0].predictions if results[0].predictions else "No predictions")
            i = i + 1
        cap.release()
    return i


if __name__ == "__main__":
    video_path = "zVzXEht1aME.mp4"  # Replace with your test video path
    frames = reduce_fps(video_path, target_fps=5, cut_video=True, video_len=10)
    print(f"Reduced to {len(frames)} frames")
    start = time.time()
    i = process_video(temp_vid, frames=frames)
    end = time.time()
    print(f"Processing time: {end - start} seconds")
    # os.remove(temp_vid)
    print(i)