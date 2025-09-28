import cv2
import threading

class Camera:
    def __init__(self, cam_id: str, source: str):
        self.cam_id = cam_id
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            #aqui se puede procesar el frame si es necesario

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
