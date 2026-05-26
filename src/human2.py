import cv2
from ultralytics import YOLO
import math
import os

# Get the base directory (root of the project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")

# 1. Load the "Brain"
# 'yolov8n.pt' is the Nano model (Fastest for drones/Raspberry Pi)
print("[INFO] Loading YOLOv8 model...")
model = YOLO(model_path) 

# 2. Open the Camera
# Use 0 for webcam. Replace with 'video.mp4' to test on drone footage.
cap = cv2.VideoCapture(0)

# Set resolution (optional, but good for speed)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        break

    # 3. The Magic Line
    # YOLOv8 does everything (resize, normalize, detect) in one line
    # stream=True makes it faster/smoother
    results = model(img, stream=True, classes=[0]) # class 0 is 'Person'

    # 4. Draw the results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100

            # Draw only if confident
            if conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Person {conf}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Drone View', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()