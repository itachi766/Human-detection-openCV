import numpy as np
import imutils
import cv2
import time
import os

# Get the base directory (root of the project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load the Model
prototxt = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# 2. Initialize the Camera
print("[INFO] starting video stream...")
# Use '0' for built-in webcam. Change to '1' if you have an external USB camera.
vs = cv2.VideoCapture(0)
time.sleep(2.0) # Allow camera sensor to warm up

# 3. Loop over frames from the video stream
while True:
    # Grab the frame from the camera
    ret, frame = vs.read()

    # If the frame was not grabbed successfully, stop the loop
    if not ret:
        break

    # Resize the frame for speed (max width 400px is faster than 600px)
    frame = imutils.resize(frame, width=400)
    
    # Grab the frame dimensions
    (h, w) = frame.shape[:2]

    # 4. Prepare the frame for the Neural Network
    # MobileNet requires 300x300 pixel input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
        (300, 300), 127.5)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # 5. Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections (confidence < 20%)
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            # Class 15 is 'Person'
            if idx == 15:
                # Compute bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the box and label
                label = "Person: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Adjust label position so it doesn't go off-screen
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Live Feed (Press 'q' to exit)", frame)
    
    # Press 'q' to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()