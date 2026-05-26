import numpy as np
import cv2
import imutils
import os
import sys

# Get the base directory (root of the project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load the pre-trained model
prototxt = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")

# Initialize the DNN module
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# 2. Load the image
image_path = os.path.join(BASE_DIR, "images", "match.jpg")
if not os.path.exists(image_path):
    # Try fallback to any JPEG/PNG/JPG in images folder or root
    images_dir = os.path.join(BASE_DIR, "images")
    candidates = []
    if os.path.exists(images_dir):
        candidates += [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    candidates += [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if candidates:
        image_path = candidates[0]
        print(f"[INFO] 'images/match.jpg' not found. Falling back to found image: '{image_path}'")
    else:
        print("[ERROR] Could not find 'images/match.jpg' or any other image in the project directory.")
        sys.exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Could not read image at '{image_path}'. Please check the file path/integrity.")
    sys.exit(1)

image = imutils.resize(image, width=600) # Resize for speed/consistency
(h, w) = image.shape[:2]


# 3. Create a "Blob" from the image
# This performs mean subtraction and scaling required by the model
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    (300, 300), 127.5)

# 4. Pass the blob through the network
net.setInput(blob)
detections = net.forward()

# 5. Loop over the detections
for i in range(0, detections.shape[2]):
    # Extract confidence (probability)
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections (e.g., < 20% confidence)
    if confidence > 0.2:
        # Extract the index of the class label
        idx = int(detections[0, 0, i, 1])

        # CHECK: Is this a person? (Class 15 is Person in MobileNet SSD)
        if idx == 15:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the box and label
            label = "Person: {:.2f}%".format(confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output
cv2.imshow("Deep Learning Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()