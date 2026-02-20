import mss
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # using the small YOLOv8 model
monitor = {"top": 100, "left": 200, "width": 800, "height": 800}

with mss.mss() as sct:
    while True:
        # Capture the region
        img = np.array(sct.grab(monitor))

        # Convert BGRA to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Run YOLO detection
        results = model(frame)

        # Annotate frame with detections
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("BlueSim YOLO Feed", annotated_frame)
        # Exit on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
