#fruit_detection.py
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def detect_image(image_path):

    results = model(image_path)

    annotated = results[0].plot()

    annotated = cv2.resize(annotated, None,fx=0.1,fy=0.1)

    cv2.imshow("Fruit Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_image("anh_test_5.jpg")
