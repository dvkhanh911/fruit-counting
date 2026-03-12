
from ultralytics import YOLO
from collections import Counter
import cv2

model = YOLO("best.pt")

def count_fruit(image_path):

    results = model(image_path,conf=0.25)

    boxes = results[0].boxes

    labels = boxes.cls.tolist()

    names = model.names

    fruits = [names[int(i)] for i in labels]

    fruit_counter = Counter(fruits)

    print("Fruit count:")

    for fruit, count in fruit_counter.items():
        print(f"{fruit}: {count}")

    print("Total fruits:", len(fruits))
  
    annotated = results[0].plot()

    # resize
    annotated = cv2.resize(annotated, None, fx=0.1,fy=0.1)

    cv2.imshow("Fruit Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    image = "anh_test_5.jpg"   
    count_fruit(image)
