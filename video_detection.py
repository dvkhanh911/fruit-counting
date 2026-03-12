from ultralytics import YOLO
import cv2
from collections import Counter

# load model
model = YOLO("best.pt")

def video_detection(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không mở được video")
        return

    cv2.namedWindow("Fruit Detection", cv2.WINDOW_NORMAL)

    prev_time = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # resize
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2)

        # detect
        results = model(frame, conf=0.4,iou =0.3)

        boxes = results[0].boxes
        labels = boxes.cls.tolist()
        names = model.names

        fruits = [names[int(i)] for i in labels]

        fruit_count = Counter(fruits)

        annotated = results[0].plot()

        # hiển thị số lượng từng loại
        y = 30
        for fruit, count in fruit_count.items():

            text = f"{fruit}: {count}"

            cv2.putText(
                annotated,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            y += 30

        # tổng số trái cây
        total = len(fruits)

        cv2.putText(
            annotated,
            f"Total: {total}",
            (10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )


        cv2.imshow("Fruit Detection", annotated)

        # nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_detection("video_fruit_1.mp4")
