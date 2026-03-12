# 🍎 Fruit Detection and Counting using YOLOv8

## 📌 Overview

This project implements a **fruit detection and counting system using YOLOv8**.  
The system can detect multiple types of fruits from **images and videos** and automatically count the number of each fruit.

The model is trained using a **custom fruit detection dataset** and deployed using Python with OpenCV for visualization.

### Main capabilities

- 🍎 Detect fruits in images  
- 🍌 Count the number of each fruit type  
- 🎥 Detect and count fruits in video  
- ⚡ Real-time object detection using YOLOv8  

---

# 🧠 Model

The project uses **YOLOv8s** from the Ultralytics library.
YOLOv8 provides fast and accurate object detection suitable for real-time applications.

---

# 📂 Project Structure
fruit-counting
│
├── train.py # Train YOLOv8 model
├── validation.py # Evaluate trained model
├── fruit_detection.py # Detect fruits in an image
├── count_fruit.py # Count fruits in an image
├── video_detection.py # Detect and count fruits in video
│
├── best.pt # Trained model weights
├── requirements.txt
├── .gitignore
└── README.md

# 📊 Dataset

The model is trained using [**Fruit Detection Dataset**](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection).
Dataset format follows the **YOLO annotation format**.

dataset
│
├── train
│ ├── images
│ └── labels
│
├── valid
│ ├── images
│ └── labels
│
├── test
│ ├── images
│ └── labels
│
└── fruit.yaml

