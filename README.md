# 🍎 Fruit Detection and Counting using YOLOv8
 
## Overview

This project implements a fruit detection and counting system using YOLOv8.  
The system can detect multiple types of fruits from images and videos and automatically count the number of each fruit.

The model is trained using a custom fruit detection dataset and deployed using Python with OpenCV for visualization.

## Main Capabilities

- Detect fruits in images  
- Count the number of each fruit type  
- Detect and count fruits in video  
- Real-time object detection using YOLOv8  

## Model

The project uses YOLOv8s from the Ultralytics library.  
YOLOv8 provides fast and accurate object detection suitable for real-time applications.

## Project Structure
```
fruit-counting
│
├── train.ipynb                  
├── fruit_detection.py    
├── count_fruit.py        
├── video_detection.py   
│
├── best.pt              
├── requirements.txt
├── .gitignore
└── README.md
```
## Dataset

The model is trained using the [Fruit Detection Dataset](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection).
The dataset follows the YOLO annotation format.
```
dataset
│
├── train
│   ├── images
│   └── labels
│
├── valid
│   ├── images
│   └── labels
│
├── test
│   ├── images
│   └── labels
│
└── fruit.yaml
```

## How to Run

- Install required dependencies.
- Run `train.ipynb` to train the model.
- Run `fruit_detection.py` to detect fruits in an image.
- Run `count_fruit.py` to detect and count fruits in an image.
- Run `video_detection.py` to detect and count fruits in a video.



