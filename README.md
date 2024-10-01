# 🚕 Computer Vision-Based Automated Taxi Stand

This project leverages **Computer Vision** and **YOLO** (via **Roboflow**) to analyze queue length and wait times at taxi stands. The goal is to automate the process of counting how many people are standing in line and for how long, using annotated coordinates from a video feed.

## 📹 What I've Done

- **Annotated Coordinates**: Marked the positions in a video to define the queue area.
- **YOLO + Roboflow**: Trained and integrated a YOLO model to detect people standing in the queue.
- **Time Calculation**: Computed how many people are present in the queue and how long they have been waiting, based on video input.

I have used a delay time of 20 sec to filterout the random people passing through the annotated area. I can change the waiting time based on the use case.
To reduce future cost, We can use pre installed Live stream video. One of the model is there inside the model folder named "YT_track". By tracking the 
people waiting in the particulare area, the CSV of waiting time has been saved simultaneously in a interval of 2 mins.


## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![YOLO](https://img.shields.io/badge/YOLO-%23FF9800.svg?style=for-the-badge&logo=AI&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-blue?style=for-the-badge&logo=AI&logoColor=white)

---

## 🏁 Steps to Get Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
