import numpy as np
import cv2
import torch
import threading
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import time
import ast
import subprocess
import signal
import sys

class VideoStreamThread(threading.Thread):
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, self.frame = self.cap.read()
            if not ret:
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()

def get_stream_url(youtube_link):
    command = ["yt-dlp", "-f", "best", "-g", youtube_link]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Error extracting stream URL:", result.stderr)
        return None

def load_yolov8_model(model_path='yolov8n.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path).to(device)
    return model

def save_csv_periodically(csv_output_path, valid_objects, time_inside_polygon, csv_data):
    while True:
        time.sleep(120)  # Save every 2 minutes
        for obj_id in valid_objects:
            if not any(d['object_id'] == obj_id for d in csv_data):
                csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV saved to {csv_output_path} at interval.")

def process_video_stream_async(url, model, csv_output_path, polygon_str, frame_skip=3, resize_dim=(800, 600)):
    try:
        polygon = np.array(ast.literal_eval(polygon_str), np.int32).reshape((-1, 1, 2))
        print(f"Polygon coordinates: {polygon}")
    except Exception as e:
        print(f"Error parsing polygon coordinates: {e}")
        return

    time_inside_polygon = defaultdict(float)
    start_times = {}
    valid_objects = set()
    csv_data = []
    object_trackers = {}
    object_id_counter = 0

    video_stream = VideoStreamThread(url)
    video_stream.start()
    frame_count = 0

    def is_point_in_polygon(point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def process_frame(frame: np.ndarray) -> np.ndarray:
        nonlocal object_id_counter, object_trackers

        results = model(frame)[0]
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        detections = results.boxes
        people_detections = [box for box in detections if box.cls[0] == 0]

        for box in people_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            
            object_id = None
            for obj_id, tracker in object_trackers.items():
                if tracker['last_position'] is not None:
                    dist = np.linalg.norm(np.array([x_center, y_center]) - np.array(tracker['last_position']))
                    if dist < 50:
                        object_id = obj_id
                        tracker['last_position'] = (x_center, y_center)
                        break
            
            if object_id is None:
                object_id_counter += 1
                object_id = object_id_counter
                object_trackers[object_id] = {'id': object_id, 'last_position': (x_center, y_center)}

            if is_point_in_polygon((x_center, y_center), polygon):
                if object_id not in start_times:
                    start_times[object_id] = time.time()
                time_inside_polygon[object_id] += time.time() - start_times[object_id]
                start_times[object_id] = time.time()

                if time_inside_polygon[object_id] > 2:
                    valid_objects.add(object_id)
            else:
                start_times.pop(object_id, None)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_text = f"{object_id}, {time_inside_polygon[object_id]:.2f}s"
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

        object_trackers = {k: v for k, v in object_trackers.items() if v['last_position'] is not None}

        return frame

    threading.Thread(target=save_csv_periodically, args=(csv_output_path, valid_objects, time_inside_polygon, csv_data), daemon=True).start()

    print("Processing live stream... Press 'Esc' to exit.")
    try:
        while True:
            frame = video_stream.frame
            if frame is None:
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, resize_dim)
            processed_frame = process_frame(frame)

            cv2.imshow("Live Stream", processed_frame)

            if cv2.waitKey(1) == 27:  # Esc key to stop
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()

        for obj_id in valid_objects:
            csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)
        print(f"Final CSV saved to {csv_output_path}")

def main():
    youtube_link = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"
    csv_output_path = 'output_live_stream_data.csv'
    polygon_str = '[[876, 768], [828, 737], [1555, 886], [1295, 1053], [1085, 640], [1555, 889], [832, 742], [834, 741], [1085, 639], [1288, 1047], [1256, 1032], [831, 740], [1212, 830], [1212, 830], [876, 768]]'

    stream_url = get_stream_url(youtube_link)
    if not stream_url:
        print("Failed to retrieve the stream URL.")
        return

    model = load_yolov8_model('yolov8n.pt')

    process_video_stream_async(stream_url, model, csv_output_path, polygon_str, frame_skip=3, resize_dim=(800, 600))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))
    main()
