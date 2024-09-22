import numpy as np
import cv2
import torch
import threading
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import time
import ast
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

def load_yolov8_model(model_path='yolov8n.pt'):
    """
    Load the YOLOv8 model from the given path and move it to the GPU if available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path).to(device)
    return model

def process_video_stream_async(url, model, csv_output_path, polygon_str, frame_skip=3, resize_dim=(800, 600)):
    """
    Process the video stream from the given URL using the YOLOv8 model asynchronously.

    :param url: URL of the video stream.
    :param model: YOLOv8 model for inference.
    :param csv_output_path: Path to save the CSV file with tracking data.
    :param polygon_str: String representing the polygon coordinates for region of interest.
    :param frame_skip: Number of frames to skip before processing the next one.
    :param resize_dim: Dimensions to resize the frame for faster processing.
    """
    # Convert the polygon coordinates from string to a numpy array
    try:
        polygon = np.array(ast.literal_eval(polygon_str), np.int32).reshape((-1, 1, 2))
        print(f"Polygon coordinates: {polygon}")
    except Exception as e:
        print(f"Error parsing polygon coordinates: {e}")
        return

    # Dictionary to store the time spent by each object inside the polygon
    time_inside_polygon = defaultdict(float)
    start_times = {}
    valid_objects = set()

    # CSV storage
    csv_data = []

    # Track objects using a basic method (tracking by bounding box ID)
    object_trackers = {}
    object_id_counter = 0

    video_stream = VideoStreamThread(url)
    video_stream.start()
    frame_count = 0

    def is_point_in_polygon(point, polygon):
        # Check if a point is inside a polygon using cv2.pointPolygonTest
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def process_frame(frame: np.ndarray) -> np.ndarray:
        nonlocal object_id_counter, object_trackers

        results = model(frame)[0]  # Get the results from YOLO model
        
        # Draw the polygon on the frame
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color in BGR format

        # Process detections and filter out only persons (class index 0 is 'person' in YOLO)
        detections = results.boxes
        people_detections = [box for box in detections if box.cls[0] == 0]

        for box in people_detections:
            # Get the bounding box coordinates and calculate the center point
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            
            # Generate a unique ID for each detection
            object_id = None
            for obj_id, tracker in object_trackers.items():
                if tracker['last_position'] is not None:
                    dist = np.linalg.norm(np.array([x_center, y_center]) - np.array(tracker['last_position']))
                    if dist < 50:  # Threshold distance to consider it the same object
                        object_id = obj_id
                        tracker['last_position'] = (x_center, y_center)
                        break
            
            if object_id is None:
                # Create a new object entry with a unique ID
                object_id_counter += 1
                object_id = object_id_counter
                object_trackers[object_id] = {'id': object_id, 'last_position': (x_center, y_center)}

            # Track time inside the polygon
            if is_point_in_polygon((x_center, y_center), polygon):
                if object_id not in start_times:
                    start_times[object_id] = time.time()
                time_inside_polygon[object_id] += time.time() - start_times[object_id]
                start_times[object_id] = time.time()
                
                # Mark object as valid if it stays inside the polygon for more than 2 seconds
                if time_inside_polygon[object_id] > 2:
                    valid_objects.add(object_id)
            else:
                start_times.pop(object_id, None)  # Remove start time if the object leaves the area

            # Draw the bounding box around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # Display the ID and time inside the bounding box (without explicitly writing "Time")
            display_text = f"{object_id}, {time_inside_polygon[object_id]:.2f}s"
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Clean up object trackers for objects no longer detected
        object_trackers = {k: v for k, v in object_trackers.items() if v['last_position'] is not None}

        return frame

    def save_csv():
        for obj_id in valid_objects:
            csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV saved to {csv_output_path}")

    print("Processing live webcam stream... Press 'Esc' to exit.")
    try:
        while True:
            frame = video_stream.frame
            if frame is None:
                continue

            frame_count += 1

            # Skip frames to reduce load
            if frame_count % frame_skip != 0:
                continue

            # Resize the frame for faster processing
            frame = cv2.resize(frame, resize_dim)

            processed_frame = process_frame(frame)

            # Display the frame
            cv2.imshow("Webcam Stream", processed_frame)

            # Break the loop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()

        # Save the CSV after processing is complete
        save_csv()

def main():
    """
    Main function to run the video stream processing application.
    """
    # Replace with your actual IP Webcam URL
    url = "http://192.168.8.152:2879/video"
    csv_output_path = 'output.csv'
    polygon_str = '[[62, 130], [596, 140], [587, 382], [24, 374], [62, 130]]'  # Replace with your actual polygon coordinates

    # Load the YOLOv8 model
    model = load_yolov8_model('yolov8n.pt')

    # Process the video stream with optimizations
    process_video_stream_async(url, model, csv_output_path, polygon_str, frame_skip=3, resize_dim=(800, 600))

if __name__ == "__main__":
    # Handling CTRL+C to gracefully stop the program
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))
    main()
