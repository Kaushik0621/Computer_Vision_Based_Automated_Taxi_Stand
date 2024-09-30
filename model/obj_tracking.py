import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import time
import ast

def object_tracking_with_yolo(input_video_path, output_video_path, csv_output_path, polygon_str):
    # Convert the polygon coordinates from string to a numpy array
    try:
        polygon = np.array(ast.literal_eval(polygon_str), np.int32).reshape((-1, 1, 2))
    except Exception as e:
        print(f"Error parsing polygon coordinates: {e}")
        return

    # Load the YOLO model
    model_path = 'yolov8n.pt'
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
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

    def is_point_in_polygon(point, polygon):
        # Check if a point is inside a polygon using cv2.pointPolygonTest
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
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
                
                # Mark object as valid if it stays inside the polygon for more than 40 seconds
                if time_inside_polygon[object_id] > 40:
                    valid_objects.add(object_id)
            else:
                start_times.pop(object_id, None)  # Remove start time if the object leaves the area

            # Display the ID and time above the head of the object
            display_time = f"ID: {object_id}, Time: {time_inside_polygon[object_id]:.2f}s"
            cv2.putText(frame, display_time, (x_center, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Clean up object trackers for objects no longer detected
        object_trackers = {k: v for k, v in object_trackers.items() if v['last_position'] is not None}

        return frame

    # After processing, save the CSV
    def save_csv():
        for obj_id in valid_objects:
            csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)

    # Process the video
    sv.process_video(
        source_path=input_video_path,
        target_path=output_video_path,
        callback=callback
    )

    # Save the CSV after processing is complete
    save_csv()
    print(f"Output video saved to {output_video_path}")
