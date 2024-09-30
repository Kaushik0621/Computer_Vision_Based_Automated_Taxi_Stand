# import numpy as np
# import cv2
# import supervision as sv
# from ultralytics import YOLO
# from collections import defaultdict
# import pandas as pd
# import time
# import ast
# import subprocess

# def get_stream_url(youtube_link):
#     command = ["yt-dlp", "-f", "best", "-g", youtube_link]
#     result = subprocess.run(command, capture_output=True, text=True)
#     if result.returncode == 0:
#         return result.stdout.strip()
#     else:
#         print("Error extracting stream URL:", result.stderr)
#         return None

# def object_tracking_with_yolo_streaming(stream_url, csv_output_path, polygon_str):
#     # Convert the polygon coordinates from string to a numpy array
#     try:
#         polygon = np.array(ast.literal_eval(polygon_str), np.int32).reshape((-1, 1, 2))
#     except Exception as e:
#         print(f"Error parsing polygon coordinates: {e}")
#         return

#     # Load the YOLO model
#     model_path = 'yolov8n.pt'
#     try:
#         model = YOLO(model_path)
#     except Exception as e:
#         print(f"Error loading YOLO model: {e}")
#         return

#     # Dictionary to store the time spent by each object inside the polygon
#     time_inside_polygon = defaultdict(float)
#     start_times = {}
#     valid_objects = set()

#     # CSV storage
#     csv_data = []

#     # Track objects using a basic method (tracking by bounding box ID)
#     object_trackers = {}
#     object_id_counter = 0

#     def is_point_in_polygon(point, polygon):
#         return cv2.pointPolygonTest(polygon, point, False) >= 0

#     # Capture the video stream
#     cap = cv2.VideoCapture(stream_url)

#     if not cap.isOpened():
#         print("Error: Could not open the video stream.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from the stream.")
#             break

#         # Process the frame with YOLO
#         results = model(frame)[0]

#         # Draw the polygon on the frame
#         cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color in BGR format

#         # Process detections and filter out only persons (class index 0 is 'person' in YOLO)
#         detections = results.boxes
#         people_detections = [box for box in detections if box.cls[0] == 0]

#         for box in people_detections:
#             # Get the bounding box coordinates and calculate the center point
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             x_center = (x1 + x2) // 2
#             y_center = (y1 + y2) // 2
            
#             # Generate a unique ID for each detection
#             object_id = None
#             for obj_id, tracker in object_trackers.items():
#                 if tracker['last_position'] is not None:
#                     dist = np.linalg.norm(np.array([x_center, y_center]) - np.array(tracker['last_position']))
#                     if dist < 50:  # Threshold distance to consider it the same object
#                         object_id = obj_id
#                         tracker['last_position'] = (x_center, y_center)
#                         break
            
#             if object_id is None:
#                 # Create a new object entry with a unique ID
#                 object_id_counter += 1
#                 object_id = object_id_counter
#                 object_trackers[object_id] = {'id': object_id, 'last_position': (x_center, y_center)}

#             # Track time inside the polygon
#             if is_point_in_polygon((x_center, y_center), polygon):
#                 if object_id not in start_times:
#                     start_times[object_id] = time.time()
#                 time_inside_polygon[object_id] += time.time() - start_times[object_id]
#                 start_times[object_id] = time.time()
                
#                 # Mark object as valid if it stays inside the polygon for more than 2 seconds
#                 if time_inside_polygon[object_id] > 2:
#                     valid_objects.add(object_id)
#             else:
#                 start_times.pop(object_id, None)  # Remove start time if the object leaves the area

#             # Display the ID and time above the head of the object
#             display_time = f"ID: {object_id}, Time: {time_inside_polygon[object_id]:.2f}s"
#             cv2.putText(frame, display_time, (x_center, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
#         # Clean up object trackers for objects no longer detected
#         object_trackers = {k: v for k, v in object_trackers.items() if v['last_position'] is not None}

#         # Display the frame
#         cv2.imshow('YOLO Live Stream Object Tracking', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()

#     # After processing, save the CSV
#     for obj_id in valid_objects:
#         csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
#     df = pd.DataFrame(csv_data)
#     df.to_csv(csv_output_path, index=False)
#     print(f"CSV saved to {csv_output_path}")

# # Replace this with your actual YouTube link
# youtube_link = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"

# # Get the stream URL
# stream_url = get_stream_url(youtube_link)

# if stream_url:
#     print(f"Processing stream from: {stream_url}")
#     object_tracking_with_yolo_streaming(
#         stream_url=stream_url,
#         csv_output_path='output_live_stream_data.csv',
#         polygon_str='[[1211, 728], [984, 1012], [361, 685], [586, 517], [502, 117], [1228, 207], [1212, 729]]'
#     )
# else:
#     print("Failed to retrieve the stream URL.")

import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import time
import ast
import subprocess
import threading

def get_stream_url(youtube_link):
    command = ["yt-dlp", "-f", "best", "-g", youtube_link]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Error extracting stream URL:", result.stderr)
        return None

def save_csv_periodically(csv_output_path, valid_objects, time_inside_polygon, csv_data):
    while True:
        time.sleep(120)  # Save every 2 minutes
        for obj_id in valid_objects:
            if not any(d['object_id'] == obj_id for d in csv_data):
                csv_data.append({"object_id": obj_id, "time_inside_marked_area": time_inside_polygon[obj_id]})
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV saved to {csv_output_path} at interval.")

def object_tracking_with_yolo_streaming(stream_url, csv_output_path, polygon_str):
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
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    # Capture the video stream
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        return

    # Reduce the resolution to ease computational load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Start a thread to save the CSV periodically
    threading.Thread(target=save_csv_periodically, args=(csv_output_path, valid_objects, time_inside_polygon, csv_data), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the stream.")
            break

        # Process the frame with YOLO
        results = model(frame)[0]

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

            # Display the ID and time above the head of the object
            display_time = f"ID: {object_id}, Time: {time_inside_polygon[object_id]:.2f}s"
            cv2.putText(frame, display_time, (x_center, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Clean up object trackers for objects no longer detected
        object_trackers = {k: v for k, v in object_trackers.items() if v['last_position'] is not None}

        # Display the frame
        cv2.imshow('YOLO Live Stream Object Tracking', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Replace this with your actual YouTube link
youtube_link = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"

# Get the stream URL
stream_url = get_stream_url(youtube_link)

if stream_url:
    print(f"Processing stream from: {stream_url}")
    object_tracking_with_yolo_streaming(
        stream_url=stream_url,
        csv_output_path='output_live_stream_data.csv',
        polygon_str='[[876, 768], [828, 737], [1555, 886], [1295, 1053], [1085, 640], [1555, 889], [832, 742], [834, 741], [1085, 639], [1288, 1047], [1256, 1032], [831, 740], [1212, 830], [1212, 830], [876, 768]]'
    )
else:
    print("Failed to retrieve the stream URL.")
