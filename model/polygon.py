import cv2
import numpy as np

# Global variables
drawing = False  # True if the mouse is being used to draw
points = []  # List to store polygon vertices

def draw_polygon(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        # If the first point is clicked
        if len(points) == 0:
            drawing = True
            points.append([x, y])
        else:
            # Check if the new point is close to the first point to close the polygon
            if np.linalg.norm(np.array(points[0]) - np.array([x, y])) < 10:
                points.append(points[0])  # Close the polygon
                drawing = False
            else:
                points.append([x, y])

    if event == cv2.EVENT_MOUSEMOVE and drawing:
        # Draw the current line segment
        temp_img = frame.copy()
        cv2.line(temp_img, tuple(points[-1]), (x, y), (255, 0, 0), 2)
        cv2.imshow("Select Polygon", temp_img)

    if event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button click to reset
        points.clear()
        drawing = False

def select_polygon_on_webcam_feed(url):
    global frame, points, drawing
    points = []  # Initialize the points list

    # Capture video stream from webcam
    cap = cv2.VideoCapture(url)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cv2.namedWindow("Select Polygon")
    cv2.setMouseCallback("Select Polygon", draw_polygon)

    print("Click to select the polygon vertices. Click near the starting point to close the polygon.")
    print("Right-click to reset. Press 'c' to confirm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=2)

        cv2.imshow("Select Polygon", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(points) > 2:  # Press 'c' to confirm selection
            if points[-1] != points[0]:  # Ensure the polygon is closed
                points.append(points[0])
            break

    cap.release()
    cv2.destroyAllWindows()

    polygon_str = str(points)
    print(f"Polygon coordinates: {polygon_str}")
    return polygon_str

# Example usage
url = "http://192.168.8.152:2879/video"  # Replace with your webcam URL or IP camera stream URL
polygon_str = select_polygon_on_webcam_feed(url)
