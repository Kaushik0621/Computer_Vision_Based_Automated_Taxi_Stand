from flask import Flask, render_template, request, jsonify
import os
import csv
from werkzeug.utils import secure_filename
from model.obj_tracking import object_tracking_with_yolo
import json
import pandas as pd

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure the output directory exists
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('model_input.html')

@app.route('/submit', methods=['POST'])
def submit():
    final_json = {}
    for i in range(1, len(request.form) // 3 + 1):
        latitude = request.form.get(f'latitude{i}')
        longitude = request.form.get(f'longitude{i}')
        polygon_str = request.form.get(f'polygon{i}')
        video = request.files.get(f'video{i}')
        if latitude and longitude and video:
            location = [float(latitude), float(longitude)]
            camera_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'camera{i}')
            os.makedirs(camera_folder, exist_ok=True)
            # Save the input video in the camera folder
            input_video_filename = secure_filename(video.filename)
            input_video_path = os.path.join(camera_folder, input_video_filename)
            video.save(input_video_path)
            # Remove the .mp4 extension from the input video filename for the CSV file
            csv_filename = os.path.splitext(input_video_filename)[0] + '.csv'
            # Define paths for the output video and CSV
            output_video_path = os.path.join(camera_folder, f'output_{input_video_filename}')
            csv_output_path = os.path.join(camera_folder, csv_filename)
            # Process the video (assumed to create the CSV file)
            object_tracking_with_yolo(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                csv_output_path=csv_output_path,
                polygon_str=polygon_str
            )
            try:
                df = pd.read_csv(csv_output_path)
                number = len(df)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                # Handle the case where the CSV is empty or doesn't exist
                number = 0

            # Create JSON for this camera
            camera_json = {
                "location": location,
                "number": number
            }
            final_json[f'camera{i}'] = camera_json

    # Save the final JSON to the root directory
    final_json_path = os.path.join(app.root_path, 'final.json')
    with open(final_json_path, 'w') as json_file:
        json.dump(final_json, json_file, indent=4)

    return jsonify({"message": "Done!", "final_json": final_json})

if __name__ == '__main__':
    app.run(debug=True)
