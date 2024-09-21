from flask import Flask, render_template, jsonify
from dotenv import load_dotenv
import os
import json
import time
from flask_sse import sse
from threading import Thread

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost:6379"
app.register_blueprint(sse, url_prefix='/stream')

def check_for_updates():
    last_data = None
    while True:
        try:
            with open('input.json', 'r') as file:
                cameras_data = json.load(file)
            data = [
                {"location": details['location'], "number": details['number']}
                for details in cameras_data.values()
            ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading input.json: {e}")
            continue

        if data != last_data:
            sse.publish({"coordinates": data}, type='update_coordinates')
            last_data = data

        time.sleep(5)


@app.route('/')
def index():
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    map_id = os.getenv('GOOGLE_MAPS_MAP_ID')
    
    # Load coordinates and numbers from input.json
    data = []
    try:
        with open('input.json', 'r') as file:
            cameras_data = json.load(file)
            data = [
                {"location": details['location'], "number": details['number']}
                for details in cameras_data.values()
            ]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading input.json: {e}")
    
    return render_template('index.html', api_key=api_key, map_id=map_id, coordinates=data)


if __name__ == '__main__':
    thread = Thread(target=check_for_updates)
    thread.start()
    app.run(debug=True)
