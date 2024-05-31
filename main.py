import os
import logging
import urllib.request
import uuid
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

mp_pose = mp.solutions.pose

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/", methods=['POST', 'GET'])
def hello_world():
    req_data = request.get_json()

    logging.debug(f"Received request data: {req_data}")

    tempfile_name = str(uuid.uuid4()) + ".mp4"
    try:
        # Download the necessary video
        url = req_data["video_url"]
        urllib.request.urlretrieve(url, tempfile_name)
        logging.debug(f"Downloaded video to {tempfile_name}")

        # Specify the video filename and create a `Pose` object as before
        file = tempfile_name
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Create VideoCapture object
            cap = cv2.VideoCapture(file)

            # Raise error if file cannot be opened
            if not cap.isOpened():
                logging.error("Error opening video stream or file")
                raise TypeError("Error opening video stream or file")

            # Get the number of frames in the video
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            data = {
                "frame_rate": float(cap.get(cv2.CAP_PROP_FPS)),
                "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                "frames": [
                    {
                        "timestamp": 0.0,
                        "pose": [[0.0, 0.0, 0.0, 0.0] for _ in range(len(mp_pose.PoseLandmark))]
                    } for _ in range(length)
                ]
            }

            # For each image in the video, extract the spatial pose data and save it in the appropriate spot in the `data` array
            frame_num = 0
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                # Remove the image flipping step
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks is None:
                    data["frames"][frame_num]["timestamp"] = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    data["frames"][frame_num]["pose"] = None
                    frame_num += 1
                    continue
                landmarks = results.pose_landmarks.landmark
                data["frames"][frame_num]["timestamp"] = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                for i in range(len(mp_pose.PoseLandmark)):
                    data["frames"][frame_num]["pose"][i][0] = float(landmarks[i].x)
                    data["frames"][frame_num]["pose"][i][1] = float(landmarks[i].y)
                    data["frames"][frame_num]["pose"][i][2] = float(landmarks[i].z)
                    data["frames"][frame_num]["pose"][i][3] = float(landmarks[i].visibility)

                frame_num += 1

            # Close the video file
            cap.release()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tempfile_name):
            os.remove(tempfile_name)
            logging.debug(f"Removed temporary file {tempfile_name}")

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
