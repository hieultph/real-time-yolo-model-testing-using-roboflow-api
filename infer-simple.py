# load config
import json
with open('roboflow_config.json') as f:
    config = json.load(f)

ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

FRAMERATE = config["FRAMERATE"]
BUFFER = config["BUFFER"]

import cv2
import base64
import numpy as np
import requests
import time

# Construct the Roboflow Infer URL
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",  # change format to json to get bounding box details
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img_resized = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img_resized)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    response = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }).json()

    # Draw bounding boxes and labels on the image
    for prediction in response['predictions']:
        # Get bounding box coordinates
        x0 = int(prediction['x'] - prediction['width'] / 2)
        y0 = int(prediction['y'] - prediction['height'] / 2)
        x1 = int(prediction['x'] + prediction['width'] / 2)
        y1 = int(prediction['y'] + prediction['height'] / 2)
        
        # Draw the bounding box
        cv2.rectangle(img_resized, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Get label and confidence score
        label = prediction['class']
        confidence = prediction['confidence']

        # Prepare label text
        label_text = f"{label} ({confidence:.2f})"

        # Draw the label text above the bounding box
        cv2.putText(img_resized, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_resized

# Main loop; infers sequentially until you press "q"
while True:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()
