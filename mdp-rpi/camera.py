import base64
import json
import os
from typing import Dict, Any
from picamera import PiCamera
import cv2
import time
from datetime import datetime

FOLDER_PATH = "/home/pi/image_rec/inference"
IMAGE_PREPROCESSED_FOLDER_PATH = "/home/pi/image_rec/inference/ImagePreProcessed"

def capture(img_pth: str) -> None:
    """
    Capture image using PiCamera and save it to the specified path.

    Parameters:
        img_pth (str): Path to save the captured image.
    """
    camera = PiCamera()
    print(img_pth)
    image_save_location = os.path.join(FOLDER_PATH, img_pth)
    camera.capture(image_save_location)
    camera.close()
    print("[Camera] Image captured")

def preprocess_img(img_pth: str) -> None:
    """
    Read image, resize it, and save the resized image.

    Parameters:
        img_pth (str): Path of the image to be preprocessed.
    """
    image_save_location = os.path.join(FOLDER_PATH, img_pth)
    img = cv2.imread(image_save_location)

    resized_img = cv2.resize(img, (640, 480))
    image_save_location = os.path.join(IMAGE_PREPROCESSED_FOLDER_PATH, img_pth)
    cv2.imwrite(image_save_location, resized_img)
    print("[Camera] Image preprocessing complete")

def get_image(final_image:bool=False) -> bytes:
    """
    Capture an image, preprocess it, and return a JSON message with the encoded image.

    Returns:
        bytes: Encoded JSON message containing image data.
    """
    formatted_time = datetime.fromtimestamp(time.time()).strftime('%d-%m_%H-%M-%S.%f')[:-3]
    img_pth = f"img_{formatted_time}.jpg"
    capture(img_pth)
    preprocess_img(img_pth)

    encoded_string = ""
    image_save_location = os.path.join(IMAGE_PREPROCESSED_FOLDER_PATH, img_pth)
    if os.path.isfile(image_save_location):
        with open(image_save_location, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    message: Dict[str, Any] = {
        "type": 'IMAGE_TAKEN',
        "final_image": final_image,
        "data": {
            "image": encoded_string
        }
    }
    return json.dumps(message).encode("utf-8")
