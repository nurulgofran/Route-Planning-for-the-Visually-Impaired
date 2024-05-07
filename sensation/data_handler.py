"""Hanldes differen types of input data like image, video or camera input"""

import os
from enum import Enum

import cv2


class InputType(Enum):
    IMAGES = 1
    VIDEO = 2
    CAMERA = 3


class InputHandler:
    def __init__(self,
                 input_path:str,
                 output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.input_images = []
        self.cap = None
        self.input_type = None
        self.check_input_type()

    def check_input_type(self):
        # Determine if input is a folder, video file, or camera
        if os.path.isdir(self.input_path):
            # Input is a folder of images
            self.input_type = InputType.IMAGES
            for filename in os.listdir(self.input_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.input_images.append(os.path.join(self.input_path, filename))
        elif self.input_path.isdigit():
            # Input is a USB camera path
            self.input_type = InputType.CAMERA
            self.cap = cv2.VideoCapture(int(self.input_path))      
        else:
            # Input is a video file
            self.input_type = InputType.VIDEO
            self.cap = cv2.VideoCapture(self.input_path)

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def save_image(self, image, image_path):        
        base_name = os.path.splitext(image_path)[0]  # Get the path without the file extension
        new_file_path = base_name + ".png"
        cv2.imwrite(os.path.join(self.output_path, os.path.basename(new_file_path)), image)

    def get_images(self) -> list:
        return self.input_images

    def get_input_type(self):
        return self.input_type

    def get_cap(self):
        return self.cap

    def get_output_path(self):
        return self.output_path
