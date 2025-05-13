# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /___   ____________   _____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""
Vision module for line following JetTec rover. Returns offset between the rover's camera 
and the line, and the processed image.
"""

import cv2
import numpy as np

THRESHOLD = 30
MIN_AREA = 500

class Vision:
    def __init__(self, image, debug=True):
        self.rgb_image = image
        self.x_offset = None
        self.debug = debug

    def calculate_offset_and_threshold(self):
        h, w, _ = self.rgb_image.shape
        clipped_rgb = self.rgb_image[int(h * 0.5):, :]
        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        processed_image = thresh.copy()
        processed_image = cv2.resize(processed_image, (84, 84))  # CNN input size

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.x_offset = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > MIN_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    center_x = clipped_rgb.shape[1] // 2
                    pixel_offset = cx - center_x
                    self.x_offset = pixel_offset / (clipped_rgb.shape[1] / 2)

        return self.x_offset, processed_image

