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
Vision processing for line following with JetTec rover.
Calculates the offset between the rover's camera and the detected line.
"""

import cv2
import numpy as np

class Vision:
    def __init__(self, image, threshold=30, min_area=500, debug=True):
        """
        Args:
            image (np.array): RGB image from the camera.
            threshold (int): Threshold value for binarization.
            min_area (int): Minimum area to consider a contour as valid.
            debug (bool): Debug mode (not used yet but could help).
        """
        self.rgb_image = image
        self.threshold = threshold
        self.min_area = min_area
        self.x_offset = None
        self.debug = debug

    def calculate_offset_and_threshold(self):
        """
        Processes the image and calculates the horizontal offset.

        Returns:
            x_offset (float): Offset normalized between -1 and 1. None if not found.
            processed_image (np.array): Thresholded and resized image.
        """
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

