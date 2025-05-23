# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /___   _______/  /_   _____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""
Vision processing for line following with JetTec rover.
Sliding Windows + Perspective Transform + Offset calculation.
Compatible with previous interface (calculate_offset_and_threshold).
"""

import cv2
import numpy as np

class Vision:
    def __init__(self, image, threshold=30, min_area=500, debug=True):
        self.rgb_image = image
        self.threshold = threshold
        self.min_area = min_area
        self.debug = debug
        self.x_offset = None

    def calculate_offset_and_threshold(self):
        """
        Processes the image and calculates the horizontal offset.
        Returns:
            x_offset (float): Offset normalized between -1 and 1. None if not found.
            processed_image (np.array): Processed 84x84 mono8 image.
        """
        h, w, _ = self.rgb_image.shape

        # Prétraitement
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 120)

        # ROI Trapèze
        mask = self.create_trapezoidal_mask(edges.shape)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Perspective transform
        birdseye, Minv = self.perspective_transform(masked_edges)

        # Sliding window
        offset, ppo_input = self.sliding_window_polyfit(birdseye)

        return offset, ppo_input

    def create_trapezoidal_mask(self, shape):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)

        bottom_left = (int(width * 0.05), height)
        bottom_right = (int(width * 0.95), height)
        top_left = (int(width * 0.3), int(height * 0.5))
        top_right = (int(width * 0.7), int(height * 0.5))

        roi_corners = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, [roi_corners], 255)
        return mask

    def perspective_transform(self, image):
        height, width = image.shape

        src = np.float32([
            [int(width * 0.4), int(height * 0.55)],
            [int(width * 0.6), int(height * 0.55)],
            [int(width * 0.95), height],
            [int(width * 0.05), height]
        ])

        dst = np.float32([
            [int(width * 0.25), 0],
            [int(width * 0.75), 0],
            [int(width * 0.75), height],
            [int(width * 0.25), height]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        birdseye = cv2.warpPerspective(image, M, (width, height))

        return birdseye, M

    def sliding_window_polyfit(self, binary_warped):
        height, width = binary_warped.shape

        histogram = np.sum(binary_warped[height//2:,:], axis=0)
        base_x = np.argmax(histogram)

        nwindows = 10
        window_height = int(height / nwindows)
        x_current = base_x
        lane_inds = []

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 60
        minpix = 50

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        offset = None

        if len(x) > 0 and len(y) > 0:
            fit = np.polyfit(y, x, 2)
            ploty = np.linspace(0, height-1, height)
            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

            # Calcul offset
            x_bottom = fit[0]*(height-1)**2 + fit[1]*(height-1) + fit[2]
            center_x = width // 2
            pixel_offset = x_bottom - center_x
            offset = pixel_offset / (width / 2)

            # Image noir + courbe blanche
            ppo_input = np.zeros((height, width), dtype=np.uint8)
            for (x_fit, y_fit) in zip(fitx, ploty):
                if 0 <= int(x_fit) < width and 0 <= int(y_fit) < height:
                    ppo_input[int(y_fit), int(x_fit)] = 255

            # Resize en 84x84
            ppo_input_resized = cv2.resize(ppo_input, (84, 84), interpolation=cv2.INTER_AREA)

            return offset, ppo_input_resized

        # Si rien trouvé
        return None, np.zeros((84, 84), dtype=np.uint8)
