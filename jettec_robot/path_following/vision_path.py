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

"""

"""

import cv2
import numpy as np

class Vision:
    def __init__(self, image, scale_factor=0.4, threshold=30, min_area=500, debug=False):
        self.rgb_image = image
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.min_area = min_area
        self.debug = debug

    def calculate_offset_and_threshold(self):
        h_orig, w_orig, _ = self.rgb_image.shape
        center_x = int(w_orig * self.scale_factor / 2)

        # 📦 Réduction de la taille pour accélérer
        scaled_image = cv2.resize(self.rgb_image, (int(w_orig * self.scale_factor), int(h_orig * self.scale_factor)))
        h, w, _ = scaled_image.shape

        # Moitié basse (0.6 à 1.0)
        clipped = scaled_image[int(h * 0.6):, :]
        clipped_h, clipped_w, _ = clipped.shape

        gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Morphologie pour remplir et fusionner
        kernel = np.ones((7,7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Contours et points bas
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ppo_input = np.zeros_like(closed)
        bottom_points = []

        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                cv2.drawContours(ppo_input, [c], -1, 255, thickness=-1)
                c = c.squeeze()
                if c.ndim == 1:
                    c = np.expand_dims(c, axis=0)
                y_max = np.max(c[:,1])
                x_at_y_max = c[c[:,1] == y_max][:,0]
                mean_x = int(np.mean(x_at_y_max))
                bottom_points.append((mean_x, y_max))
                if self.debug:
                    print(f"➡️ Point bas: (x={mean_x}, y={y_max})")

        # Calcul du line_state
        line_state = len(bottom_points) if len(bottom_points) <= 2 else 2

        # Calcul de l'offset
        if len(bottom_points) >= 2:
            avg_x = np.mean([p[0] for p in bottom_points])
            offset = (avg_x - center_x) / center_x
        elif len(bottom_points) == 1:
            avg_x = bottom_points[0][0]
            offset = (avg_x - center_x) / center_x
        else:
            offset = 0.0

        # Affichage debug si activé
        if self.debug:
            debug_display = cv2.cvtColor(ppo_input, cv2.COLOR_GRAY2BGR)
            for (x, y) in bottom_points:
                cv2.circle(debug_display, (int(x), int(y)), 5, (0,0,255), -1)
                cv2.putText(debug_display, f"({x},{y})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.imshow("Debug - Bottom Points", debug_display)
            cv2.waitKey(1)

        # Redimensionner pour PPO
        ppo_input_resized = cv2.resize(ppo_input, (84, 84), interpolation=cv2.INTER_AREA)

        return offset, ppo_input_resized, line_state
