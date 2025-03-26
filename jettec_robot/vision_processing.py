#!/usr/bin/env python3

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class VisionDistance:
    def __init__(self):
        self.bridge = CvBridge()

        # Variables pour l'image RGB et Depth
        self.rgb_image = None
        self.depth_image = None

    def image_callback(self, msg):
        """
        Callback pour recevoir l'image RGB.
        """
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Erreur de conversion: {str(e)}")

    def depth_callback(self, msg):
        """
        Callback pour recevoir l'image de profondeur.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')  # Image de profondeur en float
        except CvBridgeError as e:
            print(f"Erreur de conversion de l'image de profondeur: {str(e)}")

        # Calcul des résultats après avoir reçu les deux images
        if self.rgb_image is not None and self.depth_image is not None:
            return self.calculate_and_display_results()

    def calculate_and_display_results(self):
        """
        Calcule le centroïde, top point, vecteur theta et le point le plus proche de la ligne.
        """
        # Récupérer la partie inférieure de l'image RGB et de l'image de profondeur (zone d'intérêt)
        h, w, _ = self.rgb_image.shape
        clipped_rgb = self.rgb_image[int(h * 0.5):, :]
        clipped_depth = self.depth_image[int(h * 0.5):, :]  # Partie correspondante de l'image de profondeur

        # Conversion en niveaux de gris et flou pour réduire le bruit
        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Seuillage pour isoler la ligne (blanche sur fond noir)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)

        # Détection des contours sur l'image seuillée
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par aire pour éliminer le bruit
        min_area = 500  # à ajuster selon votre cas
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Copier l'image pour y dessiner les centroïdes et les points culminants
        image_with_centroids = clipped_rgb.copy()

        # Stocker les informations (centroïde et top point) pour chaque contour
        contour_data = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                # Le top point est celui avec le minimum y (point le plus haut)
                top_point = min(cnt, key=lambda p: p[0][1])[0]
                contour_data.append({
                    'centroid': centroid,
                    'top': (top_point[0], top_point[1]),
                    'contour': cnt
                })
                # Dessiner le centroïde
                cv2.circle(image_with_centroids, centroid, 5, (0, 0, 255), -1)
                # Dessiner le top point
                cv2.circle(image_with_centroids, (top_point[0], top_point[1]), 7, (0, 255, 0), -1)
                cv2.putText(image_with_centroids, "Top", (top_point[0] + 10, top_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculer le point le plus proche de la ligne
        closest_point, closest_distance = self.find_closest_point(thresh, clipped_depth)

        # Dessiner le point le plus proche sur l'image
        if closest_point is not None:
            cv2.circle(image_with_centroids, closest_point, 10, (0, 255, 255), -1)  # Point en jaune
            # Afficher la distance sur l'image
            cv2.putText(image_with_centroids, f"Distance: {closest_distance:.2f} m", (closest_point[0] + 10, closest_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Affichage dans deux fenêtres
        cv2.imshow('Image Threshold', thresh)  # Afficher l'image seuillée
        cv2.imshow('Image with Centroids and Top Points', image_with_centroids)  # Afficher l'image originale avec les points et le vecteur
        cv2.imshow('Depth Map', self.depth_image)  # Afficher la carte de profondeur originale
        cv2.waitKey(1)

        return centroid, top_point, closest_point, closest_distance

    def find_closest_point(self, thresh, clipped_depth):
        """
        Trouve le point de la ligne le plus proche de la caméra à partir de l'image de profondeur.
        """
        # Utiliser les pixels blancs dans `thresh` comme mask
        y_indices, x_indices = np.where(thresh == 255)

        # Initialiser la distance minimale
        min_distance = float('inf')
        closest_point = None

        # Comparer les valeurs de profondeur des pixels blancs
        for y, x in zip(y_indices, x_indices):
            # Récupérer la valeur de profondeur correspondante dans l'image de profondeur
            depth_value = clipped_depth[y, x]

            # Si la profondeur est plus petite que la distance minimale trouvée jusqu'à présent, la mettre à jour
            if depth_value < min_distance:
                min_distance = depth_value
                closest_point = (x, y)

        return closest_point, min_distance if closest_point is not None else (None, None)

