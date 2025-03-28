#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        # Souscription au topic de la caméra (ajustez le nom du topic si besoin)
        self.subscription = self.create_subscription(
            Image,
            '/rgbd_camera/image',
            self.image_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/rgbd_camera/depth_image',
            self.depth_callback,
            10
        )
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
            self.get_logger().error("Erreur de conversion: " + str(e))

    def depth_callback(self, msg):
        """
        Callback pour recevoir l'image de profondeur.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')  # Image de profondeur en float
        except CvBridgeError as e:
            self.get_logger().error("Erreur de conversion de l'image de profondeur: " + str(e))

        # Une fois l'image de profondeur reçue, calculer le point le plus proche
        if self.rgb_image is not None and self.depth_image is not None:
            self.calculate_and_display_results()

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

        # Si l'on a au moins un contour détecté, calculer le vecteur entre le centroïde et le top point
        if contour_data:
            # Ici, nous allons utiliser seulement le premier contour pour calculer le vecteur
            centroid = contour_data[0]['centroid']
            top_point = contour_data[0]['top']

            # Calcul du vecteur entre le centroïde et le top point
            vector = (top_point[0] - centroid[0], top_point[1] - centroid[1])

            # Affichage du vecteur sous forme de flèche
            cv2.arrowedLine(image_with_centroids, centroid, top_point, (255, 0, 0), 2)

            # Calcul de l'angle entre le vecteur du centroïde au top point et le vecteur vertical
            center_x = w // 2
            start_point = (center_x, h)  # Point de départ au centre bas de l'image
            end_point = (center_x, h - 100)  # Taille arbitraire de la flèche verticale (ici 100 pixels vers le haut)
            vector_vertical = (end_point[0] - start_point[0], end_point[1] - start_point[1])

            # Calcul de l'angle entre les deux vecteurs (en utilisant np.arctan2)
            angle_rad = np.arctan2(vector[1], vector[0]) - np.arctan2(vector_vertical[1], vector_vertical[0])
            
            # Affichage de l'angle dans les logs
            self.get_logger().info(f"Angle entre les vecteurs: {angle_rad:.2f} rad")

            # Affichage de l'angle sur l'image
            cv2.putText(image_with_centroids, f"Angle: {angle_rad:.2f} rad", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Calcul du vecteur unitaire vertical partant du centre bas de l'image
        center_x = w // 2
        start_point = (center_x, h)  # Point de départ au centre bas de l'image (h = hauteur)
        end_point = (center_x, h - 100)  # Taille arbitraire de la flèche verticale (ici 100 pixels vers le haut)
        
        # Dessiner un vecteur unitaire vertical (flèche) en cyan
        cv2.arrowedLine(image_with_centroids, start_point, end_point, (255, 255, 0), 2)

        # Trouver le point le plus proche de la ligne
        closest_point, closest_distance = self.find_closest_point(thresh, clipped_depth)

        # Dessiner le point le plus proche sur l'image
        if closest_point is not None:
            cv2.circle(image_with_centroids, closest_point, 10, (0, 255, 255), -1)  # Point en jaune

            # Afficher la distance sur l'image
            cv2.putText(image_with_centroids, f"Distance: {closest_distance:.2f} m", (closest_point[0] + 10, closest_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Affichage dans trois fenêtres
        cv2.imshow('Image Threshold', thresh)  # Afficher l'image seuillée
        cv2.imshow('Image with Centroids and Top Points', image_with_centroids)  # Afficher l'image originale avec les points et le vecteur
        cv2.imshow('Depth Map', self.depth_image)  # Afficher la carte de profondeur originale
        cv2.waitKey(1)

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
                
        if closest_point is not None:
            self.get_logger().info(f"Point le plus proche trouvé à la position {closest_point} avec une distance de {min_distance:.2f} mètres")


        return closest_point, min_distance if closest_point is not None else (None, None)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

