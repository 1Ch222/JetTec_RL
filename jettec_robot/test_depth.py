#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node

class ClosestPointToLine(Node):
    def __init__(self):
        super().__init__('closest_point_to_line')
        
        # Souscription aux images RGB et Depth de la caméra
        self.rgb_subscription = self.create_subscription(
            Image,
            '/rgbd_camera/image',
            self.rgb_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/rgbd_camera/depth_image',
            self.depth_callback,
            10
        )
        
        self.bridge = CvBridge()
        
        # Variables pour l'image de profondeur et l'image RGB
        self.rgb_image = None
        self.depth_image = None

    def rgb_callback(self, msg):
        """
        Callback pour recevoir les images RGB
        """
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error("Erreur de conversion de l'image RGB: " + str(e))

    def depth_callback(self, msg):
        """
        Callback pour recevoir les images de profondeur (depth).
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')  # Image de profondeur en float
        except CvBridgeError as e:
            self.get_logger().error("Erreur de conversion de l'image de profondeur: " + str(e))

        # Après avoir reçu l'image de profondeur, on peut calculer le point le plus proche de la ligne
        if self.rgb_image is not None and self.depth_image is not None:
            self.calculate_closest_point()

    def calculate_closest_point(self):
        """
        Calcule le point de la ligne le plus proche de la caméra en utilisant l'image de profondeur.
        """
        # Récupérer la partie inférieure de l'image RGB (zone d'intérêt)
        h, w, _ = self.rgb_image.shape
        clipped_rgb = self.rgb_image[int(h * 0.5):, :]
        clipped_depth = self.depth_image[int(h * 0.5):, :]  # Partie correspondante de l'image de profondeur

        # Conversion en niveaux de gris et flou pour réduire le bruit
        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Seuillage pour isoler la ligne (blanche sur fond noir)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)

        # Créer un masque binaire des pixels correspondant à la ligne
        mask = thresh == 255  # Pixels de la ligne blanche

        # Extraire les valeurs de profondeur des pixels de la ligne
        depth_values = clipped_depth[mask]

        if len(depth_values) > 0:
            # Trouver le point le plus proche de la caméra (la profondeur minimale)
            min_depth = np.min(depth_values)
            min_depth_index = np.argmin(depth_values)

            # Récupérer les coordonnées du point le plus proche
            closest_point_y, closest_point_x = np.unravel_index(min_depth_index, clipped_depth.shape)

            # Afficher les résultats
            self.get_logger().info(f"Point le plus proche trouvé : ({closest_point_x}, {closest_point_y}) avec une distance de {min_depth:.2f} mètres")

            # Dessiner le point le plus proche sur l'image
            closest_point = (closest_point_x, closest_point_y)
            cv2.circle(clipped_rgb, closest_point, 10, (0, 0, 255), -1)  # Point en rouge

        # Affichage des résultats
        cv2.imshow('Threshold Image', thresh)  # Afficher l'image seuillée
        cv2.imshow('Image with Closest Point', clipped_rgb)  # Afficher l'image avec le point le plus proche
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ClosestPointToLine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

