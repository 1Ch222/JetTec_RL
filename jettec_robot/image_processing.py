#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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
        self.bridge = CvBridge()
        # Variables pour stocker les vecteurs (droite et gauche) et l'angle theta
        self.vector_right = None
        self.vector_left = None
        self.theta = None  # angle de différence entre le vecteur normal et le vecteur somme

    def image_callback(self, msg):
        try:
            # Conversion du message ROS Image en image OpenCV (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error("Erreur de conversion: " + str(e))
            return

        # Récupération de la partie inférieure de l'image (zone d'intérêt)
        h, w, _ = cv_image.shape
        clipped = cv_image[int(h * 0.5):, :]

        # Conversion en niveaux de gris et flou pour réduire le bruit
        gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Seuillage pour isoler les zones blanches (ajustez la valeur seuil selon l'éclairage)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)

        # Détection des contours sur l'image seuillée
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par aire pour éliminer le bruit
        min_area = 500  # à ajuster selon votre cas
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Copier l'image pour y dessiner les centroïdes, top points et vecteurs
        image_with_centroids = clipped.copy()

        # Stocker les informations (centroïde et top point) pour chaque contour
        contour_data = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                # Le top point est celui avec la plus petite coordonnée y (point le plus haut)
                top_point = min(cnt, key=lambda p: p[0][1])[0]
                contour_data.append({
                    'centroid': centroid,
                    'top': (top_point[0], top_point[1]),
                    'contour': cnt
                })
                # Dessiner le centroïde et le top point
                cv2.circle(image_with_centroids, centroid, 5, (0, 0, 255), -1)
                cv2.circle(image_with_centroids, (top_point[0], top_point[1]), 7, (0, 255, 0), -1)
                cv2.putText(image_with_centroids, "Top", (top_point[0] + 10, top_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Si l'on a au moins un contour détecté, sélectionner ceux dont le centroïde est le plus à droite et le plus à gauche
        if contour_data:
            rightmost = max(contour_data, key=lambda d: d['centroid'][0])
            leftmost = min(contour_data, key=lambda d: d['centroid'][0])
            
            # Calcul des vecteurs: du centroïde au top point pour chaque contour
            vector_right = (rightmost['top'][0] - rightmost['centroid'][0],
                            rightmost['top'][1] - rightmost['centroid'][1])
            vector_left = (leftmost['top'][0] - leftmost['centroid'][0],
                           leftmost['top'][1] - leftmost['centroid'][1])
            
            # Stocker les vecteurs
            self.vector_right = vector_right
            self.vector_left = vector_left
            
            self.get_logger().info(f"Vecteur droit: {vector_right}, Vecteur gauche: {vector_left}")
            
            # Tracer les vecteurs sur l'image (en bleu)
            cv2.arrowedLine(image_with_centroids, rightmost['centroid'], rightmost['top'], (255, 0, 0), 2)
            cv2.arrowedLine(image_with_centroids, leftmost['centroid'], leftmost['top'], (255, 0, 0), 2)
        
        # Définition du base point : centre de la limite basse de l'image "clipped"
        base_point = (w // 2, clipped.shape[0] - 1)
        scale = 50  # Échelle pour visualisation
        
        # Tracé du vecteur normal unitaire (pointant vers le haut)
        unit_normal = (0, -1)  # Vecteur unitaire pointant vers le haut (y décroît en OpenCV)
        normal_end = (base_point[0] + int(unit_normal[0] * scale),
                      base_point[1] + int(unit_normal[1] * scale))
        cv2.arrowedLine(image_with_centroids, base_point, normal_end, (0, 255, 255), 2)
        cv2.putText(image_with_centroids, "Normal", (normal_end[0] + 10, normal_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Tracé du vecteur somme (direction de la somme des vecteurs droit et gauche)
        if self.vector_right is not None and self.vector_left is not None:
            # Somme des vecteurs
            vector_sum = (self.vector_right[0] + self.vector_left[0],
                          self.vector_right[1] + self.vector_left[1])
            # Calcul de la norme
            norm = np.hypot(vector_sum[0], vector_sum[1])
            if norm != 0:
                # Normalisation pour obtenir un vecteur unitaire
                unit_sum = (vector_sum[0] / norm, vector_sum[1] / norm)
                sum_end = (base_point[0] + int(unit_sum[0] * scale),
                           base_point[1] + int(unit_sum[1] * scale))
                # Tracer le vecteur somme (en magenta)
                cv2.arrowedLine(image_with_centroids, base_point, sum_end, (255, 0, 255), 2)
                cv2.putText(image_with_centroids, "Sum", (sum_end[0] + 10, sum_end[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                self.get_logger().info(f"Vecteur somme (normalisé): {unit_sum}")
                
                # Calcul de l'angle theta entre le vecteur normal et le vecteur somme
                # On utilise np.arctan2 pour obtenir l'angle en radians
                angle_normal = np.arctan2(unit_normal[1], unit_normal[0])
                angle_sum = np.arctan2(unit_sum[1], unit_sum[0])
                self.theta = angle_sum - angle_normal
                self.get_logger().info(f"Angle theta (différence): {self.theta} rad")
                
                # Tracé du vecteur somme est déjà effectué
            else:
                self.get_logger().info("La somme des vecteurs est nulle, impossible de normaliser.")

        # Affichage des images
        cv2.imshow('Image Segmentée', thresh)
        cv2.imshow('Centroides, Top Points et Vecteurs', image_with_centroids)
        cv2.waitKey(1)

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

