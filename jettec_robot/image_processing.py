import cv2
import numpy as np

class LineDetection:
    def __init__(self):
        pass

    def process_image(self, cv_image, depth_image):
        """
        Process the image to detect the black line, convert it to white, and calculate the angle theta.
        """
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un flou gaussien pour réduire le bruit
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Appliquer un seuillage pour isoler les contours de la ligne
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Créer une image vide de la même taille, remplie de noir
        output_image = np.zeros_like(cv_image)
        
        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours par aire
        min_area = 500
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Si des contours sont trouvés, dessiner la ligne sur l'image de sortie
        if filtered_contours:
            cv2.drawContours(output_image, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # Calculer l'angle theta entre la ligne et l'orientation du robot
        theta = self.compute_theta(filtered_contours)
        
        return output_image, theta
    
    def compute_theta(self, contours):
        """
        Calcule l'angle (theta) entre la ligne détectée et l'orientation du robot.
        """
        if len(contours) > 0:
            # Calculer le moment pour obtenir les centroïdes des contours
            moments = cv2.moments(contours[0])
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Estimer la direction de la ligne par rapport à l'horizon
            # Supposons que l'orientation du robot est donnée par un vecteur horizontal (par exemple [1, 0])
            line_direction = np.array([cx, cy])
            robot_orientation = np.array([1, 0])  # Orientation hypothétique du robot (horizontal)
            
            # Calculer l'angle entre la direction de la ligne et l'orientation du robot
            angle = np.arctan2(line_direction[1], line_direction[0]) - np.arctan2(robot_orientation[1], robot_orientation[0])
            angle = np.rad2deg(angle)  # Convertir en degrés
            return angle
        return 0


