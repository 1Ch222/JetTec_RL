import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
import math

class ProgressionAlongLine(Node):
    def __init__(self):
        super().__init__('progression_along_line_node')

        # Abonnement au topic /odom pour obtenir l'odométrie du robot
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Variables pour la gestion de la ligne
        self.line_points = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Exemple de points de la ligne (à adapter)
        self.previous_position = None  # Position précédente
        self.distance_travelled = 0.0  # Distance parcourue le long de la ligne

    def odom_callback(self, msg):
        """
        Callback pour recevoir les messages d'odométrie et calculer la distance parcourue le long de la ligne.
        """
        # Récupérer la position du robot à partir de l'odométrie
        current_position = msg.pose.pose.position
        current_position = (current_position.x, current_position.y)

        if self.previous_position is None:
            # Initialiser la position précédente si c'est la première itération
            self.previous_position = current_position
            return

        # Calculer la distance parcourue sur la ligne
        distance = self.calculate_distance_along_line(self.previous_position, current_position)

        # Ajouter cette distance à la distance totale parcourue le long de la ligne
        self.distance_travelled += distance

        # Mettre à jour la position précédente pour la prochaine itération
        self.previous_position = current_position

        # Afficher la progression dans les logs
        self.get_logger().info(f"Distance parcourue le long de la ligne depuis la dernière itération : {distance:.2f} m")
        self.get_logger().info(f"Distance totale parcourue le long de la ligne : {self.distance_travelled:.2f} m")

    def calculate_distance_along_line(self, prev_pos, current_pos):
        """
        Calcule la distance parcourue le long de la ligne définie par une série de points.
        Utilise la projection de la position actuelle du robot sur la ligne.
        """
        # Initialiser une très grande distance pour trouver le plus proche point de la ligne
        min_distance = float('inf')

        # Boucle sur chaque segment de la ligne
        for i in range(1, len(self.line_points)):
            # Les points successifs de la ligne
            p1 = self.line_points[i - 1]
            p2 = self.line_points[i]

            # Calculer la projection de la position actuelle du robot sur le segment de la ligne
            dist_to_segment = self.project_point_onto_line(prev_pos, current_pos, p1, p2)

            # Si la distance est plus petite, mettre à jour la distance minimale
            if dist_to_segment < min_distance:
                min_distance = dist_to_segment

        return min_distance

    def project_point_onto_line(self, prev_pos, current_pos, p1, p2):
        """
        Projette la position actuelle du robot sur un segment de la ligne défini par deux points (p1, p2).
        """
        # Vecteurs (p1 -> p2) et (p1 -> position du robot)
        line_vector = (p2[0] - p1[0], p2[1] - p1[1])
        robot_vector = (current_pos[0] - p1[0], current_pos[1] - p1[1])

        # Calculer la projection du vecteur robot sur le vecteur de la ligne
        dot_product = robot_vector[0] * line_vector[0] + robot_vector[1] * line_vector[1]
        line_length_squared = line_vector[0] ** 2 + line_vector[1] ** 2

        projection_scale = dot_product / line_length_squared

        # Calculer le point projeté sur la ligne
        projected_point = (p1[0] + projection_scale * line_vector[0], p1[1] + projection_scale * line_vector[1])

        # Calculer la distance entre le robot et le point projeté
        distance = math.sqrt((projected_point[0] - current_pos[0]) ** 2 + (projected_point[1] - current_pos[1]) ** 2)

        return distance


def main(args=None):
    rclpy.init(args=args)
    node = ProgressionAlongLine()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

