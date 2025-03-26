import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdometryListener(Node):
    def __init__(self):
        super().__init__('odometry_listener')
        # Souscrire au topic Gazebo d'odométrie
        self.subscription = self.create_subscription(
            Odometry,
            '/model/JetTec_Robot/odometry',  # Le topic d'odométrie Gazebo
            self.odom_callback,
            10  # Taille du message (10 Hz)
        )

    def odom_callback(self, msg):
        """
        Callback pour recevoir l'odométrie et afficher les informations.
        """
        # Récupérer la position et l'orientation du robot
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        orientation = msg.pose.pose.orientation

        # Afficher la position et l'orientation dans les logs
        self.get_logger().info(f'Position: x={x}, y={y}, z={z}')
        self.get_logger().info(f'Orientation: {orientation}')

def main(args=None):
    rclpy.init(args=args)

    # Créer une instance de notre node
    odometry_listener = OdometryListener()

    # Exécuter le node
    rclpy.spin(odometry_listener)

    # Nettoyage après exécution
    odometry_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

