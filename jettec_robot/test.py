#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SimpleMover(Node):
    def __init__(self):
        super().__init__('simple_mover')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # Timer pour publier toutes les 100 ms
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Simple mover node started")

    def timer_callback(self):
        twist = Twist()
        twist.linear.x = 0.5   # Avancer à vitesse linéaire constante
        twist.angular.z = 0.3  # Tourner avec une vitesse angulaire constante
        self.publisher_.publish(twist)
        self.get_logger().info("Commande publiée: linear.x=0.5, angular.z=0.3")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

