#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist

class AckermannToTwist(Node):
    def __init__(self):
        super().__init__('ackermann_to_twist')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(AckermannDrive, '/cmd_vel_raw', self.cb, 10)

    def cb(self, msg):
        twist = Twist()
        twist.linear.x = msg.speed
        twist.angular.z = msg.steering_angle
        self.pub.publish(twist)

def main():
    rclpy.init()
    node = AckermannToTwist()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

