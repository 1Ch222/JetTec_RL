#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
import argparse

class CmdVelPublisher(Node):
    def __init__(self, speed, steering, rate):
        super().__init__('cmd_vel_publisher')
        self.pub = self.create_publisher(AckermannDrive, '/cmd_vel', 10)
        self.msg = AckermannDrive()
        self.msg.speed = speed
        self.msg.steering_angle = steering
        self.timer = self.create_timer(1.0 / rate, self.publish_cmd)

    def publish_cmd(self):
        self.pub.publish(self.msg)
        self.get_logger().info(
            f'Published → speed: {self.msg.speed:.2f} m/s | steering_angle: {self.msg.steering_angle:.2f} rad'
        )

def main():
    parser = argparse.ArgumentParser(
        description='Publish AckermannDrive commands to /cmd_vel'
    )
    parser.add_argument('--speed',    type=float, default=1.0, help='Vitesse (m/s)')
    parser.add_argument('--steering', type=float, default=0.0, help='Angle de braquage (rad)')
    parser.add_argument('--rate',     type=float, default=10.0, help='Fréquence de publication (Hz)')
    args = parser.parse_args()

    rclpy.init()
    node = CmdVelPublisher(args.speed, args.steering, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

