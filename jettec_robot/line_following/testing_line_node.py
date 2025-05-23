# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /___   _______/  /_   _____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__    
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""
ROS2 Node for testing a trained PPO policy to follow a line.
Loads a trained CNNActorCritic model, subscribes to vision input, 
and commands the robot in real time based on policy inference.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import torch
import numpy as np

from jettec_robot.line_following.model_line import CNNActorCritic

# === CONFIGURATION ===
MODEL_PATH = "/home/mrsl/ros2_ws/checkpoints/20250519_173249/model_ep100.pth"
LINEAR_VELOCITY = 1.0
IMAGE_SIZE = (1, 84, 84)

class LineFollowerTestNode(Node):
    def __init__(self):
        super().__init__('test_line_node')
        self.bridge = CvBridge()

        # === Charger modèle entraîné ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNActorCritic(1, action_size=1, device=self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.get_logger().info(f"🧠 Modèle chargé depuis {MODEL_PATH}")

        # === Souscriptions et publications ===
        self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            input_tensor = torch.tensor(cv_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                action_dist, _ = self.model(input_tensor)  # action_dist est un torch.distributions.Normal
                action_sample = action_dist.sample()
                action = float(torch.tanh(action_sample)[0, 0].cpu().numpy())

            # Sécurité
            if np.isnan(action):
                self.get_logger().warn("⚠️ Action is NaN, replacing with 0.")
                action = 0.0

            self.publish_velocity(LINEAR_VELOCITY, action)

        except Exception as e:
            self.get_logger().error(f"❌ image_callback: {e}")


    def publish_velocity(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

