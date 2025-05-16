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
ROS2 Node for line following vision processing.
Subscribes to RGB images, processes them to find the line, and publishes the offset and processed image.
Compatible with the new Vision class using sliding windows and 84x84 processed image.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from jettec_robot.line_following.vision_line import Vision
import cv2

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.bridge = CvBridge()
        self.subscriber = self.create_subscription(
            Image, '/rgbd_camera/image', self.image_callback, 10
        )
        self.offset_publisher = self.create_publisher(Float32, '/line_offset', 10)
        self.image_publisher = self.create_publisher(Image, '/processed_image', 10)

        self.get_logger().info("📸 VisionNode ready, waiting for images...")

    def image_callback(self, msg):
        try:
            # Convertir l'image ROS2 en OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Appliquer le traitement de vision
            vision = Vision(cv_image, debug=True)
            offset, processed_image = vision.calculate_offset_and_threshold()

            # >>>> AJOUT pour visualiser <<<<
            cv2.imshow("Processed PPO Image (84x84)", processed_image)
            cv2.waitKey(1)

            # Publier l'offset (ou NaN si rien détecté)
            if offset is not None:
                self.offset_publisher.publish(Float32(data=offset))
            else:
                self.offset_publisher.publish(Float32(data=float('nan')))

            # Publier l'image traitée
            processed_image = processed_image.astype(np.uint8)
            ros_img = self.bridge.cv2_to_imgmsg(processed_image, encoding="mono8")
            self.image_publisher.publish(ros_img)

        except Exception as e:
            self.get_logger().error(f"❌ Callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
