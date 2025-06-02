import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# 🌿 Importe directement la classe Vision (depuis ton code)
from jettec_robot.path_following.vision_path import Vision

class VisionTestNode(Node):
    def __init__(self):
        super().__init__('vision_test_node')
        self.bridge = CvBridge()

        # 🎯 Abonnement au topic image de la ZED
        self.subscriber = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',  # Ton topic ZED
            self.image_callback,
            10
        )

        self.get_logger().info("✅ VisionTestNode ready, waiting for images...")

    def image_callback(self, msg):
        try:
            # 🌈 Convertir l'image ROS en OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # 🔎 Traiter l'image avec Vision
            vision = Vision(cv_image, debug=False)
            offset, processed_image, line_state = vision.calculate_offset_and_threshold()

            # 🎨 Affichage de l'image processée
            cv2.imshow("Processed Image (84x84)", processed_image)
            cv2.waitKey(1)

            # 🖨️ Affiche dans le terminal aussi
            self.get_logger().info(f"Line State: {line_state} | Offset: {offset:.3f}")

        except Exception as e:
            self.get_logger().error(f"❌ image_callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
