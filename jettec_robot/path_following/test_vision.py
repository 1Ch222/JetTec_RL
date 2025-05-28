import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class PathVisionCannyMergeNode(Node):
    def __init__(self):
        super().__init__('path_vision_canny_merge_node')
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, '/rgbd_camera/image', self.image_callback, 10)
        self.get_logger().info("🚀 Path Vision Canny Merge Node Ready.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            binary_filled, binary_resized, line_count, offset, clipped_rgb = self.process_image(cv_image)

            self.get_logger().info(f"📏 Offset: {offset:.3f} | 📊 Lignes détectées: {line_count}")

            # Affichage avec points bas
            display = cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)
            for (x, y) in self.last_bottom_points:
                cv2.circle(display, (int(x), int(y)), 5, (0,0,255), -1)
                cv2.putText(display, f"({x},{y})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            cv2.imshow("Clipped RGB", clipped_rgb)
            cv2.imshow("Binary Final (Canny + Merge)", display)
            cv2.imshow("Binary Resized (84x84)", binary_resized)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Erreur image_callback: {e}")

    def process_image(self, rgb_image):
        h, w, _ = rgb_image.shape
        center_x = w // 2

        # Moitié basse (Canny sur 0.6 à 1.0)
        clipped_rgb = rgb_image[int(h * 0.6):, :]
        clipped_h, clipped_w, _ = clipped_rgb.shape

        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny sur 0.6 à 1.0
        edges = cv2.Canny(blur, 50, 150)

        # Remplir les contours + fusion des blobs
        kernel = np.ones((7,7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Contours des zones fusionnées
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Image binaire finale (lignes remplies)
        binary_filled = np.zeros_like(closed)
        bottom_points = []
        line_count = 0

        for c in contours:
            if cv2.contourArea(c) > 100:
                cv2.drawContours(binary_filled, [c], -1, 255, thickness=-1)
                c = c.squeeze()
                if c.ndim == 1:
                    c = np.expand_dims(c, axis=0)

                y_max = np.max(c[:,1])
                x_at_y_max = c[c[:,1] == y_max][:,0]
                mean_x = int(np.mean(x_at_y_max))
                bottom_points.append((mean_x, y_max))
                self.get_logger().info(f"➡️ Point bas: (x={mean_x}, y={y_max})")
                line_count += 1

        # Sauvegarde pour affichage
        self.last_bottom_points = bottom_points

        # Calcul offset basé sur ces points bas
        if len(bottom_points) >= 2:
            avg_x = np.mean([p[0] for p in bottom_points])
            offset = (avg_x - center_x) / center_x
        elif len(bottom_points) == 1:
            avg_x = bottom_points[0][0]
            offset = (avg_x - center_x) / center_x
        else:
            offset = 0.0

        # Resize final pour PPO
        binary_resized = cv2.resize(binary_filled, (84, 84))

        return binary_filled, binary_resized, line_count, offset, clipped_rgb

def main(args=None):
    rclpy.init(args=args)
    node = PathVisionCannyMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
