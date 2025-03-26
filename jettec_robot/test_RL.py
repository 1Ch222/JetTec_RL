#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import math
import random
import time
import subprocess

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

# Paramètres globaux
MAX_EPISODES = 300
EPISODE_DURATION = 30  # durée d'un épisode en secondes

def reset_world():
    """
    Réinitialise le monde via ign service.
    Adaptez la commande si nécessaire selon votre configuration.
    """
    cmd = [
        "ign", "service", "-s", "/world/Line_track/reset_simulation",
        "--reqtype", "ignition.msgs.Empty",
        "--reptype", "ignition.msgs.Empty",
        "--timeout", "1000"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("World reset successfully.")
        else:
            print("Failed to reset world. Error:", result.stderr)
    except Exception as e:
        print("Exception during world reset:", e)

def reset_robot():
    """
    Réinitialise la pose du robot en utilisant le service /world/Line_track/set_pose.
    Le robot est remis à la position initiale : x=-1.0, y=-0.90, z=0.15, orientation identitaire.
    """
    cmd = [
    "ign", "service", "-s", "/world/Line_track/set_pose",
    "--reqtype", "ignition.msgs.Pose",
    "--reptype", "ignition.msgs.Boolean",
    "--timeout", "1000",
    "--req", "name: 'JetTec_Robot' position: { x: -0.5, y: -1.7, z: 0.15 } orientation: { x: 0.0, y: 0, z: 0.14, w: 1 }"
]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Robot reset successfully using set_pose service.")
        else:
            print("Failed to reset robot using set_pose service. Error:", result.stderr)
    except Exception as e:
        print("Exception during robot reset using set_pose service:", e)

class JettecEnv(Node):
    def __init__(self):
        super().__init__('jettec_env')
        # Souscription aux images de la caméra
        self.subscription = self.create_subscription(
            Image,
            '/rgbd_camera/image',
            self.image_callback,
            10
        )
        # Publication sur le topic de commande du robot
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # Paramètre pour la reward basée sur theta
        self.K = 0.2

        # Variables d'état
        self.state = None          # état courant (0,1,2)
        self.last_reward = 0.0     # dernière récompense calculée
        self.done = False          # flag de fin d'épisode
        self.last_theta = None     # dernier theta calculé

    def discretize_state(self, theta):
        if theta < -0.5:
            return 0
        elif theta > 0.5:
            return 2
        else:
            return 1

    def compute_theta(self, unit_sum):
        angle_normal = math.atan2(-1, 0)  # -pi/2
        angle_sum = math.atan2(unit_sum[1], unit_sum[0])
        theta = angle_sum - angle_normal
        theta = math.atan2(math.sin(theta), math.cos(theta))
        return theta

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error("Erreur de conversion: " + str(e))
            return

        # Conserver la moitié inférieure de l'image (zone d'intérêt)
        h, w, _ = cv_image.shape
        clipped = cv_image[int(h * 0.5):, :]

        # Prétraitement de l'image
        gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par aire
        min_area = 500
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Récupérer centroïdes et top points
        contour_data = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                top_point = min(cnt, key=lambda p: p[0][1])[0]
                contour_data.append({'centroid': centroid, 'top': (top_point[0], top_point[1])})

        if len(contour_data) >= 2:
            rightmost = max(contour_data, key=lambda d: d['centroid'][0])
            leftmost = min(contour_data, key=lambda d: d['centroid'][0])
            vector_right = (rightmost['top'][0] - rightmost['centroid'][0],
                            rightmost['top'][1] - rightmost['centroid'][1])
            vector_left = (leftmost['top'][0] - leftmost['centroid'][0],
                           leftmost['top'][1] - leftmost['centroid'][1])
            vector_sum = (vector_right[0] + vector_left[0],
                          vector_right[1] + vector_left[1])
            norm = np.hypot(vector_sum[0], vector_sum[1])
            if norm != 0:
                unit_sum = (vector_sum[0] / norm, vector_sum[1] / norm)
            else:
                unit_sum = (0, 0)
            theta = self.compute_theta(unit_sum)
            self.last_theta = theta
            self.state = self.discretize_state(theta)
            self.last_reward = self.K * np.cos(theta)
            self.done = False
        elif len(contour_data) == 1:
            self.state = 1  # état par défaut si une seule voie détectée
            self.last_reward = -0.5
            self.done = False
        else:
            self.state = 1
            self.last_reward = -1
            self.done = True

        # Affichage pour debug
        debug_image = clipped.copy()
        for data in contour_data:
            cv2.circle(debug_image, data['centroid'], 5, (0, 0, 255), -1)
            cv2.circle(debug_image, data['top'], 7, (0, 255, 0), -1)
            cv2.putText(debug_image, "Top", (data['top'][0] + 10, data['top'][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Debug', debug_image)
        cv2.waitKey(1)

    def step(self, action):
        twist = Twist()
        twist.linear.x = 1.0  # vitesse linéaire constante
        if action == 0:
            twist.angular.z = 0.0  # tout droit
        elif action == 1:
            twist.angular.z = 0.5  # tourner à gauche
        elif action == 2:
            twist.angular.z = -0.5  # tourner à droite

        self.cmd_publisher.publish(twist)
        time.sleep(0.1)
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.state, self.last_reward, self.done, {}

    def reset(self):
    	self.get_logger().info("Resetting world via gz transport...")
    	reset_world()
    	time.sleep(1.0)
    	self.get_logger().info("Resetting robot position via set_pose service...")
    	reset_robot()

    	# Attendre que la caméra détecte au moins une ligne
    	self.done = False
    	self.state = None
    	self.last_reward = 0.0
    	max_wait = 5.0  # attendre au maximum 5 secondes
    	start_wait = time.time()
    	while self.state is None and (time.time() - start_wait < max_wait):
        	rclpy.spin_once(self, timeout_sec=0.1)
        	time.sleep(0.1)
    	if self.state is None:
        	self.get_logger().warn("Aucune ligne détectée après reset, démarrage de l'épisode malgré tout.")
        	self.state = 1  # état par défaut
    	return self.state


def simulate(env, q_table, alpha, gamma, epsilon, epsilon_decay):
    rewards = []
    avgrewards = []
    for episode in range(1, MAX_EPISODES):
        env.get_logger().info("============= STARTING NEW EPISODE ===============")
        state = env.reset()
        total_reward = 0
        total_steps = 0
        start_time = time.time()

        done = False
        while not done and (time.time() - start_time < EPISODE_DURATION):
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            state = next_state
            total_steps += 1

        avg_reward = total_reward / (total_steps if total_steps > 0 else 1)
        rewards.append(avg_reward)
        avgrewards.append(np.mean(rewards[-50:]))
        env.get_logger().info(f"Episode finished: Avg reward = {avg_reward:.2f}")
        if epsilon >= 0.005:
            epsilon *= epsilon_decay

    try:
        import matplotlib.pyplot as plt
        plt.plot(avgrewards)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title("Learning Curve")
        plt.show()
    except Exception as e:
        env.get_logger().error("Erreur d'affichage du graphique: " + str(e))

def main(args=None):
    rclpy.init(args=args)
    env = JettecEnv()
    q_table = np.zeros((3, 3))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    epsilon_decay = 0.99
    try:
        simulate(env, q_table, alpha, gamma, epsilon, epsilon_decay)
    except KeyboardInterrupt:
        env.get_logger().info("Entraînement interrompu.")
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

