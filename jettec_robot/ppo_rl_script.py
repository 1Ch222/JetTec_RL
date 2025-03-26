#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import math
import random
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# --- Hyper-parameters ---

# PPO Hyperparameters
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
GAMMA = 0.99  # Discount factor for future rewards
CLIP_EPSILON = 0.2  # Clip epsilon for PPO
ACTION_STD = 0.5  # Standard deviation for action space (continuous)
EPOCHS = 10  # Number of epochs
EPISODES_PER_EPOCH = 100  # Episodes per epoch

# PID Controller Hyperparameters
Kp = 1.0  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.1  # Derivative gain

# Vision Hyperparameters
THRESHOLD = 30  # Threshold for binary image in vision processing
MIN_AREA = 500  # Minimum contour area to consider

# --- Reset Functions ---

def reset_world():
    """
    Réinitialise le monde via ign service.
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
    """
    cmd = [
        "ign", "service", "-s", "/world/Line_track/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", "name: 'JetTec_Robot' position: { x: -1.0, y: -0.9, z: 0.15 } orientation: { x: 0, y: 0, z: 0, w: 1 }"
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Robot reset successfully using set_pose service.")
        else:
            print("Failed to reset robot using set_pose service. Error:", result.stderr)
    except Exception as e:
        print("Exception during robot reset using set_pose service:", e)

# --- VisionDistance Class ---

class VisionDistance:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

    def image_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Erreur de conversion: {str(e)}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            print(f"Erreur de conversion de l'image de profondeur: {str(e)}")

        if self.rgb_image is not None and self.depth_image is not None:
            return self.calculate_and_display_results()

    def calculate_and_display_results(self):
        h, w, _ = self.rgb_image.shape
        clipped_rgb = self.rgb_image[int(h * 0.5):, :]
        clipped_depth = self.depth_image[int(h * 0.5):, :]
        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        image_with_centroids = clipped_rgb.copy()

        contour_data = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                top_point = min(cnt, key=lambda p: p[0][1])[0]
                contour_data.append({
                    'centroid': centroid,
                    'top': (top_point[0], top_point[1]),
                    'contour': cnt
                })
                cv2.circle(image_with_centroids, centroid, 5, (0, 0, 255), -1)
                cv2.circle(image_with_centroids, (top_point[0], top_point[1]), 7, (0, 255, 0), -1)
                cv2.putText(image_with_centroids, "Top", (top_point[0] + 10, top_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        closest_point, closest_distance = self.find_closest_point(thresh, clipped_depth)

        if closest_point is not None:
            cv2.circle(image_with_centroids, closest_point, 10, (0, 255, 255), -1)
            cv2.putText(image_with_centroids, f"Distance: {closest_distance:.2f} m", (closest_point[0] + 10, closest_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Image Threshold', thresh)
        cv2.imshow('Image with Centroids and Top Points', image_with_centroids)
        cv2.imshow('Depth Map', self.depth_image)
        cv2.waitKey(1)

        return centroid, top_point, closest_point, closest_distance

    def find_closest_point(self, thresh, clipped_depth):
        y_indices, x_indices = np.where(thresh == 255)

        min_distance = float('inf')
        closest_point = None

        for y, x in zip(y_indices, x_indices):
            depth_value = clipped_depth[y, x]

            if depth_value < min_distance:
                min_distance = depth_value
                closest_point = (x, y)

        return closest_point, min_distance if closest_point is not None else (None, None)

# --- PID Controller ---

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# --- Actor-Critic Network (PPO) ---

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)  # Sortie continue pour la vitesse et l'angle
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        action_mean = self.actor(x)  # Sortie continue
        state_value = self.critic(x)

        # Limiter la sortie de la vitesse linéaire entre [0, 1]
        linear_velocity = torch.sigmoid(action_mean[0])  # Sigmoïde pour la vitesse entre [0, 1]

        # Limiter l'angle entre -30 et +30 degrés
        angular_velocity = torch.tanh(action_mean[1]) * 30  # tanh pour les bornes de l'angle entre -30 et +30 degrés

        return torch.stack([linear_velocity, angular_velocity]), state_value


class PPO:
    def __init__(self, state_size, action_size, lr=LEARNING_RATE, gamma=GAMMA, clip_epsilon=CLIP_EPSILON, action_std=ACTION_STD):
        self.actor_critic = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.action_std = action_std  # Standard deviation for continuous action space

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_mean, _ = self.actor_critic(state)
        # Crée une distribution normale avec des moyennes et une déviation standard
        dist = torch.distributions.Normal(action_mean, self.action_std)
        action = dist.sample()  # Échantillonnage d'une action continue
        log_prob = dist.log_prob(action)  # Calculer le log-probabilité de l'action
        return action, log_prob

    def update(self, states, actions, log_probs, rewards, next_states, done_masks):
        returns = self.compute_returns(rewards, done_masks)
        for _ in range(EPOCHS):
            action_means, state_values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_means, self.action_std)
            entropy = dist.entropy().mean()

            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - log_probs)
            advantages = returns - state_values.detach()

            surrogate_loss = torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages)
            actor_loss = -surrogate_loss.mean()

            critic_loss = 0.5 * advantages.pow(2).mean()

            total_loss = actor_loss + critic_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    
# --- Odometry Listener Class ---

class OdometryListener(Node):
    def __init__(self):
        super().__init__('odometry_listener')
        self.subscription = self.create_subscription(
            Odometry,
            '/model/JetTec_Robot/odometry',  # Topic d'odométrie Gazebo
            self.odom_callback,
            10  # Taille du message (10 Hz)
        )
        self.current_position = (0.0, 0.0)  # Position initiale (x, y)
        self.current_orientation = 0.0  # Orientation initiale (angle en radians)

    def odom_callback(self, msg):
        """
        Callback pour recevoir l'odométrie et afficher les informations.
        """
        # Récupérer la position (x, y) du robot
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Récupérer l'orientation du robot sous forme de quaternion
        orientation_q = msg.pose.pose.orientation
        # Convertir quaternion en angle (yaw)
        _, _, yaw = self.quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.current_orientation = yaw

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convertir un quaternion en angles Euler (roll, pitch, yaw).
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # Retourner les angles Euler (roll, pitch, yaw)

# --- Line Following Environment Class ---

class LineFollowingEnv:
    def __init__(self, vision_distance, pid_controller, odometry_listener, publisher):
        self.vision_distance = vision_distance
        self.pid_controller = pid_controller
        self.odom_listener = odometry_listener
        self.publisher = publisher
        self.max_episode_duration = 30  # Durée maximale de l'épisode en secondes
        self.start_time = None  # Pour mesurer la durée de l'épisode

    def step(self, action):
        # Commencer à chronométrer l'épisode
        if self.start_time is None:
            self.start_time = time.time()

        # Le PPO a renvoyé deux valeurs : la vitesse linéaire et l'orientation (vitesse angulaire)
        desired_linear_velocity = action[0]  # Vitesse linéaire (avant-arrière)
        desired_angular_velocity = action[1]  # Vitesse angulaire (rotation)

        # Le PID régule la vitesse linéaire
        linear_velocity = self.pid_controller.compute_error(desired_linear_velocity)
        # Le PID régule aussi l'orientation (vitesse angulaire)
        angular_velocity = self.pid_controller.compute_error(desired_angular_velocity)

        # Publier les commandes du robot
        twist = Twist()
        twist.linear.x = linear_velocity  # Vitesse linéaire régulée par le PID
        twist.angular.z = angular_velocity  # Vitesse angulaire régulée par le PID
        self.publisher.publish(twist)

        # Vérification de la perte de la ligne
        if self.vision_distance.rgb_image is None or self.vision_distance.depth_image is None:
            # Si l'image RGB ou de profondeur n'est pas disponible, on considère que la ligne est perdue
            self.done = True
            self.last_reward = -1  # Punir si la ligne est perdue
            return self.state, self.last_reward, self.done, {}

        # Calculer la distance à la ligne (par exemple, en utilisant le centre de la ligne ou un autre critère)
        distance_to_line = self.calculate_distance_to_line()  # À implémenter selon ta logique

        # Calculer l'angle θ (l'orientation par rapport à la ligne idéale)
        theta = self.calculate_theta()  # À implémenter selon ta logique

        # Vérification du temps écoulé
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_episode_duration:
            self.done = True
            self.last_reward = 0  # Donne une petite récompense si le temps est écoulé mais que la ligne est toujours suivie
            return self.state, self.last_reward, self.done, {}

        # Continue l'épisode si la ligne est toujours suivie et le temps est toujours valide
        self.done = False
        
        # Calcul de la récompense basée sur la proximité de la ligne et de l'angle
        reward_angle = self.calculate_angle_reward(theta)  # Récompense en fonction de l'angle
        reward_distance = self.calculate_distance_reward(distance_to_line)  # Récompense en fonction de la distance à la ligne
        
        # **Récompense supplémentaire pour la vitesse**
        reward_speed = self.calculate_speed_reward(desired_linear_velocity)

        # L'addition des récompenses
        self.last_reward = reward_angle + reward_distance + reward_speed  # Ajouter la récompense pour la vitesse

        return self.state, self.last_reward, self.done, {}

    def calculate_speed_reward(self, desired_linear_velocity):
        """
        Donne une récompense en fonction de la vitesse choisie par PPO.
        Plus la vitesse est élevée, plus la récompense est grande.
        """
        if desired_linear_velocity > 0.8:  # Si la vitesse désirée est élevée
            return 1  # Récompense maximale
        elif desired_linear_velocity > 0.5:
            return 0.5  # Récompense modérée
        else:
            return 0  # Pas de récompense ou récompense minimale

    def calculate_angle_reward(self, theta):
        """
        Calcule la récompense basée sur l'angle θ (l'écart entre la direction du robot et la direction idéale).
        Un petit θ (proche de 0) signifie que le robot est bien aligné.
        """
        # Si l'angle est faible, la récompense est positive (meilleur alignement)
        if abs(theta) < 0.1:
            return 1  # Forte récompense pour un bon alignement
        elif abs(theta) < 0.3:
            return 0.5  # Récompense modérée pour un alignement correct
        else:
            return -0.5  # Punir un angle trop grand

    def calculate_distance_reward(self, distance_to_line):
        """
        Calcule la récompense en fonction de la distance au centre de la ligne.
        Plus la distance est petite, plus la récompense est grande.
        """
        if distance_to_line < 0.1:
            return 1  # Récompense maximale si très proche de la ligne
        elif distance_to_line < 0.3:
            return 0.5  # Récompense modérée si assez proche de la ligne
        else:
            return -0.5  # Punir si trop éloigné de la ligne


    def train_ppo(self):
        for epoch in range(EPOCHS):
            env = LineFollowingEnv(self.vision_distance, self.pid_controller, self.odom_listener, self.publisher)
            for episode in range(EPISODES_PER_EPOCH):  # 100 épisodes par époque
                state = env.reset()  # Réinitialiser l'environnement
                done = False
                total_reward = 0

                while not done:
                    action, log_prob = self.ppo.select_action(state)  # Sélectionner l'action
                    reward = env.step(action)  # Effectuer l'action et obtenir la récompense

                    next_state = state  # Calculer un nouvel état
                    self.ppo.update(state, action, log_prob, reward, next_state)  # Mettre à jour PPO

                    total_reward += reward
                    state = next_state

                    if total_reward >= 100:  # Critère de fin d'épisode
                        done = True

                self.get_logger().info(f"Episode {episode + 1} in Epoch {epoch + 1}, Total Reward: {total_reward}")
            # Sauvegarder le modèle à la fin de chaque époque
            self.save_model(epoch)

    def save_model(self, epoch):
        model_path = f'ppo_model_epoch_{epoch}.pth'  # Nom du fichier basé sur l'époque
        torch.save(self.ppo.actor_critic.state_dict(), model_path)
        print(f'Model saved at epoch {epoch}')
