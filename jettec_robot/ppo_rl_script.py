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
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
GAMMA = 0.99                # Discount factor for future rewards
CLIP_EPSILON = 0.2          # PPO clip parameter
ACTION_STD = 0.5            # Standard deviation for action distribution
EPOCHS = 1                 # Number of epochs (update iterations per PPO update)
EPISODES_PER_EPOCH = 1000    # Episodes per epoch

Kp = 5.0                    # PID proportional gain
Ki = 3.0                    # PID integral gain
Kd = 3.0                    # PID derivative gain

THRESHOLD = 30              # Threshold for binary image processing
MIN_AREA = 500              # Minimum contour area to consider

MIN_SPEED = 0               # (m/s)
MAX_SPEED = 1

# --- Nouveaux hyperparamètres pour la récompense ---
ALPHA_LINE = 1.0            # Coefficient pour la récompense de distance à la ligne
ALPHA_HEADING = 0.7         # Coefficient pour la récompense d'alignement
ALPHA_SPEED = 0.5           # Coefficient pour la récompense de vitesse
D_LINE_MIN = 0.24           # Distance de référence (m)
V_MAX = 1.5                 # Vitesse maximale pour normalisation (m/s)

# --- Reset Functions ---
def reset_world():
    cmd = [
        "ign", "service", "-s", "/world/Line_track/control",
        "--reqtype", "gz.msgs.WorldControl",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000"
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
    cmd = [
        "ign", "service", "-s", "/world/Line_track/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", "name: 'JetTec_Robot' position: { x: -1.0, y: 3.6, z: 0.15 } orientation: { x: 0, y: 0, z: 0, w: 1 }"
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
        # Création des fenêtres d'affichage en temps réel
        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Centroids", cv2.WINDOW_NORMAL)
        self.theta = None
        self.distance = None

    def image_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RGB Image", self.rgb_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(f"RGB conversion error: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # Normaliser pour affichage
            depth_norm = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = np.nan_to_num(depth_norm, nan=0.0)
            depth_norm = np.uint8(depth_norm)
            cv2.imshow("Depth Image", depth_norm)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(f"Depth conversion error: {e}")

    def is_ready(self):
        return self.rgb_image is not None and self.depth_image is not None

    def calculate_and_display_results(self):
        # Découpage de l'image
        h, w, _ = self.rgb_image.shape
        clipped_rgb = self.rgb_image[int(h * 0.5):, :]
        clipped_depth = self.depth_image[int(h * 0.5):, :]

        # Prétraitement de l'image
        gray = cv2.cvtColor(clipped_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
        # Recherche des contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        image_with_centroids = clipped_rgb.copy()
        if contours:
            # Utiliser directement le plus grand contour
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > MIN_AREA:
                # Utilisation de cv2.fitLine pour obtenir la droite de régression sur le contour
                [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                # Calcul de l'angle à partir du vecteur directeur (vx, vy)
                angle_rad = np.arctan2(vy, vx)[0]
                self.theta = angle_rad

                # Optionnel : tracer la droite ajustée pour la visualisation
                # Choisir deux points le long de la droite pour l'affichage
                lefty = int((-x0 * vy / vx) + y0)
                righty = int(((w - x0) * vy / vx) + y0)
                cv2.line(image_with_centroids, (w - 1, righty), (0, lefty), (255, 0, 0), 2)

                # Calcul de la distance (la fonction find_closest_point reste inchangée)
                _, closest_distance = self.find_closest_point(thresh, clipped_depth)
                self.distance = closest_distance
            else:
                self.theta = None
                self.distance = None
        else:
            self.theta = None
            self.distance = None

    # Option : supprimer l'affichage si non nécessaire
        cv2.imshow("Centroids", image_with_centroids)
        cv2.waitKey(1)

        if self.theta is not None and self.distance is not None:
            print(f"Calculated theta: {self.theta:.2f} rad, Distance: {self.distance:.2f} m")
        else:
            print("Theta or Distance is None.")
        return self.theta, self.distance


    def find_closest_point(self, thresh, clipped_depth):
        y_indices, x_indices = np.where(thresh == 255)
        min_distance = float('inf')
        closest_point = None
        for y, x in zip(y_indices, x_indices):
            depth_value = clipped_depth[y, x]
            if depth_value < min_distance:
                min_distance = depth_value
                closest_point = (x, y)
        if closest_point is not None:
            return closest_point, min_distance
        else:
            return None, None


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
        self.actor = nn.Linear(128, action_size)  # Deux sorties: [linear_velocity, angular_velocity]
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = self.actor(x)  # forme (batch, 2)
        state_value = self.critic(x)  # forme (batch, 1)
        linear_velocity = torch.sigmoid(action_mean[:, 0]).unsqueeze(1)
        angular_velocity = torch.tanh(action_mean[:, 1]).unsqueeze(1) * 30
        action = torch.cat([linear_velocity, angular_velocity], dim=1)
        return action, state_value

# --- PPO Class ---
class PPO:
    def __init__(self, state_size, action_size, lr=LEARNING_RATE, gamma=GAMMA,
                 clip_epsilon=CLIP_EPSILON, action_std=ACTION_STD):
        self.actor_critic = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.action_std = action_std

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mean, _ = self.actor_critic(state)
        dist = torch.distributions.Normal(action_mean, self.action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0)

    def compute_returns(self, rewards, done_masks):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(done_masks)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

    def update(self, states, actions, log_probs, rewards, next_states, done_masks):
        returns = self.compute_returns(rewards, done_masks).detach()
        states = states.detach()
        actions = actions.detach()
        log_probs = log_probs.detach()
        for i in range(EPOCHS):
            action_means, state_values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_means, self.action_std)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - log_probs)
            advantages = returns - state_values.detach()
            surrogate_loss = torch.min(ratios * advantages,
                                       torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages)
            actor_loss = -surrogate_loss.mean()
            critic_loss = 0.5 * advantages.pow(2).mean()
            total_loss = actor_loss + critic_loss - 0.001 * entropy
            self.optimizer.zero_grad()
            if i < EPOCHS - 1:
                total_loss.backward(retain_graph=True)
            else:
                total_loss.backward()
            self.optimizer.step()

# --- Odometry Listener Class ---
class OdometryListener(Node):
    def __init__(self):
        super().__init__('odometry_listener')
        self.subscription = self.create_subscription(
            Odometry,
            '/model/JetTec_Robot/odometry',
            self.odom_callback,
            10
        )
        self.current_position = (0.0, 0.0)
        self.current_orientation = 0.0

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = self.quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.current_orientation = yaw

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

# --- LineFollowingEnv Class ---
class LineFollowingEnv:
    def __init__(self, vision_distance, pid_controller, odometry_listener, publisher, node):
        self.vision_distance = vision_distance
        self.pid_controller = pid_controller
        self.odom_listener = odometry_listener
        self.publisher = publisher
        self.node = node
        self.max_episode_duration = 30  # seconds
        self.start_time = None
        self.state = None
        self.last_reward = 0
        self.previous_position = None

    def reset(self):
        reset_world()
        reset_robot()

        # Remise à zéro des roues via cmd_vel
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        time.sleep(0.1)
    
        time.sleep(2)
        self.start_time = time.time()
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.previous_position = self.odom_listener.current_position

        # Attendre que la caméra soit prête
        max_wait = 5.0
        start_wait = time.time()
        while not self.vision_distance.is_ready() and (time.time() - start_wait < max_wait):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            cv2.waitKey(1)
            time.sleep(0.1)
    
        # Attendre que la ligne soit détectée (θ non nul)
        line_detected = False
        max_line_wait = 5.0  # délai maximum d'attente pour la détection de la ligne
        line_start = time.time()
        while not line_detected and (time.time() - line_start < max_line_wait):
            theta, distance = self.vision_distance.calculate_and_display_results()
            if theta is not None:
                line_detected = True
            else:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                cv2.waitKey(1)
                time.sleep(0.1)
        if not line_detected:
            self.node.get_logger().warn("Ligne non détectée lors du reset; début de l'épisode malgré tout.")
        return self.state


    def step(self, action):
        # Utilisation de l'action (commandes directes) pour commander le robot
        desired_linear_velocity = action[0].item() if isinstance(action, torch.Tensor) else action[0]
        desired_angular_velocity = action[1].item() if isinstance(action, torch.Tensor) else action[1]
        print(desired_angular_velocity, desired_linear_velocity)
        twist = Twist()
        twist.linear.x = desired_linear_velocity
        twist.angular.z = desired_angular_velocity
        self.publisher.publish(twist)
        # On fixe l'intervalle de temps à 0.1 s
        dt = 0.1
        time.sleep(dt)
        # Calcul du déplacement réel via odométrie
        current_pos = self.odom_listener.current_position
        if self.previous_position is None:
            moved_distance = 0
        else:
            moved_distance = np.linalg.norm(np.array(current_pos) - np.array(self.previous_position))
        self.previous_position = current_pos

        # Calcul de theta et de la distance via vision
        theta, distance = self.vision_distance.calculate_and_display_results()

        # Si theta ou distance invalide, terminer l'épisode avec une forte pénalité
        if theta is None or distance is None or distance == 0:
            self.node.get_logger().warn("Line lost: theta or distance invalid. Terminating episode.")
            self.last_reward = -100
            self.done = True
            return self.state, self.last_reward, self.done, {}

        # Vérification de la durée de l'épisode
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_episode_duration:
            self.done = True
            self.last_reward = 0
            self.node.get_logger().info("Time limit reached.")
            return self.state, self.last_reward, self.done, {}

        self.done = False

        # Calcul de la vitesse réelle
        V_real = moved_distance / dt if dt != 0 else 0

        # Calcul des nouvelles récompenses
        reward_line = ALPHA_LINE * (1 - min(distance / D_LINE_MIN, 1))
        reward_heading = ALPHA_HEADING * math.cos(theta)
        reward_speed = ALPHA_SPEED * (V_real / V_MAX)
        reward = reward_line + reward_heading + reward_speed

        self.last_reward = reward
        return self.state, self.last_reward, self.done, {}

# --- LineFollowingNode (main node) ---
class LineFollowingNode(Node):
    def __init__(self):
        super().__init__('line_following_node')
        self.odom_listener = OdometryListener()
        self.vision_distance = VisionDistance()
        self.pid_controller = PIDController(Kp, Ki, Kd)
        self.ppo = PPO(state_size=4, action_size=2)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_subscription = self.create_subscription(
            Image, '/rgbd_camera/image', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/rgbd_camera/depth_image', self.depth_callback, 10)

    def image_callback(self, msg):
        self.vision_distance.image_callback(msg)

    def depth_callback(self, msg):
        self.vision_distance.depth_callback(msg)

    def train_ppo(self):
        for epoch in range(EPOCHS):
            env = LineFollowingEnv(self.vision_distance, self.pid_controller, self.odom_listener, self.publisher, self)
            for episode in range(EPISODES_PER_EPOCH):
                state = env.reset()
                done = False
                total_reward = 0
                log_probs = []
                actions = []
                rewards = []
                next_states = []
                done_masks = []
                while not done:
                    rclpy.spin_once(self, timeout_sec=0.01)
                    cv2.waitKey(1)
                    action, log_prob = self.ppo.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    log_probs.append(log_prob)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(state)
                    done_masks.append(1 if done else 0)
                    total_reward += reward
                    state = next_state
                    cv2.waitKey(1)
                    if total_reward >= 100:
                        done = True
                self.ppo.update(
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.stack(actions),
                    torch.stack(log_probs),
                    torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(done_masks, dtype=torch.float32).unsqueeze(1)
                )
                self.get_logger().info(f"Epoch {epoch+1}, Episode {episode+1}, Total Reward: {total_reward}")
            self.save_model(epoch)

    def save_model(self, epoch):
        model_path = f'ppo_model_epoch_{epoch}.pth'
        torch.save(self.ppo.actor_critic.state_dict(), model_path)
        print(f"Model saved at epoch {epoch}")

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode()
    try:
        node.train_ppo()
    except KeyboardInterrupt:
        node.get_logger().info("Training interrupted.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
