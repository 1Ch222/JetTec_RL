# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /___   _______/  /_   _____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__
#      \______/ \____/\___/__/ \____/ \___/      

import os
import sys
import time
import signal
import subprocess
import numpy as np
import torch
import cv2
import math as m
import random

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from math import isnan

from jettec_robot.path_following.agent_path import Agent
from jettec_robot.path_following.model_path import CNNPathActor, CNNPathCritic

# === TRAINING PARAMETERS ===
NUM_AGENTS = 1
ACTION_SIZE = 1
LINEAR_VELOCITY = 1.0
IMAGE_SIZE = (1, 84, 84)
SAVE_INTERVAL_EPISODES = 10
MAX_EPISODES = 10000

ROLLOUT_LENGTH = 256
BUFFER_THRESHOLD = 2048

SPAWN_POSES = [
    "name: 'JetTec_Robot' position: { x: -1.16, y: -1.55, z: 0.15 } orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }",
]

LOAD_MODEL_PATH = "/home/mrsl/ros2_ws/checkpoints/20250527_143305/checkpoint_ep1480.pth"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PPOTrainerNode(Node):
    def __init__(self):
        super().__init__('trainer_node')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"checkpoints/{run_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tensorboard_log = f"runs/jettec_run_{run_id}"
        self.get_logger().info("\U0001f9e0 PPOTrainerNode launched.")

        self.agent = Agent(NUM_AGENTS, IMAGE_SIZE[0], ACTION_SIZE)
        self.model_for_viz = CNNPathActor(1, ACTION_SIZE, device=self.agent.device)
        self.model_for_viz.load_state_dict(self.agent.actor.state_dict())
        self.model_for_viz.to(self.agent.device)

        if LOAD_MODEL_PATH:
            self.get_logger().warn(f"\U0001f4e6 Loading checkpoint from {LOAD_MODEL_PATH}")
            checkpoint = torch.load(LOAD_MODEL_PATH)
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            if 'actor_optimizer' in checkpoint:
                self.agent.actor_optim.load_state_dict(checkpoint['actor_optimizer'])
            if 'critic_optimizer' in checkpoint:
                self.agent.critic_optim.load_state_dict(checkpoint['critic_optimizer'])
            self.episode = checkpoint.get('episode', 0)
        else:
            self.episode = 0

        self.bridge = CvBridge()
        self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.rollout = []
        self.buffer = []
        self.step_counter = 0
        self.lost_steps = 0
        self.rollout_length = ROLLOUT_LENGTH

        self.latest_offset = None
        self.latest_line_state = None
        self.latest_image = None

        self.prev_angular = 0.0  # Ajouté pour le lissage

        self.create_subscription(Float32, '/line_offset', self.offset_callback, 10)
        self.create_subscription(Int32, '/line_state', self.line_state_callback, 10)
        self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.training_step)

        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def offset_callback(self, msg):
        self.latest_offset = msg.data

    def line_state_callback(self, msg):
        self.latest_line_state = msg.data

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.latest_image = torch.tensor(cv_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        except Exception as e:
            self.get_logger().error(f"\u274c image_callback: {e}")

    def training_step(self):
        if self.latest_image is None:
            return

        self.publish_velocity(0.0, 0.0)
        state = self.latest_image.to(self.agent.device)
        self.show_features(state)

        offset = self.latest_offset
        line_state = self.latest_line_state

        if line_state is None:
            reward = -1.0
        else:
            if line_state == 2:
                reward = m.exp(-(offset**2)/(2*(0.2**2))) if offset is not None and not isnan(offset) else -1.0
            elif line_state == 1:
                reward = -0.5
            else:
                reward = -1.0

        # Pénalité sur le steering angle
        penalty_factor = 0.05
        action = float(self.prev_angular)  # On utilise l'action lissée précédente
        steering_penalty = penalty_factor * abs(action)
        reward -= steering_penalty
        self.writer.add_scalar("reward/steering_penalty", -steering_penalty, self.step_counter)

        self.writer.add_scalar("reward/step", reward, self.step_counter)
        self.writer.add_scalar("tracking/lost_steps", self.lost_steps, self.step_counter)
        self.step_counter += 1

        if state.dim() == 3:
            state = state.unsqueeze(0)
        if state.dim() == 4:
            state = state.unsqueeze(0)

        actions, log_probs, values, _ = self.agent.act(state)
        raw_action = float(np.clip(actions.cpu().numpy()[0][0], -1.0, 1.0))
        self.publish_velocity(LINEAR_VELOCITY, raw_action)

        self.rollout.append([state.detach(), actions.detach(), log_probs.detach(), [reward], [1.0], values.detach()])

        if len(self.rollout) >= self.rollout_length:
            self.train_policy(state)

    def publish_velocity(self, linear, angular):
        alpha = 0.3
        filtered_angular = alpha * angular + (1 - alpha) * self.prev_angular
        self.prev_angular = filtered_angular

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = filtered_angular
        self.publisher.publish(twist)

    def train_policy(self, last_state):
        self.episode += 1

        if last_state.dim() == 3:
            last_state = last_state.unsqueeze(0)
        if last_state.dim() == 4:
            last_state = last_state.unsqueeze(0)

        _, _, last_value, _ = self.agent.act(last_state)
        pending_value = last_value.detach()
        self.rollout.append([last_state.detach(), None, None, None, None, pending_value])

        self.buffer.extend(self.rollout[:-1])

        if len(self.buffer) >= BUFFER_THRESHOLD:
            self.get_logger().info(f"\U0001f4da Training on {len(self.buffer)} transitions")
            self.buffer.append(self.rollout[-1])
            self.agent.step(self.buffer, NUM_AGENTS, self.writer, self.episode)
            self.writer.add_scalar("actor/std_scale", self.agent.actor.std_scale, self.episode)
            self.buffer.clear()

            total_reward = sum([r[3][0] for r in self.rollout[:-1]])
            self.writer.add_scalar("reward/episode", total_reward, self.episode)

            if self.episode % SAVE_INTERVAL_EPISODES == 0:
                try:
                    ckpt = f"{self.checkpoint_dir}/checkpoint_ep{self.episode}.pth"
                    torch.save({
                        'episode': self.episode,
                        'actor_state_dict': self.agent.actor.state_dict(),
                        'critic_state_dict': self.agent.critic.state_dict(),
                        'actor_optimizer': self.agent.actor_optim.state_dict(),
                        'critic_optimizer': self.agent.critic_optim.state_dict()
                    }, ckpt)
                    self.get_logger().info(f"\U0001f4be Unified checkpoint saved: {ckpt}")
                except Exception as e:
                    self.get_logger().error(f"\u274c Failed to save checkpoint: {e}")

        self.rollout.clear()

        if self.episode >= MAX_EPISODES:
            self.get_logger().info("\U0001f3c1 Training finished.")
            self.handle_exit(None, None)
        else:
            self.reset_robot()

    def reset_robot(self):
        self.get_logger().warn("\U0001f501 Resetting robot pose...")
        self.publish_velocity(0.0, 0.0)
        time.sleep(0.1)

        pose = np.random.choice(SPAWN_POSES)
        cmd = ["ign", "service", "-s", "/world/Line_track/set_pose",
               "--reqtype", "ignition.msgs.Pose",
               "--reptype", "ignition.msgs.Boolean",
               "--timeout", "1000",
               "--req", pose]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.latest_image = None
        self.latest_offset = None
        self.latest_line_state = None

    def show_features(self, state_tensor):
        try:
            fmap = self.model_for_viz.extract_feature_maps(state_tensor).cpu()
            if fmap.ndim != 3 or fmap.shape[0] == 0:
                return

            maps = []
            for i in range(min(8, fmap.shape[0])):
                m = fmap[i].detach().cpu().numpy()
                m = (m - m.min()) / (m.max() - m.min() + 1e-5)
                m = (m * 255).astype(np.uint8)
                maps.append(m)

            vis = np.concatenate(maps, axis=1)
            vis_resized = cv2.resize(vis, (vis.shape[1]*2, vis.shape[0]*2))
            cv2.namedWindow("\U0001f9e0 Feature Maps", cv2.WINDOW_NORMAL)
            cv2.imshow("\U0001f9e0 Feature Maps", vis_resized)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"\u274c show_features() failed: {e}")

    def handle_exit(self, sig, frame):
        self.get_logger().warn("\U0001f4c5 Final model saved...")
        torch.save({
            'episode': self.episode,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer': self.agent.actor_optim.state_dict(),
            'critic_optimizer': self.agent.critic_optim.state_dict()
        }, f"{self.checkpoint_dir}/checkpoint_final.pth")
        self.writer.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    set_seed(42)
    rclpy.init()
    node = PPOTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.handle_exit(None, None)
    finally:
        rclpy.shutdown()
