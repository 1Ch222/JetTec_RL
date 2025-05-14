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
ROS2 Node for training a line-following policy using PPO (Proximal Policy Optimization).
This node subscribes to vision outputs, controls the robot, collects rollouts, 
and optimizes the policy/value networks over time.

Features:
- Online reinforcement learning with rollout collection.
- Robot reset during training.
- Live feature map visualization.
- TensorBoard logging.
- Graceful shutdown with model saving.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
from torch.utils.tensorboard import SummaryWriter

from math import isnan
from datetime import datetime

from jettec_robot.line_following.agent_line import Agent
from jettec_robot.line_following.model_line import CNNActorCritic

# === CONFIGURATION ===
ROLLOUT_START = 64
ROLLOUT_MAX = 512
ROLLOUT_STEP = 16
ROLLOUT_FREQ = 600
NUM_AGENTS = 1
ACTION_SIZE = 1
LINEAR_VELOCITY = 1.0
IMAGE_SIZE = (1, 84, 84)
SAVE_INTERVAL_EPISODES = 10
MAX_EPISODES = 10000

SPAWN_POSES = [
    "name: 'JetTec_Robot' position: { x: 2.2, y: -3.3, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 0.0 }",
    "name: 'JetTec_Robot' position: { x: -3.8, y: -1.4, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 1.0 }",
    "name: 'JetTec_Robot' position: { x: 1.7, y: -1.6, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 1.0 }",
    "name: 'JetTec_Robot' position: { x: -1.0, y: 3.6, z: 0.15 } orientation: { x: 0, y: 0, z: 0, w: 1 }"
]

class PPOTrainerNode(Node):
    """ROS2 node for training the JetTec line-following agent using PPO."""

    def __init__(self, load_model_path=None):
        super().__init__('trainer_node')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Setup paths for saving
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"checkpoints/{run_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tensorboard_log = f"runs/jettec_run_{run_id}"

        self.get_logger().info("🧠 PPOTrainerNode launched.")

        # Agent setup
        self.agent = Agent(NUM_AGENTS, IMAGE_SIZE[0], ACTION_SIZE)
        self.model_for_viz = CNNActorCritic(1, ACTION_SIZE).to(self.agent.model.device)
        self.model_for_viz.load_state_dict(self.agent.model.state_dict())

        if load_model_path:
            self.get_logger().warn(f"📦 Loading model from {load_model_path}")
            self.agent.model.load_state_dict(torch.load(load_model_path))

        # Tools
        self.bridge = CvBridge()
        self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        # States
        self.rollout = []
        self.episode = 0
        self.step_counter = 0
        self.lost_steps = 0
        self.rollout_length = ROLLOUT_START
        self.latest_offset = None
        self.latest_image = None

        # ROS Communication
        self.create_subscription(Float32, '/line_offset', self.offset_callback, 10)
        self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.training_step)

        # Signal handling
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def offset_callback(self, msg):
        """Receives the latest line offset from vision."""
        self.latest_offset = msg.data

    def image_callback(self, msg):
        """Receives the latest processed image."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.latest_image = torch.tensor(cv_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        except Exception as e:
            self.get_logger().error(f"❌ image_callback: {e}")

    def training_step(self):
        """Main training loop called periodically."""
        if self.latest_image is None:
            return

        # Stop the robot momentarily
        self.publish_velocity(0.0, 0.0)

        # Process state
        state = self.latest_image.to(self.agent.model.device)
        self.show_features(state)

        # Reward computation
        reward = 1.0 - self.latest_offset ** 2 if self.latest_offset is not None and not isnan(self.latest_offset) else -1.0
        self.lost_steps = 0 if reward > -1.0 else self.lost_steps + 1

        # Log
        self.writer.add_scalar("reward/step", reward, self.step_counter)
        self.writer.add_scalar("tracking/lost_steps", self.lost_steps, self.step_counter)
        self.step_counter += 1

        # Action selection
        actions, log_probs, values = self.agent.act(state)
        action = float(np.clip(actions.cpu().numpy()[0][0], -1.0, 1.0))

        self.publish_velocity(LINEAR_VELOCITY, action)

        # Store rollout step
        self.rollout.append([
            state, actions.detach(), log_probs.detach(),
            [reward], [1.0], values.detach()
        ])

        # Train if enough rollout collected
        if len(self.rollout) >= self.rollout_length:
            self.train_policy(state)

    def publish_velocity(self, linear, angular):
        """Publishes a velocity command to the robot."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.publisher.publish(twist)

    def train_policy(self, last_state):
        """Trains the policy using collected rollouts."""
        self.get_logger().info(f"📚 Training - episode {self.episode}")
        pending_value = self.agent.model(last_state)[-1].detach()
        self.rollout.append([last_state, None, None, None, None, pending_value])

        self.agent.step(self.rollout, NUM_AGENTS, self.writer, self.episode)

        total_reward = sum([r[3][0] for r in self.rollout[:-1]])
        self.writer.add_scalar("reward/episode", total_reward, self.episode)

        # Save model
        if self.episode % SAVE_INTERVAL_EPISODES == 0:
            ckpt = f"{self.checkpoint_dir}/model_ep{self.episode}.pth"
            torch.save(self.agent.model.state_dict(), ckpt)
            self.get_logger().info(f"💾 Model saved : {ckpt}")

        # Rollout size progression
        if self.episode % ROLLOUT_FREQ == 0 and self.rollout_length < ROLLOUT_MAX:
            self.rollout_length = min(self.rollout_length + ROLLOUT_STEP, ROLLOUT_MAX)
            self.get_logger().info(f"📈 ROLLOUT_LENGTH increased to {self.rollout_length}")

        self.rollout.clear()
        self.episode += 1

        if self.episode >= MAX_EPISODES:
            self.get_logger().info("🏁 Training finished.")
            self.handle_exit(None, None)
        else:
            self.reset_robot()

    def reset_robot(self):
        """Resets the robot to a random pose in the world."""
        self.get_logger().warn("🔁 Resetting robot pose...")
        self.publish_velocity(0.0, 0.0)
        time.sleep(0.1)

        pose = np.random.choice(SPAWN_POSES)
        cmd = [
            "ign", "service", "-s", "/world/Line_track/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "1000",
            "--req", pose
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.latest_image = None
        self.latest_offset = None

    def show_features(self, state_tensor):
        """Displays extracted feature maps for visualization."""
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
            cv2.namedWindow("🧠 Feature Maps", cv2.WINDOW_NORMAL)
            cv2.imshow("🧠 Feature Maps", vis_resized)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"❌ show_features() failed: {e}")

    def handle_exit(self, sig, frame):
        """Handles clean shutdown and saves the final model."""
        self.get_logger().warn("📅 Final model saved...")
        torch.save(self.agent.model.state_dict(), f"{self.checkpoint_dir}/model_final.pth")
        self.writer.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to a .pth model", default=None)
    args, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = PPOTrainerNode(load_model_path=args.load)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.handle_exit(None, None)
    finally:
        rclpy.shutdown()
