# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /____   ____________   ____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""

"""

import os
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from torch.utils.tensorboard import SummaryWriter
from cv_bridge import CvBridge
import torch, numpy as np, cv2, signal, subprocess, time, sys, argparse
from math import isnan
from datetime import datetime
from jettec_robot.line_following.agent_line import Agent
from jettec_robot.line_following.model_line import CNNActorCritic

# === Configurations ===
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
LOAD_MODEL_PATH = None

# === Dynamic files ===
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = f"checkpoints/{RUN_ID}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TENSORBOARD_LOG = f"runs/jettec_run_{RUN_ID}"

SPAWN_POSES = [
    "name: 'JetTec_Robot' position: { x: 2.2, y: -3.3, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 0.0 }",
    "name: 'JetTec_Robot' position: { x: -3.8, y: -1.4, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 1.0 }",
    "name: 'JetTec_Robot' position: { x: 1.7, y: -1.6, z: 0.15 } orientation: { x: 0, y: 0, z: 1.0, w: 1.0 }",
    "name: 'JetTec_Robot' position: { x: -1.0, y: 3.6, z: 0.15 } orientation: { x: 0, y: 0, z: 0, w: 1 }"
]

class PPOTrainerNode(Node):
    def __init__(self):
        super().__init__('trainer_node')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.get_logger().info("🧠 PPOTrainerNode launched.")

        self.agent = Agent(NUM_AGENTS, IMAGE_SIZE[0], ACTION_SIZE)
        self.model_for_viz = CNNActorCritic(1, ACTION_SIZE).to(self.agent.model.device)
        self.model_for_viz.load_state_dict(self.agent.model.state_dict())

        if LOAD_MODEL_PATH:
            self.get_logger().warn(f"📦 Loading model from {LOAD_MODEL_PATH}")
            self.agent.model.load_state_dict(torch.load(LOAD_MODEL_PATH))

        self.bridge = CvBridge()
        self.writer = SummaryWriter(log_dir=TENSORBOARD_LOG)
        self.rollout = []
        self.episode = 0
        self.step_counter = 0
        self.lost_steps = 0
        self.rollout_length = ROLLOUT_START

        self.latest_offset = None
        self.latest_image = None

        self.create_subscription(Float32, '/line_offset', self.offset_callback, 10)
        self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.training_step)

        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def offset_callback(self, msg):
        self.latest_offset = msg.data

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.latest_image = torch.tensor(cv_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        except Exception as e:
            self.get_logger().error(f"image_callback: {e}")

    def training_step(self):
        if self.latest_image is None:
            return

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)

        state = self.latest_image.to(self.agent.model.device)
        self.show_features(state)

        reward = 1.0 - self.latest_offset ** 2 if self.latest_offset is not None and not isnan(self.latest_offset) else -1.0
        self.lost_steps = 0 if reward > -1.0 else self.lost_steps + 1

        self.writer.add_scalar("reward/step", reward, self.step_counter)
        self.writer.add_scalar("tracking/lost_steps", self.lost_steps, self.step_counter)
        self.step_counter += 1

        actions, log_probs, values = self.agent.act(state)
        action = float(np.clip(actions.cpu().numpy()[0][0], -1.0, 1.0))

        twist.linear.x = LINEAR_VELOCITY
        twist.angular.z = action
        self.publisher.publish(twist)

        self.rollout.append([
            state, actions.detach(), log_probs.detach(),
            [reward], [1.0], values.detach()
        ])

        if len(self.rollout) >= self.rollout_length:
            self.get_logger().info(f"📚 Training - episode {self.episode}")
            pending_value = self.agent.model(state)[-1].detach()
            self.rollout.append([state, None, None, None, None, pending_value])
            self.agent.step(self.rollout, NUM_AGENTS, self.writer, self.episode)
            total_reward = sum([r[3][0] for r in self.rollout[:-1]])
            self.writer.add_scalar("reward/episode", total_reward, self.episode)

            if self.episode % SAVE_INTERVAL_EPISODES == 0:
                ckpt = f"{CHECKPOINT_DIR}/model_ep{self.episode}.pth"
                torch.save(self.agent.model.state_dict(), ckpt)
                self.get_logger().info(f"💾 Model saved : {ckpt}")

            if self.episode % ROLLOUT_FREQ == 0 and self.rollout_length < ROLLOUT_MAX:
                self.rollout_length = min(self.rollout_length + ROLLOUT_STEP, ROLLOUT_MAX)
                self.get_logger().info(f"📈 ROLLOUT_LENGTH augmented to {self.rollout_length}")

            self.rollout.clear()
            self.episode += 1
            if self.episode >= MAX_EPISODES:
                self.get_logger().info("🏁 Training finished.")
                self.handle_exit(None, None)
            else:
                self.reset_robot()

    def reset_robot(self):
        self.get_logger().warn("🔁 Robot reset...")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
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
            self.get_logger().error(f"show_features() failed: {e}")

    def handle_exit(self, sig, frame):
        self.get_logger().warn("📅 Final model saved...")
        torch.save(self.agent.model.state_dict(), f"{CHECKPOINT_DIR}/model_final.pth")
        self.writer.close()
        cv2.destroyAllWindows()
        sys.exit(0)

def main(args=None):
    global LOAD_MODEL_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to a .pth model", default=None)
    args, _ = parser.parse_known_args()
    LOAD_MODEL_PATH = args.load

    rclpy.init(args=None)
    node = PPOTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.handle_exit(None, None)
    rclpy.shutdown()

