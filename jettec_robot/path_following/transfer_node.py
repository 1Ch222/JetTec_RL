# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import numpy as np
import smbus2
import time
from jettec_robot.path_following.model_path import CNNPathActor

# === PCA9685 CONFIGURATION ===
I2C_BUS = 7
I2C_ADDR = 0x40
LED0_ON_L = 0x06
MODE1 = 0x00
PRESCALE = 0xFE
FREQ = 100

SERVO_CHANNEL = 4
MOTOR_CHANNEL = 0
SERVO_CENTER = 11
SERVO_LEFT = 15
SERVO_RIGHT = 6
MOTOR_NEUTRAL = 15.0
MOTOR_FORWARD = 20.0

MODEL_PATH = "../checkpoints/checkpoint_ep7280.pth"

class PPOInferenceNode(Node):
    def __init__(self):
        super().__init__('ppo_inference_node')

        # PCA9685 init
        self.bus = smbus2.SMBus(I2C_BUS)
        self.init_pca9685(FREQ)

        # PPO Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNPathActor(1, 1, device=self.device).to(self.device)
        self.load_model()

        # ROS
        self.bridge = CvBridge()
        self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.create_subscription(String, '/robot_control', self.control_callback, 10)

        self.robot_active = False
        self.prev_angular = 0.0
        self.alpha = 0.3  # For action smoothing

        self.get_logger().info("🚗 PPOInferenceNode ready. Waiting for start...")

    def init_pca9685(self, freq):
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x00)
        time.sleep(0.005)
        prescale_val = int(round(25000000.0 / (4096 * freq)) - 1)
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x10)
        self.bus.write_byte_data(I2C_ADDR, PRESCALE, prescale_val)
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x80)
        time.sleep(0.005)
        self.get_logger().info(f"✅ PCA9685 initialized at {freq} Hz (prescale {prescale_val})")

    def load_model(self):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['actor_state_dict'])
            self.model.eval()
            self.get_logger().info(f"🎓 Model loaded from {MODEL_PATH}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")

    def set_pwm(self, channel, duty_cycle):
        on_time = 0
        off_time = int(duty_cycle * 4096 / 100)
        reg_base = LED0_ON_L + 4 * channel
        self.bus.write_byte_data(I2C_ADDR, reg_base, on_time & 0xFF)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 1, on_time >> 8)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 2, off_time & 0xFF)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 3, off_time >> 8)

    def control_callback(self, msg):
        command = msg.data.lower().strip()
        if command == "start":
            self.robot_active = True
            self.get_logger().info("✅ Robot control started.")
            self.set_pwm(MOTOR_CHANNEL, MOTOR_FORWARD)
        elif command == "pause" or command == "stop":
            self.robot_active = False
            self.stop_motors()
            self.get_logger().info(f"⏸ Robot {command}.")

    def image_callback(self, msg):
        if not self.robot_active:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            state = torch.tensor(cv_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

            with torch.no_grad():
                dist, _ = self.model(state)
                action = dist.sample().cpu().numpy()[0][0]
                action = np.clip(action, -1.0, 1.0)

            # Apply smoothing
            filtered_action = self.alpha * action + (1 - self.alpha) * self.prev_angular
            self.prev_angular = filtered_action

            # Convert to servo duty
            k = 4.0
            duty = SERVO_CENTER - k * filtered_action
            duty = max(min(duty, SERVO_LEFT), SERVO_RIGHT)
            self.set_pwm(SERVO_CHANNEL, duty)

            self.get_logger().info(f"➡️ Servo: {duty:.2f} | Action: {filtered_action:.3f}")

        except Exception as e:
            self.get_logger().error(f"❌ image_callback: {e}")

    def stop_motors(self):
        self.set_pwm(MOTOR_CHANNEL, MOTOR_NEUTRAL)
        self.set_pwm(SERVO_CHANNEL, SERVO_CENTER)

    def destroy_node(self):
        self.stop_motors()
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x10)
        self.bus.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PPOInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("🔌 Stopped by Ctrl+C")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
