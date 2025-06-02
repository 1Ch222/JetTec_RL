import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String
import smbus2
import time
import numpy as np

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

class ControlRobotNode(Node):
    def __init__(self):
        super().__init__('control_robot_node')
        self.bus = smbus2.SMBus(I2C_BUS)
        self.init_pca9685(FREQ)

        # Abonnements
        self.create_subscription(Float32, '/line_offset', self.offset_callback, 10)
        self.create_subscription(Int32, '/line_state', self.line_state_callback, 10)
        self.create_subscription(String, '/robot_control', self.control_callback, 10)

        self.latest_offset = 0.0
        self.latest_line_state = 0
        self.robot_active = False
        self.last_line_time = self.get_clock().now()

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("🚗 ControlRobotNode ready (waiting for /robot_control start command).")

    def init_pca9685(self, freq):
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x00)
        time.sleep(0.005)
        prescale_val = int(round(25000000.0 / (4096 * freq)) - 1)
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x10)
        self.bus.write_byte_data(I2C_ADDR, PRESCALE, prescale_val)
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x80)
        time.sleep(0.005)
        self.get_logger().info(f"✅ PCA9685 initialized at {freq} Hz (prescale {prescale_val})")

    def set_pwm(self, channel, duty_cycle):
        on_time = 0
        off_time = int(duty_cycle * 4096 / 100)
        reg_base = LED0_ON_L + 4 * channel
        self.bus.write_byte_data(I2C_ADDR, reg_base, on_time & 0xFF)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 1, on_time >> 8)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 2, off_time & 0xFF)
        self.bus.write_byte_data(I2C_ADDR, reg_base + 3, off_time >> 8)

    def offset_callback(self, msg):
        self.latest_offset = msg.data

    def line_state_callback(self, msg):
        self.latest_line_state = msg.data
        if self.latest_line_state == 2:
            self.last_line_time = self.get_clock().now()

    def control_callback(self, msg):
        command = msg.data.lower().strip()
        if command == "start":
            self.robot_active = True
            self.get_logger().info("✅ Robot control started.")
        elif command == "pause":
            self.robot_active = False
            self.stop_motors()
            self.get_logger().info("⏸ Robot control paused.")
        elif command == "stop":
            self.robot_active = False
            self.stop_motors()
            self.sleep_pca()
            self.get_logger().warn("❌ Robot control stopped and PCA9685 in sleep mode.")

    def control_loop(self):
        if not self.robot_active:
            return

        time_since_line = (self.get_clock().now() - self.last_line_time).nanoseconds / 1e9
        if time_since_line > 3.0:
            self.get_logger().error("🚨 No line detected for 3s - emergency stop.")
            self.robot_active = False
            self.stop_motors()
            return

        if self.latest_line_state == 2 and not (self.latest_offset is None or np.isnan(self.latest_offset)):
            k = 4.0
            duty = SERVO_CENTER - k * self.latest_offset
            duty = max(min(duty, SERVO_LEFT), SERVO_RIGHT)
            self.set_pwm(SERVO_CHANNEL, duty)
            self.set_pwm(MOTOR_CHANNEL, MOTOR_FORWARD)
            self.get_logger().info(f"➡️ Servo: {duty:.2f} | Offset: {self.latest_offset:.3f}")
        else:
            self.stop_motors()
            self.get_logger().warn("⚠️ Line lost - stopping.")

    def stop_motors(self):
        self.set_pwm(MOTOR_CHANNEL, MOTOR_NEUTRAL)
        self.set_pwm(SERVO_CHANNEL, SERVO_CENTER)

    def sleep_pca(self):
        self.bus.write_byte_data(I2C_ADDR, MODE1, 0x10)
        self.bus.close()

    def destroy_node(self):
        self.stop_motors()
        self.sleep_pca()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ControlRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("🔌 Stopped by Ctrl+C")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
