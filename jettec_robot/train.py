import rclpy
from vision_processing import VisionDistance  # Module Vision
from progression import ProgressionAlongLine  # Module Progression
from SAC import ProgressionSAC  # Module SAC
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.node import Node
import numpy as np
import subprocess
from stable_baselines3 import SAC  # Importation du modèle SAC depuis stable_baselines3


class RobotTrainer(Node):
    def __init__(self):
        super().__init__('robot_trainer')

        # Initialiser les modules
        self.vision = VisionDistance()
        self.progression = ProgressionAlongLine()

        # Initialisation de l'environnement et du modèle SAC
        self.env = ProgressionSAC()
        self.model = self.initialize_sac_model()

        # Abonnement aux topics ROS nécessaires
        self.subscription_rgb = self.create_subscription(
            Image,
            '/rgbd_camera/image',
            self.vision.image_callback,
            10
        )

        self.subscription_depth = self.create_subscription(
            Image,
            '/rgbd_camera/depth_image',
            self.vision.depth_callback,
            10
        )

        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.progression.odom_callback,
            10
        )

        self.reset_world()

    def initialize_sac_model(self):
        """
        Crée et initialise le modèle SAC pour l'entraînement.
        """
        model = SAC('MlpPolicy', self.env, verbose=1)
        return model

    def reset_world(self):
        """
        Réinitialise la simulation à chaque entraînement.
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

    def reset_robot(self):
        """
        Réinitialise la position du robot dans le monde de simulation.
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
                print("Robot reset successfully.")
            else:
                print("Failed to reset robot. Error:", result.stderr)
        except Exception as e:
            print("Exception during robot reset:", e)

    def run_training(self):
        """
        Exécute l'entraînement du robot en utilisant SAC avec des données de vision et de progression.
        """
        for _ in range(1000):  # Entraîner pendant 1000 itérations (ajustez selon vos besoins)
            state = self.env.reset()  # Réinitialiser l'état de l'environnement
            done = False
            total_reward = 0  # Pour accumuler la récompense

            while not done:
                action, _states = self.model.predict(state)
                next_state, reward, done, _ = self.env.step(action)

                # Accumuler la récompense
                total_reward += reward

                # Entraînement du modèle SAC à chaque itération
                self.model.learn(total_timesteps=1)
                
                # Mettre à jour l'état
                state = next_state

            # Afficher des informations de progression si nécessaire
            self.get_logger().info(f"Iteration complete. Total reward: {total_reward}")

if __name__ == '__main__':
    rclpy.init()
    node = RobotTrainer()

    try:
        node.run_training()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

