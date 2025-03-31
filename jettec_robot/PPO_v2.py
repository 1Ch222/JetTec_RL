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

# --------- reset_robot_position function ---------

def reset_robot_position():
    cmd = [
        "ign", "service", "-s", "/world/Line_track/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", "name: 'JetTec_Robot' position: { x: -1.0, y: 3.6, z: 0.15 } orientation: { x: 0, y: 0, z: 0, w: 1 }"
    ]
    try :
        command = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if command.returncode == 0:
            print("The robot's position has been reset succesfully.")
        else:
            print("Failed to reset robot using set_pose service. Error:", command.stderr)
    except Exception as e :
        print("Exception during robot reset using set_pose service:", e)



