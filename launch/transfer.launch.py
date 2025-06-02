from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('load', default_value='', description='Chemin vers un modèle à charger'),

        Node(
            package='jettec_robot',
            executable='vision_path_node',
            name='vision_path_node',
            output='screen',
        ),

        Node(
            package='jettec_robot',
            executable='transfer_node',
            name='transfer_node',
            output='screen',
        ),
    ])
