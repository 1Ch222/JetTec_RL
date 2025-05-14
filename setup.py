from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'jettec_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.xml')),
        ('share/' + package_name + '/world', glob('world/*.sdf')),
        ('share/' + package_name + '/model', glob('model/*.sdf')),
        ('share/' + package_name + '/world', glob('world/*.png')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),    
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mrsl',
    maintainer_email='137825924+1Ch222@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_line_node = jettec_robot.line_following.vision_line_node:main',
            'training_line_node = jettec_robot.line_following.training_line_node:main',
            'testing_line_node = jettec_robot.line_following.testing_line_node:main',
            'vision_path_node = jettec_robot.path_following.vision_path_node:main',
            'training_path_node = jettec_robot.path_following.training_path_node:main',
            'testing_path_node = jettec_robot.path_following.testing_path_node:main',
        ],
    },
)
