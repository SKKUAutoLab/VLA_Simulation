#!/usr/bin/env python3

import os
import subprocess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

# Force Gazebo to use NVIDIA GPU instead of software renderer (llvmpipe)
nvidia_env = {
    '__NV_PRIME_RENDER_OFFLOAD': '1',
    '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
    '__VK_LAYER_NV_optimus': 'NVIDIA_only',
}

def generate_launch_description():

    subprocess.run(['killall', 'gzserver'])
    subprocess.run(['killall', 'gzclient'])

    package_dir = get_package_share_directory('simulation_pkg')
    world_file = os.path.join(package_dir, 'worlds', 'track.world')

    return LaunchDescription([

        ExecuteProcess(
            cmd=['gazebo', world_file, '-s', 'libgazebo_ros_factory.so'],
            additional_env=nvidia_env,
            output='screen'),

        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            output='screen',
        ),

        # Qwen3-VL-2B: 카메라 이미지만 보고 직접 MotionCommand 퍼블리시
        # yolov8_node, lane_info_extractor_node, motion_planner_node 대체
        Node(
            package='qwen_vl_pkg',
            executable='qwen_vl_driver_node',
            output='screen',
        ),

        Node(
            package='simulation_pkg',
            executable='sim_simulation_sender_node',
            output='screen',
        ),

    ])
