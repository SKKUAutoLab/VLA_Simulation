#!/usr/bin/env python3
"""
teleop_sim.launch.py — perception/planner 없이 차+카메라+sender만 띄운다.
키보드 teleop로 사람이 직접 운전하며 LoRA 데모를 수집하기 위한 최소 구성.
(driving_sim 과 달리 motion_planner 미실행 → topic_control_signal 충돌 없음)
"""
import os
import subprocess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

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
        Node(package='simulation_pkg', executable='load_ego_car_node',
             output='screen'),
        Node(package='simulation_pkg', executable='load_traffic_light_node',
             output='screen'),
        Node(package='simulation_pkg', executable='sim_simulation_sender_node',
             output='screen'),
    ])
