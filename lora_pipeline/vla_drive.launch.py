#!/usr/bin/env python3
"""
VLA 주행 통합 런치 — gazebo + ego + sender + drive(+brain +gui) 한 번에.

실행:
  source /opt/ros/humble/setup.bash && source ~/VLA_simulation/install/setup.bash
  ros2 launch ~/VLA_simulation/lora_pipeline/vla_drive.launch.py

옵션(런치 인자) — 기본 전부 true:
  brain:=false    자연어/장면 브레인(Qwen3-VL) 끄기 (끄면 카메라 FPS↑, 부드러운 주행)
  gui:=false      PySide6 명령 GUI 끄기
  gzclient:=false gazebo 3D 뷰 끄기(headless)
예) 빠른 주행만: ros2 launch .../vla_drive.launch.py brain:=false
"""
import os
from launch import LaunchDescription
from launch.actions import (ExecuteProcess, TimerAction, SetEnvironmentVariable,
                            DeclareLaunchArgument)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

WS = "/home/autolab/VLA_simulation"
LORA = os.path.join(WS, "lora_pipeline")
WORLD = os.path.join(WS, "install/simulation_pkg/share/simulation_pkg/worlds/track.world")


def _py(script, env_extra=None, condition=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    kw = {"cmd": ["python3", "-u", os.path.join(LORA, script)],
          "cwd": LORA, "output": "screen", "additional_env": env}
    if condition is not None:
        kw["condition"] = condition
    return ExecuteProcess(**kw)


def generate_launch_description():
    brain = LaunchConfiguration("brain")
    gui = LaunchConfiguration("gui")
    gzclient = LaunchConfiguration("gzclient")

    return LaunchDescription([
        DeclareLaunchArgument("brain", default_value="true",
                              description="자연어/장면 브레인(Qwen3-VL) 켜기 (false면 카메라 FPS↑)"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="PySide6 명령 GUI 켜기"),
        DeclareLaunchArgument("gzclient", default_value="true",
                              description="gazebo 3D 뷰 띄우기 (false면 headless)"),

        # gazebo 렌더/모델 환경 (소프트웨어 렌더 유지 — PRIME offload는 카메라 readback 느려서 금지)
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", "/opt/ros/humble/lib:" + os.environ.get("GAZEBO_PLUGIN_PATH", "")),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", os.path.join(WS, "src/simulation_pkg/models") + ":" + os.environ.get("GAZEBO_MODEL_PATH", "")),
        SetEnvironmentVariable("DISPLAY", os.environ.get("DISPLAY", ":1")),

        # 1) gzserver (headless, factory 플러그인) — gazebo 기본환경 source 필수(팩토리 플러그인 로드)
        ExecuteProcess(
            cmd=["bash", "-c",
                 "source /usr/share/gazebo/setup.sh && "
                 f"export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH && "
                 f"export GAZEBO_MODEL_PATH={os.path.join(WS,'src/simulation_pkg/models')}:$GAZEBO_MODEL_PATH && "
                 f"exec gzserver {WORLD} -s libgazebo_ros_factory.so"],
            output="screen"),
        # 1b) gzclient (옵션, 3D 뷰)
        TimerAction(period=5.0, actions=[
            ExecuteProcess(cmd=["gzclient"], output="screen", condition=IfCondition(gzclient))]),

        # 2) ego 차량 스폰 (gazebo factory 뜬 뒤)
        TimerAction(period=6.0, actions=[
            Node(package="simulation_pkg", executable="load_ego_car_node", output="screen")]),

        # 3) 명령 변환 sender (topic_control_signal → cmd_vel)
        TimerAction(period=7.0, actions=[
            Node(package="simulation_pkg", executable="sim_simulation_sender_node", output="screen")]),

        # 4) drive 노드 (비전 차선추종 + 기하접근/복구) — 단 하나만
        TimerAction(period=9.0, actions=[_py("vla_lora_drive_node.py")]),

        # 5) brain (옵션, 자연어/장면) — 약 4분 로딩
        TimerAction(period=9.0, actions=[
            _py("vla_brain_node.py", env_extra={"SCENE_HZ": "1.5"}, condition=IfCondition(brain))]),

        # 6) GUI (옵션)
        TimerAction(period=12.0, actions=[
            _py("vla_gui.py", condition=IfCondition(gui))]),
    ])
