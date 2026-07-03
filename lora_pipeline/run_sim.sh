#!/usr/bin/env bash
# VLA 주행 시뮬레이션 기동 — gazebo(headless) + ego 차량 + 명령변환 sender
# 사용: bash run_sim.sh        (gzclient 3D뷰도 원하면: GUI=1 bash run_sim.sh)
# 종료: Ctrl+C (자식 프로세스 정리)
set -e
WS=/home/autolab/ros2_autonomous_vehicle_simulation
cd "$WS"

source /opt/ros/humble/setup.bash 2>/dev/null
source "$WS/install/setup.bash" 2>/dev/null
source /usr/share/gazebo/setup.sh 2>/dev/null
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=$WS/src/simulation_pkg/models:$GAZEBO_MODEL_PATH
export DISPLAY=:1

echo "[1/4] 기존 gzserver 정리..."
killall -9 gzserver gzclient 2>/dev/null || true
sleep 2

echo "[2/4] gzserver 기동(headless, 소프트웨어 렌더)..."
gzserver install/simulation_pkg/share/simulation_pkg/worlds/track.world \
    -s libgazebo_ros_factory.so &
GZ_PID=$!

# spawn_entity 서비스 대기
echo "      gazebo factory 대기..."
for i in $(seq 1 30); do
    ros2 service list 2>/dev/null | grep -q /spawn_entity && break
    sleep 1
done

echo "[3/4] ego 차량 스폰..."
ros2 run simulation_pkg load_ego_car_node 2>&1 | grep -i 'spawn status' || true

[ "$GUI" = "1" ] && { echo "      gzclient(3D뷰) 기동..."; gzclient & }

echo "[4/4] sim_sender 기동(topic_control_signal → cmd_vel)..."
trap 'echo; echo "정리 중..."; kill $GZ_PID 2>/dev/null; killall -9 gzserver gzclient 2>/dev/null; exit 0' INT TERM
ros2 run simulation_pkg sim_simulation_sender_node
