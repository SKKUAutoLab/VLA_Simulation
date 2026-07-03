#!/usr/bin/env bash
set -e
cd /home/autolab/ros2_autonomous_vehicle_simulation/lora_pipeline
source /opt/ros/humble/setup.bash 2>/dev/null || true
source /home/autolab/ros2_autonomous_vehicle_simulation/install/setup.bash 2>/dev/null || true

# 1) 주행 노드 종료(텔레포트 충돌 방지)
for p in $(pgrep -f wp_drive_node.py); do kill -9 "$p" 2>/dev/null || true; done
sleep 2

# 2) 현재 LANES 확인 출력
echo "[LANES in generate_wp_data.py]"
grep -A2 'LANES = ' generate_wp_data.py | head -3

# 3) 복구 WP 재생성
rm -f dataset/labels_wpL0.csv dataset/labels_wpL1.csv
python3 -u generate_wp_data.py
echo "[GEN done] $(wc -l < dataset/labels_wpL0.csv) + $(wc -l < dataset/labels_wpL1.csv)"

# 4) 학습
python3 -u train_wp.py
echo "[ALL DONE]"
