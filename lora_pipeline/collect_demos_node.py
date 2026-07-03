#!/usr/bin/env python3
"""
Stage 1 — 데모 수집 노드 (LoRA 학습용)
=====================================
고전 AD 스택(driving_sim.launch.py: YOLO→path_planner→motion_planner)을 expert로,
주행 중 (카메라 이미지 → expert의 steering/speed) 쌍을 디스크에 저장한다.

함께 실행:
    ros2 launch simulation_pkg driving_sim.launch.py        # expert 주행
    python3 lora_pipeline/collect_demos_node.py             # 본 수집 노드

저장:
    lora_pipeline/dataset/images/<run>_<seq>.jpg
    lora_pipeline/dataset/labels.csv  (fname,steering,left_speed,right_speed,t)

규칙:
  - 카메라 프레임마다 최신 MotionCommand를 라벨로 부착(ZOH). 제어 메시지가
    0.5s 이상 끊기면 그 프레임은 버림(스택 미동작 구간 제외).
  - SAVE_HZ 로 다운샘플. SKIP_STOPPED=True면 속도 0 프레임 제외(과다한 정지 라벨 방지).
"""
import os, csv, time, argparse
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from interfaces_pkg.msg import MotionCommand
from cv_bridge import CvBridge

CAMERA_TOPIC  = "camera/image_raw"
CONTROL_TOPIC = "topic_control_signal"
OUT_DIR       = os.path.join(os.path.dirname(__file__), "dataset")
IMG_DIR       = os.path.join(OUT_DIR, "images")
CSV_PATH      = os.path.join(OUT_DIR, "labels.csv")

SAVE_HZ        = 8.0     # 저장 다운샘플 (카메라가 더 빨라도 이 주기로만 저장)
CTRL_TIMEOUT   = 0.5     # 이 시간 내 제어 메시지 없으면 프레임 버림 [s]
SKIP_STOPPED   = True    # speed==0 프레임 제외


class CollectNode(Node):
    def __init__(self, run_tag: str, control_topic: str = CONTROL_TOPIC):
        super().__init__("collect_demos_node")
        global CONTROL_TOPIC
        CONTROL_TOPIC = control_topic
        os.makedirs(IMG_DIR, exist_ok=True)
        self.run_tag = run_tag
        self.bridge = CvBridge()
        self.cmd = None              # (steering, lspeed, rspeed)
        self.cmd_stamp = 0.0
        self.seq = 0
        self.saved = 0
        self.last_save = 0.0

        # CSV 헤더 (없을 때만)
        self.csv_f = open(CSV_PATH, "a", newline="")
        self.csv_w = csv.writer(self.csv_f)
        if os.path.getsize(CSV_PATH) == 0:
            self.csv_w.writerow(["fname", "steering", "left_speed", "right_speed", "t"])

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST,
                                durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        ctrl_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                              history=QoSHistoryPolicy.KEEP_LAST,
                              durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(MotionCommand, CONTROL_TOPIC, self._cmd_cb, ctrl_qos)
        self.create_subscription(Image, CAMERA_TOPIC, self._img_cb, sensor_qos)
        self.get_logger().info(
            f"수집 시작: run={run_tag}  out={IMG_DIR}  SAVE_HZ={SAVE_HZ}")

    def _cmd_cb(self, msg: MotionCommand):
        self.cmd = (msg.steering, msg.left_speed, msg.right_speed)
        self.cmd_stamp = time.monotonic()

    def _img_cb(self, msg: Image):
        now = time.monotonic()
        if now - self.last_save < 1.0 / SAVE_HZ:
            return
        if self.cmd is None or (now - self.cmd_stamp) > CTRL_TIMEOUT:
            return  # expert 미동작 → 버림
        st, ls, rs = self.cmd
        if SKIP_STOPPED and ls == 0 and rs == 0:
            return
        self.last_save = now
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        fname = f"{self.run_tag}_{self.seq:06d}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, fname), cv_img)
        self.csv_w.writerow([fname, st, ls, rs, f"{now:.3f}"])
        self.csv_f.flush()
        self.seq += 1
        self.saved += 1
        if self.saved % 25 == 0:
            self.get_logger().info(f"저장 {self.saved}장  (마지막 st={st} sp={ls})")

    def destroy_node(self):
        try:
            self.csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="run 태그(미지정 시 시작 시각)")
    ap.add_argument("--control-topic", default=CONTROL_TOPIC,
                    help="라벨 기록 토픽 (복구수집은 vla/expert_label)")
    args, _ = ap.parse_known_args()
    rclpy.init()
    tag = args.tag or f"run{int(time.time())}"
    node = CollectNode(tag, args.control_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"종료. 총 {node.saved}장 저장 → {CSV_PATH}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
