#!/usr/bin/env python3
"""
키보드 teleop — topic_control_signal(MotionCommand) 발행. 사람이 직접 운전.
LoRA 데모 수집용. collect_demos_node.py 와 함께 실행.

조작 (이 터미널에 포커스):
    w / s : 속도 +/-          a / d : 조향 좌/우(-/+)
    space : 정지(속도0)        c    : 조향 중앙
    q     : 종료
조향은 키 안 누르면 매 틱 0으로 서서히 복귀(직선 라벨 정확).
조향 방향이 반대로 느껴지면 --invert.
"""
import os, sys, termios, tty, select, time, argparse, threading
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from interfaces_pkg.msg import MotionCommand

CONTROL_TOPIC = "topic_control_signal"
PUB_HZ      = 15.0
MAX_SPEED   = 255      # MotionCommand 최대(0~255). 풀스피드 허용
SPEED_STEP  = 20       # 한 번에 더 크게 가감속
STEER_MAX   = 7
STEER_DT    = float(os.environ.get("STEER_DT", "0.18"))  # 꾹 누를 때 1단계당 시간[s](클수록 느림)
SPEED_DT    = float(os.environ.get("SPEED_DT", "0.18"))  # 꾹 누를 때 속도 1스텝당 시간[s](조향과 동일 페이스)
HOLD_WIN    = 0.15       # 마지막 키 후 이 시간까지 연속 이동(키반복이 갱신)


class Teleop(Node):
    def __init__(self, invert: bool):
        super().__init__("teleop_keyboard")
        self.invert = invert
        self.steering = 0
        self.speed = 0
        self.dir = 0            # 현재 조향 이동 방향(-1/+1/0)
        self.hold_until = 0.0   # 이 시각까지 dir로 연속 이동(키 반복이 갱신)
        self.last_move = 0.0
        self.sdir = 0           # 현재 속도 변화 방향(-1/+1/0)
        self.shold_until = 0.0  # 이 시각까지 sdir로 연속 가감속
        self.slast_move = 0.0
        # 키 반복 빠르게(눌러서 연속 조향). 실패해도 무해.
        os.system("xset r rate 200 12 >/dev/null 2>&1")
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.create_timer(1.0 / PUB_HZ, self._tick)
        self._run = True
        threading.Thread(target=self._key_loop, daemon=True).start()
        self.get_logger().info("teleop 준비. w/s=속도 a/d=조향 space=정지 c=중앙 q=종료")

    def _tick(self):
        # 꾹 누르면 키 반복이 hold_until을 갱신 → dir로 연속 이동. 떼면 값 유지.
        now = time.monotonic()
        if now >= self.hold_until:
            self.dir = 0   # 키 떼짐(반복 끊김) → 다음 탭이 즉시 반응하도록
        elif self.dir != 0 and now - self.last_move >= STEER_DT:
            self.steering = max(-STEER_MAX, min(STEER_MAX, self.steering + self.dir))
            self.last_move = now
            sys.stdout.write(f"\r st={self.steering:+d} sp={self.speed}    "); sys.stdout.flush()
        # 속도도 꾹 누름 시 SPEED_DT 페이스로 가감속(조향과 동일)
        if now >= self.shold_until:
            self.sdir = 0
        elif self.sdir != 0 and now - self.slast_move >= SPEED_DT:
            self.speed = max(-MAX_SPEED, min(MAX_SPEED, self.speed + self.sdir * SPEED_STEP))
            self.slast_move = now
            sys.stdout.write(f"\r st={self.steering:+d} sp={self.speed}    "); sys.stdout.flush()
        st = self.steering
        if self.speed < 0:          # 후진 시 조향 반대 보정(진행방향 기준 직관적 조작)
            st = -st
        if self.invert:
            st = -st
        m = MotionCommand()
        m.steering = int(st)
        m.left_speed = int(self.speed)
        m.right_speed = int(self.speed)
        self.pub.publish(m)

    def _key_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._run:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':   # ESC 시퀀스(화살표키 등) — 소비하고 무시
                        select.select([sys.stdin], [], [], 0.01)
                        try: sys.stdin.read(2)
                        except Exception: pass
                        continue
                    c = ch.lower(); now = time.monotonic()
                    if c == 'q':
                        self._run = False; rclpy.shutdown(); break
                    elif c == 'w':
                        if self.sdir != 1: self.speed = min(MAX_SPEED, self.speed + SPEED_STEP); self.slast_move = now
                        self.sdir = 1; self.shold_until = now + HOLD_WIN
                    elif c == 's':
                        if self.sdir != -1: self.speed = max(-MAX_SPEED, self.speed - SPEED_STEP); self.slast_move = now
                        self.sdir = -1; self.shold_until = now + HOLD_WIN
                    elif c == ' ': self.speed = 0; self.sdir = 0
                    elif c == 'x': self.steering = 0; self.speed = 0; self.dir = 0; self.sdir = 0
                    elif c == 'a':
                        # 방향 바뀌면 즉시 1단계 반응, 그 외는 tick이 STEER_DT 속도로 이동
                        if self.dir != -1: self.steering = max(-STEER_MAX, self.steering-1); self.last_move = now
                        self.dir = -1; self.hold_until = now + HOLD_WIN
                    elif c == 'd':
                        if self.dir != 1: self.steering = min(STEER_MAX, self.steering+1); self.last_move = now
                        self.dir = 1; self.hold_until = now + HOLD_WIN
                    else:
                        continue
                    sys.stdout.write(f"\r st={self.steering:+d} sp={self.speed}    ")
                    sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--invert", action="store_true", help="조향 부호 반전")
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = Teleop(args.invert)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
