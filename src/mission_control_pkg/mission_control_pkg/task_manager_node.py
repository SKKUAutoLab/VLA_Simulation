#!/usr/bin/env python3
"""
task_manager_node.py
====================
미션 태스크 관리 및 순차 실행 노드

기능:
  1. 사용자 명령 파싱 → Task 큐 생성
  2. Task 순차 실행 (spin_in_place, navigate_to, stop, wait)
  3. 장애물 감지 시 차선 변경 회피 (navigate_to + avoid_obstacles=True)
  4. Task 완료/실패 시 상태 퍼블리시
  5. 수행 불가 Task는 reason과 함께 로깅

구독 토픽:
  /mission/command        (std_msgs/String)   - 사용자 명령
  /odom                   (nav_msgs/Odometry) - 자차 위치/방향
  lidar_obstacle_info     (std_msgs/Bool)     - 장애물 감지 여부
  lidar_processed         (sensor_msgs/LaserScan) - LiDAR 원본
  vla/status              (std_msgs/String)   - VLA 목표 도달 상태

퍼블리시 토픽:
  vla/goal_cmd            (std_msgs/String)   - VLA 브레인 명령
  topic_control_signal    (interfaces_pkg/MotionCommand) - 모터 직접 제어
  /mission/task_status    (std_msgs/String)   - JSON 형식 태스크 상태
  /mission/reasoning      (std_msgs/String)   - 실패 이유 설명
  /mission/feedback       (std_msgs/String)   - 실시간 피드백
"""

import json
import math
import time
from collections import deque
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                        QoSDurabilityPolicy, QoSReliabilityPolicy)

from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from interfaces_pkg.msg import MotionCommand

from .task_parser import Task, TaskParser
from .track_data import LANE_CHANGE_PARAMS, LANDMARKS

import numpy as np


# ─── 차선 변경 상태 머신 ─────────────────────────────────────────────────
class LaneChangeState(Enum):
    IDLE        = auto()
    CHANGING    = auto()   # 회피 차선으로 이동 중
    PASSING     = auto()   # 장애물 옆 지나치는 중
    RETURNING   = auto()   # 원래 차선으로 복귀 중


# ─── 노드 ─────────────────────────────────────────────────────────────────
class TaskManagerNode(Node):

    SPIN_SPEED     = 35     # 제자리 회전 속도
    SPIN_STEER     = 7      # 제자리 회전 조향 (최대)
    NAV_TIMEOUT    = 120.0  # 이동 태스크 타임아웃 [초]
    TIMER_HZ       = 20.0   # 메인 루프 주파수

    def __init__(self):
        super().__init__("task_manager_node")

        self._parser = TaskParser()
        self._task_queue: deque[Task] = deque()
        self._current_task: Task | None = None
        self._completed_tasks: list[Task] = []
        self._failed_tasks:    list[Task] = []

        # ── odom 상태 ──
        self._pos_x     = 0.0
        self._pos_y     = 0.0
        self._yaw       = 0.0
        self._odom_ready = False

        # ── 제자리 회전 ──
        self._spin_initial_yaw   = 0.0
        self._spin_accum_deg     = 0.0
        self._spin_prev_yaw      = 0.0

        # ── 이동 태스크 ──
        self._nav_start_time     = 0.0
        self._nav_goal_reached   = False
        self._nav_target         = ""

        # ── LiDAR 장애물 ──
        self._obstacle_detected  = False
        self._obstacle_min_dist  = 999.0

        # ── 차선 변경 ──
        self._lc_state           = LaneChangeState.IDLE
        self._lc_origin_x        = 0.0
        self._lc_target_x        = 0.0
        self._lc_pass_start_time = 0.0
        self._lc_return_dist     = LANE_CHANGE_PARAMS["clear_distance"]

        # ── 대기 태스크 ──
        self._wait_start_time    = 0.0

        # ── QoS ──
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── 구독자 ──
        self.create_subscription(String,   "/mission/command",
                                 self._cmd_cb,      reliable_qos)
        self.create_subscription(Odometry, "/odom",
                                 self._odom_cb,     best_effort_qos)
        self.create_subscription(Bool,     "lidar_obstacle_info",
                                 self._obstacle_cb, reliable_qos)
        self.create_subscription(LaserScan,"lidar_processed",
                                 self._lidar_cb,    best_effort_qos)
        self.create_subscription(String,   "vla/status",
                                 self._vla_status_cb, reliable_qos)

        # ── 퍼블리셔 ──
        self._pub_vla    = self.create_publisher(String,        "vla/goal_cmd",          reliable_qos)
        self._pub_motion = self.create_publisher(MotionCommand, "topic_control_signal",  reliable_qos)
        self._pub_status = self.create_publisher(String,        "/mission/task_status",  reliable_qos)
        self._pub_reason = self.create_publisher(String,        "/mission/reasoning",    reliable_qos)
        self._pub_fb     = self.create_publisher(String,        "/mission/feedback",     reliable_qos)

        # ── 메인 루프 타이머 ──
        self._timer = self.create_timer(1.0 / self.TIMER_HZ, self._tick)

        self.get_logger().info(
            "✅ TaskManagerNode 시작\n"
            "  명령 입력: ros2 topic pub /mission/command std_msgs/msg/String "
            "\"data: '제자리에서 한 바퀴 돌고 횡단보도로 가줘'\" --once"
        )

    # ─────────────────────────────────────────────────────────────────────
    # 콜백
    # ─────────────────────────────────────────────────────────────────────
    def _cmd_cb(self, msg: String):
        """사용자 명령 수신 → 파싱 → 큐에 추가."""
        command = msg.data.strip()
        self.get_logger().info(f"📥 명령 수신: '{command}'")

        tasks, reason = self._parser.parse_with_reason(command)

        if not tasks:
            self.get_logger().warn(f"⚠️ 파싱 실패:\n{reason}")
            self._pub_reason.publish(String(data=reason))
            self._pub_fb.publish(String(data=f"명령 인식 실패: {command}"))
            return

        for t in tasks:
            self._task_queue.append(t)

        desc = " → ".join(TaskParser.describe_task(t) for t in tasks)
        self.get_logger().info(f"📋 태스크 큐 추가: {desc}")
        self._pub_fb.publish(String(data=f"태스크 등록: {desc}"))
        self._publish_status()

    def _odom_cb(self, msg: Odometry):
        """오도메트리 갱신."""
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)
        self._odom_ready = True

    def _obstacle_cb(self, msg: Bool):
        self._obstacle_detected = msg.data

    def _lidar_cb(self, msg: LaserScan):
        """LiDAR에서 전방 최소 거리 계산."""
        ranges = np.array(msg.ranges, dtype=float)
        n = len(ranges)
        if n == 0:
            return
        step = max(1, int(round(np.radians(20) / msg.angle_increment))) if msg.angle_increment > 0 else 20
        front = list(range(0, step + 1)) + list(range(n - step, n))
        valid = ranges[front]
        valid = valid[(np.isfinite(valid)) & (valid >= 0.8) & (valid <= 12.0)]
        self._obstacle_min_dist = float(np.min(valid)) if len(valid) > 0 else 999.0

    def _vla_status_cb(self, msg: String):
        """VLA 브레인 상태 수신 → 이동 완료 판정."""
        data = msg.data
        # "REACHED:target" 형태
        if data.startswith("REACHED:"):
            reached_target = data.split(":", 1)[1]
            if (self._current_task is not None
                    and self._current_task.task_type == "navigate_to"
                    and reached_target == self._nav_target):
                self._nav_goal_reached = True
                self.get_logger().info(f"✅ 목표 도달 확인: {reached_target}")

    # ─────────────────────────────────────────────────────────────────────
    # 메인 루프
    # ─────────────────────────────────────────────────────────────────────
    def _tick(self):
        """20 Hz 메인 실행 루프."""
        # 현재 태스크가 없으면 큐에서 꺼냄
        if self._current_task is None:
            if not self._task_queue:
                return
            self._current_task = self._task_queue.popleft()
            self._start_task(self._current_task)

        # 현재 태스크 실행 및 완료 확인
        self._execute_current_task()
        self._publish_status()

    # ─────────────────────────────────────────────────────────────────────
    # 태스크 시작
    # ─────────────────────────────────────────────────────────────────────
    def _start_task(self, task: Task):
        task.status = "in_progress"
        name = TaskParser.describe_task(task)
        self.get_logger().info(f"▶ 태스크 시작: [{task.task_id}] {name}")
        self._pub_fb.publish(String(data=f"시작: {name}"))

        t_type = task.task_type

        if t_type == "stop":
            self._publish_motion(0, 0, 0)
            self._complete_task(task, "즉시 정지 완료")

        elif t_type == "resume":
            # VLA 브레인에 재개 명령
            self._pub_vla.publish(String(data="출발"))
            self._complete_task(task, "재개 명령 전송 완료")

        elif t_type == "spin_in_place":
            if not self._odom_ready:
                self._fail_task(task,
                    "제자리 회전 불가: Odometry 데이터 없음. "
                    "/odom 토픽이 활성화되어 있는지 확인하세요.")
                return
            self._spin_initial_yaw = self._yaw
            self._spin_prev_yaw    = self._yaw
            self._spin_accum_deg   = 0.0

        elif t_type == "navigate_to":
            target = task.params.get("target", "traffic_light")
            mode   = task.params.get("mode",   "track")
            self._nav_target       = target
            self._nav_goal_reached = False
            self._nav_start_time   = time.time()
            self._lc_state         = LaneChangeState.IDLE
            # VLA 브레인에 목표 명령 전송
            cmd_text = self._build_vla_command(target, mode)
            self._pub_vla.publish(String(data=cmd_text))
            self.get_logger().info(f"🎯 VLA 명령 전송: '{cmd_text}'")

        elif t_type == "wait":
            self._wait_start_time = time.time()

        else:
            self._fail_task(task, f"지원하지 않는 태스크 유형: {t_type}")

    # ─────────────────────────────────────────────────────────────────────
    # 태스크 실행 (매 tick)
    # ─────────────────────────────────────────────────────────────────────
    def _execute_current_task(self):
        task = self._current_task
        if task is None or task.status != "in_progress":
            return

        t_type = task.task_type

        if t_type == "spin_in_place":
            self._execute_spin(task)

        elif t_type == "navigate_to":
            self._execute_navigate(task)

        elif t_type == "wait":
            elapsed = time.time() - self._wait_start_time
            seconds = task.params.get("seconds", 1)
            if elapsed >= seconds:
                self._complete_task(task, f"{seconds}초 대기 완료")
            else:
                remaining = seconds - elapsed
                self._pub_fb.publish(String(data=f"대기 중... {remaining:.1f}초 남음"))

    # ─────────────────────────────────────────────────────────────────────
    # 제자리 회전 실행
    # ─────────────────────────────────────────────────────────────────────
    def _execute_spin(self, task: Task):
        target_deg = task.params.get("degrees", 360)

        if not self._odom_ready:
            self._fail_task(task, "제자리 회전 불가: Odometry 수신 없음")
            return

        # 누적 회전각 계산
        delta = self._normalize_angle_deg(
            math.degrees(self._yaw) - math.degrees(self._spin_prev_yaw)
        )
        self._spin_accum_deg += abs(delta)
        self._spin_prev_yaw   = self._yaw

        progress = min(100, self._spin_accum_deg / target_deg * 100)

        if self._spin_accum_deg >= target_deg:
            # 완료
            self._publish_motion(0, 0, 0)
            self._complete_task(task,
                f"제자리 {target_deg}도 회전 완료 (실제: {self._spin_accum_deg:.1f}도)")
        else:
            # 계속 회전 (반시계 방향)
            self._publish_motion(self.SPIN_STEER, self.SPIN_SPEED, self.SPIN_SPEED)
            self._pub_fb.publish(String(
                data=f"회전 중: {self._spin_accum_deg:.0f}/{target_deg}도 ({progress:.0f}%)"
            ))

    # ─────────────────────────────────────────────────────────────────────
    # 이동 태스크 실행 (차선 변경 포함)
    # ─────────────────────────────────────────────────────────────────────
    def _execute_navigate(self, task: Task):
        avoid = task.params.get("avoid_obstacles", False)
        elapsed = time.time() - self._nav_start_time

        # 타임아웃
        if elapsed > self.NAV_TIMEOUT:
            target = task.params.get("target", "?")
            self._fail_task(task,
                f"이동 타임아웃: {target}에 {self.NAV_TIMEOUT}초 내 도달 실패. "
                "VLA 브레인이 활성화되어 있는지, 목적지가 올바른지 확인하세요.")
            return

        # 목표 도달
        if self._nav_goal_reached:
            target = task.params.get("target", "목적지")
            self._complete_task(task, f"{target} 도달 완료")
            return

        # 장애물 회피 모드
        if avoid:
            self._handle_obstacle_avoidance(task)
        else:
            # 일반 주행 (VLA 브레인이 제어)
            dist_text = f"{self._obstacle_min_dist:.1f}m" if self._obstacle_min_dist < 10 else "없음"
            self._pub_fb.publish(String(
                data=f"이동 중: {task.params.get('target','?')} | "
                     f"경과: {elapsed:.0f}s | 전방 장애물: {dist_text}"
            ))

    def _handle_obstacle_avoidance(self, task: Task):
        """차선 변경 회피 상태 머신 실행."""
        state = self._lc_state

        if state == LaneChangeState.IDLE:
            if (self._obstacle_detected
                    and self._obstacle_min_dist < 5.0
                    and self._obstacle_min_dist > LANE_CHANGE_PARAMS["obstacle_stop_dist"]):
                # 장애물 감지 → 차선 변경 시작
                self._lc_origin_x = self._pos_x
                offset = LANE_CHANGE_PARAMS["avoid_offset_x"]
                self._lc_target_x = self._pos_x + offset
                self._lc_state    = LaneChangeState.CHANGING
                self.get_logger().info(
                    f"🔄 차선 변경 시작: x {self._pos_x:.1f} → {self._lc_target_x:.1f} "
                    f"(전방 {self._obstacle_min_dist:.1f}m)"
                )
                self._pub_fb.publish(String(data="장애물 감지 → 차선 변경 중..."))

        elif state == LaneChangeState.CHANGING:
            # 목표 X로 이동 (비례 조향)
            x_err = self._lc_target_x - self._pos_x
            if abs(x_err) < 0.3:
                # 차선 변경 완료 → 장애물 추월 시작
                self._lc_state         = LaneChangeState.PASSING
                self._lc_pass_start_time = time.time()
                self.get_logger().info("🔄 차선 변경 완료 → 장애물 추월 중")
                self._pub_fb.publish(String(data="차선 변경 완료 → 장애물 추월 중..."))
            else:
                steer = int(np.clip(x_err * 3.0, -7, 7))
                self._publish_motion(steer,
                    LANE_CHANGE_PARAMS["lane_change_speed"],
                    LANE_CHANGE_PARAMS["lane_change_speed"])

        elif state == LaneChangeState.PASSING:
            # 장애물이 없어지면 복귀
            if self._obstacle_min_dist > self._lc_return_dist:
                self._lc_state = LaneChangeState.RETURNING
                self.get_logger().info("🔄 장애물 통과 → 원래 차선 복귀 중")
                self._pub_fb.publish(String(data="장애물 통과 → 원래 차선 복귀 중..."))
            else:
                # 장애물 옆 추월 속도
                self._publish_motion(0,
                    LANE_CHANGE_PARAMS["passing_speed"],
                    LANE_CHANGE_PARAMS["passing_speed"])
                elapsed = time.time() - self._lc_pass_start_time
                if elapsed > 10.0:
                    # 10초 넘으면 강제 복귀
                    self._lc_state = LaneChangeState.RETURNING

        elif state == LaneChangeState.RETURNING:
            # 원래 차선으로 복귀
            x_err = self._lc_origin_x - self._pos_x
            if abs(x_err) < 0.3:
                self._lc_state = LaneChangeState.IDLE
                # VLA 브레인 재개
                target = task.params.get("target", "traffic_light")
                mode   = task.params.get("mode", "track")
                cmd    = self._build_vla_command(target, mode)
                self._pub_vla.publish(String(data=cmd))
                self.get_logger().info("✅ 원래 차선 복귀 완료 → 주행 재개")
                self._pub_fb.publish(String(data="차선 복귀 완료 → 주행 재개"))
            else:
                steer = int(np.clip(x_err * 3.0, -7, 7))
                self._publish_motion(steer,
                    LANE_CHANGE_PARAMS["lane_change_speed"],
                    LANE_CHANGE_PARAMS["lane_change_speed"])

    # ─────────────────────────────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────────────────────────────
    def _complete_task(self, task: Task, msg: str = ""):
        task.status = "completed"
        task.reason = msg
        self._completed_tasks.append(task)
        name = TaskParser.describe_task(task)
        self.get_logger().info(f"✅ 완료: [{task.task_id}] {name} — {msg}")
        self._pub_fb.publish(String(data=f"완료: {name}"))
        self._current_task = None
        self._lc_state = LaneChangeState.IDLE

    def _fail_task(self, task: Task, reason: str):
        task.status = "failed"
        task.reason = reason
        self._failed_tasks.append(task)
        name = TaskParser.describe_task(task)
        self.get_logger().error(f"❌ 실패: [{task.task_id}] {name}\n  이유: {reason}")
        self._pub_reason.publish(String(data=f"[실패] {name}\n이유: {reason}"))
        self._pub_fb.publish(String(data=f"실패: {name}"))
        self._current_task = None
        # 모터 정지
        self._publish_motion(0, 0, 0)

    def _publish_motion(self, steering: int, left_speed: int, right_speed: int):
        msg = MotionCommand()
        msg.steering    = int(np.clip(steering,    -7, 7))
        msg.left_speed  = int(np.clip(left_speed,   0, 255))
        msg.right_speed = int(np.clip(right_speed,  0, 255))
        self._pub_motion.publish(msg)

    def _publish_status(self):
        """JSON 형식으로 현재 태스크 상태 퍼블리시."""
        status = {
            "current": self._current_task.to_dict() if self._current_task else None,
            "queue":   [t.to_dict() for t in self._task_queue],
            "completed": [t.to_dict() for t in self._completed_tasks[-5:]],
            "failed":    [t.to_dict() for t in self._failed_tasks[-5:]],
            "vehicle": {
                "x":   round(self._pos_x, 2),
                "y":   round(self._pos_y, 2),
                "yaw": round(math.degrees(self._yaw), 1),
            },
            "obstacle": {
                "detected": self._obstacle_detected,
                "min_dist": round(self._obstacle_min_dist, 2),
                "lc_state": self._lc_state.name,
            },
        }
        self._pub_status.publish(String(data=json.dumps(status, ensure_ascii=False)))

    def _build_vla_command(self, target: str, mode: str) -> str:
        """VLA 브레인 명령 문자열 생성."""
        cmd_map = {
            "crosswalk":     "신호등까지 가줘",
            "traffic_light": "신호등까지 가줘",
            "start":         "출발점으로 돌아가",
            "obstacle_1":    "장애물1 직접 가줘",
            "obstacle_2":    "장애물2 직접 가줘",
            "mid":           "중간",
        }
        base_cmd = cmd_map.get(target, f"{target} 가줘")
        if mode == "direct":
            return base_cmd.replace("까지 가줘", " 직접 가줘").replace("가줘", "직접 가줘")
        return base_cmd

    @staticmethod
    def _normalize_angle_deg(deg: float) -> float:
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
