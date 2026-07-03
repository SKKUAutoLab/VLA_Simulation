"""
task_parser.py
==============
자연어 명령 → Task 시퀀스 파서

사용 예시:
  "제자리에서 한 바퀴를 돌고 횡단보도로 가줘"
  → [Task(spin_in_place, 360°), Task(navigate_to, crosswalk, avoid=True)]

  "횡단보도까지 트랙을 따라 주행하되 중간에 만나는 장애물 차량이 있다면 차선 변경을 하여 회피해서 도착지점에 도달하게 해줘"
  → [Task(navigate_to, crosswalk, avoid=True, mode=track)]
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ─── Task 데이터 클래스 ───────────────────────────────────────────────────
@dataclass
class Task:
    task_type: str                           # 작업 유형
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"                  # pending | in_progress | completed | failed
    reason: str = ""                         # 실패 이유
    task_id: int = 0                         # 고유 ID

    def to_dict(self) -> dict:
        return {
            "task_id":   self.task_id,
            "task_type": self.task_type,
            "params":    self.params,
            "status":    self.status,
            "reason":    self.reason,
        }


# ─── 지원 목적지 키워드 ──────────────────────────────────────────────────
TARGET_KEYWORDS = {
    "crosswalk":     ["횡단보도", "crosswalk", "건널목"],
    "traffic_light": ["신호등", "traffic_light", "traffic light"],
    "start":         ["출발점", "처음", "스폰", "원점", "돌아가", "복귀", "start"],
    "obstacle_1":    ["장애물1", "장애물 1", "obstacle_1", "obstacle1"],
    "obstacle_2":    ["장애물2", "장애물 2", "obstacle_2", "obstacle2"],
    "mid":           ["중간", "midpoint", "mid"],
}

# ─── 인간 친화적 이름 ─────────────────────────────────────────────────────
TARGET_NAMES = {
    "crosswalk":     "횡단보도",
    "traffic_light": "신호등",
    "start":         "출발점",
    "obstacle_1":    "장애물 구역 1",
    "obstacle_2":    "장애물 구역 2",
    "mid":           "중간 지점",
}

TASK_TYPE_NAMES = {
    "spin_in_place":   "제자리 회전",
    "navigate_to":     "목적지 이동",
    "stop":            "정지",
    "resume":          "재개",
    "lane_change":     "차선 변경",
    "wait":            "대기",
}


class TaskParser:
    """자연어 명령 → Task 리스트 변환기."""

    _id_counter = 0

    @classmethod
    def _new_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def parse(self, command: str) -> List[Task]:
        """
        자연어 명령을 파싱하여 Task 리스트를 반환합니다.
        파싱 불가능한 경우 빈 리스트와 함께 이유를 별도 반환합니다.
        """
        if not command or not command.strip():
            return []

        segments = self._split_into_segments(command)
        tasks = []

        for seg in segments:
            task = self._parse_segment(seg.strip())
            if task:
                task.task_id = self._new_id()
                tasks.append(task)

        return tasks

    def parse_with_reason(self, command: str) -> tuple:
        """
        파싱 결과와 분석 이유를 함께 반환합니다.
        Returns: (tasks: List[Task], reason: str)
        """
        tasks = self.parse(command)
        if not tasks:
            reason = self._explain_parse_failure(command)
            return [], reason
        return tasks, ""

    def _split_into_segments(self, command: str) -> List[str]:
        """
        명령을 개별 작업 단위로 분리합니다.
        분리 패턴: "~하고", "~고", "그리고", "그 후", "다음에", "이후", ","(쉼표) 등

        한국어 "~고" 연결: "돌고 횡단보도로" → ["돌", "횡단보도로"]
        """
        # 접속사 기반 분리 패턴 (우선순위 순)
        patterns = [
            # 명시적 시간 연결사
            r'\s*그\s*다음에\s*',
            r'\s*그리고\s*나서\s*',
            r'\s*그\s*후\s*(?:에)?\s*',
            r'\s*이후\s*(?:에)?\s*',
            r'\s*다음에\s*',
            r'\s+then\s+',
            r'\s*그리고\s*',
            # 동사 연결 "~고" (한국어 핵심 패턴)
            # "돌고 ", "가고 ", "하고 " 등 — 뒤에 공백+한글이 올 때만 분리
            r'(?<=[가-힣])\s*고\s+(?=[가-힣])',
            # 쉼표
            r'\s*,\s*(?=.{5,})',
        ]

        result = [command]
        for pattern in patterns:
            new_result = []
            for seg in result:
                parts = re.split(pattern, seg, flags=re.IGNORECASE)
                new_result.extend([p.strip() for p in parts if p.strip()])
            result = new_result

        # 너무 짧은 조각 제거 (의미 있는 길이 이상)
        return [r for r in result if len(r) >= 2]

    def _parse_segment(self, seg: str) -> Optional[Task]:
        """단일 명령 세그먼트를 Task로 변환."""
        lower = seg.lower()

        # ① 정지 명령
        if any(k in lower for k in ["멈춰", "정지해", "정지", "stop", "멈춤"]):
            return Task("stop")

        # ② 재개 명령 (목표 없는 경우만)
        if any(k in lower for k in ["계속 가", "계속가", "출발해", "출발", "재개", "go", "resume"]):
            if not self._detect_target(lower):
                return Task("resume")

        # ③ 제자리 회전
        spin_keywords = ["제자리", "spin", "제자리서", "제자리에서", "그 자리", "그자리"]
        if any(k in lower for k in spin_keywords):
            degrees = self._detect_degrees(lower)
            return Task("spin_in_place", params={"degrees": degrees})

        # ④ 회전 (거리/목표 없는 단순 회전)
        if re.search(r'(360|한\s*바퀴|full\s*circle)', lower):
            degrees = 360
            return Task("spin_in_place", params={"degrees": degrees})
        if re.search(r'(180|반\s*바퀴|half\s*circle|u.?turn)', lower):
            return Task("spin_in_place", params={"degrees": 180})

        # ⑤ 목적지 이동
        target = self._detect_target(lower)
        if target:
            mode = "direct" if any(k in lower for k in ["직접", "바로", "direct"]) else "track"
            avoid = any(k in lower for k in [
                "회피", "avoid", "피해", "차선 변경", "차선변경",
                "lane change", "장애물", "obstacle", "바꿔", "우회"
            ])
            return Task("navigate_to", params={
                "target":          target,
                "mode":            mode,
                "avoid_obstacles": avoid,
            })

        # ⑥ 대기 명령
        wait_match = re.search(r'(\d+)\s*초\s*(?:동안)?\s*(?:기다려|대기|wait)', lower)
        if wait_match:
            seconds = int(wait_match.group(1))
            return Task("wait", params={"seconds": seconds})

        return None  # 인식 불가

    def _detect_target(self, lower: str) -> Optional[str]:
        """목적지 키워드 탐지."""
        for target_key, keywords in TARGET_KEYWORDS.items():
            if any(k in lower for k in keywords):
                return target_key
        return None

    def _detect_degrees(self, lower: str) -> int:
        """회전 각도 탐지."""
        if re.search(r'360|한\s*바퀴|full', lower):
            return 360
        if re.search(r'180|반\s*바퀴|half|u.?turn', lower):
            return 180
        if re.search(r'90|직각|quarter', lower):
            return 90
        if re.search(r'270', lower):
            return 270
        match = re.search(r'(\d+)\s*(?:도|degree)', lower)
        if match:
            return int(match.group(1))
        return 360  # 기본값: 한 바퀴

    def _explain_parse_failure(self, command: str) -> str:
        """파싱 실패 이유 설명."""
        lines = [f"명령을 인식할 수 없습니다: '{command}'"]
        lines.append("")
        lines.append("지원하는 명령 유형:")
        lines.append("  • 제자리 회전: '제자리에서 한 바퀴 돌아줘', '360도 회전'")
        lines.append("  • 목적지 이동: '횡단보도로 가줘', '신호등까지 트랙 따라가줘'")
        lines.append("  • 장애물 회피: '...장애물 회피하면서 횡단보도로 가줘'")
        lines.append("  • 정지: '멈춰', '정지'")
        lines.append("  • 재개: '출발', '계속 가줘'")
        lines.append("  • 순차 실행: '한 바퀴 돌고 횡단보도로 가줘'")
        return "\n".join(lines)

    @staticmethod
    def describe_task(task: Task) -> str:
        """Task를 인간 친화적 문자열로 변환."""
        if task.task_type == "spin_in_place":
            deg = task.params.get("degrees", 360)
            return f"제자리에서 {deg}도 회전"
        elif task.task_type == "navigate_to":
            target = task.params.get("target", "unknown")
            name = TARGET_NAMES.get(target, target)
            mode = task.params.get("mode", "track")
            avoid = task.params.get("avoid_obstacles", False)
            mode_str = "(직접 경로)" if mode == "direct" else "(트랙 추종)"
            avoid_str = " + 장애물 회피" if avoid else ""
            return f"{name}로 이동 {mode_str}{avoid_str}"
        elif task.task_type == "stop":
            return "정지"
        elif task.task_type == "resume":
            return "재개"
        elif task.task_type == "wait":
            sec = task.params.get("seconds", 1)
            return f"{sec}초 대기"
        return TASK_TYPE_NAMES.get(task.task_type, task.task_type)

    @staticmethod
    def describe_reason(task: Task) -> str:
        """실패 이유 반환 (있을 경우)."""
        if task.reason:
            return task.reason
        # 기본 실패 이유 생성
        reasons = {
            "spin_in_place": "제자리 회전 실행 불가: Odometry 데이터가 없거나 IMU 이상",
            "navigate_to":   "목적지 이동 불가: 경로를 찾을 수 없거나 VLA 브레인이 비활성 상태",
            "stop":          "정지 명령 전달 실패",
            "resume":        "재개 명령 전달 실패",
            "wait":          "대기 실행 중 오류",
        }
        return reasons.get(task.task_type, "알 수 없는 오류")
