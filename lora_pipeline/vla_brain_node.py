#!/usr/bin/env python3
"""
C+D: VLM 고수준 브레인 (dual-system의 slow 경로).
Qwen3-VL이 ① 자유 자연어 명령 해석(C) ② 카메라 주기적 장면추론으로 미학습객체
(빨간불/장애물) 제로샷 판단(D) → 표준 주행명령을 vla/command 로 발행.
fast 경로(WP-LoRA 주행 노드)가 실제 차선 주행/변경을 수행.

토픽: 구독 camera/image_raw, nl_command(자유명령) / 발행 vla/command
환경: SCENE_HZ(장면추론 주기, 기본 0.7Hz), NO_SCENE=1 이면 언어해석만(C).
"""
import os, re, math, threading, time
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from PIL import Image as PILImage
import cv2, numpy as np

QWEN = "Qwen/Qwen3-VL-2B-Instruct"
SCENE_PERIOD = 1.0 / float(os.environ.get("SCENE_HZ", "0.7"))
DO_SCENE = os.environ.get("NO_SCENE") != "1"
AVOID_COOLDOWN = float(os.environ.get("AVOID_COOLDOWN", "9.0"))  # 회피 차선변경 후 재명령 금지(초) — 플립플롭 방지
PIXELS = 320 * 240

LANG_SYS = ("Convert the driving instruction to ONE line: LANE=<1|2|none> ACT=<keep|change|stop|go> "
            "LAPS=<int|inf> STOP_AT=<none|start|crosswalk|mid|parking>. "
            "RULES: 즉시(아무데서나) 멈추라는 명령만 ACT=stop(예 '멈춰''그만 서''정지'). "
            "'N바퀴 돌고 멈춰/그만'은 완주 후 자동정지이므로 ACT=go LAPS=N (stop 아님). "
            "숫자 없이 '돌고 멈춰/돌다 서/go around and stop/lap and stop'이면 LAPS=1 (무한 아님, 한바퀴 후 정지). "
            "특정 지점에서 멈추라면 ACT=go 로 두고 STOP_AT 지정: '출발점/출발지'->start, "
            "'횡단보도/신호등/건널목'->crosswalk, '중간/가운데/중앙'->mid, '주차/주차구역'->parking. "
            "바꿔/변경/옆차선/change->change; 1차선->1;2차선->2; 한바퀴->1,두바퀴->2,세바퀴->3,계속/무한->inf. "
            "예 '2차선으로 바꿔'->LANE=2 ACT=change LAPS=1 STOP_AT=none | "
            "'횡단보도에서 정지'->LANE=none ACT=go LAPS=inf STOP_AT=crosswalk | "
            "'출발점에서 멈춰'->ACT=go STOP_AT=start | "
            "'go around lane2 and stop'->LANE=2 ACT=go LAPS=1 STOP_AT=none | '그만 서'->ACT=stop STOP_AT=none")
SCENE_Q = ("You are a driving safety monitor. Look at the front camera. "
           "Reply EXACTLY one token: HAZARD=<none|redlight|obstacle>. "
           "redlight = a red traffic light ahead. obstacle = a car/object blocking the lane ahead. "
           "else none.")


STOP_AT_KOR = {"start": "출발점", "crosswalk": "횡단보도", "mid": "중간", "parking": "주차구역"}


def build_cmd(lane, act, laps, cur_lane, stop_at="none"):
    if act == "stop":
        return "멈춰"
    if act == "change":
        return f"{lane if lane in (1,2) else (2 if cur_lane==1 else 1)}차선으로 변경"
    if stop_at in STOP_AT_KOR:        # 목적지 정지: 그 차선 따라가서 지점 도달 시 정지
        ln = lane if lane in (1,2) else cur_lane
        return f"{ln}차선 {STOP_AT_KOR[stop_at]}에서 정지"
    laps_s = "계속" if laps == "inf" else f"{laps}바퀴"
    return f"{lane if lane in (1,2) else cur_lane}차선 {laps_s} 돌아"


class BrainNode(Node):
    def __init__(self):
        super().__init__("vla_brain_node")
        self.get_logger().info("Qwen3-VL 브레인 로딩...")
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        self.proc = Qwen3VLProcessor.from_pretrained(QWEN, min_pixels=PIXELS, max_pixels=PIXELS)
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
        self.bridge = None; self.img = None
        self.cur_lane = 1            # 현재 주행 차선(1/2)
        self.hazard = "none"         # 마지막 위험상태(중복발행 방지)
        self._last_avoid = 0.0       # 마지막 회피 차선변경 시각(쿨다운용)
        self.lock = threading.Lock()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(String, "vla/command", qos)
        self.create_subscription(Image, "camera/image_raw", self._img, be)
        self.create_subscription(String, "nl_command", self._nl, qos)
        # drive 노드의 실제 주행차선 구독 → cur_lane 동기화(회피 시 반대차선 오판 방지)
        self.create_subscription(String, "vla/cur_lane",
                                 lambda m: setattr(self, "cur_lane", int(m.data)) if m.data in ("1", "2") else None, qos)
        if DO_SCENE:
            self.create_timer(SCENE_PERIOD, self._scene_tick)
        self.get_logger().info(f"브레인 준비. 언어=/nl_command, 장면추론={'on' if DO_SCENE else 'off'}")

    def _img(self, msg):
        if self.bridge is None:
            from cv_bridge import CvBridge; self.bridge = CvBridge()
        self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    @torch.inference_mode()
    def _ask(self, prompt, with_img):
        content = [{"type": "text", "text": prompt}]
        images = None
        if with_img and self.img is not None:
            pil = PILImage.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            content = [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]
            images = [pil]
        t = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                          tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[t], images=images, return_tensors="pt").to("cuda:0")
        out = self.qwen.generate(**inp, max_new_tokens=20, do_sample=False)
        return self.proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    def _nl(self, msg):
        # 목적지 정지(랜드마크+정지)는 2B LLM이 불안정 → 결정론적 우회: 원문을 drive 노드로 직통(네이티브 파싱).
        t = msg.data.lower()
        LM_KEYS = ("출발", "스타트", "start", "횡단보도", "건널목", "crosswalk", "신호등", "traffic",
                   "중간", "가운데", "중앙", "mid", "middle", "주차", "parking")
        if any(k in t for k in LM_KEYS) and any(k in t for k in ("정지", "멈춰", "stop", "세워")):
            self.get_logger().info(f"[언어] '{msg.data}' → (목적지정지 직통) {msg.data}")
            self.pub.publish(String(data=msg.data)); return
        # 직진(차선무시) 모드도 2B 불안정 → 랜드마크+직진키워드면 원문 직통(drive 노드 네이티브 파싱)
        if any(k in t for k in LM_KEYS) and any(k in t for k in ("직진", "차선 무시", "차선무시", "가로질러", "straight to", "곧장")):
            self.get_logger().info(f"[언어] '{msg.data}' → (직진 직통) {msg.data}")
            self.pub.publish(String(data=msg.data)); return
        # 후진·속도조절: LANG 문법(go/stop/change) 밖 동작 → 원문 직통(drive 노드 네이티브 파싱)
        if any(k in t for k in ("후진", "뒤로", "back up", "backward")) and "역방향" not in t:
            self.get_logger().info(f"[언어] '{msg.data}' → (후진 직통)")
            self.pub.publish(String(data=msg.data)); return
        if any(k in t for k in ("천천히", "서행", "빨리", "빠르게", "전속", "slow down", "faster")):
            self.get_logger().info(f"[언어] '{msg.data}' → (속도조절 직통)")
            self.pub.publish(String(data=msg.data)); return
        try:
            resp = self._ask(LANG_SYS + "\n\n명령: " + msg.data, with_img=False)
            la = re.search(r'LANE=(\w+)', resp); ac = re.search(r'ACT=(\w+)', resp); lp = re.search(r'LAPS=(\w+)', resp)
            sa = re.search(r'STOP_AT=(\w+)', resp); stop_at = sa.group(1) if sa else "none"
            lane = int(la.group(1)) if la and la.group(1).isdigit() else None
            act = ac.group(1) if ac else "go"; laps = lp.group(1) if lp else "1"
            # 결정론적 ACT/LAPS 보정(2B 흔들림 방지). 차선변경(change)은 건드리지 않음.
            tl = msg.data.lower()
            has_drive = any(k in tl for k in ("돌", "바퀴", "주행", "go", "drive", "around", "lap"))
            has_stop = any(k in tl for k in ("stop", "멈춰", "그만", "정지", "세워"))
            has_forever = any(k in tl for k in ("forever", "endless", "계속", "무한", "쭉"))
            knum = next((v for k, v in {"한":1,"두":2,"세":3,"네":4,"다섯":5}.items()
                         if k+"바퀴" in tl or k+" 바퀴" in tl), None)
            numm = re.search(r"(\d+)\s*(바퀴|lap)", tl)
            if act != "change":
                if has_drive:                       # 주행 의도
                    act = "go"
                    if has_forever: laps = "inf"
                    elif numm: laps = numm.group(1)
                    elif knum: laps = str(knum)
                    elif has_stop: laps = "1"        # 숫자없는 '돌고 멈춰' = 1바퀴 후 정지
                elif has_stop:                       # 주행없이 멈춤만 = 즉시정지
                    act = "stop"
            if lane in (1, 2) and act in ("keep", "go", "change"):
                self.cur_lane = lane if act != "change" else lane
            cmd = build_cmd(lane, act, laps, self.cur_lane, stop_at)
            self.get_logger().info(f"[언어] '{msg.data}' → {cmd}")
            self.pub.publish(String(data=cmd))
        except Exception as e:
            self.get_logger().warn(f"언어해석 실패: {e}")

    def _scene_tick(self):
        if self.img is None:
            return
        try:
            resp = self._ask(SCENE_Q, with_img=True)
            m = re.search(r'HAZARD=(\w+)', resp)
            hz = m.group(1) if m else "none"
            if hz not in ("none", "redlight", "obstacle"):
                hz = "none"
        except Exception:
            return
        if hz == self.hazard:
            return                              # 상태 변화 없으면 발행 안 함
        with self.lock:
            prev = self.hazard; self.hazard = hz
        if hz == "redlight":
            # 일시정지(랩 진행상태 보존) — 사용자 '멈춰'와 달리 초록불에 같은 랩 이어감
            self.get_logger().info("[장면] 🔴 빨간불 → 일시정지"); self.pub.publish(String(data="일시정지"))
        elif hz == "obstacle":
            # 장애물 회피는 라이다(drive 노드)가 전담 → brain은 명령 안 냄(LLM 스웨브 충돌 방지)
            self.get_logger().info("[장면] 🚧 장애물 감지(회피는 라이다가 처리)")
        else:  # none (위험 해소)
            if prev == "redlight":
                self.get_logger().info("[장면] 🟢 초록불 → 재개"); self.pub.publish(String(data="재개"))


def main():
    rclpy.init(); node = BrainNode()
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
