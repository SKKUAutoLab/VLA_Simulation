#!/usr/bin/env python3
"""
C(G2): 언어명령 해석층 — 자유 자연어 → 표준 주행명령 변환.
nl_command(String, 자유 한국어/영어) 수신 → Qwen3-VL LM 해석 →
vla/command(String, 표준)로 발행. 주행 노드(vla_lora_drive_node)는 그대로 사용.
표준 명령: "1차선/2차선 N바퀴 돌아", "1차선/2차선으로 변경", "멈춰".
사용: 이 노드 + vla_lora_drive_node 동시 실행.
  ros2 topic pub -1 /nl_command std_msgs/String "{data: '2차선으로 바꿔서 두바퀴 돌아'}"
"""
import os, re, json
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

QWEN = "Qwen/Qwen3-VL-2B-Instruct"
SYS = ("You convert a driving instruction into ONE line. Output EXACTLY:\n"
       "LANE=<1|2|none> ACT=<keep|change|stop|go> LAPS=<integer|inf>\n"
       "Rules: '멈춰/정지/stop'->ACT=stop. '바꿔/변경/change/차선변경'->ACT=change. "
       "'1차선/일차선/inner/lane one'->LANE=1. '2차선/이차선/outer/lane two'->LANE=2. "
       "'한/1바퀴'->LAPS=1,'두/2바퀴'->2,'계속/무한/forever'->LAPS=inf, 명시없으면 LAPS=1. "
       "변경(change)이면 LANE=목표차선. 예) '2차선으로 바꿔'->LANE=2 ACT=change LAPS=1")


def keyword_fallback(t):
    t = t.lower()
    if any(k in t for k in ("멈춰", "정지", "stop")):
        return "멈춰"
    lane = 2 if any(k in t for k in ("2차선", "이차선", "outer", "lane two", "lane2", "lane 2")) else \
           (1 if any(k in t for k in ("1차선", "일차선", "inner", "lane one", "lane1", "lane 1")) else 1)
    laps = "계속" if any(k in t for k in ("계속", "무한", "forever")) else None
    m = re.search(r"(\d+)\s*바퀴", t)
    if any(k in t for k in ("바꿔", "변경", "change")):
        return f"{lane}차선으로 변경"
    laps_s = "계속" if laps else (f"{m.group(1)}바퀴" if m else "한바퀴")
    return f"{lane}차선 {laps_s} 돌아"


def build_cmd(lane, act, laps):
    if act == "stop":
        return "멈춰"
    if act == "change":
        return f"{lane if lane in (1,2) else 2}차선으로 변경"
    laps_s = "계속" if laps == "inf" else f"{laps}바퀴"
    L = lane if lane in (1, 2) else 1
    return f"{L}차선 {laps_s} 돌아"


class LangNode(Node):
    def __init__(self):
        super().__init__("vla_lang_node")
        self.get_logger().info("Qwen 언어해석 로딩...")
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        self.proc = Qwen3VLProcessor.from_pretrained(QWEN)
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=1, durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.pub = self.create_publisher(String, "vla/command", qos)
        self.create_subscription(String, "nl_command", self._cb, qos)
        self.get_logger().info("언어해석 준비. /nl_command 로 자유 명령 보내세요.")

    @torch.inference_mode()
    def _interpret(self, text):
        msgs = [{"role": "user", "content": [{"type": "text", "text": SYS + "\n\n명령: " + text}]}]
        t = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[t], return_tensors="pt").to("cuda:0")
        out = self.qwen.generate(**inp, max_new_tokens=24, do_sample=False)
        return self.proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    def _cb(self, msg):
        text = msg.data
        try:
            resp = self._interpret(text)
            la = re.search(r'LANE=(\w+)', resp); ac = re.search(r'ACT=(\w+)', resp); lp = re.search(r'LAPS=(\w+)', resp)
            lane = int(la.group(1)) if la and la.group(1).isdigit() else None
            act = ac.group(1) if ac else "go"
            laps = lp.group(1) if lp else "1"
            cmd = build_cmd(lane, act, laps)
            self.get_logger().info(f"'{text}' → (raw:{resp.strip()!r}) → 표준:'{cmd}'")
        except Exception as e:
            cmd = keyword_fallback(text)
            self.get_logger().warn(f"Qwen 실패({e}) → 폴백:'{cmd}'")
        self.pub.publish(String(data=cmd))


def main():
    rclpy.init(); node = LangNode()
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
