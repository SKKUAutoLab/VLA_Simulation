#!/usr/bin/env python3
"""
순수 VLA(LoRA) 정적 지각 검증 — 차선 위 여러 지점에 차를 텔레포트(정지)하고
그 깨끗한 카메라 화면에서 VLA가 곡률에 맞는 조향을 뱉는지 본다.
정적(분포 내 유사)서 잘 뱉으면 → 지각은 됨, 실패는 closed-loop 전이 문제.
정적서도 st=0만 뱉으면 → 지각/매핑 자체가 약함.
"""
import os, math, json, time
import cv2, numpy as np, re
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "adapter"); BASE = "Qwen/Qwen3-VL-2B-Instruct"
INPUT_W, INPUT_H = 320, 240; PIXELS = INPUT_W*INPUT_H
PROMPT = ('Drive a car using the front camera. Follow the gray road, stay between the '
          'lane lines. Reply ONE line only: D <st> <sp>  st=-7..7 (NEGATIVE=left, '
          'POSITIVE=right), sp=0..100; S=stop. Ex: D -3 40')
CL = os.path.expanduser("~/track_gt_lane1_demo.json")
YAW_OFFSET = math.pi/2


def parse_st(resp):
    s = resp.strip().lstrip("`*: ")
    if s[:1].upper() == "D": s = s[1:]
    m = re.findall(r'-?\d+', s)
    return int(max(-7, min(7, int(m[0])))) if m else None


def main():
    cl = [(float(a), float(b)) for a, b in json.load(open(CL))["centerline_world"]]
    N = len(cl)
    # 곡률(다음 ~20idx 헤딩변화)로 기대 조향 부호
    def expected(i):
        a = cl[i]; b = cl[(i+10) % N]; c = cl[(i+20) % N]
        h1 = math.atan2(b[1]-a[1], b[0]-a[0]); h2 = math.atan2(c[1]-b[1], c[0]-b[0])
        d = (h2-h1+math.pi) % (2*math.pi)-math.pi   # +좌회전,-우회전(월드)
        return d
    print("VLA(2B+LoRA) 로딩...")
    base = Qwen3VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                           device_map="cuda:0", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=PIXELS, max_pixels=PIXELS)

    rclpy.init(); n = Node("vstat"); br = CvBridge(); img = {"v": None}
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: img.__setitem__("v", br.imgmsg_to_cv2(m, "bgr8")),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)

    def infer():
        bgr = cv2.resize(img["v"], (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], images=[pil], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False, use_cache=True)
        return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

    idxs = list(range(0, N, N//16))
    print(f"\n{'idx':>4} {'곡률(deg)':>9} {'기대':>5} {'VLA출력':>10} {'VLA조향':>7}")
    rows = []
    for i in idxs:
        x0, y0 = cl[i]; nx, ny = cl[(i+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+YAW_OFFSET
        r = SetEntityState.Request(); r.state.name = "ego_vehicle"
        r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
        r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
        rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
        t0 = time.time()
        while time.time()-t0 < 0.7:
            rclpy.spin_once(n, timeout_sec=0.02)
        if img["v"] is None:
            continue
        d = math.degrees(expected(i)); exp = "좌-" if d > 5 else ("우+" if d < -5 else "직0")
        resp = infer(); st = parse_st(resp)
        rows.append((d, st))
        print(f"{i:>4} {d:>9.1f} {exp:>5} {resp!r:>10} {str(st):>7}")
    # 요약
    sts = [st for _, st in rows if st is not None]
    nz = sum(1 for s in sts if s != 0)
    correct = sum(1 for d, s in rows if s is not None and ((d > 5 and s < 0) or (d < -5 and s > 0) or (abs(d) <= 5 and s == 0)))
    print(f"\n요약: 출력 {len(sts)}개 중 비-직진 {nz}개, 곡률부호 일치 {correct}/{len(rows)}")
    print("→ 정적서도 거의 st=0면: 지각/매핑 자체 약함. 다양·일치 높으면: 지각OK·closed-loop전이가 문제.")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
