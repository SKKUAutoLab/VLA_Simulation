#!/usr/bin/env python3
"""CoT 탐지 비교 — 같은 장면(파란/빨간 차 9m)에 terse(CoT끔) vs CoT(켬) 프롬프트를 걸어
제로샷 장애물 탐지가 CoT로 개선되는지 통제 비교."""
import math, json, os, re, time, rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
import torch
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

QWEN = "Qwen/Qwen3-VL-2B-Instruct"; PIXELS = 320*240
TERSE = ("You are a driving safety monitor. Look at the front camera. "
         "Reply EXACTLY one token: HAZARD=<none|redlight|obstacle>. "
         "redlight = a red traffic light ahead. obstacle = a car/object blocking the lane ahead. else none.")
COT = ("You are a driving safety monitor looking at the front camera. Think step by step:\n"
       "1) List the objects you see ahead.\n"
       "2) Is any vehicle or object blocking the driving lane ahead? (any color)\n"
       "3) Is there a red traffic light ahead?\n"
       "Then on the FINAL line output exactly: HAZARD=<none|redlight|obstacle> "
       "(obstacle if a vehicle/object blocks the lane regardless of color).")
ANCHOR = ("You are a driving safety monitor on a moving car. Look at the front camera. "
          "RULE: if ANY vehicle is visible ahead on the road in front of you — even if it faces away, "
          "looks parked, or is any color — it is an obstacle you must avoid. "
          "A red traffic light ahead means stop. "
          "Output exactly one token: HAZARD=<none|redlight|obstacle>.")

lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane)
rclpy.init(); n = Node("cotdet")
setc = n.create_client(SetEntityState, "/gazebo/set_entity_state"); setc.wait_for_service(timeout_sec=10)
spc = n.create_client(SpawnEntity, "/spawn_entity"); spc.wait_for_service(timeout_sec=10)
delc = n.create_client(DeleteEntity, "/delete_entity"); delc.wait_for_service(timeout_sec=10)
from cv_bridge import CvBridge; br = CvBridge(); cur = {"img": None}
n.create_subscription(Image, "camera/image_raw", lambda m: cur.__setitem__("img", br.imgmsg_to_cv2(m, "bgr8")), 10)

print("Qwen3-VL 로딩...")
proc = Qwen3VLProcessor.from_pretrained(QWEN, min_pixels=PIXELS, max_pixels=PIXELS)
qwen = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()


@torch.inference_mode()
def ask(prompt, bgr, mx):
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    content = [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]
    t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[pil], return_tensors="pt").to("cuda:0")
    t0 = time.time()
    out = qwen.generate(**inp, max_new_tokens=mx, do_sample=False)
    dt = time.time()-t0
    return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True), dt


def grab(model_name):
    i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)
    r = SetEntityState.Request(); r.state.name = "ego_vehicle"
    r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
    ey = yaw+math.pi/2; r.state.pose.orientation.z = math.sin(ey/2); r.state.pose.orientation.w = math.cos(ey/2); r.state.reference_frame = "world"
    rclpy.spin_until_future_complete(n, setc.call_async(r), timeout_sec=2)
    d = 0.0; j = i0
    while d < 9.0:
        a = lane[j % N]; b = lane[(j+1) % N]; d += math.dist(a, b); j += 1
    ox, oy = lane[j % N]
    sp = SpawnEntity.Request(); sp.name = "hazard_obj"
    sp.xml = open(os.path.expanduser(f"~/.gazebo/models/{model_name}/model.sdf")).read()
    p = Pose(); p.position.x = ox; p.position.y = oy; p.position.z = 0.05
    obyaw = math.atan2(lane[(j+1) % N][1]-oy, lane[(j+1) % N][0]-ox)
    p.orientation.z = math.sin(obyaw/2); p.orientation.w = math.cos(obyaw/2); sp.initial_pose = p
    rclpy.spin_until_future_complete(n, spc.call_async(sp), timeout_sec=5)
    cur["img"] = None; t = time.time()
    while time.time()-t < 4:
        rclpy.spin_once(n, timeout_sec=0.1)
    img = cur["img"]
    dr = DeleteEntity.Request(); dr.name = "hazard_obj"
    rclpy.spin_until_future_complete(n, delc.call_async(dr), timeout_sec=3)
    return img


def haz(text):
    m = re.findall(r'HAZARD=(\w+)', text)
    return m[-1] if m else "?"


for model_name in ["hatchback_red", "hatchback_blue"]:
    img = grab(model_name)
    if img is None:
        print(f"\n### {model_name}: 카메라 프레임 없음"); continue
    cv2.imwrite(f"/tmp/cot_{model_name}.png", img)
    tt, td = ask(TERSE, img, 20)
    ct, cd = ask(COT, img, 200)
    at, ad = ask(ANCHOR, img, 20)
    print(f"\n########## {model_name} (9m 전방) ##########")
    print(f"[CoT 끔]    {td:.1f}s → HAZARD={haz(tt)}   (원문: {tt.strip()[:60]})")
    print(f"[CoT 켬]    {cd:.1f}s → HAZARD={haz(ct)}")
    print(f"   추론: {ct.strip()}")
    print(f"[앵커 강제] {ad:.1f}s → HAZARD={haz(at)}   (원문: {at.strip()[:60]})")
n.destroy_node(); rclpy.shutdown()
