#!/usr/bin/env python3
"""
순수 VLA 주행 (비전 LoRA) — Qwen 비전(LoRA 비동결, fp32) → mean-pool → WP 헤드 → pure-pursuit.
vla_lora_adapter/ + vla_lora_head.pt 로드. 바퀴카운트·EMA·영어명령 포함.
"""
import os, math, json, re, threading, time
import cv2, numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand
from gazebo_msgs.srv import SetEntityState
from PIL import Image as PILImage
from peft import PeftModel
from vla_vision import load_vision, _dummy, FEAT_DIM
from train_vla_lora import Head

HERE = os.path.dirname(__file__)
ADAPTER = os.environ.get("VLA_ADAPTER", os.path.join(HERE, "vla_lora_adapter"))
HEAD_PT = os.environ.get("VLA_HEAD", os.path.join(HERE, "vla_lora_head_fast.pt"))  # 기본=공간헤드(frozen)
CAMERA_TOPIC, CONTROL_TOPIC, ODOM_TOPIC = "camera/image_raw", "topic_control_signal", "/odom"
LANE_FILE = {0: os.path.expanduser("~/track_gt_lane1_demo.json"),
             1: os.path.expanduser("~/track_gt_lane0_demo.json")}
KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
GAIN = float(os.environ.get("PP_GAIN", "13.0"))
LOOKAHEAD = float(os.environ.get("PP_LD", "0.55"))
STEER_EMA = float(os.environ.get("STEER_EMA", "0.5"))
CRUISE = int(os.environ.get("CRUISE", "70"))
CRUISE_TURN = int(os.environ.get("CRUISE_TURN", "50"))
REV_SPEED = int(os.environ.get("REV_SPEED", "40"))          # 후진 속도(음수로 발행)
REV_DIST = float(os.environ.get("REV_DIST", "2.0"))         # 후진 거리(m) 후 자동 정지
REV_MAX_SECS = float(os.environ.get("REV_MAX_SECS", "5.0")) # 후진 시간 상한(odom 미수신 안전장치)
SPEED_SLOW = float(os.environ.get("SPEED_SLOW", "0.6"))     # 서행 배율
SPEED_FAST = float(os.environ.get("SPEED_FAST", "1.3"))     # 빠르게 배율
CHANGE_SECS = float(os.environ.get("CHANGE_SECS", "12.0"))  # 차선변경 전이 지속시간(저속 2.8m 횡단)
CHANGE_LD = float(os.environ.get("CHANGE_LD", "1.8"))       # 전이 중 lookahead(더 멀리=B쪽 WP 조준→강한 변경)
NTOK = 70
# 라이다 전방 정지(전방장착 후 차 전진=스캔 268°, 차체가림 43~136°와 안겹침)
LIDAR_FWD_DEG = int(os.environ.get("LIDAR_FWD_DEG", "268"))   # 차 전진방향에 해당하는 스캔각
LIDAR_ARC = int(os.environ.get("LIDAR_ARC", "22"))            # 전방 ±아크(도)
LIDAR_STOP = float(os.environ.get("LIDAR_STOP", "2.5"))       # 이 거리 이내 전방 장애물이면 정지(m)
LIDAR_AVOID_DIST = float(os.environ.get("LIDAR_AVOID_DIST", "8.5"))  # 전방 이 거리부터 회피 시도(정지 전, 비킬 공간 확보)
LIDAR_SIDE_CLEAR = float(os.environ.get("LIDAR_SIDE_CLEAR", "4.0"))  # 다른차선 앞이 이만큼 트이면 회피변경
LIDAR_AVOID_CD = float(os.environ.get("LIDAR_AVOID_CD", "6.0"))      # 회피 차선변경 쿨다운(초)
AVOID_HOLD = float(os.environ.get("AVOID_HOLD", "12.0"))            # 비켜난 차선 복귀금지 시간(플립플롭 방지)
AVOID_SIDE_OFF = float(os.environ.get("AVOID_SIDE_OFF", "32"))      # 옆차선 체크: 전방에서 이만큼 오프셋(도). 좌=268-, 우=268+
AVOID_SIDE_ARC = float(os.environ.get("AVOID_SIDE_ARC", "14"))      # 옆차선 체크 콘 ±아크(도)
AVOID_SPEED = int(os.environ.get("AVOID_SPEED", "45"))              # 회피 차선변경 시 속도(완만)
AVOID_STEER_CAP = int(os.environ.get("AVOID_STEER_CAP", "6"))       # 회피 시 조향 상한(오버슈트 방지)
# 목적지 정지용 랜드마크(odom 월드좌표). 주행 중 그 차선에서 가장 가까운 점에 도달하면 정지.
LANDMARKS = {
    "출발점":   (("출발", "스타트", "start"),                          (-2.55, -22.71)),
    "횡단보도": (("횡단보도", "건널목", "crosswalk", "신호등", "traffic"), (-5.63,  17.90)),
    "중간":     (("중간", "가운데", "중앙", "mid", "middle"),           (-2.55,  -2.00)),
    "주차구역": (("주차", "parking"),                                  (-1.21, -15.77)),
}


class VLALoraNode(Node):
    def __init__(self):
        super().__init__("vla_lora_drive_node")
        # 시작 즉시 정지 보장: 모델 로딩(~6초) 동안 gazebo ackermann이 이전 setpoint(예: 속도70)를
        # 유지해 차가 제멋대로 주행하는 것을 차단. 첫 명령(active=True) 전까진 계속 (0,0)로 대기.
        _ctrl_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                               durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, _ctrl_qos)
        _stop = MotionCommand(); _stop.steering = 0; _stop.left_speed = 0; _stop.right_speed = 0
        for _ in range(6):                      # 디스커버리 유실 방지 위해 간격두고 반복(~0.5초)
            self.pub.publish(_stop); time.sleep(0.08)
        self.get_logger().info("Qwen 비전 + LoRA 어댑터 로딩...")
        vis, self.proc = load_vision()
        vis = vis.float()
        # 기본=베이스 비전(LoRA 없음)+공간헤드(head_fast). VLA_USE_ADAPTER=1 이면 LoRA 어댑터 사용.
        if os.environ.get("VLA_USE_ADAPTER") == "1":
            self.vis = PeftModel.from_pretrained(vis, ADAPTER).eval(); self.get_logger().info("LoRA 어댑터 모드")
        else:
            self.vis = vis.eval(); self.get_logger().info("베이스 비전+공간헤드(기본) 모드")
        ck = torch.load(HEAD_PT, map_location="cuda:0")
        self.head = Head(ck["nout"]).float().to("cuda:0"); self.head.load_state_dict(ck["state_dict"]); self.head.eval()
        self.wp_n = ck["wp_n"]; self.wp_scale = ck["wp_scale"]
        self.bridge = None; self.lane = 0; self.active = False
        self.keep_lane = 0          # 랩카운트/중심선용 목적차선(0/1). lane은 FiLM조건(0/1유지,2/3변경)
        self.change_until = 0.0     # 이 시각까지 전이(lane 2/3) 유지 후 keep복귀
        self.paused = False         # 일시정지(빨간불 등) — 랩 상태 유지하며 0속도
        self.cur_p = None           # 현재 위치(x,y)
        self.cur_yaw = None         # 현재 yaw(odom) — 기하 접근 제어용
        self.approach_lane = None   # 기하 접근 목표차선(0/1): 멀면 GT경로로 데려옴, 도달하면 None→비전 keep
        self.ddir = -1              # 주행방향 기본=정방향(CCW,-index). +1=역방향(CW,+index). 아커만 U턴불가→방향전환은 recover로 스냅
        self.stop_at_idx = None     # 목적지 정지: keep_lane의 이 인덱스 도달 시 정지
        self.stop_name = None; self.stop_traveled = 0.0; self.prev_idx = None
        self.lidar_block = False; self.lidar_dist = 99.0   # 전방 라이다 장애물 감지(정지)
        self.direct_target = None   # (x,y) 직진모드: 차선 무시하고 이 점으로 기하 직선주행
        self.last_scan = None; self._last_avoid = 0.0; self._avoiding = False  # 라이다 회피용
        self._avoid_from = None; self._avoid_from_t = 0.0  # 방금 비켜난 차선(되돌아가기 금지)
        self.steering = 0; self.speed = 0; self.steer_f = 0.0
        self.reversing = False; self.rev_from = None    # 후진 모드 + 시작위치(거리측정)
        self.speed_scale = 1.0                          # 속도 조절 배율(서행/빠르게)
        self.inferring = False; self.latest = None
        self.lock = threading.Lock()
        self.cl = {k: [(float(a), float(b)) for a, b in json.load(open(p))["centerline_world"]]
                   for k, p in LANE_FILE.items()}
        self.target_laps = 0; self.laps_done = 0; self.start_idx = None; self.visited = set(); self.fps = 0.0; self._last = None
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, CAMERA_TOPIC, self._img, be)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom, be)
        self.create_subscription(LaserScan, "scan", self._scan, be)   # 전방 라이다 정지
        self.create_subscription(String, "vla/command", self._cmd, qos)
        # self.pub 는 시작 즉시 정지용으로 위에서 이미 생성됨(중복 생성 제거)
        # 현재 주행차선(1/2)을 발행 → brain이 구독해 회피 시 올바른 반대차선 판단(불일치 방지)
        self.lane_pub = self.create_publisher(String, "vla/cur_lane", qos)
        self._pub_lane = None
        self.reset_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")  # 복구(텔레포트)용
        self.create_timer(0.05, self._pub)
        self.get_logger().info(f"VLA-LoRA drive ready. GAIN={GAIN} Ld={LOOKAHEAD} CRUISE={CRUISE}")

    @torch.inference_mode()
    def _feat(self, bgr):
        pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(text=[_dummy(self.proc)], images=[pil], return_tensors="pt").to("cuda:0")
        out = self.vis(inp["pixel_values"].float(), grid_thw=inp["image_grid_thw"])
        feat = out[0] if isinstance(out, (list, tuple)) else out
        return feat.float().view(1, NTOK, FEAT_DIM)   # (1,70,2048) 공간보존(학습과 동일)

    def _img(self, msg):
        self.latest = msg
        if not self.inferring:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with self.lock:
            if self.inferring:
                return
            self.inferring = True
        try:
            # 직진 모드: 차선 무시하고 목표점으로 기하 직선주행(도달 시 정지). 비전보다 우선.
            if (self.active and self.direct_target is not None
                    and self.cur_p is not None and self.cur_yaw is not None):
                tx, ty = self.direct_target
                if math.dist(self.cur_p, (tx, ty)) < 1.5:
                    with self.lock:
                        self.active = False; self.direct_target = None
                    self.get_logger().info("[lora] 🎯 직진 목표 도달 → 정지")
                    return
                car_fwd = self.cur_yaw          # p3d odom: cur_yaw=실제 전진방향(오프셋 없음)
                desired = math.atan2(ty - self.cur_p[1], tx - self.cur_p[0])
                he = math.atan2(math.sin(desired - car_fwd), math.cos(desired - car_fwd))
                self.steer_f = STEER_EMA*(-he*GAIN) + (1-STEER_EMA)*self.steer_f
                st = int(max(-7, min(7, round(self.steer_f))))
                with self.lock:
                    self.steering = st; self.speed = CRUISE_TURN
                return
            # 드리프트 안전망: 비전 추종 중 목표차선서 1.5m 넘게 벗어나면(커브 컷 등) 기하접근으로 복귀
            if (self.active and self.approach_lane is None and self.cur_p is not None
                    and min(math.dist(self.cur_p, c) for c in self.cl[self.keep_lane]) > 1.5):
                self.approach_lane = self.keep_lane
                self.get_logger().info("[lora] 🎯 차선이탈 감지 → 기하접근 복귀")
            # 기하 접근 모드: 목표차선서 멀면 GT경로로 데려옴(비전 대신, 어디서든 복귀/변경)
            if self.approach_lane is not None and self.cur_p is not None and self.cur_yaw is not None:
                he = self._approach_steer()
                if he is not None:
                    cap, sp = (AVOID_STEER_CAP, AVOID_SPEED) if self._avoiding else (7, CRUISE_TURN)
                    self.steer_f = STEER_EMA*(-he*GAIN) + (1-STEER_EMA)*self.steer_f
                    st = int(max(-cap, min(cap, round(self.steer_f))))
                    with self.lock:
                        self.steering = st; self.speed = sp
                    return
                # he is None = 목표차선 도달 → approach_lane 해제됨, 아래 비전 keep로 진행
            # 비전 주행(차선 추종)
            if self.bridge is None:
                from cv_bridge import CvBridge; self.bridge = CvBridge()
            bgr = self.bridge.imgmsg_to_cv2(self.latest, "bgr8")
            t0 = time.time()
            feat = self._feat(bgr)
            lane_t = torch.tensor([self.lane], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                out = self.head(feat, lane_t)[0].cpu().numpy() * self.wp_scale
            self.fps = 0.9*self.fps + 0.1*(1.0/max(1e-3, time.time()-t0))
            pts = [(out[2*k], out[2*k+1]) for k in range(self.wp_n)]
            ex, ey = next(((a, b) for a, b in pts if a >= LOOKAHEAD), pts[-1])
            he = math.atan2(ey, max(0.3, ex))
            self.steer_f = STEER_EMA*(-he*GAIN) + (1-STEER_EMA)*self.steer_f
            st = int(max(-7, min(7, round(self.steer_f))))
            with self.lock:
                self.steering = st; self.speed = CRUISE if abs(st) < 3 else CRUISE_TURN
        finally:
            self.inferring = False

    def _recover(self, ddir=None, force_lane=None):
        """가장 가까운(또는 지정) 차선의 가장 가까운 점에 지정 방향으로 텔레포트.
        아커만은 조향으로 U턴 불가 → 길잃음/역주행/방향전환을 이 스냅으로 처리.
        ddir: +1=역방향(중앙선 +index 진행), -1=정방향(-index). 기본은 현재 self.ddir."""
        if ddir is None:
            ddir = self.ddir
        with self.lock:
            self.active = False; self.paused = False; self.target_laps = 0
            self._last = None; self.approach_lane = None; self.direct_target = None
        if self.cur_p is None:
            self.get_logger().warn("[lora] 복구 실패: 위치 미수신"); return False
        px, py = self.cur_p
        if force_lane is not None:
            cl = self.cl[force_lane]; N = len(cl); ln = force_lane
            i = min(range(N), key=lambda k: (cl[k][0]-px)**2 + (cl[k][1]-py)**2)
        else:
            best = None
            for l, c in self.cl.items():
                M = len(c); j = min(range(M), key=lambda k: (c[k][0]-px)**2 + (c[k][1]-py)**2)
                d = math.dist((px, py), c[j])
                if best is None or d < best[0]:
                    best = (d, l, j, c, M)
            _, ln, i, cl, N = best
        x0, y0 = cl[i]; nx, ny = cl[(i + ddir) % N]      # 진행방향 접선(ddir로 +/-index)
        eyaw = math.atan2(ny-y0, nx-x0) + math.pi/2
        req = SetEntityState.Request(); req.state.name = "ego_vehicle"
        req.state.pose.position.x = x0; req.state.pose.position.y = y0; req.state.pose.position.z = 0.05
        req.state.pose.orientation.z = math.sin(eyaw/2); req.state.pose.orientation.w = math.cos(eyaw/2)
        req.state.reference_frame = "world"
        self.reset_cli.call_async(req)
        with self.lock:
            self.keep_lane = ln; self.lane = ln; self.ddir = ddir
            self.start_idx = None; self.visited = set()
        self.get_logger().info(f"[lora] 🛟 복구 → {ln+1}차선 {'역방향' if ddir>0 else '정방향'} 스냅(정지)")
        return True

    def _approach_steer(self):
        """목표차선(GT)까지 기하 유도. 헤딩오차 기반이라 역방향도 돌려세움.
        목표차선 0.7m 이내 + 진행방향 정렬(<0.6rad)되면 approach_lane 해제→비전 인계."""
        cl = self.cl[self.approach_lane]; N = len(cl)
        px, py = self.cur_p; car_fwd = self.cur_yaw   # p3d odom: cur_yaw=실제 전진방향(오프셋 없음)
        i = min(range(N), key=lambda k: (cl[k][0]-px)**2 + (cl[k][1]-py)**2)
        # 차선 진행방향(접선) — 주행방향 ddir 반영(+1=+index, -1=-index)
        a, b = cl[i], cl[(i + self.ddir*3) % N]
        lane_fwd = math.atan2(b[1]-a[1], b[0]-a[0])
        align = abs(math.atan2(math.sin(lane_fwd-car_fwd), math.cos(lane_fwd-car_fwd)))  # 0=정렬, π=역방향
        if math.dist((px, py), cl[i]) < 0.7 and align < 0.6:
            self.get_logger().info(f"[lora] ✅ {self.approach_lane+1}차선 도달·정렬 → 비전 추종")
            self.approach_lane = None
            self.start_idx = None; self.visited = set()
            return None
        # 목표점: 진행방향 앞. 회피 차선변경은 더 멀리 잡아 완만한 S자(오버슈트 방지)
        tgt = cl[(i + self.ddir*(16 if self._avoiding else 6)) % N]
        desired = math.atan2(tgt[1]-py, tgt[0]-px)            # 차→목표점 월드 방위
        he = math.atan2(math.sin(desired-car_fwd), math.cos(desired-car_fwd))  # 헤딩오차(-π..π)
        return he

    def _parse_laps(self, t):
        if any(k in t for k in ("계속", "무한", "forever", "endless")):
            return 0
        m = re.search(r"(\d+)\s*바퀴", t) or re.search(r"(\d+)\s*lap", t)
        if m:
            return int(m.group(1))
        for k, v in KOR_NUM.items():
            if k+"바퀴" in t or k+" 바퀴" in t:
                return v
        return 1

    def _cmd(self, msg):
        t = msg.data.lower()
        # 속도 조절(지속 배율) — 다른 주행 명령과 함께 와도 배율만 갱신하고 계속 진행(return 안 함).
        if any(k in t for k in ("천천히", "서행", "슬로우", "slow")):
            self.speed_scale = SPEED_SLOW; self.get_logger().info(f"[lora] 🐢 서행 ({SPEED_SLOW}x)")
        elif any(k in t for k in ("빨리", "빠르게", "전속", "fast")):
            self.speed_scale = SPEED_FAST; self.get_logger().info(f"[lora] 🐇 빠르게 ({SPEED_FAST}x)")
        elif any(k in t for k in ("보통 속도", "기본 속도", "원래 속도", "normal speed")):
            self.speed_scale = 1.0; self.get_logger().info("[lora] 속도 기본(1.0x)")
        # 후진 — 짧게 뒤로 후 자동 정지. '역방향'(트랙 방향전환)과 구분.
        if any(k in t for k in ("후진", "뒤로", "back up", "backward")) and \
                not any(k in t for k in ("역방향", "반대로", "거꾸로")):
            with self.lock:
                self.reversing = True; self.rev_from = self.cur_p; self.rev_start = time.time()
                self.active = False; self.paused = False; self.target_laps = 0
                self.approach_lane = None; self.direct_target = None
                self.stop_at_idx = None; self.stop_name = None; self._last = None
            self.get_logger().info(f"[lora] ◀ 후진 시작 (뒤로 {REV_DIST}m 후 정지)"); return
        # 일시정지/재개 — 랩 진행상태 보존(빨간불 등 자율 정지용). 사용자 '멈춰'와 구분.
        if any(k in t for k in ("일시정지", "pause", "잠깐", "hold")):
            with self.lock:
                self.paused = True
            self.get_logger().info("[lora] ⏸ 일시정지(랩 상태 유지)"); return
        if any(k in t for k in ("재개", "resume", "출발", "go on")):
            with self.lock:
                self.paused = False
            self.get_logger().info("[lora] ▶ 재개"); return
        # 직진 모드: "<랜드마크>로 직진/차선무시" — 차선 무시하고 그 좌표로 기하 직선주행.
        is_direct = any(k in t for k in ("직진", "차선 무시", "차선무시", "가로질러", "straight to", "곧장"))
        lm_d = next(((nm, xy) for nm, (keys, xy) in LANDMARKS.items() if any(k in t for k in keys)), None)
        if is_direct and lm_d is not None:
            nm, (lx, ly) = lm_d
            with self.lock:
                self.direct_target = (lx, ly); self.active = True; self.target_laps = 0; self.paused = False
                self.approach_lane = None; self.stop_at_idx = None; self.stop_name = None; self._last = None
            self.get_logger().info(f"[lora] ▶ {nm}로 직진(차선무시) → 도달 시 정지 ({lx:.1f},{ly:.1f})")
            return
        # 목적지 정지: "<랜드마크>에서 정지/멈춰" — 그 차선 따라가서 지점 도달 시 정지. 즉시정지보다 먼저 검사.
        has_stop = any(k in t for k in ("정지", "멈춰", "stop", "세워"))
        lm = next(((nm, xy) for nm, (keys, xy) in LANDMARKS.items() if any(k in t for k in keys)), None)
        if lm is not None and has_stop:
            nm, (lx, ly) = lm
            d2 = any(k in t for k in ("2차선", "이차선", "outer", "lane2", "lane 2", "second"))
            d1 = any(k in t for k in ("1차선", "일차선", "inner", "lane1", "lane 1", "first"))
            dest_lane = 1 if d2 else (0 if d1 else self.keep_lane)
            cl = self.cl[dest_lane]; N = len(cl)
            sidx = min(range(N), key=lambda k: (cl[k][0]-lx)**2 + (cl[k][1]-ly)**2)
            need_appr = self.cur_p is not None and min(math.dist(self.cur_p, c) for c in cl) > 0.8
            with self.lock:
                self.keep_lane = dest_lane; self.lane = dest_lane
                self.stop_at_idx = sidx; self.stop_name = nm; self.stop_traveled = 0.0; self.prev_idx = None
                self.approach_lane = dest_lane if need_appr else None
                self.active = True; self.target_laps = 0; self.paused = False
                self.laps_done = 0; self.start_idx = None; self.visited = set(); self._last = None
            self.get_logger().info(f"[lora] ▶ {dest_lane+1}차선 주행 → {nm} 도착 시 정지 (idx{sidx})")
            return
        if any(k in t for k in ("멈춰", "정지", "stop")):
            with self.lock:
                self.active = False; self.target_laps = 0; self._last = None; self.paused = False
                self.stop_at_idx = None; self.stop_name = None; self.direct_target = None
                self.reversing = False      # 후진 중이면 취소
            self.get_logger().info("[lora] 정지"); return
        # 복구: 길잃음/역방향 시 가장 가까운 차선에 정방향으로 스냅(아커만은 U턴 불가 → 텔레포트)
        if any(k in t for k in ("복구", "리셋", "reset", "제자리")):
            self._recover(); return
        to2 = any(k in t for k in ("2차선", "이차선", "outer", "lane2", "lane 2", "lane two", "second"))
        to1 = any(k in t for k in ("1차선", "일차선", "inner", "lane1", "lane 1", "lane one", "first"))
        # 차선 명령(주행/변경 통합): 어디 있든 그 차선으로 가서(멀면 GT 기하접근) 비전 추종.
        # 방향키워드(정방향/역방향)로 주행방향 지정 — 바뀌면 아커만 U턴불가라 스냅 후 추종.
        if to1 or to2:
            dest = 1 if to2 else 0       # keep_lane 인덱스(0=1차선 안, 1=2차선 밖)
            laps = self._parse_laps(t)
            want_fwd = any(k in t for k in ("정방향", "forward", "앞으로"))
            want_rev = any(k in t for k in ("역방향", "반대로", "거꾸로", "reverse"))
            new_ddir = -1 if want_fwd else (1 if want_rev else self.ddir)
            dname = "역방향" if new_ddir > 0 else "정방향"
            # 방향 전환: 목적차선·새 방향으로 텔레포트 스냅 후 주행(조향으론 못 돌림)
            if new_ddir != self.ddir:
                if not self._recover(ddir=new_ddir, force_lane=dest):
                    return
                with self.lock:
                    self.keep_lane = dest; self.lane = dest
                    self.active = True; self.target_laps = laps; self.approach_lane = None
                    self.laps_done = 0; self.start_idx = None; self.visited = set()
                    self._last = ("go", dest, laps, new_ddir); self.direct_target = None
                self.get_logger().info(f"[lora] ▶ {dest+1}차선 {dname} ({'무한' if laps==0 else str(laps)+'바퀴'}) — 방향전환 스냅")
                return
            # 같은 방향: 어디 있든 그 차선으로(멀면 기하접근) 비전 추종
            need_approach = False
            if self.cur_p is not None:
                cl = self.cl[dest]; N = len(cl); px, py = self.cur_p
                i = min(range(N), key=lambda k: (cl[k][0]-px)**2 + (cl[k][1]-py)**2)
                need_approach = math.dist((px, py), cl[i]) > 0.8    # 목표차선서 멀면 기하접근
            key = ("go", dest, laps, self.ddir)
            if self.active and getattr(self, "_last", None) == key and self.approach_lane in (None, dest) and not need_approach:
                return
            with self.lock:
                self._last = key
                self.keep_lane = dest; self.lane = dest      # 비전 keep FiLM(0/1)
                self.approach_lane = dest if need_approach else None
                self.active = True; self.target_laps = laps
                self.laps_done = 0; self.start_idx = None; self.visited = set(); self.direct_target = None
            mode = "🎯 기하접근→" if need_approach else "▶ "
            self.get_logger().info(f"[lora] {mode}{dest+1}차선 {dname} ({'무한' if laps==0 else str(laps)+'바퀴'})")
            return

    def _lidar_clear(self, center_deg, arc_deg):
        """스캔 center_deg±arc_deg 아크의 최소거리(트임 정도). 스캔 없으면 99."""
        m = self.last_scan
        if m is None or len(m.ranges) == 0:
            return 99.0
        n = len(m.ranges)
        c = int((math.radians(center_deg % 360) - m.angle_min) / m.angle_increment)
        w = max(1, int(math.radians(arc_deg) / m.angle_increment))
        mn = 99.0
        for k in range(-w, w + 1):
            r = m.ranges[(c + k) % n]
            if m.range_min < r < m.range_max and r < mn:
                mn = r
        return mn

    def _scan(self, msg):
        """전방 아크(차 전진=스캔 LIDAR_FWD_DEG) 최소거리 → LIDAR_STOP 이내면 정지.
        막히면 다른차선 앞을 라이다로 확인해 트였으면 회피변경, 양쪽 막히면 정지."""
        self.last_scan = msg
        mn = self._lidar_clear(LIDAR_FWD_DEG, LIDAR_ARC)
        self.lidar_dist = mn
        if mn < LIDAR_AVOID_DIST:        # 정지 전, 멀리서 미리 회피 시도(비킬 공간 확보)
            self._maybe_avoid()
        elif self._avoiding:             # 전방 다시 트임 → 회피 완료(다음 장애물 대비 리셋)
            self._avoiding = False
        block = mn < LIDAR_STOP
        if block and not self.lidar_block:
            self.get_logger().info(f"[lora] 🛑 전방 라이다 {mn:.1f}m → 정지")
        elif not block and self.lidar_block:
            self.get_logger().info(f"[lora] ✅ 전방 트임({mn:.1f}m) → 재개")
        self.lidar_block = block

    def _maybe_avoid(self):
        """전방 막힘 시 다른차선 앞쪽 라이다 클리어런스 확인 → 트였으면 그 차선으로 회피변경."""
        now = time.time()
        # 직진(차선무시)모드만 회피 제외. 목적지정지 중엔 회피하고 목표인덱스를 새 차선으로 재계산.
        if not (self.active and not self.paused and self.direct_target is None and self.lane in (0, 1)):
            return
        # 비킨 뒤 '지나가는 중'(막히지 않음)이면 재변경 금지(플립플롭 방지). 단 비킨 직후 또 막혀 '갇히면'
        # 쿨다운 후 재평가 허용(다른 차선 트였으면 다시 회피, 둘 다 막히면 정지유지).
        if (self._avoiding and not self.lidar_block) or now - self._last_avoid < LIDAR_AVOID_CD \
                or self.cur_p is None or self.cur_yaw is None \
                or self.approach_lane is not None:   # 차선변경 진행 중이면 재결정 금지(플립플롭 차단)
            return
        other = 1 - self.keep_lane
        # 방금 비켜난 차선으로 되돌아가기 금지(플립플롭 방지): 그 차선이 지금 트여보여도
        # 거긴 방금 피한 장애물이 있던 곳 → 복귀하면 다시 박음. 그동안 막히면 그냥 정지(전방정지가 처리).
        if other == self._avoid_from and now - self._avoid_from_t < AVOID_HOLD:
            return
        cl = self.cl[other]; N = len(cl)
        # 차 전진방위 = keep_lane 접선(ddir 방향). odom yaw 규약(±90°오프셋)에 의존 안 함 → 신뢰성.
        keep = self.cl[self.keep_lane]; Nk = len(keep)
        j = min(range(Nk), key=lambda k: (keep[k][0]-self.cur_p[0])**2 + (keep[k][1]-self.cur_p[1])**2)
        bx, by = keep[j]; fx, fy = keep[(j + 3*self.ddir) % Nk]
        fwd = math.atan2(fy-by, fx-bx)
        # 다른차선이 차의 좌/우? (월드 외적). 라이다 전방=scan268, 좌=scan268-OFF, 우=268+OFF (실측 보정).
        oi = min(range(N), key=lambda k: (cl[k][0]-self.cur_p[0])**2 + (cl[k][1]-self.cur_p[1])**2)
        ox, oy = cl[oi]
        cross = math.cos(fwd)*(oy-self.cur_p[1]) - math.sin(fwd)*(ox-self.cur_p[0])
        side = LIDAR_FWD_DEG - AVOID_SIDE_OFF if cross > 0 else LIDAR_FWD_DEG + AVOID_SIDE_OFF
        clr = self._lidar_clear(side, AVOID_SIDE_ARC)        # 옆차선 쪽(전방-오프셋 콘) 클리어런스
        if clr > LIDAR_SIDE_CLEAR:
            self._last_avoid = now; self._avoiding = True
            self._avoid_from = self.keep_lane; self._avoid_from_t = now  # 떠나는 차선 = 복귀금지
            with self.lock:
                if self.stop_at_idx is not None:            # 목적지정지 중이면 목표점을 새 차선 인덱스로 재계산
                    old_pt = self.cl[self.keep_lane][self.stop_at_idx]
                    self.stop_at_idx = min(range(N), key=lambda k: (cl[k][0]-old_pt[0])**2 + (cl[k][1]-old_pt[1])**2)
                self.keep_lane = other; self.lane = other
                self.approach_lane = other                  # 기하접근으로 다른차선 이동
                self.start_idx = None; self.visited = set()
            self.get_logger().info(f"[lora] 🛑→↪ 전방막힘 → {other+1}차선({'좌' if cross>0 else '우'}) 트임({clr:.1f}m) 회피변경")
        else:
            self.get_logger().info(f"[lora] 🛑 양쪽 막힘({other+1}차선도 {clr:.1f}m, scan{side:.0f}°) → 정지유지")

    def _odom(self, msg):
        p = msg.pose.pose.position
        self.cur_p = (p.x, p.y)          # 현재 위치 항상 추적
        q = msg.pose.pose.orientation    # yaw 추출(기하 접근 제어용)
        self.cur_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        with self.lock:
            # 목적지 정지: 진행거리 누적 후 목표 인덱스 근접 시 정지(랩수 무관)
            if self.active and self.stop_at_idx is not None:
                cl = self.cl[self.keep_lane]; N = len(cl)
                i = min(range(N), key=lambda k: (cl[k][0]-p.x)**2 + (cl[k][1]-p.y)**2)
                if self.prev_idx is not None:
                    dd = i - self.prev_idx
                    if dd > N/2: dd -= N
                    elif dd < -N/2: dd += N
                    self.stop_traveled += abs(dd)
                self.prev_idx = i
                cd = abs(i - self.stop_at_idx); cd = min(cd, N-cd)
                if self.stop_traveled >= 8 and cd <= 5:
                    nm = self.stop_name
                    self.active = False; self.stop_at_idx = None; self.stop_name = None
                    self.get_logger().info(f"[lora] 🎯 {nm} 도착 → 정지")
            if not self.active or self.target_laps == 0:
                return
            cl = self.cl[self.keep_lane]; N = len(cl)   # 전이중(lane2/3)에도 목적차선 기준
            i = min(range(N), key=lambda k: (cl[k][0]-p.x)**2 + (cl[k][1]-p.y)**2)
            if self.start_idx is None:
                self.start_idx = i
            self.visited.add(i // 20)
            d = abs(i-self.start_idx); d = min(d, N-d)
            if len(self.visited) >= 34 and d < 12:
                self.laps_done += 1
                self.get_logger().info(f"🔁 {self.laps_done}/{self.target_laps}바퀴 (~{self.fps:.0f}FPS)")
                self.visited = set(); self.start_idx = i
                if self.laps_done >= self.target_laps:
                    self.active = False; self.get_logger().info(f"✅ {self.target_laps}바퀴 완료 — 정지")

    def _pub(self):
        # 후진 모드: 라이다(전방) 무시하고 뒤로. REV_DIST 만큼 물러나면 자동 정지.
        if self.reversing:
            with self.lock:
                moved = (self.cur_p is not None and self.rev_from is not None
                         and math.dist(self.cur_p, self.rev_from) >= REV_DIST)
                timeout = (time.time() - getattr(self, "rev_start", 0)) >= REV_MAX_SECS
                done = moved or timeout
                if done:
                    self.reversing = False
                st, sp = (0, 0) if done else (0, -REV_SPEED)
            if done:
                self.get_logger().info("[lora] ◀ 후진 완료 → 정지")
            m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
            self.pub.publish(m); return
        with self.lock:
            # 전이(lane 2/3): 목적차선 도달 즉시 복귀(오버슈트 방지), 아니면 시간초과 시 복귀
            if self.lane in (2, 3):
                arrived = False
                if self.cur_p is not None:
                    dt = min(math.dist(self.cur_p, c) for c in self.cl[self.keep_lane])
                    elapsed = time.time() - (self.change_until - CHANGE_SECS)
                    arrived = (dt < 0.6 and elapsed > 2.0)   # 목적차선 0.6m 이내 + 최소 2초 경과
                if arrived or time.time() > self.change_until:
                    self.lane = self.keep_lane
                    self.start_idx = None; self.visited = set()
                    self.get_logger().info(f"[lora] ✅ 변경완료 → {self.keep_lane+1}차선 유지")
            st, sp = (self.steering, self.speed) if (self.active and not self.paused) else (0, 0)
            if self.lidar_block:          # 전방 라이다 장애물 → 비상정지(조향유지, 속도0)
                # 단, 회피 차선변경 중(목표차선 트임 확인됨)엔 저속 전진 허용 → 옆으로 빠져 통과(교착 방지)
                sp = AVOID_SPEED if (self._avoiding and self.approach_lane is not None) else 0
            sp = int(sp * self.speed_scale)   # 속도 조절 배율(서행/빠르게) 적용
            kl = self.keep_lane
        m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
        self.pub.publish(m)
        if kl != self._pub_lane:                 # 차선 바뀔 때만 발행(brain 동기화)
            self._pub_lane = kl
            self.lane_pub.publish(String(data=str(kl + 1)))


def main():
    rclpy.init(); node = VLALoraNode()
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
