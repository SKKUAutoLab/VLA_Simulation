#!/usr/bin/env python3
"""
lane_picker.py — 학생용 차선 마스킹 도구 (PySide6, 폴리곤 면 → 중앙선 추출)

트랙 위에서 점을 클릭하면 선으로 이어지고, 선이 닫혀 이룬 '면(폴리곤 내부)'만 반투명하게
색칠된다 = 그 차선의 범위(정답지). 다 그린 뒤 '중앙선 추출'을 누르면 그 면의 가운데선을
자동 계산하고, 학습·주행이 바로 쓰는 GT(JSON)로 저장한다.

버튼(또는 단축키):
    Lane 1 / Lane 2  차선 선택(1/2, 빨강/파랑)   중앙선 추출  면의 가운데선 계산(e)
    Load  JSON/PNG 불러오기(l)   Undo 되돌리기(u/Ctrl+Z)
    Save  중앙선+라벨마스크 저장(s)   Clear 현재 차선 지우기(c)   Reset view 뷰 리셋(r)

마우스:
    빈 곳 딸깍  점 추가       선 위(+) 딸깍  두 점 사이 삽입
    점 좌드래그  점 이동        점 우클릭      점 삭제
    빈 곳 좌드래그  화면 이동    휠  확대/축소
"""
import os, json, copy, shutil, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TRACK = os.path.join(WS, "src/simulation_pkg/models/race_track/materials/textures/track.png")
IMG_W, IMG_H = 1180.0, 884.0
N_SAVE = 720
HIT, EDGE, SNAP = 14, 9, 18
LANE_FILE = {1: os.path.expanduser("~/track_gt_lane1_demo.json"),
             2: os.path.expanduser("~/track_gt_lane0_demo.json")}
LANE_COLOR = {1: (40, 40, 220), 2: (220, 90, 40)}


def pixel_to_world(px, py):
    return -20.237 + (py / IMG_H) * 40.473, -26.915 + (px / IMG_W) * 53.83


def resample_closed(pts, n, smooth=15):
    P = np.array(pts, float); P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]); total = s[-1]
    if total < 1e-6:
        return P[:-1]
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, s, P[:, 0]); y = np.interp(t, s, P[:, 1])
    if smooth >= 3:
        k = np.ones(smooth) / smooth
        xx = np.concatenate([x[-smooth:], x, x[:smooth]]); yy = np.concatenate([y[-smooth:], y, y[:smooth]])
        x = np.convolve(xx, k, "same")[smooth:-smooth]; y = np.convolve(yy, k, "same")[smooth:-smooth]
    return np.stack([x, y], 1)


def poly_mask(pts):
    m = np.zeros((int(IMG_H), int(IMG_W)), np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m


def extract_centerline(mask):
    """칠한 밴드(면) → 진짜 가운데선. 바깥/안쪽 경계선 사이 '중점'을 이어 계산.
    (S자·길쭉한 모양에서도 정확히 가운데) 반환 [[x,y]..] 또는 None."""
    if int((mask > 0).sum()) < 300:
        return None
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return None
    hier = hier[0]; areas = [cv2.contourArea(c) for c in cnts]
    outers = [(areas[i], i) for i in range(len(cnts)) if hier[i][3] == -1]
    if not outers:
        return None
    oi = max(outers)[1]; outer = cnts[oi].reshape(-1, 2).astype(float)
    holes = [(areas[i], i) for i in range(len(cnts)) if hier[i][3] == oi]
    if not holes:
        return None                                       # 링(구멍)이 아니면 가운데선 못 뽑음
    inner = cnts[max(holes)[1]].reshape(-1, 2).astype(float)
    step = max(1, len(outer) // 720)
    O = outer[::step]                                     # 바깥 경계(순서대로)
    D = np.linalg.norm(O[:, None, :] - inner[None, :, :], axis=2)   # 각 바깥점 → 안쪽점 거리
    mids = (O + inner[D.argmin(1)]) / 2.0                 # 두 경계의 중점 = 가운데선
    dense = resample_closed(mids, 720)
    return [[int(x), int(y)] for x, y in dense]


def save_lane(lane, pts, path):
    dense = resample_closed(pts, N_SAVE)
    world = [pixel_to_world(px, py) for px, py in dense]
    out = {"meta": {"tool": "lane_picker", "lane": lane,
                    "formula": {"world_x": "-20.237+(py/884)*40.473", "world_y": "-26.915+(px/1180)*53.83"}},
           "centerline_pixels": [[int(x), int(y)] for x, y in dense],
           "centerline_world":  [[round(wx, 4), round(wy, 4)] for wx, wy in world]}
    if os.path.exists(path) and not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    return f"{lane}차선 저장 → {path} (720점)"


def _nearest(pts, x, y, r):
    best, bd = None, r * r
    for i, (px, py) in enumerate(pts):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            bd = d; best = i
    return best


def _nearest_segment(pts, x, y, r):
    n = len(pts)
    if n < 2:
        return None
    best, bd = None, r * r
    for i in range(n):
        ax, ay = pts[i]; bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay; L2 = dx*dx + dy*dy
        if L2 < 1e-9:
            continue
        t = ((x - ax) * dx + (y - ay) * dy) / L2
        if not (0.08 < t < 0.92):
            continue
        qx, qy = ax + t*dx, ay + t*dy; d2 = (x-qx)**2 + (y-qy)**2
        if d2 < bd:
            bd = d2; best = (i, (int(qx), int(qy)))
    return best


# ───────────────────────── GUI (PySide6) ─────────────────────────
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton,   # noqa: E402
                               QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFileDialog)
from PySide6.QtGui import QPainter, QImage, QColor                                # noqa: E402
from PySide6.QtCore import Qt, QRectF                                             # noqa: E402


def _qimg(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w, _ = rgb.shape
    return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)


class Canvas(QWidget):
    def __init__(self, base, status):
        super().__init__()
        self.base = base; self.H, self.W = base.shape[:2]; self.status = status
        self.pts = {1: [], 2: []}; self.center = {1: None, 2: None}; self.lane = 1
        self.zoom = 1.0; self.ox = 0.0; self.oy = 0.0
        self.drag = None; self.pan = None; self.pending = None; self.hover = None; self.ins = None
        self.history = []
        self.setMinimumSize(560, 420); self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _visWH(self):
        return self.W / self.zoom, self.H / self.zoom

    def _target(self):
        Wd, Hd = max(1, self.width()), max(1, self.height()); ar = self.W / self.H
        if Wd / Hd > ar:
            fh = Hd; fw = fh * ar
        else:
            fw = Wd; fh = fw / ar
        return (Wd - fw) / 2, (Hd - fh) / 2, fw, fh

    def _in(self, wx, wy):
        tx, ty, fw, fh = self._target()
        return tx <= wx <= tx + fw and ty <= wy <= ty + fh

    def w2i(self, wx, wy):
        tx, ty, fw, fh = self._target(); vw, vh = self._visWH()
        return self.ox + ((wx - tx) / fw) * vw, self.oy + ((wy - ty) / fh) * vh

    def clampview(self):
        vw, vh = self._visWH()
        self.ox = min(max(0.0, self.ox), max(0.0, self.W - vw))
        self.oy = min(max(0.0, self.oy), max(0.0, self.H - vh))

    def push(self):
        self.history.append((copy.deepcopy(self.pts), copy.deepcopy(self.center)))
        if len(self.history) > 60:
            self.history.pop(0)

    def compose(self):
        track = self.base.copy()
        # ① 폴리곤 내부 '면'만 반투명 채움
        for ln in (1, 2):
            if len(self.pts[ln]) >= 3:
                m = poly_mask(self.pts[ln])
                color = np.zeros_like(track); color[m > 0] = LANE_COLOR[ln]; sel = m > 0
                track[sel] = cv2.addWeighted(track, 0.45, color, 0.55, 0)[sel]
        # ② 경계선(폴리곤) + 꼭짓점 + 추출된 중앙선(흰 선)
        for ln in (1, 2):
            P = self.pts[ln]; col = LANE_COLOR[ln]; cur = (ln == self.lane)
            if len(P) >= 2:
                cv2.polylines(track, [np.array(P, np.int32).reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
            for (px, py) in P:
                r = 6 if cur else 4
                cv2.circle(track, (px, py), r, col, -1, cv2.LINE_AA)
                cv2.circle(track, (px, py), r, (255, 255, 255), 1, cv2.LINE_AA)
            if cur and len(P) >= 1:                        # 시작점(닫는 기준) 초록 표시
                fx, fy = P[0]
                cv2.circle(track, (fx, fy), 10, (0, 210, 0), 2, cv2.LINE_AA)
            if self.center[ln]:
                cv2.polylines(track, [np.array(self.center[ln], np.int32).reshape(-1, 1, 2)],
                              True, (255, 255, 255), 2, cv2.LINE_AA)
            if cur and self.hover is not None and self.hover < len(P):
                hx, hy = P[self.hover]; cv2.circle(track, (hx, hy), 11, (0, 255, 255), 2, cv2.LINE_AA)
            if cur and self.ins is not None:
                ix, iy = self.ins
                cv2.circle(track, (ix, iy), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(track, (ix-4, iy), (ix+4, iy), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(track, (ix, iy-4), (ix, iy+4), (0, 255, 255), 2, cv2.LINE_AA)
        return track

    def paintEvent(self, e):
        qi = _qimg(self.compose()); vw, vh = self._visWH(); tx, ty, fw, fh = self._target()
        p = QPainter(self); p.fillRect(self.rect(), QColor(24, 24, 24))
        p.drawImage(QRectF(tx, ty, fw, fh), qi, QRectF(self.ox, self.oy, vw, vh)); p.end()
        n = len(self.pts[self.lane]); cl = self.center[self.lane]
        self.status(f"lane {self.lane}   점 {n}   면 {'채움' if n >= 3 else '미완성'}   "
                    f"중앙선 {'추출됨' if cl else '미추출'}   확대 x{self.zoom:.1f}")

    def mousePressEvent(self, e):
        wx, wy = e.position().x(), e.position().y()
        if not self._in(wx, wy):
            return
        ix, iy = self.w2i(wx, wy); px, py = int(ix), int(iy); P = self.pts[self.lane]
        if e.button() == Qt.LeftButton:
            i = _nearest(P, px, py, HIT)
            if i is not None:
                self.push(); self.drag = i
            else:
                seg = _nearest_segment(P, px, py, EDGE)
                if seg is not None:
                    self.push(); si, (qx, qy) = seg; P.insert(si+1, [qx, qy]); self.drag = si+1
                else:
                    self.pending = (wx, wy, px, py)
        elif e.button() == Qt.RightButton:
            i = _nearest(P, px, py, HIT)
            if i is not None:
                self.push(); P.pop(i); self.hover = None
        self.update()

    def mouseMoveEvent(self, e):
        wx, wy = e.position().x(), e.position().y()
        ix, iy = self.w2i(wx, wy); px, py = int(ix), int(iy); P = self.pts[self.lane]
        self.hover = _nearest(P, px, py, HIT); self.ins = None
        if self.hover is None and self.drag is None and self.pan is None:
            seg = _nearest_segment(P, px, py, EDGE); self.ins = seg[1] if seg else None
        if self.drag is not None:
            P[self.drag] = [px, py]; self.ins = None
            if self.drag != 0 and len(P) > 0 and (P[0][0]-px)**2 + (P[0][1]-py)**2 <= SNAP*SNAP:
                P[self.drag] = [P[0][0], P[0][1]]         # 시작점 근처로 끌면 스냅
        elif self.pan is not None:
            apx, apy = self.pan; tx, ty, fw, fh = self._target(); vw, vh = self._visWH()
            self.ox = apx - ((wx - tx) / fw) * vw; self.oy = apy - ((wy - ty) / fh) * vh; self.clampview()
        elif self.pending is not None:
            psx, psy, apx, apy = self.pending
            if abs(wx - psx) + abs(wy - psy) > 4:
                self.pan = (apx, apy); self.pending = None; self.ins = None
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.pending is not None:
                _, _, ax, ay = self.pending; P = self.pts[self.lane]
                if len(P) >= 2 and (P[0][0]-ax)**2 + (P[0][1]-ay)**2 <= SNAP*SNAP:
                    ax, ay = P[0]                          # 시작점 근처 클릭 = 시작점에 스냅(닫기)
                self.push(); P.append([ax, ay])
            self.drag = None; self.pan = None; self.pending = None
        self.update()

    def wheelEvent(self, e):
        wx, wy = e.position().x(), e.position().y()
        if not self._in(wx, wy):
            return
        ix, iy = self.w2i(wx, wy)
        f = 1.25 if e.angleDelta().y() > 0 else 1/1.25
        self.zoom = min(6.0, max(1.0, self.zoom * f))
        tx, ty, fw, fh = self._target(); vw, vh = self._visWH()
        self.ox = ix - ((wx - tx) / fw) * vw; self.oy = iy - ((wy - ty) / fh) * vh; self.clampview(); self.update()

    def set_lane(self, ln): self.lane = ln; self.hover = self.ins = None; self.update()

    def do_extract(self):
        n = 0
        for ln in (1, 2):
            if len(self.pts[ln]) >= 3:
                c = extract_centerline(poly_mask(self.pts[ln]))
                if c:
                    self.center[ln] = c; n += 1
        print(f"중앙선 추출: {n}개 차선" if n else "먼저 점을 찍어 면을 만드세요(각 3점+).")
        self.update()

    def do_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "불러오기 (GT JSON 또는 라벨 PNG)",
                                              os.path.expanduser("~"), "GT/Mask (*.json *.png)")
        if not path:
            return
        self.push()
        if path.endswith(".png"):
            lab = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if lab is None:
                print("불러오기 실패"); return
            if lab.shape != (self.H, self.W):
                lab = cv2.resize(lab, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            for ln in (1, 2):
                cnts, _ = cv2.findContours((lab == ln).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    big = max(cnts, key=cv2.contourArea)
                    ap = cv2.approxPolyDP(big, 4, True).reshape(-1, 2)
                    self.pts[ln] = [[int(x), int(y)] for x, y in ap]
            print(f"불러오기: {os.path.basename(path)} (라벨마스크 → 폴리곤)")
        else:
            d = json.load(open(path))
            pix = d.get("centerline_pixels") or [list(self._w2p(wx, wy)) for wx, wy in d["centerline_world"]]
            self.center[self.lane] = [[int(x), int(y)] for x, y in pix]
            print(f"불러오기: {os.path.basename(path)} → {self.lane}차선 중앙선(참고)")
        self.update()

    @staticmethod
    def _w2p(wx, wy):
        return int((wy + 26.915) / 53.83 * IMG_W), int((wx + 20.237) / 40.473 * IMG_H)

    def do_undo(self):
        if self.history:
            self.pts, self.center = self.history.pop()
        self.update()

    def do_save(self):
        if self.center[self.lane] is None and len(self.pts[self.lane]) >= 3:
            self.center[self.lane] = extract_centerline(poly_mask(self.pts[self.lane]))
        if self.center[self.lane] is None:
            print("현재 차선의 면을 그리고 '중앙선 추출' 후 저장하세요."); return
        path, _ = QFileDialog.getSaveFileName(self, f"{self.lane}차선 저장",
                                              LANE_FILE[self.lane], "GT JSON (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        msg = save_lane(self.lane, self.center[self.lane], path)
        lab = np.zeros((self.H, self.W), np.uint8)         # 라벨마스크 = 폴리곤 면
        if len(self.pts[2]) >= 3:
            lab[poly_mask(self.pts[2]) > 0] = 2
        if len(self.pts[1]) >= 3:
            lab[poly_mask(self.pts[1]) > 0] = 1
        labp = path[:-5] + "_label.png"; cv2.imwrite(labp, lab)
        print(msg + f"  + {os.path.basename(labp)}"); self.update()

    def do_clear(self):
        self.push(); self.pts[self.lane] = []; self.center[self.lane] = None; self.update()

    def reset_view(self): self.zoom = 1.0; self.ox = self.oy = 0.0; self.update()

    def keyPressEvent(self, e):
        t = e.text().lower()
        if t == '1': self.set_lane(1)
        elif t == '2': self.set_lane(2)
        elif t == 'e': self.do_extract()
        elif t == 'l': self.do_load()
        elif t == 's': self.do_save()
        elif t == 'c': self.do_clear()
        elif t == 'r': self.reset_view()
        elif t == 'u' or (e.key() == Qt.Key_Z and e.modifiers() & Qt.ControlModifier): self.do_undo()
        elif t == 'q': self.window().close()


class Window(QMainWindow):
    def __init__(self, base):
        super().__init__()
        self.setWindowTitle("lane_picker")
        self.status_lbl = QLabel("")
        c = self.canvas = Canvas(base, self.status_lbl.setText)
        bar = QHBoxLayout(); bar.setContentsMargins(6, 4, 6, 4); bar.setSpacing(5)
        for label, fn in [("1차선", lambda: c.set_lane(1)), ("2차선", lambda: c.set_lane(2)),
                          ("중앙선 추출", c.do_extract), ("Load", c.do_load),
                          ("Undo", c.do_undo), ("Save", c.do_save), ("Clear", c.do_clear),
                          ("Reset view", c.reset_view)]:
            b = QPushButton(label); b.setFocusPolicy(Qt.NoFocus); b.clicked.connect(fn); bar.addWidget(b)
        bar.addStretch(1); bar.addWidget(self.status_lbl)
        top = QWidget(); top.setLayout(bar); top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root = QVBoxLayout(); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(top, 0); root.addWidget(c, 1)
        w = QWidget(); w.setLayout(root); self.setCentralWidget(w); c.setFocus()


def main():
    base = cv2.imread(TRACK)
    if base is None:
        raise SystemExit(f"트랙 이미지를 못 찾음: {TRACK}")
    base = cv2.resize(base, (int(IMG_W), int(IMG_H)))
    app = QApplication.instance() or QApplication(sys.argv)
    win = Window(base); win.resize(1180, 820); win.show()
    app.exec()


if __name__ == "__main__":
    main()
