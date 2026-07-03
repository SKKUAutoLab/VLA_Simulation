#!/usr/bin/env python3
"""
gt_gui_shot.py — 실제 GT 어노테이터(PySide6) 창을 띄워 스크린샷을 캡처.

GT를 수동점으로 주입하고, 선택적으로 도로/차선 마스킹(V-range 검출)을 실행한 뒤 창을 grab한다.
X 디스플레이가 필요하다 (헤드리스면 DISPLAY=:1, QT_QPA_PLATFORM=xcb).

사용법:
    DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py [GT_JSON] [-o OUT] [--mask]

예:
    # GT 중심선만
    DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json -o /tmp/gt_gui.png
    # 도로/차선 마스킹까지 (1차선 빨강 / 2차선 파랑)
    DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json --mask -o /tmp/gt_mask.png
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.join(HERE, "..", "src", "gui_pkg", "gui_pkg")
sys.path.insert(0, os.path.abspath(GUI_DIR))

from PySide6.QtWidgets import QApplication       # noqa: E402
from PySide6.QtCore import QTimer                # noqa: E402
import gt_annotator as ga                        # noqa: E402

DEF_IN = os.path.expanduser("~/track_gt_manual.json")


def main():
    ap = argparse.ArgumentParser(description="Screenshot the GT annotator window")
    ap.add_argument("gt_json", nargs="?", default=DEF_IN)
    ap.add_argument("-o", "--out", default="/tmp/gt_gui.png")
    ap.add_argument("--mask", action="store_true",
                    help="run V-range road detection + lane split (도로/차선 마스킹)")
    a = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    win = ga.GTAnnotator()                       # 생성자에서 track.png 자동 로드

    d = json.load(open(os.path.expanduser(a.gt_json)))
    if d.get("centerline_pixels"):
        pts = [tuple(p) for p in d["centerline_pixels"]]
    else:
        pts = [ga.world_to_pixel(wx, wy) for wx, wy in d["centerline_world"]]
    win._canvas.set_manual(pts)
    win._canvas.set_show_lane("both")            # 반드시 문자열 "both"/"inner"/"outer"
    try:
        win._export_combo.setCurrentIndex(2)     # "수동 점만" 뷰
        win._update_count()
    except Exception:
        pass

    win.resize(1500, 920)
    win.show()
    app.processEvents()

    if a.mask:
        win._run("v_range")                      # 도로 검출 + 차선 분리 (비동기 QThread)

    def shot():
        if a.mask and (win._canvas._inner is None or win._canvas._outer is None):
            QTimer.singleShot(300, shot)         # 검출 완료까지 폴링
            return
        win._canvas._refresh()
        app.processEvents()
        win.grab().save(a.out)
        print("saved", a.out, "(mask)" if a.mask else "(centerline)")
        app.quit()

    QTimer.singleShot(1000, shot)                # 레이아웃 안정화 대기
    app.exec()


if __name__ == "__main__":
    main()
