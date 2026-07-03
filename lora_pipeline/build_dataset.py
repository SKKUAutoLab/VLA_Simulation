#!/usr/bin/env python3
"""
Stage 2 — 데이터셋 빌드 + 분포 점검
==================================
수집된 labels.csv → 학습용 manifest(train/val).
  - steering 분포 출력 (커브 커버리지 점검: 비-제로 비율이 핵심)
  - 직진(steering==0) 과다 시 다운샘플(--zero-keep 비율)
  - train/val 분할
타깃 포맷은 추론 노드와 동일한 terse "D <st> <sp>".

사용:
    python3 lora_pipeline/build_dataset.py            # 분포만 보고 빌드
    python3 lora_pipeline/build_dataset.py --zero-keep 0.3 --val 0.1
"""
import os, csv, json, argparse, glob
from collections import Counter

HERE      = os.path.dirname(__file__)
OUT_DIR   = os.path.join(HERE, "dataset")
IMG_DIR   = os.path.join(OUT_DIR, "images")
CSV_PATH  = os.path.join(OUT_DIR, "labels.csv")
TRAIN_OUT = os.path.join(OUT_DIR, "train.jsonl")
VAL_OUT   = os.path.join(OUT_DIR, "val.jsonl")

# 추론 노드가 쓸 프롬프트와 반드시 동일하게 유지할 것 (train/infer 일치)
PROMPT = (
    "Drive a car using the front camera. Follow the gray road, stay between the "
    "lane lines. Reply ONE line only: D <st> <sp>  st=-7..7 (NEGATIVE=left, "
    "POSITIVE=right), sp=0..100; S=stop. Ex: D -3 40"
)

MAX_SPEED = 55  # 추론 노드 상한과 맞춤


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-keep", type=float, default=1.0,
                    help="steering==0 프레임 유지 비율(0~1). 직진 과다 시 0.3 등")
    ap.add_argument("--val", type=float, default=0.1, help="검증셋 비율")
    ap.add_argument("--cap", type=int, default=0,
                    help="steering 값별 최대 샘플 수(0=무제한). 최빈값 상수붕괴 방지")
    ap.add_argument("--quant", type=int, default=0,
                    help="조향 양자화 간격(예 3 → {-6,-3,0,3,6}). 라벨 지터 흡수")
    args, _ = ap.parse_known_args()

    # labels*.csv 모두 합침 (labels_inner.csv + labels_outer.csv + labels.csv 등)
    csv_files = sorted(glob.glob(os.path.join(OUT_DIR, "labels*.csv")))
    print("읽는 CSV:", [os.path.basename(p) for p in csv_files])
    rows = []
    for cf in csv_files:
        with open(cf) as f:
            for r in csv.DictReader(f):
                img = os.path.join(IMG_DIR, r["fname"])
                if not os.path.exists(img):
                    continue
                st = clamp(int(r["steering"]), -7, 7)
                if args.quant > 0:   # 거친 클래스로 양자화 (지터 흡수)
                    st = clamp(int(round(st / args.quant)) * args.quant, -6, 6)
                sp = clamp(int(r["left_speed"]), 0, 100)
                rows.append({"image": img, "steering": st, "speed": sp})

    if not rows:
        print(f"데이터 없음: {CSV_PATH} 확인. 먼저 collect_demos_node.py로 수집.")
        return

    dist = Counter(r["steering"] for r in rows)
    n = len(rows)
    nz = sum(v for k, v in dist.items() if k != 0)
    print(f"총 {n} 프레임 | 비-제로 조향(커브) {nz} ({100*nz/n:.1f}%)")
    print("steering 분포:", dict(sorted(dist.items())))
    if nz / n < 0.15:
        print("\n⚠️  경고: 커브(비-제로 조향) 비율이 너무 낮음. expert가 직진만 "
              "하고 있을 수 있음 → 학습해도 커브 못 배움. 수집 구간/스택 점검 필요.")

    # 직진 다운샘플 (결정적: 인덱스 모듈러로 균등 추림 — 난수 미사용)
    if args.zero_keep < 1.0:
        kept, zc = [], 0
        keep_every = max(1, round(1.0 / args.zero_keep))
        for r in rows:
            if r["steering"] == 0:
                if zc % keep_every == 0:
                    kept.append(r)
                zc += 1
            else:
                kept.append(r)
        print(f"직진 다운샘플: {n} → {len(kept)} (zero-keep={args.zero_keep})")
        rows = kept

    # steering 값별 상한 (상수붕괴 방지: 최빈값이 손실 최소해가 되지 않게)
    if args.cap > 0:
        seen = Counter()
        capped = []
        for r in rows:
            s = r["steering"]
            if seen[s] < args.cap:
                capped.append(r); seen[s] += 1
        print(f"빈 상한 cap={args.cap}: {len(rows)} → {len(capped)}")
        print("  균형 후 분포:", dict(sorted(Counter(r['steering'] for r in capped).items())))
        rows = capped

    # target 텍스트 부여
    # 속도는 고정 cruise로 단순화 → 모델은 조향만 학습 (출력공간 축소)
    CRUISE = 30
    for r in rows:
        r["prompt"] = PROMPT
        r["target"] = f"D {r['steering']} {CRUISE}"

    # 결정적 train/val 분할 (매 1/val 번째를 val로)
    val_every = max(2, round(1.0 / args.val)) if args.val > 0 else 0
    train, val = [], []
    for i, r in enumerate(rows):
        (val if (val_every and i % val_every == 0) else train).append(r)

    with open(TRAIN_OUT, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(VAL_OUT, "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ train {len(train)} → {TRAIN_OUT}\n✓ val {len(val)} → {VAL_OUT}")


if __name__ == "__main__":
    main()
