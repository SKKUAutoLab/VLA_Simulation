#!/usr/bin/env bash
# 사전 학습 산출물 번들(zip) 생성 — GitHub Release 업로드용
#   사용: bash tools/make_pretrained_bundle.sh
#   출력: ~/vla_pretrained_bundle.zip
set -e
LP="$HOME/ros2_autonomous_vehicle_simulation/lora_pipeline"
STAGE=$(mktemp -d)/vla_pretrained
mkdir -p "$STAGE"

cp "$LP/vla_lora_head_fast.pt" "$STAGE/"
cp "$LP/vla_lora_head.pt"      "$STAGE/"
cp -r "$LP/vla_lora_adapter"   "$STAGE/"

OUT="$HOME/vla_pretrained_bundle.zip"
( cd "$(dirname "$STAGE")" && zip -qr "$OUT" vla_pretrained )
echo "생성 완료: $OUT"
du -sh "$OUT"
