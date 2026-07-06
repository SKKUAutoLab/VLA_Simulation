#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 사전 학습 산출물 설치 — 데이터 수집·학습을 건너뛰고 바로 추론/주행
#
#   사용:  bash lora_pipeline/setup_pretrained.sh [번들.zip 경로]
#   - 인자 없으면 GitHub Release 에서 자동 다운로드
#
# 배포 내용(약 76MB): vla_lora_head_fast.pt, vla_lora_head.pt, vla_lora_adapter/
# 주의: 추론(주행) 자체는 여전히 NVIDIA GPU(VRAM 6GB+)가 필요합니다.
#       GPU가 없으면 Colab/랩 서버에서 실행하거나 데모 영상으로 결과를 확인하세요.
# ─────────────────────────────────────────────────────────────
set -e
DEST="$HOME/ros2_autonomous_vehicle_simulation/lora_pipeline"
HF_REPO="${HF_REPO:-hoonsy/VLA_Simulation-pretrained}"   # HuggingFace 모델 repo
ZIP="${1:-}"

if [ -n "$ZIP" ] && [ -f "$ZIP" ]; then
  echo "[1/3] 로컬 번들 사용: $ZIP"
  tmp=$(mktemp -d); unzip -q "$ZIP" -d "$tmp"
  cp "$tmp"/vla_pretrained/vla_lora_head_fast.pt "$DEST"/
  cp "$tmp"/vla_pretrained/vla_lora_head.pt      "$DEST"/
  cp -r "$tmp"/vla_pretrained/vla_lora_adapter   "$DEST"/
  rm -rf "$tmp"
else
  echo "[1/3] HuggingFace Hub 에서 산출물 로드 (모델 import 방식): $HF_REPO"
  pip install -q -U huggingface_hub
  python3 - "$HF_REPO" "$DEST" <<'PY'
import sys, shutil
from huggingface_hub import hf_hub_download, snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
for f in ("vla_lora_head_fast.pt", "vla_lora_head.pt"):
    p = hf_hub_download(repo_id=repo, filename=f)      # 자동 캐싱 (~/.cache/huggingface)
    shutil.copy(p, f"{dest}/{f}")
# LoRA 어댑터 폴더 통째로
d = snapshot_download(repo_id=repo, allow_patterns="vla_lora_adapter/*")
shutil.copytree(f"{d}/vla_lora_adapter", f"{dest}/vla_lora_adapter", dirs_exist_ok=True)
print("HF 로드 완료")
PY
fi

echo "[2/3] 배치 완료 → $DEST"

echo "[3/3] 완료. 이제 데이터 수집·학습 없이 바로 주행할 수 있습니다:"
echo "    cd ~/ros2_autonomous_vehicle_simulation"
echo "    ros2 launch lora_pipeline/vla_drive.launch.py"
