#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 사전 학습 산출물을 HuggingFace 개인 계정에 업로드 (모델 repo)
#
#   먼저:  huggingface-cli login        # 쓰기 토큰 입력 (채팅에 붙여넣지 말 것)
#   실행:  bash tools/upload_pretrained_hf.sh <HF_USERNAME>
#          (또는 HF_USER=<username> bash tools/upload_pretrained_hf.sh)
#
# 업로드: vla_lora_head_fast.pt, vla_lora_head.pt, vla_lora_adapter/
# 결과 repo:  <HF_USERNAME>/VLA_Simulation-pretrained
# ─────────────────────────────────────────────────────────────
set -e
USER_IN="${1:-$HF_USER}"
if [ -z "$USER_IN" ]; then
  echo "사용법: bash tools/upload_pretrained_hf.sh <HF_USERNAME>"; exit 1
fi
REPO="$USER_IN/VLA_Simulation-pretrained"
LP="$HOME/ros2_autonomous_vehicle_simulation/lora_pipeline"

pip install -q -U huggingface_hub
python3 - "$REPO" "$LP" <<'PY'
import sys, os
from huggingface_hub import HfApi, create_repo
repo, lp = sys.argv[1], sys.argv[2]
api = HfApi()
create_repo(repo, repo_type="model", exist_ok=True)
print("repo 준비:", repo)
for f in ("vla_lora_head_fast.pt", "vla_lora_head.pt"):
    api.upload_file(path_or_fileobj=os.path.join(lp, f), path_in_repo=f, repo_id=repo)
    print("업로드:", f)
api.upload_folder(folder_path=os.path.join(lp, "vla_lora_adapter"),
                  path_in_repo="vla_lora_adapter", repo_id=repo)
print("업로드: vla_lora_adapter/")
# 간단한 모델 카드
card = f"""---
license: mit
tags: [vla, autonomous-driving, qwen3-vl, ros2, gazebo]
---
# VLA_Simulation — Pretrained artifacts

SKKUAutoLab/VLA_Simulation 실습용 사전 학습 산출물.

| 파일 | 역할 |
|------|------|
| vla_lora_head_fast.pt | 주행 노드 기본 헤드 (비전 고정 + spatial head) |
| vla_lora_head.pt | LoRA 학습용 헤드 |
| vla_lora_adapter/ | 비전 LoRA 어댑터 |

## 사용
```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("{repo}", "vla_lora_head_fast.pt")
```
또는 저장소에서: `bash lora_pipeline/setup_pretrained.sh`
(주의: 추론은 NVIDIA GPU 필요. 데이터 수집·학습은 이 산출물로 생략 가능.)
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
print("완료 → https://huggingface.co/"+repo)
PY
