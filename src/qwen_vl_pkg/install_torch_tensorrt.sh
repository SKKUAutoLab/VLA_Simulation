#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# torch-tensorrt 설치 스크립트
#
# 환경 요건:
#   - PyTorch   2.10.0+cu128  (현재 설치됨)
#   - TensorRT  10.x          (현재 설치됨)
#   - CUDA      12.8          (현재 설치됨)
#
# 실행:
#   chmod +x install_torch_tensorrt.sh
#   ./install_torch_tensorrt.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=== torch-tensorrt 설치 ==="
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(nvcc --version | grep release | awk '{print $6}')"
echo "TRT:     $(python3 -c 'import tensorrt; print(tensorrt.__version__)')"

# torch-tensorrt 버전은 PyTorch 버전과 반드시 일치해야 함
# PyTorch 2.10.0 → torch-tensorrt 2.10.0
pip install torch-tensorrt==2.10.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --no-deps   # TRT/torch 버전 충돌 방지

echo ""
echo "=== 설치 확인 ==="
python3 -c "
import torch_tensorrt
print('torch_tensorrt:', torch_tensorrt.__version__)
print('backends:', torch_tensorrt.enabled_features())
"

echo ""
echo "=== 사용 방법 ==="
echo "  # TRT 노드 실행"
echo "  ros2 run qwen_vl_pkg qwen_vl_trt_node"
echo ""
echo "  # TRT 비활성화 (inductor fallback)"
echo "  QWEN_TRT=0 ros2 run qwen_vl_pkg qwen_vl_trt_node"
