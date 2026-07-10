# 사전 학습 산출물 배포 (Pre-trained Artifacts)

데이터 수집(수동주행)과 GPU 학습은 시간·용량 부담이 큽니다
(데이터셋 66GB, 학습 GPU 필요). **학습된 결과물을 미리 배포**하면 학생들은
이 단계를 건너뛰고 바로 추론/주행에 들어갈 수 있습니다.

## 무엇을 배포하나 (약 76MB)
| 파일 | 크기 | 역할 |
|------|------|------|
| `vla_lora_head_fast.pt` | 9.9MB | 주행 노드 기본 헤드(비전 고정 + spatial head) ★ |
| `vla_lora_head.pt` | 9.9MB | LoRA 학습용 헤드 |
| `vla_lora_adapter/` | 56MB | 비전 LoRA 어댑터 |

> 배포 제외: 특징 캐시 `vla_tokfeat_cache.pt`(9.7GB)·`vla_px_cache.pt`(32GB),
> 데이터셋/이미지 — 용량이 너무 커서 배포하지 않습니다.

## 배포 방법 (조교/운영자)
1. 번들 생성:
   ```bash
   bash tools/make_pretrained_bundle.sh    # → ~/vla_pretrained_bundle.zip (약 69MB)
   ```
2. GitHub Release 로 업로드:
   `VLA_Simulation` → Releases → Draft new release → tag `pretrained-v1`
   → `vla_pretrained_bundle.zip` 첨부.
   (또는 HuggingFace/드라이브 링크로 배포하고 `setup_pretrained.sh`의 URL 수정)

## 학생 사용법
```bash
cd ~/VLA_simulation
bash lora_pipeline/setup_pretrained.sh            # Release에서 자동 다운로드·배치
# 또는 로컬 zip:  bash lora_pipeline/setup_pretrained.sh ~/vla_pretrained_bundle.zip

# 데이터 수집·학습 없이 바로 주행
ros2 launch lora_pipeline/vla_drive.launch.py
```

## 사양별 참여 경로 (솔직한 안내)
| 학생 사양 | 배포 산출물로 가능한 것 |
|-----------|------------------------|
| NVIDIA GPU(VRAM 6GB+) 노트북 | 배포 받고 **바로 주행/추론** (데이터·학습 생략) ✅ |
| GPU 없음 → Colab/랩 서버 | Colab(T4 무료)이나 랩 GPU 서버에 배포물 올려 추론 |
| GPU 전혀 없음 | **데모 영상**으로 결과 확인 + 코드/원리 학습 + CNN 저사양 트랙 |

> ⚠️ 중요: 배포 산출물은 **데이터 수집·학습 단계를 없애줄 뿐**, 추론(주행) 자체는
> Qwen3-VL-2B 비전 인코더를 GPU에 올려야 하므로 여전히 GPU가 필요합니다.
> GPU가 없는 학생은 Colab/랩 서버를 쓰거나, 결과는 데모 영상으로 확인하고
> 원리·코드 학습에 집중하는 경로를 권장합니다.
