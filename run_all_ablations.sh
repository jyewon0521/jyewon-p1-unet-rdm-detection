#!/usr/bin/env bash
# P01 Ablation B — 경량 모델 (base_ch=16) 재현 스크립트
# 사용법: bash run_all_ablations.sh
# 주의: run_assignment.sh 실행 후 데이터가 있어야 함
set -euo pipefail

cd projects/p01_unet_detector

echo "=== [1/3] Ablation B 학습 (base_ch=16, 30 epoch) ==="
python train.py --epochs 30 --base_ch 16 \
    --artifact_dir artifacts/unet_base16

echo "=== [2/3] Ablation B 평가 (val → 정책 고정 → test) ==="
python evaluate_unet.py --split val \
    --checkpoint artifacts/unet_base16/best_model.pt \
    --base_ch 16 \
    --out artifacts/unet_base16/p01_unet_base16_threshold_sweep_val.json

python evaluate_unet.py --split test \
    --checkpoint artifacts/unet_base16/best_model.pt \
    --base_ch 16 \
    --policy artifacts/unet_base16/p01_unet_base16_selected_val.json \
    --out artifacts/unet_base16/p01_unet_base16_selected_test.json

echo "=== [3/3] 비교 그림 재생성 ==="
python make_report_figures.py

echo ""
echo "=== Ablation B 완료 ==="
echo "결과: projects/p01_unet_detector/artifacts/unet_base16/"
