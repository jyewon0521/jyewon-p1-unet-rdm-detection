#!/usr/bin/env bash
# P01 U-Net FMCW 표적 탐지 — 전체 실행 스크립트
# 사용법: bash run_assignment.sh
# 실행 위치: 레포 루트 (radar-ai-projects-main/)
set -euo pipefail

cd projects/p01_unet_detector

echo "=== [1/5] 데이터 생성 ==="
python train.py --generate --epochs 0

echo "=== [2/5] U-Net 학습 (base_ch=32, 30 epoch) ==="
python train.py --epochs 30

echo "=== [3/5] CA-CFAR 평가 (val → 정책 고정 → test) ==="
python evaluate_cfar.py --split val
python evaluate_cfar.py --split test \
    --policy artifacts/verified_p01/p01_cfar_selected_val.json

echo "=== [4/5] U-Net 평가 (val → 정책 고정 → test) ==="
python evaluate_unet.py --split val
python evaluate_unet.py --split test \
    --policy artifacts/verified_p01/p01_unet_selected_val.json

echo "=== [5/5] 보고서 그림 생성 ==="
python make_report_figures.py

echo ""
echo "=== 완료 ==="
echo "결과: projects/p01_unet_detector/artifacts/verified_p01/"
echo "그림: projects/p01_unet_detector/artifacts/report_figures/"
