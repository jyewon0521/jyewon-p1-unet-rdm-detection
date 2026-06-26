# Radar AI Project - 14주차 최종 과제 브리핑

## 과제 개요

충남대학교 REMI Lab "레이다 신호처리 및 인공지능" 대학원 강의 최종 과제.
P1~P4 중 **1개 사례를 선택**하여 baseline 재현 + 최소 1개 ablation 실험 수행 후 보고서 제출.

- **제출물**: 최종 보고서 PDF (`학번_이름_W14.pdf`) + 재현용 코드 저장소 링크
- **본문 분량**: 6~10쪽 (참고문헌 제외, 그림/표 포함)
- **평가**: 보고서 단독 (발표/시험 없음)
- **강의 페이지**: https://remilab.cnu.ac.kr/lectures/grad-radar-ai/weeks/week14/index.html

---

## 프로젝트 코드 위치

```
C:\Users\Remi0521\Downloads\radar-ai-projects-main\
```

### 디렉토리 구조

```
radar-ai-projects-main/
├── common/           # 공유 CLI, HDF5 I/O, metrics, train utilities
├── shared/           # FMCW 시뮬레이터, micro-Doppler, SAR 등 재사용 모듈
├── projects/
│   ├── p01_unet_detector/       # P1: RDM 표적 탐지
│   ├── p02_resnet18_har/        # P2: micro-Doppler 인체 행동 인식
│   ├── p03_radar_cube_doa/      # P3: DoA 추정 + 매핑
│   ├── p04_dncnn_sar/           # P4: SAR 디스페클링
│   ├── p05_waveform_classification/     # (과제 대상 아님)
│   └── p06_target_signature_classification/  # (과제 대상 아님)
├── docs/
├── requirements.txt
└── README.md
```

---

## 환경 셋업 (VS Code 터미널에서)

```bash
cd C:\Users\Remi0521\Downloads\radar-ai-projects-main
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# GPU 없으면 CPU PyTorch 먼저:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

---

## P1~P4 사례별 요약

### P1: U-Net FMCW Range-Doppler 표적 탐지

| 항목 | 내용 |
|------|------|
| **문제** | RDM에서 이동 표적 픽셀 단위 탐지 (세그멘테이션) |
| **모델** | U-Net 5단 인코더/디코더, skip connection, ~7.7M params |
| **입력** | (B, 2, Nd, Nr) — RDM log-magnitude + phase |
| **출력** | (B, 1, Nd, Nr) — 탐지 확률 맵 |
| **손실** | FocalDiceLoss (alpha=0.75, gamma=2.0, dice_weight=0.5) |
| **베이스라인** | CA-CFAR (guard/train window, Pfa 설계) |
| **데이터** | 시뮬레이터 생성, train 5만 / val 5천 / test 5천 |
| **평가 지표** | Pd, Pfa, Precision, F1, target_recall, false_alarms_per_rdm |
| **ablation 후보** | 입력 채널(2ch vs 1ch), loss 조합, threshold 정책, label gate, base_ch |

**실행 명령:**
```bash
cd projects/p01_unet_detector
python train.py --generate --smoke          # smoke test
python train.py --generate --epochs 30      # 본 학습
python train.py --eval_only --checkpoint artifacts/best_model.pt  # 평가
```

---

### P2: ResNet-18 Micro-Doppler 인체 행동 인식 (HAR)

| 항목 | 내용 |
|------|------|
| **문제** | 스펙트로그램에서 6개 인체 행동 분류 |
| **모델** | ResNet-18 (1채널), ~11.2M params / TinyCNN 옵션 |
| **입력** | (B, 1, 128, 128) — micro-Doppler 스펙트로그램 |
| **출력** | (B, 6) — 클래스 logits |
| **클래스** | walk, run, sit_down, fall, wave, idle |
| **손실** | CrossEntropyLoss (label_smoothing=0.1) |
| **베이스라인** | SVM/LogReg + 핸드크래프트 특징 |
| **데이터** | Boulic 인체 모델 시뮬레이터, train 3만 / val 3천 / test 3천 |
| **평가 지표** | accuracy, per_class accuracy, confusion_matrix |
| **ablation 후보** | STFT 윈도우, log vs linear, augmentation, aspect 일반화 stress |

**실행 명령:**
```bash
cd projects/p02_resnet18_har
python train.py --generate --smoke          # smoke test
python train.py --generate --epochs 30      # 본 학습
python evaluate_feature_baseline.py --data_dir data --model rbf_svm --max_train 10000  # SVM 베이스라인
```

---

### P3: RadarCubeDoANet 매핑/DoA

| 항목 | 내용 |
|------|------|
| **문제** | 안테나 벡터에서 DoA 추정 → BEV 점유 맵 생성 |
| **모델** | 1D ResidualConvNet (Stem + 6 ResBlock + FC head), width=128 |
| **입력** | (B, 2, 8) — 복소 안테나 벡터 [real, imag] |
| **출력** | (B, 181) — -90~+90도 DoA 스펙트럼 |
| **손실** | SpectrumLoss = BCE(pos_weight=8) + 0.25*MSE |
| **베이스라인** | Angle-FFT, single-snapshot MUSIC |
| **데이터** | 이동 ego 시나리오 (77GHz, BW=200MHz, 8 Rx) |
| **평가 지표** | DoA MAE/RMSE, 1/2/5도 정확도, OGM IoU, point error |
| **ablation 후보** | DoA 알고리즘 교체, snapshot 수, occlusion mask |

**참고 결과 (200MHz 기준):**

| Method | DoA MAE | Point-grid IoU | Mean point error |
|---|---:|---:|---:|
| Oracle GT | 0.000° | 0.969 | 0.000 m |
| MUSIC | 0.346° | 0.517 | 0.104 m |
| DoANet | 0.434° | 0.525 | 0.126 m |
| Angle FFT | 3.888° | 0.225 | 1.097 m |

**실행 명령:**
```bash
cd projects/p03_radar_cube_doa
python train.py --mapping --generate --smoke   # smoke test
python train.py --mapping --generate --epochs 30 --batch_size 1024  # 본 학습
```

---

### P4: DnCNN-SAR 스페클 제거

| 항목 | 내용 |
|------|------|
| **문제** | SAR 영상 스페클 잡음 제거 (영상 복원) |
| **모델** | DnCNN 17층 잔차 CNN, 64 필터, ~556K params |
| **입력** | (B, 1, 256, 256) — 정규화 log/dB SAR 패치 |
| **출력** | clean = input - predicted_residual |
| **손실** | 0.8*Charbonnier + 0.2*SSIM |
| **베이스라인** | Lee filter, Frost filter, Median filter |
| **데이터** | 실제 Sentinel-1 GRD/SLC 패치 (smoke는 번들 데이터 사용 가능) |
| **평가 지표** | PSNR, SSIM, ENL (smoothness proxy), EPI (edge preservation) |
| **ablation 후보** | loss 비교(L1/Charbonnier/SSIM/혼합), patch 크기, depth, residual learning |

**참고 결과:**

| Method | PSNR | SSIM |
|---|---:|---:|
| Median filter | 26.34 dB | 0.621 |
| **DnCNN-SAR** | **31.10 dB** | **0.794** |

**실행 명령:**
```bash
cd projects/p04_dncnn_sar
python train.py --generate --smoke          # smoke test (번들 데이터)
python train.py --generate --epochs 100 --batch_size 32 --lr 5e-4 --no_amp  # 본 학습
python train.py --eval_only --checkpoint artifacts/best_model.pt  # 평가
```

---

## 평가 기준 (Rubric)

| 평가 축 | 비중 | 핵심 질문 |
|--------|------|---------|
| **Method clarity** | 25% | 다른 사람이 같은 실험을 재현할 수 있는가? |
| **Experiment rigor** | 30% | 비교가 공정하고 통제 변수가 고정되어 있는가? |
| **Result interpretation** | 30% | "왜 이 결과인가" 분석 + 실패 모드 분석이 있는가? |
| **Code quality** | 15% | 실행 명령 한 줄로 결과가 재현되는가? |

### 가산 항목
- Ablation 2개 이상 (독립 변수 하나씩만 변경)
- Stress 실험 (다른 SNR/aspect/시점)
- 외부 공개 데이터셋 활용

### 감점 항목
- baseline 재현 수치 누락 (강의 자료 수치만 인용)
- test split에서 정책 튜닝 (val 없이 test에서 최적화)
- 코드 저장소 미제출 (15% 자동 0점)

---

## 보고서 구성 (6개 섹션)

1. **Abstract** (5~8문장): 문제, 방법, 핵심 결과 수치 1개, 결론
2. **Introduction** (0.5~1쪽): 사례 선택 사유, research question
3. **Method** (1.5~2.5쪽): 모델 구조도, 데이터 split, 손실 함수, 하이퍼파라미터, seed
4. **Experiments** (0.5~1쪽): baseline + ablation 표, 평가 지표, 평가 split
5. **Results & Discussion** (2~3쪽): 비교 표, 학습 곡선, 분석 문단, 실패 사례
6. **Conclusion & References** (0.5~1쪽): 요약, 한계 2~3개, 참고문헌 5편 이상

### 필수 시각화 4종
- 학습 곡선 (train/val loss vs epoch)
- 비교 표 (baseline vs proposed vs ablation)
- Ablation 막대 그래프
- 대표 사례 시각화 (성공 1개 + 실패 1개)

---

## 실험 설계 4원칙

1. **한 번에 하나만 바꾼다** — 독립 변수 1개만 변경, 나머지 통제
2. **baseline 재현이 출발점** — 본인 환경에서 직접 돌린 수치 사용
3. **실패 모드 분석** — 모델이 틀린 사례와 그 이유 분석
4. **재현성 확보** — seed 고정, requirements.txt, 실행 명령 1줄, 학습 로그 저장

---

## 작업 순서 제안

1. 환경 셋업 (venv + requirements)
2. 사례 선택 (P1~P4 중 1개)
3. smoke test 통과 확인
4. baseline 본 학습 + 재현 수치 확보
5. ablation 실험 1~2개 설계 및 수행
6. 결과 시각화 (학습 곡선, 비교 표, 대표 사례)
7. 보고서 작성 (PDF)
8. 코드 저장소 정리 (README + seed + requirements)
