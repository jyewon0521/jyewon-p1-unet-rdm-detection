# P1. U-Net FMCW Range-Doppler 표적 탐지

**레이다 신호처리 및 인공지능** | Week 14 최종 과제  
충남대학교 REMI Lab | 정예원 | [학번 직접 기입]

---

## 개요

FMCW 레이다로 생성한 Range-Doppler Map(RDM)에서 **U-Net 기반 딥러닝 모델**로 표적을 탐지하고,  
전통적인 **CA-CFAR** 알고리즘과 성능을 비교합니다.

**핵심 결과 요약**

| 방법 | Pd | Precision | F1 | FA/RDM |
|---|---|---|---|---|
| CA-CFAR | 0.529 | 0.355 | 0.601 | 2.38 |
| **U-Net (base\_ch=32, 기본)** | **0.767** | **0.910** | **0.831** | **0.810** |
| U-Net (base\_ch=16, Ablation B) | 0.762 | 0.909 | 0.830 | 0.763 |

> U-Net F1 **0.831** — CA-CFAR 대비 **+23.0%p 향상**  
> Ablation B: 파라미터 75% 감소(7.76M → 1.94M)에도 F1 **0.830** 유지

---

## 환경

| 항목 | 값 |
|---|---|
| Python | 3.12 (Anaconda) |
| PyTorch | 2.6.0+cu124 |
| GPU | NVIDIA GTX 1660 Ti (6GB) |
| OS | Windows 11 |

```bash
# 의존 패키지 설치
pip install -r requirements.txt
```

---

## 실행 방법

### 1. 데이터 생성

```bash
cd projects/p01_unet_detector
python train.py --generate --epochs 0
```

### 2. 모델 학습 (기본, base_ch=32)

```bash
python train.py --generate --epochs 30
```

### 3. Ablation B — 경량 모델 (base_ch=16)

```bash
python train.py --epochs 30 --base_ch 16 --artifact_dir artifacts/unet_base16
```

### 4. 평가 — CA-CFAR

```bash
python evaluate_cfar.py --split val
python evaluate_cfar.py --split test --policy artifacts/verified_p01/p01_cfar_selected_val.json
```

### 5. 평가 — U-Net

```bash
python evaluate_unet.py --split val
python evaluate_unet.py --split test --policy artifacts/verified_p01/p01_unet_selected_val.json
```

### 6. 시각화 그림 생성

```bash
python make_report_figures.py
# → artifacts/report_figures/ 에 fig1~fig4 저장
```

---

## 모델 구조

**UNetDetector** — 인코더-디코더 구조 (skip connection 포함)

```
입력: [B, 2, H, W]  (실수부 + 허수부 채널)
  └─ Encoder: Conv→BN→ReLU×4 블록, 각 단계에서 2× 다운샘플
  └─ Bottleneck
  └─ Decoder: TransposeConv + skip concat → Conv 블록
출력: [B, 1, H, W]  (픽셀별 표적 확률, sigmoid)
```

| 설정 | 파라미터 수 | 학습 시간 |
|---|---|---|
| base\_ch=32 (기본) | 7,762,753 | 약 130분 (30 epoch) |
| base\_ch=16 (Ablation B) | 1,940,705 | 약 67분 (30 epoch) |

**손실 함수:** FocalDiceLoss = Focal(α=0.75, γ=2.0) + 0.5 × Dice  
**옵티마이저:** Adam (lr=3e-4) + CosineAnnealingLR  
**배치 크기:** 32 | **에폭:** 30 | **시드:** 42

---

## Ablation 실험

**목적:** 모델 크기(`base_ch`) 축소가 탐지 성능에 미치는 영향 분석

| 설정 | base\_ch | 파라미터 | F1 | Pd | FA/RDM |
|---|---|---|---|---|---|
| 기본 (Baseline) | 32 | 7.76M | **0.831** | 0.767 | 0.810 |
| **Ablation B** | **16** | **1.94M** | **0.830** | 0.762 | 0.763 |

**결론:** 파라미터를 75% 줄여도 F1 감소는 0.001에 불과. 경량화 효과가 뚜렷함.

---

## Validation-Locked 정책

과적합 방지를 위해 **검증 세트(val)에서만** threshold / CFAR 파라미터를 선택하고,  
해당 설정을 **테스트 세트(test)에 한 번만** 적용하는 방식을 사용했습니다.

- U-Net 최적 threshold: **0.25** (기본) / **0.55** (Ablation B)
- CA-CFAR 최적 Pfa: val 기반 선택 → test 적용

---

## 실험 결과 그림

| 그림 | 내용 |
|---|---|
| `fig1_learning_curves.png` | 기본 / Ablation B 학습 곡선 (Train·Val Loss) |
| `fig2_ablation_bar.png` | CA-CFAR · U-Net-32 · U-Net-16 성능 비교 막대 그래프 |
| `fig3_case_studies.png` | 성공 2건 + 실패 2건 탐지 사례 (RDM · GT · CFAR · U-Net) |
| `fig4_pd_pfa.png` | Pd vs Pfa 산점도 |

그림 파일 위치: `projects/p01_unet_detector/artifacts/report_figures/`

---

## 파일 구조

```
radar-ai-projects-main/
├── README.md                        ← 이 파일
├── requirements.txt
├── shared/                          ← 공통 유틸 (fmcw_simulator 등)
└── projects/
    └── p01_unet_detector/
        ├── train.py                 ← 학습 스크립트
        ├── model.py                 ← UNetDetector + FocalDiceLoss
        ├── evaluate_cfar.py         ← CA-CFAR 평가
        ├── evaluate_unet.py         ← U-Net 평가
        ├── make_report_figures.py   ← 시각화 그림 생성 (직접 작성)
        ├── p01_deep_learning_code.py ← 딥러닝 코드 설명 (직접 작성)
        ├── make_report_pdf.py       ← PDF 보고서 생성 (직접 작성)
        └── artifacts/
            ├── history.json         ← 기본 모델 학습 로그
            ├── unet_base16/
            │   └── history.json     ← Ablation B 학습 로그
            ├── verified_p01/        ← 평가 결과 JSON
            └── report_figures/      ← fig1~fig4 PNG
```

---

## 참고

- 과제 안내: `radar_ai_project_brief.md`
- 보고서: `P01_Report_Draft.pdf`
- 담당 교수: REMI Lab, 충남대학교
