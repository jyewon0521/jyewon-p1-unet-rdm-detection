#!/usr/bin/env python3
"""
P1: U-Net FMCW Range-Doppler 표적 탐지 — 딥러닝 코드 설명

이 파일은 P1 딥러닝 파이프라인의 핵심 코드를 설명용으로 정리한 것입니다.
실제 학습은 train.py를 실행하세요.

목차:
  1. 문제 정의
  2. U-Net 모델 구조
  3. 손실 함수 (FocalDiceLoss)
  4. 데이터 로더
  5. 학습 루프
  6. 평가 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

# ==============================================================================
# 1. 문제 정의
# ==============================================================================
#
# 입력: Range-Doppler Map (RDM) — 2채널 (log-magnitude + phase)
#   Shape: (Batch, 2, 64, 200)
#           ↑     ↑   ↑    ↑
#         배치  채널  속도축 거리축
#
# 출력: 표적 탐지 확률 맵
#   Shape: (Batch, 1, 64, 200)
#   값: 0 ~ 1 (1에 가까울수록 표적일 가능성 높음)
#
# 평가: 확률 맵에 임계값(threshold) 적용 → 이진 탐지 맵
#   예) threshold=0.25 → 0.25 이상인 픽셀을 표적으로 판단


# ==============================================================================
# 2. U-Net 모델 구조
# ==============================================================================

class ConvBlock(nn.Module):
    """기본 빌딩 블록: Conv2D × 2 + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDetector(nn.Module):
    """
    U-Net 기반 레이더 표적 탐지기.

    구조 다이어그램 (base_ch=32):
    ┌────────────────────────────────────────────────────┐
    │ 입력 (B, 2, 64, 200)                               │
    │                                                    │
    │ Encoder:                  Decoder:                 │
    │  enc1: 2→32  ──────────→  dec1: 64→32             │
    │  enc2: 32→64  ─────────→  dec2: 128→64            │
    │  enc3: 64→128  ────────→  dec3: 256→128           │
    │  enc4: 128→256  ───────→  dec4: 512→256           │
    │         ↓                                          │
    │  bottleneck: 256→512                               │
    │                                                    │
    │ 출력 (B, 1, 64, 200) → Sigmoid → 확률 맵           │
    └────────────────────────────────────────────────────┘

    핵심 아이디어: skip connection
    - 인코더의 특징맵을 디코더로 직접 연결
    - 위치 정보(어디에 표적이 있는지)가 보존됨
    """

    def __init__(self, in_channels: int = 2, base_ch: int = 32, dropout: float = 0.3):
        super().__init__()

        # 채널 수: base_ch를 기준으로 각 단계마다 2배씩 증가
        # base_ch=32 → [32, 64, 128, 256, 512]
        # base_ch=16 → [16, 32,  64, 128, 256]  ← Ablation B
        ch = [base_ch * (2 ** i) for i in range(5)]

        # ── 인코더 (특징 추출, 공간 크기 축소) ────────────────
        self.enc1 = ConvBlock(in_channels, ch[0])   # 입력 → 32ch
        self.enc2 = ConvBlock(ch[0], ch[1])          # 32 → 64ch
        self.enc3 = ConvBlock(ch[1], ch[2])          # 64 → 128ch
        self.enc4 = ConvBlock(ch[2], ch[3])          # 128 → 256ch
        self.pool = nn.MaxPool2d(2)                  # 공간 크기 절반으로

        # ── 병목 (bottleneck) ─────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBlock(ch[3], ch[4]),    # 256 → 512ch
            nn.Dropout2d(dropout),      # 과적합 방지
        )

        # ── 디코더 (공간 크기 복원 + skip connection 결합) ─────
        self.up4  = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)  # 업샘플링
        self.dec4 = ConvBlock(ch[3] * 2, ch[3])   # skip connection → 채널 2배 입력

        self.up3  = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch[2] * 2, ch[2])

        self.up2  = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch[1] * 2, ch[1])

        self.up1  = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch[0] * 2, ch[0])

        # ── 최종 출력 ──────────────────────────────────────────
        self.out_conv = nn.Conv2d(ch[0], 1, kernel_size=1)  # 1×1 conv → 1채널

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # 16의 배수로 패딩 (MaxPool2d × 4 = 16배 축소)
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 병목
        b = self.bottleneck(self.pool(e4))

        # 디코더 + skip connection (torch.cat으로 채널 방향 결합)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # 확률값 출력 (0~1)
        out = torch.sigmoid(self.out_conv(d1))

        # 패딩 제거
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out


# ==============================================================================
# 3. 손실 함수: FocalDiceLoss
# ==============================================================================
#
# 왜 일반 CrossEntropy를 안 쓰나?
#   RDM에서 표적 픽셀은 전체의 0.1% 미만 → 극심한 클래스 불균형
#   일반 BCE는 "전부 배경"이라고 예측해도 loss가 낮아지는 문제 발생
#
# FocalDiceLoss = Focal Loss + Dice Loss 혼합
#   Focal Loss: 맞추기 어려운 샘플(표적)에 더 높은 가중치
#   Dice Loss:  예측 마스크와 정답 마스크의 겹치는 비율 최대화

class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss 혼합 손실 함수.

    Parameters:
        alpha      : 표적(양성) 클래스 가중치 (0.75 → 표적에 75% 가중)
        gamma      : focusing 파라미터 (2.0 → 어려운 샘플 강조)
        dice_weight: Dice Loss 비중 (0.5 → Focal 50% + Dice 50%)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.dice_weight = dice_weight
        self.smooth      = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c = pred.clamp(1e-6, 1 - 1e-6)

        # ── Focal Loss ────────────────────────────────────────
        bce = -target * torch.log(pred_c) - (1 - target) * torch.log(1 - pred_c)
        pt      = target * pred_c + (1 - target) * (1 - pred_c)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal   = alpha_t * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        # ── Dice Loss ─────────────────────────────────────────
        pred_flat   = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice      = (2.0 * intersection + self.smooth) / \
                    (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        # ── 혼합 ──────────────────────────────────────────────
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


# ==============================================================================
# 4. 데이터셋
# ==============================================================================

class RDMDataset(Dataset):
    """
    HDF5 파일에서 RDM 데이터를 로드하는 데이터셋.

    HDF5 스키마:
        x: (N, 2, 64, 200) float16  ← 입력 (log-mag + phase)
        y: (N, 1, 64, 200) uint8    ← 정답 마스크 (0 or 1)
    """

    def __init__(self, h5_path: str):
        self.path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.n = len(f["x"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        with h5py.File(self.path, "r") as f:
            x = torch.as_tensor(f["x"][idx].astype(np.float32))
            y = torch.as_tensor(f["y"][idx].astype(np.float32))
        return x, y


# ==============================================================================
# 5. 학습 루프 (핵심 구조)
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """한 epoch 학습. train loss 반환."""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()       # 기울기 초기화
        pred = model(x)             # 순전파 (forward)
        loss = criterion(pred, y)   # 손실 계산
        loss.backward()             # 역전파 (backward)
        optimizer.step()            # 파라미터 업데이트

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """검증 loss 계산. 기울기 계산 없음."""
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


def train(data_dir: str = "data", epochs: int = 30, base_ch: int = 32,
          batch_size: int = 32, lr: float = 3e-4, device_str: str = "cuda"):
    """
    U-Net 학습 메인 함수.

    핵심 하이퍼파라미터:
        epochs=30     : 전체 데이터 30번 반복
        base_ch=32    : U-Net 채널 수 (Ablation B: 16)
        batch_size=32 : 한 번에 32개 RDM씩 처리
        lr=3e-4       : Adam 학습률
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    # 데이터 로더
    train_ds = RDMDataset(f"{data_dir}/det_train.h5")
    val_ds   = RDMDataset(f"{data_dir}/det_val.h5")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # 모델 / 손실 / 옵티마이저
    model     = UNetDetector(in_channels=2, base_ch=base_ch).to(device)
    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}"
              + (" *" if improved else ""))

    print(f"\n학습 완료. Best val loss: {best_val_loss:.4f}")
    return model


# ==============================================================================
# 6. 평가 함수
# ==============================================================================

@torch.no_grad()
def evaluate(model, data_path: str, threshold: float = 0.25,
             device_str: str = "cuda") -> dict:
    """
    U-Net 탐지 성능 평가.

    threshold: 이 값 이상이면 표적으로 판단
               val sweep에서 최적값 선택 → test에 적용
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    tp = fp = fn = tn = 0

    with h5py.File(data_path, "r") as f:
        n = len(f["x"])
        bs = 32
        for start in range(0, n, bs):
            x = torch.as_tensor(
                f["x"][start:start+bs].astype(np.float32)
            ).to(device)
            y = f["y"][start:start+bs].astype(bool)

            prob = model(x).cpu().numpy()[:, 0]   # (B, H, W)
            pred = prob > threshold                 # 이진 탐지

            tp += int((pred &  y).sum())
            fp += int((pred & ~y).sum())
            fn += int((~pred &  y).sum())
            tn += int((~pred & ~y).sum())

    pd_val  = tp / (tp + fn + 1e-9)
    pfa_val = fp / (fp + tn + 1e-9)
    prec    = tp / (tp + fp + 1e-9)
    f1      = 2 * prec * pd_val / (prec + pd_val + 1e-9)

    metrics = {"Pd": pd_val, "Pfa": pfa_val, "Precision": prec, "F1": f1,
               "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    print(f"Pd={pd_val:.4f}  Pfa={pfa_val:.2e}  Prec={prec:.4f}  F1={f1:.4f}")
    return metrics


# ==============================================================================
# 실행 예시
# ==============================================================================
if __name__ == "__main__":
    # 모델 구조 확인
    print("=" * 60)
    print("U-Net 모델 구조 확인")
    print("=" * 60)

    for base_ch, name in [(32, "Baseline"), (16, "Ablation B")]:
        model = UNetDetector(in_channels=2, base_ch=base_ch)
        x_test = torch.randn(2, 2, 64, 200)
        y_test = model(x_test)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] base_ch={base_ch}")
        print(f"  입력:  {tuple(x_test.shape)}")
        print(f"  출력:  {tuple(y_test.shape)}")
        print(f"  파라미터: {n_params:,}개 ({n_params/1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("손실 함수 동작 확인")
    print("=" * 60)
    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)
    pred   = torch.rand(2, 1, 64, 200)   # 랜덤 예측
    target = torch.zeros(2, 1, 64, 200)
    target[:, :, 10:15, 50:55] = 1.0     # 작은 표적 영역만 1
    loss = criterion(pred, target)
    print(f"  Loss 값: {loss.item():.4f}")
    print("  (랜덤 예측이라 높게 나오는 게 정상)")
