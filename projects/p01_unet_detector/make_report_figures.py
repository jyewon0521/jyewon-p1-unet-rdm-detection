#!/usr/bin/env python3
"""P01 Report Figures Generator.

Figures:
  1. Learning curves (baseline vs ablation B)
  2. Ablation bar chart (CA-CFAR / U-Net-32 / U-Net-16)
  3. Pd/Pfa comparison
  4. Case studies (success x2 + failure x2)

Usage:
  python make_report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import UNetDetector
from shared.fmcw_simulator import ca_cfar_2d

ROOT = Path(__file__).parent
OUT  = ROOT / "artifacts" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

C_CFAR   = "#2563eb"
C_BASE32 = "#dc2626"
C_BASE16 = "#16a34a"

def load(p: Path) -> dict:
    return json.loads(p.read_text())

def sel(d: dict) -> dict:
    """selected_policy가 있으면 그걸, 없으면 d 자체를 반환."""
    return d.get("selected_policy", d)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: 학습 곡선
# ══════════════════════════════════════════════════════════════════════════════
def fig_learning_curves():
    h32 = load(ROOT / "artifacts" / "history.json")
    h16 = load(ROOT / "artifacts" / "unet_base16" / "history.json")

    epochs32 = list(range(1, len(h32["train_loss"]) + 1))
    epochs16 = list(range(1, len(h16["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig.suptitle("Figure 1. Learning Curves (Train / Val Loss)", fontsize=13, fontweight="bold")

    for ax, h, epochs, label, color in [
        (axes[0], h32, epochs32, "U-Net base_ch=32 (Baseline, 7.7M params)", C_BASE32),
        (axes[1], h16, epochs16, "U-Net base_ch=16 (Ablation B, 1.9M params)", C_BASE16),
    ]:
        ax.plot(epochs, h["train_loss"], color=color, lw=2, label="Train loss")
        ax.plot(epochs, h["val_loss"],   color=color, lw=2, ls="--", alpha=0.7, label="Val loss")
        best_ep = int(np.argmin(h["val_loss"])) + 1
        best_vl = min(h["val_loss"])
        ax.axvline(best_ep, color="gray", ls=":", lw=1.2, alpha=0.8)
        ax.scatter([best_ep], [best_vl], s=60, color=color, zorder=5,
                   label=f"Best val epoch={best_ep} ({best_vl:.4f})")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (FocalDiceLoss)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(epochs))

    path = OUT / "fig1_learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Ablation 막대 그래프
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_bar():
    cfar = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_cfar_selected_test.json"))
    u32  = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_unet_selected_test.json"))
    u16  = sel(load(ROOT / "artifacts" / "unet_base16"  / "p01_unet_base16_selected_test.json"))

    methods = ["CA-CFAR", "U-Net\nbase_ch=32\n(7.7M)", "U-Net\nbase_ch=16\n(1.9M)"]
    colors  = [C_CFAR, C_BASE32, C_BASE16]

    metrics = {
        "Pd (Detection Rate)":       [cfar["Pd"],        u32["Pd"],        u16["Pd"]],
        "Precision":                  [cfar["Precision"], u32["Precision"], u16["Precision"]],
        "F1 Score":                   [cfar["F1"],        u32["F1"],        u16["F1"]],
        "False Alarms / RDM\n(lower is better)": [cfar["false_alarms_per_rdm"],
                                                   u32["false_alarms_per_rdm"],
                                                   u16["false_alarms_per_rdm"]],
    }

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle("Figure 2. Ablation Study Comparison (Test Set, 5,000 samples)", fontsize=13, fontweight="bold")

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=1.2, alpha=0.88)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(metric_name, fontsize=10)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.18)
        ax.tick_params(axis="x", labelsize=8)

    path = OUT / "fig2_ablation_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: 대표 사례 (성공 2 + 실패 2)
# ══════════════════════════════════════════════════════════════════════════════
def fig_case_studies():
    cfar_pol  = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_cfar_selected_test.json"))
    unet_pol  = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_unet_selected_test.json"))
    guard     = tuple(cfar_pol["guard"])
    train_w   = tuple(cfar_pol["train"])
    pfa       = float(cfar_pol["pfa_design"])
    threshold = float(unet_pol["threshold"])

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDetector(in_channels=2, base_ch=32).to(device)
    ckpt  = ROOT / "artifacts" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # 테스트 데이터에서 사례 선택
    data_path = ROOT / "data" / "det_test.h5"
    with h5py.File(data_path, "r") as f:
        snr_all = f["snr_db"][:]
        n = len(snr_all)
        # 높은 SNR (성공 사례 후보) / 낮은 SNR (실패 사례 후보)
        order = np.argsort(snr_all)
        # 상위 SNR 2개, 하위 SNR 2개
        high_idx = [int(order[int(n * 0.95)]), int(order[int(n * 0.85)])]
        low_idx  = [int(order[int(n * 0.05)]), int(order[int(n * 0.15)])]
        picks = high_idx + low_idx
        labels_case = ["Success 1 (High SNR)", "Success 2 (High SNR)",
                       "Failure 1 (Low SNR)",  "Failure 2 (Low SNR)"]

        scenes = []
        for idx in picks:
            x       = f["x"][idx].astype(np.float32)
            gt      = f["y"][idx, 0] > 0.5
            rdm_mag = f["rdm_mag_linear"][idx]
            snr     = float(f["snr_db"][idx])
            scenes.append((x, gt, rdm_mag, snr))

    # 4 rows x 4 cols: row=case, col=RDM/GT/CFAR/UNet
    fig = plt.figure(figsize=(14, 13))
    fig.suptitle("Figure 3. Case Studies: CA-CFAR vs U-Net", fontsize=13, fontweight="bold")
    col_titles = ["Input RDM (log-mag)", "Ground Truth Mask", "CA-CFAR Detection", "U-Net Detection"]

    for row, ((x, gt, rdm_mag, snr), case_label) in enumerate(zip(scenes, labels_case)):
        # CFAR 탐지
        cfar_det = ca_cfar_2d(rdm_mag, guard=guard, train=train_w, pfa=pfa)
        # U-Net 추론
        with torch.no_grad():
            tx   = torch.as_tensor(x[None], dtype=torch.float32, device=device)
            prob = model(tx).cpu().numpy()[0, 0]
        unet_det = prob > threshold

        panels = [x[0], gt.astype(float), cfar_det.astype(float), unet_det.astype(float)]
        cmaps  = ["viridis", "Greens", "Oranges", "Reds"]

        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = fig.add_subplot(4, 4, row * 4 + col + 1)
            ax.imshow(panel, aspect="auto", cmap=cmap, origin="lower")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{case_label}\nSNR={snr:.1f} dB", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Show per-sample metrics on the U-Net panel
        tp = int((unet_det & gt).sum())
        fp = int((unet_det & ~gt).sum())
        fn = int((~unet_det & gt).sum())
        pd_val  = tp / (tp + fn + 1e-9)
        pre_val = tp / (tp + fp + 1e-9)
        ax_last = fig.add_subplot(4, 4, row * 4 + 4)
        ax_last.text(0.5, -0.12, f"Pd={pd_val:.2f}  Prec={pre_val:.2f}",
                     ha="center", va="top", transform=ax_last.transAxes, fontsize=7, color="darkred")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "fig3_case_studies.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def fig_pd_pfa():
    cfar = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_cfar_selected_test.json"))
    u32  = sel(load(ROOT / "artifacts" / "verified_p01" / "p01_unet_selected_test.json"))
    u16  = sel(load(ROOT / "artifacts" / "unet_base16"  / "p01_unet_base16_selected_test.json"))

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    fig.suptitle("Figure 4. Pd vs Pfa (Test Set)", fontsize=13, fontweight="bold")

    for result, label, color, marker in [
        (cfar, "CA-CFAR",              C_CFAR,   "o"),
        (u32,  "U-Net base_ch=32",     C_BASE32,  "s"),
        (u16,  "U-Net base_ch=16",     C_BASE16,  "^"),
    ]:
        pd_val  = result["Pd"]
        pfa_val = result["Pfa"]
        f1_val  = result["F1"]
        ax.scatter([pfa_val], [pd_val], s=120, color=color, marker=marker, zorder=5,
                   label=f"{label}  Pd={pd_val:.3f}  F1={f1_val:.3f}")
        ax.annotate(f"  F1={f1_val:.3f}", xy=(pfa_val, pd_val), fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Pfa (False Alarm Rate, lower is better)", fontsize=10)
    ax.set_ylabel("Pd (Detection Rate, higher is better)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(0, 1.05)

    path = OUT / "fig4_pd_pfa.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


if __name__ == "__main__":
    print("=== P01 Report Figures ===")
    print()
    print("[1/4] Learning curves...")
    fig_learning_curves()

    print("[2/4] Ablation bar chart...")
    fig_ablation_bar()

    print("[3/4] Pd/Pfa comparison...")
    fig_pd_pfa()

    print("[4/4] 대표 사례 시각화 (시간이 조금 걸립니다)...")
    fig_case_studies()

    print()
    print(f"=== Done! Figures saved to: {OUT} ===")
