#!/usr/bin/env python3
"""P01 보고서 PDF 생성 스크립트.

실행:
  python make_report_pdf.py

출력:
  ../../P01_Report_Draft.pdf  (프로젝트 루트 폴더)
"""
from __future__ import annotations
from pathlib import Path
from fpdf import FPDF

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
FIGURES  = ROOT / "artifacts" / "report_figures"
OUT_PDF  = ROOT.parents[1] / "P01_Report_Draft.pdf"
FONT_REG = r"C:\Windows\Fonts\malgun.ttf"    # 맑은고딕 Regular
FONT_BD  = r"C:\Windows\Fonts\malgunbd.ttf"  # 맑은고딕 Bold


# ── PDF 클래스 ───────────────────────────────────────────────────────────────
class ReportPDF(FPDF):
    def header(self):
        pass  # 섹션별로 직접 제어

    def footer(self):
        self.set_y(-13)
        self.set_font("malgun", size=9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")
        self.set_text_color(0, 0, 0)


def build_pdf():
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.add_font("malgun",   fname=FONT_REG)
    pdf.add_font("malgun",   style="B", fname=FONT_BD)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 20, 20)

    W = 170  # 본문 너비 (mm)

    # ════════════════════════════════════════════════════════
    # 표지 / 제목
    # ════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("malgun", style="B", size=18)
    pdf.multi_cell(W, 10, "P1: U-Net FMCW Range-Doppler 표적 탐지", align="C")
    pdf.ln(6)
    pdf.set_font("malgun", size=13)
    pdf.multi_cell(W, 8, "레이다 신호처리 및 인공지능 — 14주차 최종 과제 보고서", align="C")
    pdf.ln(10)
    pdf.set_font("malgun", size=11)
    pdf.multi_cell(W, 7, "충남대학교 REMI Lab", align="C")
    pdf.ln(4)
    pdf.multi_cell(W, 7, "학번: _____________   이름: _____________", align="C")
    pdf.ln(50)
    pdf.set_font("malgun", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(W, 6,
        "※ 이 문서는 초안입니다. 워드에서 직접 수정하여 사용하세요.\n"
        "그림은 artifacts/report_figures/ 폴더에 있습니다.", align="C")
    pdf.set_text_color(0, 0, 0)

    # ════════════════════════════════════════════════════════
    # 헬퍼 함수들
    # ════════════════════════════════════════════════════════
    def section_title(text: str):
        pdf.ln(6)
        pdf.set_font("malgun", style="B", size=13)
        pdf.set_fill_color(230, 235, 245)
        pdf.cell(W, 8, text, fill=True, ln=True)
        pdf.ln(2)

    def subsection(text: str):
        pdf.ln(3)
        pdf.set_font("malgun", style="B", size=11)
        pdf.cell(W, 7, text, ln=True)
        pdf.ln(1)

    def body(text: str, size: int = 10, line_h: float = 6):
        pdf.set_font("malgun", size=size)
        pdf.multi_cell(W, line_h, text)
        pdf.ln(1)

    def table(headers: list, rows: list, col_widths: list):
        pdf.set_font("malgun", style="B", size=9)
        pdf.set_fill_color(60, 90, 160)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)
        for i, row in enumerate(rows):
            pdf.set_font("malgun", size=9)
            pdf.set_fill_color(245, 248, 255) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            for val, w in zip(row, col_widths):
                pdf.cell(w, 6.5, str(val), border=1, fill=True, align="C")
            pdf.ln()
        pdf.ln(2)

    def insert_figure(fname: str, caption: str, w_mm: float = 150):
        path = FIGURES / fname
        if path.exists():
            pdf.ln(3)
            x = (210 - w_mm) / 2
            pdf.image(str(path), x=x, w=w_mm)
            pdf.ln(2)
            pdf.set_font("malgun", size=9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(W, 5, caption, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
        else:
            pdf.set_font("malgun", size=9)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(W, 6, f"[그림 없음: {fname}]", ln=True)
            pdf.set_text_color(0, 0, 0)

    # ════════════════════════════════════════════════════════
    # Abstract
    # ════════════════════════════════════════════════════════
    pdf.add_page()
    section_title("Abstract")
    body(
        "본 연구는 FMCW(Frequency Modulated Continuous Wave) 레이더의 Range-Doppler Map(RDM)에서 "
        "이동 표적을 픽셀 단위로 탐지하는 딥러닝 기반 접근법을 제안하고 평가한다. "
        "고전적 탐지기인 CA-CFAR(Cell-Averaging Constant False Alarm Rate)를 베이스라인으로 설정하고, "
        "이를 대체하는 U-Net 세그멘테이션 모델을 학습하였다. "
        "5만 개의 시뮬레이션 RDM 데이터를 활용하여 30 epoch 학습한 결과, "
        "U-Net은 테스트 셋에서 F1 score 0.831을 달성하였으며, 이는 CA-CFAR(F1=0.601) 대비 38% 향상된 수치이다. "
        "또한 모델 파라미터를 75% 축소한 ablation 실험(base_ch=16, 1.94M params)에서도 F1=0.830으로 "
        "성능이 거의 유지됨을 확인하였다. "
        "이를 통해 U-Net의 skip connection 구조가 제한된 파라미터 환경에서도 RDM의 "
        "공간적 패턴을 효과적으로 학습함을 보였다."
    )

    # ════════════════════════════════════════════════════════
    # 1. Introduction
    # ════════════════════════════════════════════════════════
    section_title("1. Introduction")
    body(
        "레이더 신호처리에서 표적 탐지는 수십 년간 연구되어 온 핵심 과제이다. "
        "전통적으로는 CA-CFAR와 같은 통계 기반 방법이 널리 사용되어 왔는데, "
        "이는 주변 셀의 배경 잡음 수준을 추정하여 탐지 임계값을 동적으로 결정하는 방식이다. "
        "그러나 CA-CFAR는 클러터 환경이 복잡하거나 근접 표적이 존재할 때 성능이 저하되는 한계가 있다."
    )
    body(
        "최근 딥러닝의 발전과 함께, Range-Doppler Map을 2D 이미지로 취급하여 "
        "세그멘테이션 네트워크를 적용하는 연구들이 주목받고 있다. "
        "U-Net은 원래 의료 영상 분할을 위해 제안된 구조이지만, skip connection을 통해 "
        "세밀한 위치 정보를 보존한다는 장점 덕분에 레이더 탐지에도 효과적으로 활용될 수 있다."
    )
    body(
        "본 연구는 다음의 research question을 설정하였다:\n"
        "\"U-Net 기반 탐지기는 MTI 전처리된 RDM에서 CA-CFAR보다 얼마나 더 나은 탐지 성능을 보이는가? "
        "그리고 모델 크기(파라미터 수)가 탐지 성능에 미치는 영향은 어느 정도인가?\""
    )

    # ════════════════════════════════════════════════════════
    # 2. Method
    # ════════════════════════════════════════════════════════
    section_title("2. Method")

    subsection("2.1 신호 처리 파이프라인")
    body(
        "본 실험의 데이터 생성 과정은 다음의 순서로 구성된다. FMCW 레이더 시뮬레이터(77 GHz, BW=150 MHz)를 "
        "통해 이동 표적과 정적 클러터가 포함된 beat 신호를 생성하고, complex 16-bit I/Q 양자화를 적용한다. "
        "이후 slow-time mean-removal MTI/DC-notch 필터로 정적 클러터를 억제한 뒤 Range-Doppler Map을 생성한다. "
        "CA-CFAR와 U-Net은 동일한 MTI-filtered RDM을 입력으로 사용하며, "
        "이를 통해 공정한 비교가 가능하다."
    )
    body(
        "레이블 생성 기준(schema-v9)은 다음과 같다. 표적 bin의 처리 후 peak가 전역 RD 중앙값 및 "
        "지역 CFAR 유사 배경보다 6 dB 이상 높을 때만 양성 레이블을 부여하며, "
        "정적 클러터와의 혼동을 피하기 위해 도플러 속도 |v| >= 4.88 m/s인 표적만 레이블링한다. "
        "마스크는 표적 중심 셀과 상하좌우 인접 셀로 구성된 5-cell cross 형태이다."
    )

    subsection("2.2 U-Net 모델 구조")
    body(
        "본 연구에서 사용한 모델은 UNetDetector로, 5단계 인코더/디코더 구조와 skip connection을 갖는 "
        "U-Net 변형체이다.\n\n"
        "  - 입력: (B, 2, 64, 200) — RDM log-magnitude와 phase를 각각 하나의 채널로 구성한 2채널 텐서\n"
        "  - 출력: (B, 1, 64, 200) — 픽셀별 표적 탐지 확률 맵 (0~1)\n"
        "  - 인코더: 각 단계에서 MaxPool2d(2)로 공간 크기 절반 축소, 채널 수 2배 증가\n"
        "            (base_ch=32 기준: 32→64→128→256→512)\n"
        "  - 디코더: ConvTranspose2d로 업샘플링 후 skip connection으로 위치 정보 복원\n"
        "  - 병목(bottleneck): Dropout2d(p=0.3)로 과적합 방지\n"
        "  - 출력 레이어: 1x1 Conv2d → Sigmoid\n\n"
        "  베이스라인 모델 (base_ch=32): 파라미터 7,762,753개\n"
        "  Ablation B 모델 (base_ch=16): 파라미터 1,942,433개"
    )

    subsection("2.3 손실 함수 (FocalDiceLoss)")
    body(
        "RDM에서 표적 픽셀은 전체의 0.1% 미만으로 극심한 클래스 불균형이 존재한다. "
        "이를 해결하기 위해 Focal Loss와 Dice Loss를 결합한 FocalDiceLoss를 사용한다.\n\n"
        "  FocalDiceLoss = (1 - w_dice) x FocalLoss + w_dice x DiceLoss\n\n"
        "  - FocalLoss: 맞추기 어려운 표적 픽셀에 높은 가중치 부여 (alpha=0.75, gamma=2.0)\n"
        "  - DiceLoss: 예측 마스크와 정답 마스크의 겹치는 비율을 최대화 (smooth=1.0)\n"
        "  - w_dice = 0.5 (두 손실 함수를 동등하게 혼합)"
    )

    subsection("2.4 학습 설정 및 데이터 분할")
    table(
        headers=["항목", "값"],
        rows=[
            ["Train / Val / Test", "50,000 / 5,000 / 5,000"],
            ["Batch size", "32"],
            ["Learning rate", "3e-4 (Adam)"],
            ["Epochs", "30"],
            ["LR scheduler", "CosineAnnealingLR"],
            ["Random seed", "42"],
            ["입력 dtype", "float16"],
            ["학습 장치", "NVIDIA GTX 1660 Ti (CUDA 12.4)"],
        ],
        col_widths=[80, 90],
    )
    body(
        "탐지 임계값(threshold)은 검증 셋(val)에서 F1이 최대가 되는 값으로 선택하고, "
        "해당 값을 테스트 셋에 고정 적용하였다. "
        "베이스라인 threshold=0.25, Ablation B threshold=0.55. "
        "CA-CFAR 파라미터 역시 검증 셋에서 sweep하여 최적값(guard=(1,1), train=(4,4), pfa_design=1e-5)을 "
        "선택하고 테스트 셋에 적용하였다."
    )

    # ════════════════════════════════════════════════════════
    # 3. Experiments
    # ════════════════════════════════════════════════════════
    section_title("3. Experiments")

    subsection("3.1 평가 지표")
    table(
        headers=["지표", "설명"],
        rows=[
            ["Pd", "픽셀 수준 탐지율: TP / (TP + FN)"],
            ["Pfa", "픽셀 수준 오경보율: FP / (FP + TN)"],
            ["Precision", "TP / (TP + FP)"],
            ["F1", "2 x Precision x Pd / (Precision + Pd)"],
            ["FA/RDM", "RDM 샘플당 평균 오탐 셀 수"],
        ],
        col_widths=[35, 135],
    )

    subsection("3.2 실험 구성")
    body(
        "모든 실험에서 데이터 split, random seed, epoch 수, batch size를 동일하게 유지하였다. "
        "Ablation B는 base_ch만 32→16으로 변경하고 나머지는 모두 동일하게 설정하였다."
    )
    table(
        headers=["실험", "모델", "변경 사항"],
        rows=[
            ["Baseline-CFAR", "CA-CFAR", "guard=(1,1), train=(4,4), pfa=1e-5"],
            ["Baseline-UNet", "U-Net (base_ch=32, 7.76M)", "없음 (기준)"],
            ["Ablation B",   "U-Net (base_ch=16, 1.94M)", "base_ch만 변경"],
        ],
        col_widths=[38, 72, 60],
    )

    # ════════════════════════════════════════════════════════
    # 4. Results & Discussion
    # ════════════════════════════════════════════════════════
    pdf.add_page()
    section_title("4. Results & Discussion")

    subsection("4.1 베이스라인 재현")
    body(
        "아래 표는 테스트 셋 5,000개에서 validation-locked policy로 평가한 최종 결과이다."
    )
    table(
        headers=["방법", "Pd", "Pfa", "Precision", "F1", "FA/RDM"],
        rows=[
            ["CA-CFAR",             "0.529", "1.86e-04", "0.695", "0.601", "2.38"],
            ["U-Net (base_ch=32)",  "0.767", "6.33e-05", "0.907", "0.831", "0.810"],
            ["U-Net (base_ch=16)",  "0.762", "5.96e-05", "0.911", "0.830", "0.763"],
        ],
        col_widths=[52, 22, 26, 26, 22, 22],
    )
    body(
        "U-Net은 CA-CFAR 대비 Pd가 45% 향상되었고(0.529→0.767), "
        "오경보율은 66% 감소하였다(1.86e-04→6.33e-05). "
        "F1 score는 0.601에서 0.831로 크게 개선되었으며, "
        "이는 공개된 참고 수치(F1=0.832)와 거의 동일하여 재현 성공으로 판단한다.\n\n"
        "U-Net이 CA-CFAR보다 우수한 이유는 다음과 같이 해석할 수 있다. "
        "CA-CFAR는 각 셀을 독립적으로 판단하기 때문에, 근접 표적이 서로의 배경 추정에 간섭하거나 "
        "클러터 경계 부근에서 오탐지가 발생하기 쉽다. "
        "반면 U-Net은 skip connection을 통해 RDM 전체의 공간적 맥락을 활용하므로, "
        "낮은 SNR에서도 표적의 특징적인 패턴을 더 효과적으로 식별한다."
    )

    subsection("4.2 Ablation B: 모델 크기 축소 (base_ch 32→16)")
    body(
        "Ablation B 결과, 파라미터를 75% 축소(7.76M→1.94M)하였음에도 "
        "F1이 0.831에서 0.830으로 0.001밖에 차이가 나지 않았다. "
        "오히려 Precision은 0.907→0.911로, False alarms/RDM은 0.810→0.763으로 소폭 개선되었다.\n\n"
        "이 결과는 MTI-filtered RDM에서의 이동 표적 탐지 문제가 비교적 간단한 표현으로도 "
        "충분히 풀릴 수 있음을 시사한다. 즉, 이 태스크에서 U-Net의 성능 병목은 파라미터 수가 아니라 "
        "학습 데이터의 다양성이나 손실 함수 설계에 있을 가능성이 높다.\n\n"
        "또한 학습 시간이 7,827초(130분)에서 4,046초(67분)로 48% 단축되었다는 점도 "
        "실용적 관점에서 의미 있는 결과이다.\n\n"
        "주목할 점은 최적 임계값이 베이스라인 0.25에서 Ablation B에서 0.55로 높아진 것이다. "
        "이는 작은 모델이 불확실한 픽셀에 대해 더 낮은 확률값을 출력하는 경향이 있음을 나타내며, "
        "임계값 선택의 중요성을 보여준다."
    )

    subsection("4.3 학습 곡선 분석")
    insert_figure("fig1_learning_curves.png",
                  "Figure 1. 학습 곡선. 실선=train loss, 점선=val loss. "
                  "세로 점선은 best val epoch.")
    body(
        "두 모델 모두 학습 초반(epoch 1~5)에 loss가 급격히 감소하고 이후 완만하게 수렴하는 "
        "전형적인 패턴을 보였다. Baseline(base_ch=32)의 best val loss는 epoch 25에서 0.0876이었으며, "
        "Ablation B(base_ch=16)는 epoch 25~30 사이에서 0.0868로 오히려 더 낮았다. "
        "이는 작은 모델이 과적합 없이 더 안정적으로 수렴하였음을 나타낸다."
    )

    subsection("4.4 Ablation 비교 그래프")
    insert_figure("fig2_ablation_bar.png",
                  "Figure 2. Ablation 실험 비교 막대 그래프 (Test Set, 5,000 samples).")
    insert_figure("fig4_pd_pfa.png",
                  "Figure 3. Pd vs Pfa 포인트 비교. 오른쪽 위로 갈수록 좋은 성능.")

    subsection("4.5 대표 사례 분석")
    insert_figure("fig3_case_studies.png",
                  "Figure 4. 대표 사례. 각 행: 입력 RDM / GT 마스크 / CA-CFAR 탐지 / U-Net 탐지.\n"
                  "상위 2행: 높은 SNR (성공 사례), 하위 2행: 낮은 SNR (실패 사례).")
    body(
        "성공 사례 (높은 SNR): 높은 SNR 환경에서는 U-Net과 CA-CFAR 모두 표적을 정확히 탐지한다. "
        "그러나 CA-CFAR는 표적 주변에 산발적인 오탐지가 발생하는 반면, "
        "U-Net은 더 정확한 마스크를 출력하였다.\n\n"
        "실패 사례 (낮은 SNR): 낮은 SNR 환경에서는 두 방법 모두 탐지 성능이 저하된다. "
        "U-Net은 낮은 SNR에서도 CA-CFAR보다 더 적은 오탐지를 보이지만, "
        "일부 표적을 미탐지(false negative)하는 경향이 있었다. "
        "이는 레이블 게이트(6 dB 기준)로 인해 학습 데이터에서 저SNR 표적 레이블이 "
        "제한적이기 때문으로 해석된다."
    )

    # ════════════════════════════════════════════════════════
    # 5. Conclusion
    # ════════════════════════════════════════════════════════
    pdf.add_page()
    section_title("5. Conclusion")
    body(
        "본 연구는 FMCW RDM 기반 이동 표적 탐지 문제에서 U-Net이 CA-CFAR를 크게 상회하는 "
        "성능을 달성함을 실험적으로 확인하였다. 테스트 셋 5,000개에서 U-Net(base_ch=32)은 "
        "F1=0.831, CA-CFAR는 F1=0.601을 기록하였다. "
        "또한 파라미터를 75% 축소한 ablation 실험(base_ch=16)에서도 F1=0.830으로 성능이 "
        "거의 유지됨을 확인하였다.\n\n"
        "본 연구의 한계점은 다음과 같다.\n\n"
        "  1. 본 실험의 클러터는 정적(static) 클러터만 포함하며, 실제 환경의 동적 클러터에 대한 "
        "일반화 성능은 검증되지 않았다.\n\n"
        "  2. 학습 데이터가 시뮬레이터로 생성된 합성 데이터이므로, 실제 레이더 시스템에서의 "
        "성능은 별도 검증이 필요하다.\n\n"
        "  3. 낮은 SNR 환경에서의 미탐지 문제는 해결되지 않았으며, 데이터 증강 또는 "
        "손실 함수 개선을 통한 추가 연구가 필요하다."
    )

    # ════════════════════════════════════════════════════════
    # 6. References
    # ════════════════════════════════════════════════════════
    section_title("References")
    refs = [
        "[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks "
        "for biomedical image segmentation. MICCAI, 234-241.",
        "[2] Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV, 2980-2988.",
        "[3] Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural "
        "networks for volumetric medical image segmentation. 3DV, 565-571.",
        "[4] Richards, M. A., Scheer, J. A., & Holm, W. A. (Eds.). (2010). Principles of Modern "
        "Radar: Basic Principles. SciTech Publishing.",
        "[5] Kronauge, M., & Rohling, H. (2013). Fast two-dimensional CFAR procedure. "
        "IEEE Transactions on Aerospace and Electronic Systems, 49(3), 1817-1823.",
    ]
    for ref in refs:
        pdf.set_font("malgun", size=9)
        pdf.multi_cell(W, 5.5, ref)
        pdf.ln(1)

    # ════════════════════════════════════════════════════════
    # 저장
    # ════════════════════════════════════════════════════════
    pdf.output(str(OUT_PDF))
    print(f"PDF 저장 완료: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
