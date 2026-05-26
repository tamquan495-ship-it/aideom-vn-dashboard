# -*- coding: utf-8 -*-
"""
AIDEOM-VN Decision Models Dashboard
Web Streamlit cho toàn bộ 12 bài thực hành Mô hình ra quyết định
Phát triển kinh tế Việt Nam trong kỷ nguyên AI.

Chạy:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog, minimize


# =============================================================================
# 1. CẤU HÌNH GIAO DIỆN
# =============================================================================

st.set_page_config(
    page_title="VN AIDEOM-VN | 12 bài mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0f172a"
CARD = "#111827"
CARD2 = "#182235"
BORDER = "rgba(148, 163, 184, 0.22)"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
PINK = "#ec407a"
PURPLE = "#7c3aed"
CYAN = "#22d3ee"
GREEN = "#34d399"
YELLOW = "#fbbf24"
RED = "#fb7185"


def inject_css() -> None:
    """CSS mô phỏng giao diện nền tối giống ảnh mẫu."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 5%, rgba(236,64,122,0.18) 0, transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(124,58,237,0.15) 0, transparent 32%),
                linear-gradient(180deg, #0b1020 0%, #0f172a 55%, #0b1020 100%);
            color: {TEXT};
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #111827 0%, #172033 58%, #111827 100%);
            border-right: 1px solid {BORDER};
        }}

        section[data-testid="stSidebar"] * {{
            color: #e5e7eb;
        }}

        .block-container {{
            padding-top: 3.8rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }}

        h1, h2, h3 {{
            letter-spacing: -0.03em;
            color: #fff;
        }}

        h1 {{
            font-size: 2.35rem !important;
            font-weight: 800 !important;
            line-height: 1.14 !important;
        }}

        h2 {{
            font-size: 1.5rem !important;
            margin-top: 1.4rem !important;
        }}

        h3 {{
            font-size: 1.18rem !important;
        }}

        .subtitle {{
            color: #dbeafe;
            font-size: 1rem;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }}

        .muted {{
            color: {MUTED};
            font-size: 0.9rem;
        }}

        .hero {{
            padding: 0.8rem 0 0.3rem 0;
        }}

        .badge-row {{
            display: flex;
            gap: .65rem;
            align-items: center;
            flex-wrap: wrap;
            margin: .7rem 0 1rem 0;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .36rem .72rem;
            border-radius: 999px;
            background: rgba(15,23,42,.86);
            border: 1px solid {BORDER};
            color: #e2e8f0;
            font-size: .78rem;
            font-weight: 700;
        }}

        .badge-hot {{
            background: linear-gradient(90deg, {PINK}, {PURPLE});
            color: white;
            border: 0;
        }}

        .card {{
            background: linear-gradient(180deg, rgba(30,41,59,0.86), rgba(15,23,42,0.76));
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 20px 45px rgba(0,0,0,0.23);
            margin-bottom: 1rem;
        }}

        .metric-card {{
            background: rgba(15,23,42,.72);
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1rem;
            min-height: 104px;
        }}

        .metric-label {{
            color: {MUTED};
            font-size: .80rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .4rem;
        }}

        .metric-value {{
            font-size: 1.8rem;
            line-height: 1.2;
            color: white;
            font-weight: 800;
        }}

        .metric-help {{
            color: #bfdbfe;
            font-size: .82rem;
            margin-top: .35rem;
        }}

        div.stButton > button:first-child {{
            border: 0;
            color: white;
            font-weight: 800;
            border-radius: 999px;
            padding: .55rem 1.1rem;
            background: linear-gradient(90deg, {PINK}, {PURPLE});
            box-shadow: 0 10px 24px rgba(236,64,122,.25);
        }}

        div.stButton > button:first-child:hover {{
            transform: translateY(-1px);
            filter: brightness(1.08);
            color: white;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid {BORDER};
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            background: rgba(15,23,42,.65);
            border: 1px solid {BORDER};
            border-bottom: 0;
            border-radius: 12px 12px 0 0;
            color: #cbd5e1;
            font-weight: 800;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, {PINK}, {PURPLE});
            color: white;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 16px;
            overflow: hidden;
        }}

        hr {{
            border: 0;
            border-top: 1px solid {BORDER};
            margin: 1.3rem 0;
        }}

        .formula {{
            background: rgba(2, 6, 23, .45);
            border: 1px solid {BORDER};
            color: #e0f2fe;
            border-radius: 14px;
            padding: .8rem 1rem;
            margin: .8rem 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: .95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_block(title: str, level: str, badges: List[str], subtitle: str) -> None:
    badges_html = "".join(
        f'<span class="badge {"badge-hot" if i == 0 else ""}">{b}</span>'
        for i, b in enumerate(badges)
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <div class="badge-row">
                <span class="badge badge-hot">{level}</span>{badges_html}
            </div>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(text: str) -> None:
    st.markdown(f'<div class="card">{text}</div>', unsafe_allow_html=True)


def plotly_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        margin=dict(l=25, r=25, t=45, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter", color="#e5e7eb"),
    )
    return fig


# =============================================================================
# 2. DỮ LIỆU MẪU TỪ ĐỀ BÀI
# =============================================================================


def macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "K_trillion_VND": [16500, 17800, 19600, 21300, 23500, 25900],
            "L_million": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            "D_digital_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "AI_firms_thousand": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            "H_trained_labor_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        }
    )


def sectors_df() -> pd.DataFrame:
    names = [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Khai khoáng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
        "Y tế",
    ]
    return pd.DataFrame(
        {
            "sector": names,
            "growth": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
            "productivity": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
            "spillover": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
            "export": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
            "labor": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
            "ai_readiness": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
            "automation_risk": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
        }
    )


def regions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": [
                "Trung du miền núi phía Bắc",
                "Đồng bằng sông Hồng",
                "Bắc Trung Bộ + DH Trung Bộ",
                "Tây Nguyên",
                "Đông Nam Bộ",
                "Đồng bằng sông Cửu Long",
            ],
            "short": ["NMM", "RRD", "NCC", "CH", "SE", "MD"],
            "grdp_pc": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
            "fdi": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
            "digital_index": [38, 78, 55, 32, 82, 48],
            "ai_readiness": [22, 68, 40, 18, 75, 30],
            "trained_labor": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
            "rd": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
            "internet": [72, 92, 84, 68, 94, 78],
            "gini": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
        }
    )


REGION_SHORTS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
ITEMS = ["I", "D", "AI", "H"]
BETA_REGION_ITEM = np.array(
    [
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ]
)

SCENARIOS = {
    "S1. Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2. Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
    "S3. AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
    "S4. Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
    "S5. Tối ưu cân bằng": np.array([0.40, 0.25, 0.15, 0.20]),
}


# =============================================================================
# 3. HÀM TÍNH TOÁN DÙNG CHUNG
# =============================================================================


def normalize_good(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng else 1)


def normalize_bad(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s.max() - s) / (rng if rng else 1)


def cobb_douglas(Y=None, K=None, L=None, D=None, AI=None, H=None, A=1.0,
                  alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07):
    if Y is None:
        return A * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
    return Y / ((K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta))


def topsis(df: pd.DataFrame, criteria: List[str], is_benefit: List[bool], w: np.ndarray) -> pd.DataFrame:
    X = df[criteria].values.astype(float)
    denom = np.sqrt((X**2).sum(axis=0))
    denom[denom == 0] = 1
    R = X / denom
    V = R * w
    benefit = np.array(is_benefit)
    A_star = np.where(benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C_star = S_neg / (S_star + S_neg + 1e-12)
    out = df.copy()
    out["TOPSIS_score"] = C_star
    out["rank"] = out["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values("TOPSIS_score", ascending=False)


def entropy_weights(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    # Dịch chuyển để tránh số âm khi dùng entropy
    X = X - X.min(axis=0) + 1e-9
    P = X / (X.sum(axis=0) + 1e-12)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def gini_like(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.mean(x) == 0:
        return 0.0
    return np.abs(x - x.mean()).mean() / (2 * x.mean())


# =============================================================================
# 4. SIDEBAR
# =============================================================================


def sidebar() -> str:
    st.sidebar.markdown("### 🇻🇳 AIDEOM-VN")
    st.sidebar.markdown('<div class="muted">Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI</div>', unsafe_allow_html=True)
    st.sidebar.divider()

    pages = [
        "🏠 Trang chủ",
        "🌱 Bài 1 — Cobb-Douglas + AI",
        "💰 Bài 2 — LP ngân sách số",
        "📊 Bài 3 — Priority 10 ngành",
        "🗺️ Bài 4 — LP ngành-vùng",
        "🎯 Bài 5 — MIP 15 dự án",
        "🏆 Bài 6 — TOPSIS 6 vùng",
        "🌐 Bài 7 — NSGA-II Pareto",
        "⏳ Bài 8 — Động 2026-2035",
        "👷 Bài 9 — Lao động & AI",
        "🎲 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🇻🇳 Bài 12 — AIDEOM tích hợp",
    ]
    choice = st.sidebar.radio("", pages, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("**📁 Dữ liệu:** NSO, MoST, MIC, MPI, WB, GII 2025")
    st.sidebar.markdown("**🛠️ Stack:** Streamlit · Plotly · SciPy · PuLP style · RL tabular")
    st.sidebar.markdown("**📘 Dựa trên:** giáo trình AIDEOM-VN 2026")
    return choice


# =============================================================================
# 5. CÁC TRANG BÀI TẬP
# =============================================================================


def page_home() -> None:
    title_block(
        "VN AIDEOM-VN Dashboard 12 bài",
        "BỘ BÀI TẬP THỰC HÀNH",
        ["Python", "Tối ưu hóa", "Học tăng cường", "Streamlit"],
        "Website mô phỏng đầy đủ 12 bài theo đề bài, có giao diện nền tối, sidebar trái và dashboard giống mẫu ảnh.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Số bài", "12", "Từ dễ đến khó")
    with c2: metric_card("Dữ liệu", "2020–2025", "Việt Nam")
    with c3: metric_card("Kỹ thuật", "LP · MIP · RL", "Tích hợp mô hình")
    with c4: metric_card("Dashboard", "6 module", "M1 đến M6")

    st.markdown("---")
    st.subheader("📌 Bản đồ kỹ năng")
    roadmap = pd.DataFrame(
        {
            "Cấp độ": ["Dễ", "Dễ", "Dễ", "Trung bình", "Trung bình", "Trung bình", "Khá khó", "Khá khó", "Khá khó", "Khó", "Khó", "Khó"],
            "Bài": [f"Bài {i}" for i in range(1, 13)],
            "Chủ đề": [
                "Cobb-Douglas mở rộng AI",
                "LP ngân sách số 4 hạng mục",
                "Priority Index cho 10 ngành",
                "LP phân bổ ngành-vùng",
                "MIP chọn 15 dự án",
                "TOPSIS 6 vùng",
                "Pareto đa mục tiêu",
                "Tối ưu động 2026-2035",
                "Tác động AI tới lao động",
                "Quy hoạch ngẫu nhiên 2 giai đoạn",
                "Q-learning chính sách thích nghi",
                "AIDEOM-VN tích hợp 6 module",
            ],
            "Công cụ": ["NumPy, Pandas", "SciPy linprog", "Pandas", "SciPy/PuLP", "MIP brute force/PuLP", "TOPSIS", "NSGA-II mô phỏng", "SciPy SLSQP", "LP", "Stochastic LP", "Tabular RL", "Streamlit + Plotly"],
        }
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

    fig = px.bar(roadmap, x="Bài", color="Cấp độ", title="12 bài theo cấp độ", template=PLOTLY_TEMPLATE)
    st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)


def page_bai1() -> None:
    title_block(
        "🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI",
        "CẤP ĐỘ DỄ",
        ["numpy", "pandas", "growth accounting"],
        "Tính TFP A_t, so sánh GDP thực tế và dự báo, phân rã tăng trưởng, mô phỏng GDP 2030.",
    )
    st.markdown('<div class="formula">Y = A · K^α · L^β · D^γ · AI^δ · H^θ, với α+β+γ+δ+θ = 1</div>', unsafe_allow_html=True)

    colp = st.columns(5)
    alpha = colp[0].number_input("α vốn", 0.0, 1.0, 0.33, 0.01)
    beta = colp[1].number_input("β lao động", 0.0, 1.0, 0.42, 0.01)
    gamma = colp[2].number_input("γ số hóa", 0.0, 1.0, 0.10, 0.01)
    delta = colp[3].number_input("δ AI", 0.0, 1.0, 0.08, 0.01)
    theta = colp[4].number_input("θ nhân lực", 0.0, 1.0, 0.07, 0.01)

    df = macro_df()
    df["A_TFP"] = cobb_douglas(
        Y=df["GDP_trillion_VND"].values,
        K=df["K_trillion_VND"].values,
        L=df["L_million"].values,
        D=df["D_digital_GDP_pct"].values,
        AI=df["AI_firms_thousand"].values,
        H=df["H_trained_labor_pct"].values,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, theta=theta,
    )
    A_bar = df["A_TFP"].mean()
    df["Y_hat"] = cobb_douglas(
        K=df["K_trillion_VND"].values,
        L=df["L_million"].values,
        D=df["D_digital_GDP_pct"].values,
        AI=df["AI_firms_thousand"].values,
        H=df["H_trained_labor_pct"].values,
        A=A_bar,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, theta=theta,
    )
    mape = np.mean(np.abs((df["GDP_trillion_VND"] - df["Y_hat"]) / df["GDP_trillion_VND"])) * 100

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("TFP trung bình", f"{A_bar:.2f}", "A_t 2020–2025")
    with c2: metric_card("MAPE", f"{mape:.2f}%", "Sai số dự báo")
    with c3: metric_card("GDP 2025", f"{df.iloc[-1]['GDP_trillion_VND']:,.1f}", "nghìn tỷ VND")

    tab1, tab2, tab3 = st.tabs(["📈 TFP & dự báo", "📊 Phân rã", "🚀 GDP 2030"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year"], y=df["GDP_trillion_VND"], name="GDP thực tế", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=df["year"], y=df["Y_hat"], name="GDP dự báo", mode="lines+markers"))
        st.plotly_chart(plotly_layout(fig, 400), use_container_width=True)
        st.dataframe(df.round(3), use_container_width=True, hide_index=True)
    with tab2:
        growth = np.log(df[["GDP_trillion_VND", "K_trillion_VND", "L_million", "D_digital_GDP_pct", "AI_firms_thousand", "H_trained_labor_pct"]]).diff().dropna()
        contrib = pd.DataFrame({
            "Y_growth": growth["GDP_trillion_VND"],
            "K": alpha * growth["K_trillion_VND"],
            "L": beta * growth["L_million"],
            "D": gamma * growth["D_digital_GDP_pct"],
            "AI": delta * growth["AI_firms_thousand"],
            "H": theta * growth["H_trained_labor_pct"],
        })
        contrib["TFP"] = contrib["Y_growth"] - contrib[["K", "L", "D", "AI", "H"]].sum(axis=1)
        avg = contrib[["K", "L", "D", "AI", "H", "TFP"]].mean() * 100
        fig = px.bar(x=avg.index, y=avg.values, labels={"x": "Yếu tố", "y": "Điểm % đóng góp"}, title="Đóng góp tăng trưởng bình quân", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
        st.dataframe((contrib * 100).round(2), use_container_width=True)
    with tab3:
        years = np.arange(2025, 2031)
        K0, L0, A0 = df.iloc[-1]["K_trillion_VND"], df.iloc[-1]["L_million"], A_bar
        rows = []
        for y in years:
            step = y - 2025
            K = K0 * (1.06 ** step)
            L = L0 * (1.01 ** step)
            D = 19.5 + (30 - 19.5) * step / 5
            AI = 80.1 + (100 - 80.1) * step / 5
            H = 29.2 + (35 - 29.2) * step / 5
            A = A0 * (1.012 ** step)
            Y = cobb_douglas(K=K, L=L, D=D, AI=AI, H=H, A=A, alpha=alpha, beta=beta, gamma=gamma, delta=delta, theta=theta)
            rows.append([y, K, L, D, AI, H, A, Y])
        sim = pd.DataFrame(rows, columns=["year", "K", "L", "D", "AI", "H", "A", "GDP_forecast"])
        metric_card("GDP dự báo 2030", f"{sim.iloc[-1]['GDP_forecast']:,.1f}", "nghìn tỷ VND")
        fig = px.line(sim, x="year", y="GDP_forecast", markers=True, title="Mô phỏng GDP đến 2030", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)


def page_bai2() -> None:
    title_block(
        "💰 Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số",
        "CẤP ĐỘ DỄ",
        ["LP", "scipy.optimize.linprog", "shadow price"],
        "Tối đa hóa tăng GDP kỳ vọng với 4 biến: hạ tầng số, AI dữ liệu, nhân lực số, R&D.",
    )
    budget = st.slider("Ngân sách tổng B, nghìn tỷ VND", 80, 160, 100, 10)
    min_human = st.slider("Sàn nhân lực số x₃", 20, 50, 20, 5)
    c = -np.array([0.85, 1.20, 0.95, 1.35])
    A_ub = np.array([
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ])
    b_ub = np.array([budget, -25, -15, -min_human, -10, 0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    if not res.success:
        st.error("Bài toán không khả thi với tham số hiện tại.")
        return

    sol = pd.DataFrame({"Hạng mục": ["Hạ tầng số I", "AI và dữ liệu", "Nhân lực số H", "R&D công nghệ"], "Phân bổ": res.x, "Hệ số": -c, "GDP gain": res.x * (-c)})
    z = -res.fun
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Z* tối ưu", f"{z:.2f}", "nghìn tỷ VND GDP gain")
    with c2: metric_card("AI + R&D", f"{sol.loc[[1,3], 'Phân bổ'].sum():.1f}", "nghìn tỷ")
    with c3: metric_card("Tỷ trọng chiến lược", f"{sol.loc[[1,3], 'Phân bổ'].sum()/sol['Phân bổ'].sum()*100:.1f}%", "≥ 35%")

    tab1, tab2 = st.tabs(["📌 Kết quả tối ưu", "📉 Độ nhạy ngân sách"])
    with tab1:
        st.dataframe(sol.round(3), use_container_width=True, hide_index=True)
        fig = px.pie(sol, names="Hạng mục", values="Phân bổ", hole=0.45, title="Cơ cấu phân bổ tối ưu", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)
    with tab2:
        rows = []
        for B in range(80, 161, 10):
            b2 = b_ub.copy(); b2[0] = B
            r = linprog(c, A_ub=A_ub, b_ub=b2, bounds=[(0, None)] * 4, method="highs")
            rows.append([B, -r.fun if r.success else np.nan])
        sens = pd.DataFrame(rows, columns=["Budget", "Z_star"])
        fig = px.line(sens, x="Budget", y="Z_star", markers=True, title="Đường cong Z*(B)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)


def page_bai3() -> None:
    title_block(
        "📊 Bài 3 — Tính chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam",
        "CẤP ĐỘ DỄ",
        ["MCDM", "min-max", "policy weights"],
        "Chuẩn hóa 7 tiêu chí, tính Priority và phân tích độ nhạy trọng số AI Readiness.",
    )
    df = sectors_df()
    st.sidebar.markdown("### Trọng số Priority")
    w_growth = st.sidebar.slider("Tăng trưởng", 0.0, 0.5, 0.15, 0.01)
    w_prod = st.sidebar.slider("Năng suất", 0.0, 0.5, 0.15, 0.01)
    w_spill = st.sidebar.slider("Lan tỏa", 0.0, 0.5, 0.20, 0.01)
    w_export = st.sidebar.slider("Xuất khẩu", 0.0, 0.5, 0.15, 0.01)
    w_labor = st.sidebar.slider("Việc làm", 0.0, 0.5, 0.10, 0.01)
    w_ai = st.sidebar.slider("AI Readiness", 0.0, 0.5, 0.20, 0.01)
    w_risk = st.sidebar.slider("Giảm rủi ro", 0.0, 0.5, 0.15, 0.01)
    weights = np.array([w_growth, w_prod, w_spill, w_export, w_labor, w_ai, w_risk])
    weights = weights / weights.sum()

    good_cols = ["growth", "productivity", "spillover", "export", "labor", "ai_readiness"]
    norm = df[good_cols].apply(normalize_good)
    norm["risk_inv"] = normalize_bad(df["automation_risk"])
    df["Priority"] = norm.values @ weights
    ranked = df.sort_values("Priority", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Top 1", ranked.iloc[0]["sector"], f"Priority {ranked.iloc[0]['Priority']:.3f}")
    with c2: metric_card("Top 2", ranked.iloc[1]["sector"], f"Priority {ranked.iloc[1]['Priority']:.3f}")
    with c3: metric_card("Top 3", ranked.iloc[2]["sector"], f"Priority {ranked.iloc[2]['Priority']:.3f}")

    tab1, tab2, tab3 = st.tabs(["🏆 Xếp hạng", "🔥 Ma trận chuẩn hóa", "🧪 Độ nhạy AI"])
    with tab1:
        fig = px.bar(ranked, x="Priority", y="sector", orientation="h", color="Priority", title="Xếp hạng Priority 10 ngành", template=PLOTLY_TEMPLATE)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(plotly_layout(fig, 480), use_container_width=True)
        st.dataframe(ranked[["rank", "sector", "Priority", "growth", "productivity", "ai_readiness", "automation_risk"]].round(3), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(norm.round(3).assign(sector=df["sector"]).set_index("sector"), use_container_width=True)
        fig = px.imshow(norm.T, x=df["sector"], y=norm.columns, aspect="auto", title="Heatmap chuẩn hóa", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 480), use_container_width=True)
    with tab3:
        rows = []
        for ai_w in np.arange(0.05, 0.401, 0.05):
            base = np.array([0.15, 0.15, 0.20, 0.15, 0.10, ai_w, 0.15])
            base = base / base.sum()
            pr = norm.values @ base
            tmp = pd.DataFrame({"sector": df["sector"], "Priority": pr}).sort_values("Priority", ascending=False)
            for k, sec in enumerate(tmp.head(3)["sector"], start=1):
                rows.append([ai_w, k, sec])
        sens = pd.DataFrame(rows, columns=["w_AI", "rank", "sector"])
        st.dataframe(sens, use_container_width=True, hide_index=True)


def solve_region_lp(with_fairness: bool = True) -> Tuple[pd.DataFrame, float, bool]:
    n = 24
    c = -BETA_REGION_ITEM.flatten()
    A_ub, b_ub = [], []
    # Tổng ngân sách <= 50000
    A_ub.append(np.ones(n)); b_ub.append(50000)
    # Mỗi vùng <= 12000 và >= 5000
    for r in range(6):
        row = np.zeros(n); row[r * 4:(r + 1) * 4] = 1
        A_ub.append(row); b_ub.append(12000)
        A_ub.append(-row); b_ub.append(-5000)
    # Sàn nhân lực >= 12000
    row = np.zeros(n); row[3::4] = -1
    A_ub.append(row); b_ub.append(-12000)
    # Fairness đơn giản: D_after của mọi vùng >= 70% vùng mạnh nhất ban đầu SE 82
    if with_fairness:
        D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)
        gamma = 0.002
        M_ref = 82
        for r in range(6):
            # D0 + gamma*x_D >= 0.7*M_ref
            row = np.zeros(n); row[r * 4 + 1] = -gamma
            A_ub.append(row); b_ub.append(-(0.7 * M_ref - D0[r]))
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * n, method="highs")
    if not res.success:
        return pd.DataFrame(), np.nan, False
    mat = res.x.reshape(6, 4)
    out = pd.DataFrame(mat, index=REGION_SHORTS, columns=ITEMS)
    return out, -res.fun, True


def page_bai4() -> None:
    title_block(
        "🗺️ Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng",
        "TRUNG BÌNH",
        ["LP", "6 vùng", "4 hạng mục", "fairness"],
        "Phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục, có sàn, trần và ràng buộc công bằng vùng miền.",
    )
    with_fairness = st.toggle("Bật ràng buộc công bằng vùng miền", value=True)
    alloc, z, ok = solve_region_lp(with_fairness)
    if not ok:
        st.error("Mô hình không khả thi.")
        return
    alloc_long = alloc.reset_index().melt(id_vars="index", var_name="Hạng mục", value_name="Ngân sách").rename(columns={"index": "Vùng"})
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Z*", f"{z:,.0f}", "GDP gain kỳ vọng")
    with c2: metric_card("Vùng nhận nhiều nhất", alloc.sum(axis=1).idxmax(), f"{alloc.sum(axis=1).max():,.0f} tỷ")
    with c3: metric_card("Hạng mục lớn nhất", alloc.sum(axis=0).idxmax(), f"{alloc.sum(axis=0).max():,.0f} tỷ")

    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap phân bổ", "📋 Ma trận", "⚖️ Chi phí công bằng"])
    with tab1:
        fig = px.imshow(alloc, labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"), title="Heatmap phân bổ tối ưu", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 440), use_container_width=True)
    with tab2:
        st.dataframe(alloc.round(2), use_container_width=True)
        fig = px.bar(alloc_long, x="Vùng", y="Ngân sách", color="Hạng mục", barmode="stack", title="Cơ cấu ngân sách theo vùng", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)
    with tab3:
        a1, z1, _ = solve_region_lp(True)
        a0, z0, _ = solve_region_lp(False)
        metric_card("Chi phí công bằng", f"{z0 - z1:,.0f}", f"Z không công bằng {z0:,.0f}, Z có công bằng {z1:,.0f}")
        cmp = pd.DataFrame({"Có công bằng": a1.sum(axis=1), "Không công bằng": a0.sum(axis=1)})
        fig = px.bar(cmp, barmode="group", title="So sánh ngân sách vùng", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)


def page_bai5() -> None:
    title_block(
        "🎯 Bài 5 — Quy hoạch nguyên hỗn hợp lựa chọn 15 dự án chuyển đổi số",
        "TRUNG BÌNH",
        ["MIP", "binary", "knapsack"],
        "Chọn tập dự án tối ưu dưới ràng buộc ngân sách tổng, ngân sách năm 1-2, loại trừ và tiên quyết.",
    )
    projects = pd.DataFrame(
        {
            "id": [f"P{i}" for i in range(1, 16)],
            "name": ["TT dữ liệu Hòa Lạc", "TT dữ liệu phía Nam", "5G toàn quốc", "VNeID 2.0", "DVC quốc gia v3", "Y tế số", "Giáo dục số K-12", "Trung tâm AI quốc gia", "Sandbox fintech", "Logistics thông minh", "Nông nghiệp số ĐBSCL", "50.000 kỹ sư AI/bán dẫn", "Khu CN bán dẫn", "SOC an ninh mạng", "Open Data quốc gia"],
            "field": ["Hạ tầng", "Hạ tầng", "Hạ tầng", "Chính phủ số", "Chính phủ số", "Y tế", "Giáo dục", "AI", "Tài chính", "Logistics", "Nông nghiệp", "Nhân lực", "Bán dẫn", "An ninh", "Dữ liệu"],
            "cost": [12000, 11500, 18000, 4500, 3200, 5800, 6500, 15000, 2500, 7200, 4800, 8500, 20000, 3800, 1500],
            "benefit": [21500, 20800, 32500, 9200, 6800, 11400, 12200, 28500, 5800, 13800, 8500, 16200, 35000, 7500, 3800],
            "year12": [8500, 7500, 12000, 3500, 2500, 4000, 4500, 9000, 1800, 5000, 3500, 5500, 13000, 2800, 1200],
        }
    )
    budget = st.slider("Ngân sách 5 năm", 60000, 110000, 80000, 5000)
    require_p1p2 = st.toggle("Bắt buộc chọn cả P1 và P2", value=False)
    best_val = -1
    best_mask = None
    for mask in itertools.product([0, 1], repeat=15):
        y = np.array(mask)
        if projects["cost"].values @ y > budget: continue
        if projects["year12"].values @ y > 40000: continue
        if y[0] + y[1] > 1 and not require_p1p2: continue
        if require_p1p2 and not (y[0] == 1 and y[1] == 1): continue
        if y[7] > y[11]: continue
        if y[12] > y[11]: continue
        if y[3] + y[4] < 1: continue
        if y[13] < 1: continue
        if y.sum() < 7 or y.sum() > 11: continue
        val = projects["benefit"].values @ y
        if val > best_val:
            best_val = val; best_mask = y
    if best_mask is None:
        st.error("Không tìm thấy nghiệm khả thi với ràng buộc hiện tại.")
        st.dataframe(projects, use_container_width=True, hide_index=True)
        return
    projects["selected"] = best_mask.astype(bool)
    chosen = projects[projects["selected"]].copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Số dự án", str(len(chosen)), "được chọn")
    with c2: metric_card("Tổng chi phí", f"{chosen['cost'].sum():,.0f}", "tỷ VND")
    with c3: metric_card("Tổng lợi ích", f"{chosen['benefit'].sum():,.0f}", "tỷ VND NPV")
    with c4: metric_card("NPV/Cost", f"{chosen['benefit'].sum()/chosen['cost'].sum():.2f}", "hiệu quả biên")

    tab1, tab2 = st.tabs(["✅ Dự án được chọn", "📦 Tất cả dự án"])
    with tab1:
        st.dataframe(chosen, use_container_width=True, hide_index=True)
        fig = px.bar(chosen, y="id", x="benefit", color="field", orientation="h", title="Lợi ích các dự án được chọn", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 480), use_container_width=True)
    with tab2:
        st.dataframe(projects, use_container_width=True, hide_index=True)
        fig = px.scatter(projects, x="cost", y="benefit", color="selected", size="benefit", hover_name="name", title="Chi phí - lợi ích 15 dự án", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)


def page_bai6() -> None:
    title_block(
        "🏆 Bài 6 — TOPSIS xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI",
        "TRUNG BÌNH",
        ["MCDM", "TOPSIS", "Entropy weights"],
        "Tính hệ số gần gũi C* và xếp hạng 6 vùng theo 8 tiêu chí sẵn sàng AI.",
    )
    df = regions_df()
    criteria = ["grdp_pc", "fdi", "digital_index", "ai_readiness", "trained_labor", "rd", "internet", "gini"]
    is_benefit = [True, True, True, True, True, True, True, False]
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    mode = st.radio("Chọn trọng số", ["Chuyên gia", "Entropy khách quan"], horizontal=True)
    if mode == "Entropy khách quan":
        X = df[criteria].values.copy()
        X[:, -1] = X[:, -1].max() - X[:, -1]
        w = entropy_weights(X)
    else:
        w = w_expert
    ranked = topsis(df, criteria, is_benefit, w)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Top 1", ranked.iloc[0]["region"], f"C*={ranked.iloc[0]['TOPSIS_score']:.3f}")
    with c2: metric_card("Top 2", ranked.iloc[1]["region"], f"C*={ranked.iloc[1]['TOPSIS_score']:.3f}")
    with c3: metric_card("Top 3", ranked.iloc[2]["region"], f"C*={ranked.iloc[2]['TOPSIS_score']:.3f}")

    tab1, tab2, tab3 = st.tabs(["🏆 Ranking", "⚖️ Trọng số", "🧪 Nhạy cảm w_AI"])
    with tab1:
        fig = px.bar(ranked, x="TOPSIS_score", y="region", color="TOPSIS_score", orientation="h", title=f"Xếp hạng TOPSIS - {mode}", template=PLOTLY_TEMPLATE)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(plotly_layout(fig, 440), use_container_width=True)
        st.dataframe(ranked[["rank", "region", "TOPSIS_score"] + criteria].round(3), use_container_width=True, hide_index=True)
    with tab2:
        wdf = pd.DataFrame({"criteria": criteria, "weight": w})
        st.dataframe(wdf.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(wdf, x="criteria", y="weight", title="Trọng số tiêu chí", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)
    with tab3:
        rows = []
        for w_ai in np.arange(0.10, 0.401, 0.05):
            ww = w_expert.copy(); ww[3] = w_ai; ww = ww / ww.sum()
            rr = topsis(df, criteria, is_benefit, ww)
            rows.append([w_ai, ", ".join(rr.head(3)["short"])])
        st.dataframe(pd.DataFrame(rows, columns=["w_AI", "Top 3 vùng"]), use_container_width=True, hide_index=True)


def generate_pareto_samples(n: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    rows = []
    for k in range(n):
        # ngẫu nhiên phân bổ ngân sách nhưng giữ tổng 50.000 và mỗi vùng khoảng 5.000-12.000
        region_budget = rng.dirichlet(np.ones(6)) * 50000
        region_budget = np.clip(region_budget, 5000, 12000)
        region_budget *= 50000 / region_budget.sum()
        X = np.vstack([rng.dirichlet(np.ones(4)) * b for b in region_budget])
        f1 = (BETA_REGION_ITEM * X).sum()
        f2 = gini_like(X.sum(axis=1))
        f3 = (e * (X[:, 0] + X[:, 2])).sum()
        f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
        rows.append([k, f1, f2, f3, f4])
    df = pd.DataFrame(rows, columns=["id", "growth", "inequality", "emission", "cyber_risk"])
    # Pareto filter: maximize growth, minimize remaining
    obj = np.column_stack([df["growth"].values, -df["inequality"].values, -df["emission"].values, -df["cyber_risk"].values])
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        dominates = np.all(obj >= obj[i], axis=1) & np.any(obj > obj[i], axis=1)
        if dominates.any():
            is_pareto[i] = False
    df["Pareto"] = is_pareto
    return df


def page_bai7() -> None:
    title_block(
        "🌐 Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II",
        "KHÁ KHÓ",
        ["multi-objective", "Pareto", "NSGA-II mô phỏng"],
        "Minh họa đánh đổi giữa tăng trưởng, bao trùm, phát thải và rủi ro dữ liệu bằng tập nghiệm Pareto.",
    )
    n = st.slider("Số nghiệm mô phỏng", 300, 2000, 900, 100)
    df = generate_pareto_samples(n=n)
    pareto = df[df["Pareto"]].copy()
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Số nghiệm Pareto", f"{len(pareto)}", "không bị trội")
    with c2: metric_card("Growth max", f"{pareto['growth'].max():,.0f}", "GDP gain")
    with c3: metric_card("Emission min", f"{pareto['emission'].min():,.0f}", "CO₂ tương đối")

    tab1, tab2 = st.tabs(["🫧 Pareto 3D", "🧮 Nghiệm thỏa hiệp TOPSIS"])

    with tab1:
        # Plotly không cho size âm, nên tạo cột kích thước luôn dương
        df["cyber_size"] = df["cyber_risk"] - df["cyber_risk"].min() + 5

        fig = px.scatter_3d(
            df,
            x="growth",
            y="inequality",
            z="emission",
            color="Pareto",
            size="cyber_size",
            hover_data={"cyber_risk": True, "cyber_size": False},
            title="Tập nghiệm mô phỏng và biên Pareto",
            template=PLOTLY_TEMPLATE
        )

        st.plotly_chart(plotly_layout(fig, 600), use_container_width=True)

    with tab2:
        x = pareto[["growth", "inequality", "emission", "cyber_risk"]].copy()
        # TOPSIS trên tập Pareto: growth lợi ích, còn lại chi phí
        tmp = pareto.copy()
        ranked = topsis(tmp, ["growth", "inequality", "emission", "cyber_risk"], [True, False, False, False], np.array([0.40, 0.25, 0.20, 0.15]))
        best = ranked.iloc[0]
        metric_card("Nghiệm thỏa hiệp", f"ID {int(best['id'])}", f"C*={best['TOPSIS_score']:.3f}")
        st.dataframe(ranked.head(20).round(3), use_container_width=True, hide_index=True)
        fig = px.parallel_coordinates(ranked.head(100), dimensions=["growth", "inequality", "emission", "cyber_risk", "TOPSIS_score"], color="TOPSIS_score", title="Parallel coordinates cho nghiệm Pareto", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 520), use_container_width=True)


def page_bai8() -> None:
    title_block(
        "⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
        "KHÁ KHÓ",
        ["dynamic", "Cobb-Douglas", "welfare"],
        "Mô phỏng quỹ đạo K, D, AI, H, Y, C theo ba chiến lược: trải đều, front-load, và cân bằng tối ưu.",
    )
    strategy = st.selectbox("Chiến lược", ["Đầu tư trải đều", "Front-load 3 năm đầu", "Cân bằng tối ưu"])
    years = np.arange(2026, 2036)
    K, D, AI, H, A = 27500.0, 20.3, 86.0, 30.0, 22.0
    rows = []
    rho = 0.97
    for idx, year in enumerate(years):
        if strategy == "Đầu tư trải đều":
            inv = np.array([0.35, 0.25, 0.15, 0.25]) * 2500
        elif strategy == "Front-load 3 năm đầu":
            scale = 1.65 if idx < 3 else 0.72
            inv = np.array([0.30, 0.30, 0.20, 0.20]) * 2500 * scale
        else:
            # tự cân bằng dần: nhiều H và D lúc đầu, AI tăng sau
            ai_share = min(0.28, 0.10 + idx * 0.02)
            h_share = max(0.20, 0.32 - idx * 0.01)
            d_share = 0.28
            k_share = 1 - ai_share - h_share - d_share
            inv = np.array([k_share, d_share, ai_share, h_share]) * 2500
        Y = cobb_douglas(K=K, L=54.0, D=D, AI=AI, H=H, A=A)
        C = max(1, Y - inv.sum())
        welfare = (rho ** idx) * np.log(C)
        rows.append([year, K, D, AI, H, A, Y, C, inv[0], inv[1], inv[2], inv[3], welfare])
        K = 0.95 * K + inv[0]
        D = 0.88 * D + inv[1] / 100
        AI = 0.85 * AI + inv[2] / 20
        H = H + 0.8 * inv[3] / 200 - 0.02 * H
        A = A * (1 + 0.003 * D / 100 + 0.002 * AI / 100 + 0.004 * H / 100)
    sim = pd.DataFrame(rows, columns=["year", "K", "D", "AI", "H", "A", "Y", "C", "I_K", "I_D", "I_AI", "I_H", "welfare"])
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("GDP 2035", f"{sim.iloc[-1]['Y']:,.0f}", "nghìn tỷ VND")
    with c2: metric_card("Tiêu dùng 2035", f"{sim.iloc[-1]['C']:,.0f}", "C_t")
    with c3: metric_card("Welfare tổng", f"{sim['welfare'].sum():.2f}", "chiết khấu")

    tab1, tab2 = st.tabs(["📈 Quỹ đạo trạng thái", "💸 Cơ cấu đầu tư"])
    with tab1:
        fig = px.line(sim, x="year", y=["K", "D", "AI", "H", "Y", "C"], markers=True, title=f"Quỹ đạo tối ưu - {strategy}", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 500), use_container_width=True)
        st.dataframe(sim.round(2), use_container_width=True, hide_index=True)
    with tab2:
        long = sim.melt(id_vars="year", value_vars=["I_K", "I_D", "I_AI", "I_H"], var_name="Investment", value_name="Value")
        fig = px.area(long, x="year", y="Value", color="Investment", title="Cơ cấu đầu tư theo thời gian", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)


def page_bai9() -> None:
    title_block(
        "👷 Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
        "KHÁ KHÓ",
        ["LP", "NetJob", "Retraining"],
        "Tối đa hóa NetJob ròng theo ngành, bảo đảm tốc độ tự động hóa không vượt năng lực đào tạo lại.",
    )
    sectors = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"]
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    N = 8
    # Variables: xAI[0..7], xH[0..7]
    coeff_net = np.concatenate([a1 - c1 * risk, b1])
    c = -coeff_net
    A_ub = [np.ones(2 * N)]; b_ub = [30000]
    # NetJob >= 0 => -(coef_AI*xAI + b1*xH) <= 0
    for i in range(N):
        row = np.zeros(2 * N); row[i] = -(a1[i] - c1[i] * risk[i]); row[N + i] = -b1[i]
        A_ub.append(row); b_ub.append(0)
        # Displaced <= RetrainCap => c1*risk*xAI - d1*xH <=0
        row2 = np.zeros(2 * N); row2[i] = c1[i] * risk[i]; row2[N + i] = -d1[i]
        A_ub.append(row2); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * (2 * N), method="highs")
    if not res.success:
        st.error("Bài toán không khả thi.")
        return
    xAI, xH = res.x[:N], res.x[N:]
    NewJob = a1 * xAI
    Upgrade = b1 * xH
    Displaced = c1 * risk * xAI
    NetJob = NewJob + Upgrade - Displaced
    out = pd.DataFrame({"sector": sectors, "x_AI": xAI, "x_H": xH, "NewJob": NewJob, "UpgradeJob": Upgrade, "DisplacedJob": Displaced, "NetJob": NetJob})
    c1m, c2m, c3m = st.columns(3)
    with c1m: metric_card("Tổng NetJob", f"{NetJob.sum():,.0f}", "việc làm")
    with c2m: metric_card("Đầu tư AI", f"{xAI.sum():,.0f}", "tỷ VND")
    with c3m: metric_card("Đào tạo lại", f"{xH.sum():,.0f}", "tỷ VND")
    tab1, tab2 = st.tabs(["📋 Phân bổ tối ưu", "🌊 Luồng việc làm"])
    with tab1:
        st.dataframe(out.round(2), use_container_width=True, hide_index=True)
        fig = px.bar(out, x="sector", y=["NewJob", "UpgradeJob", "DisplacedJob", "NetJob"], barmode="group", title="Tạo việc làm, nâng cấp, dịch chuyển và NetJob", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 520), use_container_width=True)
    with tab2:
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=["Lao động phổ thông", "Tự động hóa", "Đào tạo lại", "Việc làm AI mới", "NetJob dương"]),
            link=dict(source=[0, 0, 2, 3], target=[1, 2, 4, 4], value=[Displaced[[0, 2, 3]].sum(), xH[[0, 2, 3]].sum(), Upgrade[[0, 2, 3]].sum(), NewJob[[0, 2, 3]].sum()])
        )])
        fig.update_layout(title="Sankey minh họa nhóm dễ bị tổn thương: ngành 1, 3, 4")
        st.plotly_chart(plotly_layout(fig, 500), use_container_width=True)


def page_bai10() -> None:
    title_block(
        "🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
        "KHÓ",
        ["two-stage SP", "scenario", "VSS EVPI"],
        "Quyết định first-stage x và recourse y theo 4 kịch bản: lạc quan, cơ sở, bi quan, khủng hoảng.",
    )
    J = ["I", "D", "AI", "H"]
    S = ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"]
    p = np.array([0.30, 0.45, 0.20, 0.05])
    beta = np.array([1.00, 1.10, 1.25, 0.95])
    beta_s = np.array([[1.25, 1.35, 1.55, 1.05], [1.00, 1.10, 1.25, 0.95], [0.75, 0.85, 0.90, 1.00], [0.40, 0.50, 0.55, 1.10]])
    # Variables x4 + y16
    c = -np.concatenate([beta, (p[:, None] * beta_s).flatten()])
    A_ub, b_ub = [], []
    row = np.zeros(20); row[:4] = 1
    A_ub.append(row); b_ub.append(65000)
    for s in range(4):
        row = np.zeros(20); row[4 + s * 4:4 + (s + 1) * 4] = 1
        A_ub.append(row); b_ub.append(15000)
        # y_AI_s <= 0.5 x_H
        row = np.zeros(20); row[4 + s * 4 + 2] = 1; row[3] = -0.5
        A_ub.append(row); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * 20, method="highs")
    x = res.x[:4]
    y = res.x[4:].reshape(4, 4)
    z = -res.fun
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Expected Z*", f"{z:,.0f}", "GDP gain kỳ vọng")
    with c2: metric_card("First-stage", f"{x.sum():,.0f}", "tỷ VND")
    with c3: metric_card("Reserve used", f"{y.sum(axis=1).mean():,.0f}", "bình quân theo scenario")
    tab1, tab2, tab3 = st.tabs(["📌 First-stage", "🌳 Scenario recourse", "📏 VSS & EVPI"])
    with tab1:
        xdf = pd.DataFrame({"Hạng mục": J, "x first-stage": x})
        st.dataframe(xdf.round(2), use_container_width=True, hide_index=True)
        fig = px.pie(xdf, names="Hạng mục", values="x first-stage", hole=0.45, title="Quyết định here-and-now", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)
    with tab2:
        ydf = pd.DataFrame(y, index=S, columns=J)
        st.dataframe(ydf.round(2), use_container_width=True)
        fig = px.imshow(ydf, title="Recourse y theo từng kịch bản", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)
    with tab3:
        # EV dùng beta kỳ vọng cho y
        beta_ev = (p[:, None] * beta_s).sum(axis=0)
        ev_value = 65000 * beta.max() + 15000 * beta_ev.max()
        # PI: mỗi scenario biết trước, first + recourse theo beta_s tốt nhất
        pi_value = sum(p[s] * (65000 * beta.max() + 15000 * beta_s[s].max()) for s in range(4))
        vss = max(0, z - ev_value)
        evpi = max(0, pi_value - z)
        m1, m2 = st.columns(2)
        with m1: metric_card("VSS", f"{vss:,.0f}", "giá trị lời giải ngẫu nhiên")
        with m2: metric_card("EVPI", f"{evpi:,.0f}", "giá trị thông tin hoàn hảo")
        st.info("Hai chỉ số được minh họa theo công thức đơn giản hóa để phục vụ dashboard tương tác.")


class SimpleVietnamEnv:
    actions = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),
        1: np.array([0.40, 0.25, 0.15, 0.20]),
        2: np.array([0.25, 0.45, 0.15, 0.15]),
        3: np.array([0.20, 0.20, 0.45, 0.15]),
        4: np.array([0.30, 0.20, 0.10, 0.40]),
    }
    action_names = ["a0 Truyền thống", "a1 Cân bằng", "a2 Số hóa nhanh", "a3 AI dẫn dắt", "a4 Bao trùm"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.T = 10
        self.reset()

    def reset(self):
        self.state = np.array([1, 1, 0, 1])
        self.K, self.D, self.AI, self.H = 27500.0, 20.3, 86.0, 30.0
        self.Y_prev = cobb_douglas(K=self.K, L=54, D=self.D, AI=self.AI, H=self.H, A=22.0)
        self.t = 0
        return self.state.copy()

    def step(self, action: int):
        a = self.actions[action]
        budget = 1000
        self.K += a[0] * budget
        self.D += a[1] * budget / 100
        self.AI += a[2] * budget / 20
        self.H += a[3] * budget / 200
        Y = cobb_douglas(K=self.K, L=54, D=self.D, AI=self.AI, H=self.H, A=22.0)
        dgdp = (Y - self.Y_prev) / self.Y_prev * 100
        unem_delta = max(0, 0.9 * a[2] - 0.7 * a[3])
        cyber = 2.5 * a[2] + 0.6 * max(0, 1 - self.H / 35)
        emission = 2.0 * a[0] + 1.4 * a[2]
        reward = 0.40 * dgdp - 0.25 * unem_delta - 0.20 * cyber - 0.15 * emission
        self.Y_prev = Y
        self.state = np.array([
            0 if dgdp < 2.0 else 1 if dgdp < 3.3 else 2,
            0 if self.D < 22 else 1 if self.D < 27 else 2,
            0 if self.AI < 92 else 1 if self.AI < 105 else 2,
            2 if unem_delta > 0.10 else 1 if unem_delta > 0.04 else 0,
        ])
        self.t += 1
        return self.state.copy(), reward, self.t >= self.T


def train_q_learning(episodes: int = 1800, seed: int = 42) -> Tuple[np.ndarray, pd.DataFrame]:
    env = SimpleVietnamEnv(seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5))
    rewards = []
    for ep in range(episodes):
        s = env.reset(); total = 0
        eps = max(0.05, 1.0 - ep / (episodes * 0.65))
        while True:
            if rng.random() < eps:
                a = rng.integers(0, 5)
            else:
                a = int(np.argmax(Q[tuple(s)]))
            s2, r, done = env.step(a)
            Q[tuple(s) + (a,)] += 0.10 * (r + 0.95 * Q[tuple(s2)].max() - Q[tuple(s) + (a,)])
            total += r; s = s2
            if done: break
        rewards.append(total)
    curve = pd.DataFrame({"episode": np.arange(episodes), "reward": rewards})
    curve["smoothed"] = curve["reward"].rolling(80, min_periods=1).mean()
    return Q, curve


def page_bai11() -> None:
    title_block(
        "🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
        "CẤP ĐỘ KHÓ",
        ["RL tabular", "81 trạng thái", "5 hành động"],
        "MDP gồm GDP growth, Digital Index, AI capacity, Unemployment risk; học chính sách π* bằng Q-learning.",
    )
    st.markdown('<div class="formula">Reward: R = w₁ΔGDP − w₂ΔU − w₃CyberRisk − w₄Emission, w=(0.40,0.25,0.20,0.15)</div>', unsafe_allow_html=True)
    episodes = st.slider("Số episode huấn luyện", 500, 5000, 1800, 100)
    seed = st.number_input("Seed", value=42, step=1)
    if st.button("🚀 Train Q-learning") or "q_curve" not in st.session_state:
        Q, curve = train_q_learning(episodes=episodes, seed=int(seed))
        st.session_state["Q"] = Q
        st.session_state["q_curve"] = curve
    Q = st.session_state["Q"]
    curve = st.session_state["q_curve"]
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Mean reward 100 ep cuối", f"{curve['reward'].tail(100).mean():.2f}", "cao hơn là tốt")
    with c2: metric_card("Trạng thái", "81", "3⁴")
    with c3: metric_card("Hành động", "5", "a0 đến a4")
    tab1, tab2 = st.tabs(["📈 Learning curve", "🧠 Chính sách π*(s)"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["episode"], y=curve["reward"], name="Per-episode", opacity=0.35))
        fig.add_trace(go.Scatter(x=curve["episode"], y=curve["smoothed"], name="Smoothed"))
        fig.update_layout(title="Learning curve")
        st.plotly_chart(plotly_layout(fig, 470), use_container_width=True)
    with tab2:
        states = {
            "VN 2026 thực tế": (1, 1, 0, 1),
            "GDP thấp, D thấp, U cao": (0, 0, 0, 2),
            "GDP cao, AI cao, U thấp": (2, 2, 2, 0),
            "D cao, AI thấp, U trung bình": (1, 2, 0, 1),
            "Khủng hoảng chuyển đổi": (0, 1, 2, 2),
        }
        rows = []
        for name, s in states.items():
            action = int(np.argmax(Q[s]))
            rows.append([name, s, SimpleVietnamEnv.action_names[action], Q[s][action]])
        st.dataframe(pd.DataFrame(rows, columns=["Trạng thái", "Mã trạng thái", "π*(s)", "Q max"]).round(3), use_container_width=True, hide_index=True)


def run_scenario(name: str, alloc: np.ndarray) -> Dict[str, float]:
    K, D, AI, H, A = 25900.0, 19.5, 80.1, 29.2, 22.0
    for _ in range(5):
        budget = 2600
        K = 0.96 * K + alloc[0] * budget
        D = 0.90 * D + alloc[1] * budget / 120
        AI = 0.86 * AI + alloc[2] * budget / 18
        H = H + 0.75 * alloc[3] * budget / 220 - 0.015 * H
        A = A * (1 + 0.0025 * D / 100 + 0.002 * AI / 100 + 0.0035 * H / 100)
    GDP = cobb_douglas(K=K, L=55, D=D, AI=AI, H=H, A=A)
    digital = D
    risk = 100 * (0.5 * alloc[2] + 0.25 * max(0, 0.35 - alloc[3]) + 0.15 * alloc[0])
    netjob = 100000 * (0.45 * alloc[3] + 0.30 * alloc[1] + 0.20 * alloc[2] - 0.12 * alloc[2])
    emission = 100 * (0.50 * alloc[0] + 0.25 * alloc[2])
    return {"Kịch bản": name, "GDP_2030": GDP, "Digital_Index": digital, "AI_capacity": AI, "H_labor": H, "Risk": risk, "NetJob": netjob, "Emission": emission}


def page_bai12() -> None:
    title_block(
        "🇻🇳 Bài 12 — AIDEOM-VN Dashboard tích hợp",
        "ĐỒ ÁN TÍCH HỢP",
        ["6 module M1-M6", "5 kịch bản chính sách", "dashboard"],
        "Tích hợp M1 dự báo, M2 sẵn sàng số, M3 phân bổ, M4 lao động, M5 rủi ro và M6 dashboard ra quyết định.",
    )
    results = pd.DataFrame([run_scenario(name, alloc) for name, alloc in SCENARIOS.items()])
    selected = st.selectbox("Chọn kịch bản để xem nhanh", list(SCENARIOS.keys()), index=4)
    row = results[results["Kịch bản"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("GDP 2030", f"{row['GDP_2030']:,.0f}", "nghìn tỷ VND")
    with c2: metric_card("Digital Index", f"{row['Digital_Index']:.1f}", "điểm")
    with c3: metric_card("NetJob", f"{row['NetJob']:,.0f}", "việc làm")
    with c4: metric_card("Risk", f"{row['Risk']:.1f}", "cảnh báo")

    tab1, tab2, tab3, tab4 = st.tabs(["📌 Tổng quan M1-M2", "💸 Phân bổ M3", "📊 5 kịch bản M6", "⚠️ Cảnh báo M4-M5"])
    with tab1:
        design = pd.DataFrame({
            "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "Tên": ["Dự báo kinh tế", "Sẵn sàng số", "Tối ưu phân bổ", "Mô phỏng lao động", "Đánh giá rủi ro", "Dashboard ra QĐ"],
            "Đầu ra": ["GDP, TFP 2030", "Digital + AI Index", "Phân bổ ngành-vùng", "NetJob từng ngành", "Cyber, Env, Dependency", "Trực quan kịch bản"],
            "Kỹ thuật": ["Cobb-Douglas", "TOPSIS", "LP + Dynamic", "LP + Markov", "Pareto + SP", "Streamlit + Plotly"],
        })
        st.dataframe(design, use_container_width=True, hide_index=True)
        fig = px.bar(results, x="Kịch bản", y="GDP_2030", color="Kịch bản", title="M1 - GDP dự báo 2030 theo kịch bản", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)
    with tab2:
        alloc_df = pd.DataFrame(SCENARIOS).T.reset_index().rename(columns={"index": "Kịch bản", 0: "K", 1: "D", 2: "AI", 3: "H"})
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        fig = px.bar(alloc_df, x="Kịch bản", y=["K", "D", "AI", "H"], barmode="stack", title="Đặc điểm phân bổ 5 kịch bản", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)
    with tab3:
        st.dataframe(results.round(2), use_container_width=True, hide_index=True)
        fig = px.scatter(results, x="Risk", y="GDP_2030", size="NetJob", color="Kịch bản", hover_data=["Digital_Index", "Emission"], title="Đánh đổi GDP - Risk - NetJob", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 500), use_container_width=True)
    with tab4:
        warn = results.copy()
        warn["Cảnh báo"] = np.where(warn["Risk"] > 22, "Đỏ", np.where(warn["Risk"] > 17, "Vàng", "Xanh"))
        st.dataframe(warn[["Kịch bản", "Risk", "Emission", "NetJob", "Cảnh báo"]].round(2), use_container_width=True, hide_index=True)
        fig = px.bar(warn, x="Kịch bản", y=["Risk", "Emission"], barmode="group", color_discrete_sequence=None, title="M5 - Cảnh báo rủi ro", template=PLOTLY_TEMPLATE)
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)


# =============================================================================
# 6. ROUTER
# =============================================================================


def main() -> None:
    inject_css()
    choice = sidebar()
    if choice.startswith("🏠"):
        page_home()
    elif "Bài 1" in choice:
        page_bai1()
    elif "Bài 2" in choice:
        page_bai2()
    elif "Bài 3" in choice:
        page_bai3()
    elif "Bài 4" in choice:
        page_bai4()
    elif "Bài 5" in choice:
        page_bai5()
    elif "Bài 6" in choice:
        page_bai6()
    elif "Bài 7" in choice:
        page_bai7()
    elif "Bài 8" in choice:
        page_bai8()
    elif "Bài 9" in choice:
        page_bai9()
    elif "Bài 10" in choice:
        page_bai10()
    elif "Bài 11" in choice:
        page_bai11()
    elif "Bài 12" in choice:
        page_bai12()


if __name__ == "__main__":
    main()
