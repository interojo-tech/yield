#!/usr/bin/env python
"""
인터로조 공정기술팀 - 프리미엄 통합 수율 대시보드 (v2.0)
==================================================
원본의 모든 로직(주차 계산, 복합 그래프, 데이터 토글)을 유지하며
디자인을 기업용 관제 시스템 스타일로 업그레이드했습니다.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 고급 시각화 도구 설정
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False

# 불량 대시보드 연동
try:
    from defect_dashboard import run_defect_dashboard
except Exception:
    def run_defect_dashboard():
        st.subheader("⚠️ 불량 대시보드")
        st.info("defect_dashboard.py 모듈을 찾을 수 없습니다.")

# --- 전역 상수 및 브랜드 컬러 ---
PROCESS_ORDER = ["사출조립", "분리", "하이드레이션/전면검사", "접착/멸균", "누수/규격검사"]
BRAND_BLUE = "#004A99"  # Interojo Identity Color
BG_COLOR = "#F8F9FA"

# --- [원본 로직] 데이터 처리 함수군 ---
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name="생산실적현황")
    if not np.issubdtype(df["생산일자"].dtype, np.datetime64):
        df["생산일자"] = pd.to_datetime(df["생산일자"], errors="coerce")
    return df

def map_process_codes(df: pd.DataFrame) -> pd.DataFrame:
    split_df = df["공정코드"].astype(str).str.split("]", n=1, expand=True)
    processes = split_df[1].fillna(split_df[0]).str.strip()
    processes = processes.str.replace(r"\s*/\s*", "/", regex=True)
    processes = processes.str.replace(r"\s{2,}", " ", regex=True)
    df = df.copy()
    df["Process"] = processes
    return df

def compute_yields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["yield"] = df["양품수량"].astype(float) / df["생산수량"].astype(float)
    return df

def format_numbers(df_in: pd.DataFrame) -> pd.DataFrame:
    """숫자 천단위 콤마 및 소수점 포맷팅"""
    df_out = df_in.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            if pd.api.types.is_float_dtype(df_out[col]):
                df_out[col] = df_out[col].apply(lambda x: f"{x:,.2f}")
            else:
                df_out[col] = df_out[col].apply(lambda x: f"{int(x):,}")
    return df_out

def summarise_by_group(df: pd.DataFrame, group_cols: Tuple[str, ...]) -> pd.DataFrame:
    agg = df.groupby(list(group_cols) + ["공장", "Process"]).agg({"양품수량": "sum", "생산수량": "sum"})
    agg["yield"] = agg["양품수량"] / agg["생산수량"]
    agg = agg.reset_index()
    pivot = agg.pivot_table(index=list(group_cols) + ["공장"], columns="Process", values="yield", aggfunc="mean").fillna(1.0)
    # 종합 수율 계산 (모든 공정 수율의 곱)
    pivot["overall_yield"] = pivot.apply(lambda row: np.prod([row[c] for c in row.index if c in PROCESS_ORDER]), axis=1)
    return pivot

# --- [원본 로직] 복합 차트 데이터 계산 ---
def compute_combined_summaries(full_df: pd.DataFrame):
    def _get_yield(group):
        p_agg = group.groupby('Process').agg(g=('양품수량', 'sum'), p=('생산수량', 'sum'))
        y = 1.0
        for proc in PROCESS_ORDER:
            if proc in p_agg.index and p_agg.loc[proc, 'p'] > 0:
                y *= (p_agg.loc[proc, 'g'] / p_agg.loc[proc, 'p'])
        return y

    # 월별 요약
    m_records = []
    for (y, m), group in full_df.groupby(['년도', '월']):
        res = {'연-월': f"{int(y)}-{m}", '양품량': group['양품수량'].sum(), '종합수율': _get_yield(group)}
        for p_name in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
            sub = group[group['공장'] == p_name]
            res[p_name.split('(')[0]] = _get_yield(sub) if not sub.empty else 1.0
        m_records.append(res)
    
    # 주차별 요약 (W2 이상)
    w_records = []
    full_df['_w_num'] = full_df['주차'].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(float)
    for (y, w), group in full_df.groupby(['년도', '_w_num']):
        if w < 2: continue
        res = {'주차': f"W{int(w)}", '양품량': group['양품수량'].sum(), '종합수율': _get_yield(group)}
        for p_name in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
            sub = group[group['공장'] == p_name]
            res[p_name.split('(')[0]] = _get_yield(sub) if not sub.empty else 1.0
        w_records.append(res)
    
    return pd.DataFrame(m_records).set_index('연-월'), pd.DataFrame(w_records).set_index('주차')

# --- [디자인] 커스텀 스타일 적용 ---
def apply_custom_style():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG_COLOR}; }}
    
    /* KPI 카드 디자인 */
    div[data-testid="stMetric"] {{
        background-color: #ffffff;
        border: 1px solid #e1e8ed;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
    }}
    
    /* 헤더 섹션 */
    .header-box {{
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid {BRAND_BLUE};
    }}
    .main-title {{ font-size: 2.5rem; font-weight: 800; color: #1e293b; }}
    .sub-title {{ font-size: 1.1rem; color: #64748b; }}
    
    /* 사이드바 스타일 */
    .stSidebar {{ background-color: #ffffff; border-right: 1px solid #eee; }}
    
    /* 탭 디자인 */
    .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #ffffff; border-radius: 10px 10px 0 0;
        padding: 10px 25px; font-weight: 700; color: #475569;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {BRAND_BLUE} !important; color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="인터로조 공정기술팀 Dashboard", layout="wide")
    apply_custom_style()

    # 데이터 로드
    candidates = ["공정기술팀 대시보드(수율).xlsx", "공정기술팀 대시보드_260312.xlsx", "공정기술팀 대시보드.xlsx", "공정기술팀 대시보드_260308.xlsx"]
    file_path = next((os.path.join(os.path.dirname(__file__), n) for n in candidates if os.path.exists(os.path.join(os.path.dirname(__file__), n))), None)
    
    if not file_path:
        st.error("데이터 파일을 찾을 수 없습니다. (엑셀 파일 확인 필요)")
        return

    @st.cache_data
    def get_processed_data(path):
        data = load_data(path)
        data = map_process_codes(data)
        data = compute_yields(data)
        data["년도"] = data["생산일자"].dt.year
        data["월"] = data["생산일자"].dt.month.astype(str) + "월"
        data["주차"] = data["생산일자"].dt.isocalendar().week.astype(int)
        return data.dropna(subset=["yield"])

    df = get_processed_data(file_path)

    # --- 사이드바 (원본 필터 100% 구현) ---
    with st.sidebar:
        st.markdown(f"<h1 style='color:{BRAND_BLUE}; font-size:24px;'>INTEROJO BI</h1>", unsafe_allow_html=True)
        st.divider()
        
        plants = ["전체 공장"] + sorted(df["공장"].dropna().astype(str).unique().tolist())
        selected_plant = st.radio("🏭 공장 선택", options=plants, index=0)
        
        years = sorted(df["년도"].dropna().unique().tolist())
        months = sorted(df["월"].dropna().unique().tolist())
        weeks = [f"W{w}" for w in sorted(df["주차"].unique().tolist()) if w >= 2]
        
        st.divider()
        selected_year = st.selectbox("📅 년도", ["전체"] + [str(y) for y in years])
        selected_month = st.selectbox("📆 월", ["전체"] + months)
        selected_week = st.selectbox("📅 주차", ["전체"] + weeks)
        
        st.divider()
        st.markdown("### 🔍 불량 세부 필터")
        defect_types = ["전체", "파손", "엣지기포", "엣지", "미분리", "뜯김", "리드지 불량", "블리스터 불량"]
        selected_defect_type = st.selectbox("불량 유형", defect_types)

    # --- 메인 타이틀 ---
    st.markdown(f"""
        <div class="header-box">
            <div class="main-title">인터로조 공정기술팀</div>
            <div class="sub-title">통합 수율 시뮬레이션 및 생산 분석 시스템</div>
        </div>
    """, unsafe_allow_html=True)

    # 데이터 필터링
    f_df = df.copy()
    if selected_plant != "전체 공장": f_df = f_df[f_df["공장"] == selected_plant]
    if selected_year != "전체": f_df = f_df[f_df["년도"].astype(str) == selected_year]
    if selected_month != "전체": f_df = f_df[f_df["월"] == selected_month]
    if selected_week != "전체":
        w_num = int(selected_week.replace("W", ""))
        f_df = f_df[f_df["주차"] == w_num]

    # 대시보드 모드 전환
    m_tab = st.radio("대시보드 모드 선택", ["📊 수율 대시보드", "⚠️ 불량 대시보드"], horizontal=True)
    
    if "불량" in m_tab:
        run_defect_dashboard()
        return

    # --- KPI 지표 (Premium Metrics) ---
    st.subheader("공정별 수율 현황 (KPI)")
    p_summ = f_df.groupby("Process").agg({"양품수량": "sum", "생산수량": "sum"})
    p_summ["yield"] = p_summ["양품수량"] / p_summ["생산수량"]
    
    kpi_cols = st.columns(len(PROCESS_ORDER) + 1)
    total_y = 1.0
    for i, p_name in enumerate(PROCESS_ORDER):
        val = p_summ.loc[p_name, "yield"] if p_name in p_summ.index else 1.0
        total_y *= val
        kpi_cols[i].metric(label=p_name, value=f"{val*100:.2f}%")
    
    with kpi_cols[-1]:
        st.metric(label="🏆 종합 수율", value=f"{total_y*100:.2f}%", delta=None)

    st.divider()

    # --- 종합 트렌드 차트 (원본 Combined Chart 복구) ---
    st.subheader("생산 트렌드 분석")
    m_comb, w_comb = compute_combined_summaries(df) # 트렌드는 전체 흐름을 위해 원본 데이터 사용

    def draw_combined_plotly(comb_df, title):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # 양품량 Bar (M단위)
        fig.add_trace(go.Bar(x=comb_df.index, y=comb_df['양품량']/1_000_000, name="양품량(M)", 
                             marker_color='#86b3d1', opacity=0.4), secondary_y=True)
        # 공장별 수율 Line
        colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
        for col, color in zip(['A관', 'C관', 'S관', '종합수율'], colors):
            if col in comb_df.columns:
                fig.add_trace(go.Scatter(x=comb_df.index, y=comb_df[col]*100, name=col, 
                                         line=dict(color=color, width=2.5), marker=dict(size=6)), secondary_y=False)
        
        fig.update_layout(title=title, plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="수율 (%)", range=[50, 105], secondary_y=False, gridcolor="#eee")
        fig.update_yaxes(title_text="양품량 (M)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: draw_combined_plotly(m_comb, "월간 생산량 및 수율 추이")
    with c2: draw_combined_plotly(w_comb, "주간 생산량 및 수율 추이")

    # --- 하단 세부 분석 탭 ---
    st.divider()
    t1, t2, t3 = st.tabs(["🗓️ 일별 상세 추이", "📦 공정별 비중", "📋 로우 데이터"])

    with t1:
        y_range = st.slider("그래프 Y축 범위 (%)", 0, 100, (80, 100))
        d_pivot = f_df.groupby(["생산일자", "Process"])["yield"].mean().unstack().fillna(1.0) * 100
        fig_d = px.line(d_pivot.reset_index(), x="생산일자", y=d_pivot.columns, markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_d.update_yaxes(range=[y_range[0], y_range[1]])
        st.plotly_chart(fig_d, use_container_width=True)

    with t2:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write("### 공정별 생산량 비중")
            st.plotly_chart(px.pie(f_df, values="생산수량", names="Process", hole=0.4), use_container_width=True)
        with col_p2:
            st.write("### 공장별 생산량 비중")
            st.plotly_chart(px.pie(f_df, values="생산수량", names="공장"), use_container_width=True)

    with t3:
        # 원본의 '전체 데이터 보기' 버튼 로직 100% 복구
        if 'show_data' not in st.session_state:
            st.session_state.show_data = False
        
        btn_col1, btn_col2 = st.columns([1, 5])
        with btn_col1:
            if not st.session_state.show_data:
                if st.button("전체 데이터 보기", type="primary"):
                    st.session_state.show_data = True
                    st.rerun()
            else:
                if st.button("데이터 숨기기"):
                    st.session_state.show_data = False
                    st.rerun()

        if st.session_state.show_data:
            display_df = f_df.copy()
            display_df["수율(%)"] = (display_df["yield"] * 100).round(2)
            st.dataframe(format_numbers(display_df), use_container_width=True)

if __name__ == "__main__":
    main()
