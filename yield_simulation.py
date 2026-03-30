#!/usr/bin/env python
"""
인터로조 공정기술팀 - 프리미엄 통합 수율 대시보드
원본의 모든 기능과 로직을 유지하며 디자인만 현대적으로 개선했습니다.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# 시각화 라이브러리
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
        st.info("⚠️ 불량 대시보드 모듈(defect_dashboard.py)을 불러올 수 없습니다.")

# --- 전역 상수 및 설정 ---
PROCESS_ORDER = ["사출조립", "분리", "하이드레이션/전면검사", "접착/멸균", "누수/규격검사"]
BRAND_BLUE = "#004A99"
ACCENT_BLUE = "#EBF3FF"
BG_COLOR = "#F4F7F9"

# --- [원본 유지] 데이터 로직 함수 ---
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

@st.cache_data
def _load_and_prepare(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name="생산실적현황")
    if not np.issubdtype(df["생산일자"].dtype, np.datetime64):
        df["생산일자"] = pd.to_datetime(df["생산일자"], errors="coerce")
    df = map_process_codes(df)
    df = compute_yields(df)
    df["년도"] = df["생산일자"].dt.year
    df["월"] = df["생산일자"].dt.month.astype(str) + "월"
    df["주차"] = df["생산일자"].dt.isocalendar().week.astype(int)
    df["day"] = df["생산일자"].dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["yield"])

def summarise_by_group(df: pd.DataFrame, group_cols: Tuple[str, ...]) -> pd.DataFrame:
    agg = df.groupby(list(group_cols) + ["공장", "Process"]).agg({"양품수량": "sum", "생산수량": "sum"})
    agg["yield"] = agg["양품수량"] / agg["생산수량"]
    agg = agg.reset_index()
    pivot = agg.pivot_table(index=list(group_cols) + ["공장"], columns="Process", values="yield", aggfunc="mean").fillna(1.0)
    pivot["overall_yield"] = pivot.apply(lambda row: np.prod([row[c] for c in row.index if c in PROCESS_ORDER]), axis=1)
    return pivot

def format_numbers(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            if pd.api.types.is_float_dtype(df_out[col]):
                df_out[col] = df_out[col].apply(lambda x: f"{x:,.2f}")
            else:
                df_out[col] = df_out[col].apply(lambda x: f"{int(x):,}")
    return df_out

# --- [그래프 업그레이드] 원본 로직 기반 시각화 ---
def _plot_combined_yield_chart(summary_df: pd.DataFrame, title: str):
    """원본의 복합 그래프 로직을 Plotly로 세련되게 재현"""
    if summary_df.empty:
        st.warning(f"{title} 데이터가 없습니다.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. 양품량 Bar (M 단위)
    fig.add_trace(go.Bar(
        x=summary_df.index, y=summary_df['양품량']/1_000_000, 
        name="양품량(M)", marker_color='#AABBCB', opacity=0.4
    ), secondary_y=True)
    
    # 2. 공장별/종합 수율 Line
    colors = {'A관': '#E6194B', 'C관': '#3CB44B', 'S관': '#4363D8', '종합수율': '#F58231'}
    for col in ['A관', 'C관', 'S관', '종합수율']:
        if col in summary_df.columns:
            fig.add_trace(go.Scatter(
                x=summary_df.index, y=summary_df[col]*100, 
                name=col, line=dict(color=colors.get(col, '#999'), width=2.5),
                marker=dict(size=7)
            ), secondary_y=False)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B')),
        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fig.update_yaxes(title_text="수율 (%)", range=[50, 105], secondary_y=False, gridcolor="#EEE")
    fig.update_yaxes(title_text="양품량 (M)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

# --- [디자인] CSS Injection ---
def apply_custom_style():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG_COLOR}; }}
    
    /* 카드 디자인 */
    div[data-testid="stMetric"] {{
        background-color: #ffffff; border: 1px solid #E1E8ED; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
    }}
    
    /* 타이틀 섹션 */
    .header-box {{
        padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 3px solid {BRAND_BLUE};
    }}
    .main-title {{ font-size: 2.6rem; font-weight: 800; color: #1E293B; }}
    .sub-title {{ font-size: 1.1rem; color: #64748B; }}

    /* 탭 디자인 */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #ffffff; border-radius: 10px 10px 0 0;
        padding: 8px 24px; font-weight: 700; color: #475569;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {BRAND_BLUE} !important; color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 메인 실행 ---
def main():
    st.set_page_config(page_title="인터로조 수율 대시보드", layout="wide")
    apply_custom_style()

    # 파일 로드 (모든 후보군 유지)
    candidates = ["공정기술팀 대시보드(수율).xlsx", "공정기술팀 대시보드_260312.xlsx", "공정기술팀 대시보드.xlsx", "공정기술팀 대시보드_260308.xlsx"]
    file_path = next((os.path.join(os.path.dirname(__file__), n) for n in candidates if os.path.exists(os.path.join(os.path.dirname(__file__), n))), None)
    
    if not file_path:
        st.error("데이터 파일을 찾을 수 없습니다. (xlsx 파일 필요)")
        return

    df = _load_and_prepare(file_path)

    # --- 사이드바 필터 ---
    with st.sidebar:
        st.markdown(f"<h1 style='color:{BRAND_BLUE};'>INTEROJO BI</h1>", unsafe_allow_html=True)
        st.divider()
        
        plants = ["전체 공장"] + sorted(df["공장"].unique().tolist())
        selected_plant = st.radio("🏭 공장 선택", plants, index=0)
        
        years = sorted(df["년도"].unique().tolist())
        months = sorted(df["월"].unique().tolist())
        weeks = [f"W{w}" for w in sorted(df["주차"].unique().tolist()) if w >= 2]
        
        st.divider()
        selected_year = st.selectbox("📅 년도", ["전체"] + [str(y) for y in years])
        selected_month = st.selectbox("📆 월", ["전체"] + months)
        selected_week = st.selectbox("📅 주차", ["전체"] + weeks)
        
        st.divider()
        st.markdown("🔍 **기타 필터**")
        selected_defect = st.selectbox("불량 유형", ["전체", "파손", "엣지기포", "엣지", "미분리", "뜯김", "리드지 불량"])

    # --- 헤더 ---
    st.markdown(f"""
        <div class="header-box">
            <div class="main-title">인터로조 공정기술팀</div>
            <div class="sub-title">통합 수율 모니터링 및 생산 분석 대시보드</div>
        </div>
    """, unsafe_allow_html=True)

    # 필터링
    f_df = df.copy()
    if selected_plant != "전체 공장": f_df = f_df[f_df["공장"] == selected_plant]
    if selected_year != "전체": f_df = f_df[f_df["년도"].astype(str) == selected_year]
    if selected_month != "전체": f_df = f_df[f_df["월"] == selected_month]
    if selected_week != "전체":
        f_df = f_df[f_df["주차"] == int(selected_week.replace("W", ""))]

    # 대시보드 전환
    mode = st.radio("Dashboard Mode", ["수율 분석", "불량 분석"], horizontal=True)
    if "불량" in mode:
        run_defect_dashboard()
        return

    # --- KPI Section ---
    st.subheader("Key Performance Indicators")
    p_summ = f_df.groupby("Process").agg({"양품수량": "sum", "생산수량": "sum"})
    p_summ["yield"] = p_summ["양품수량"] / p_summ["생산수량"]
    
    k_cols = st.columns(len(PROCESS_ORDER) + 1)
    total_y = 1.0
    for i, p_name in enumerate(PROCESS_ORDER):
        val = p_summ.loc[p_name, "yield"] if p_name in p_summ.index else 1.0
        total_y *= val
        k_cols[i].metric(label=p_name, value=f"{val*100:.2f}%")
    k_cols[-1].metric(label="🏆 종합 수율", value=f"{total_y*100:.2f}%")

    st.divider()

    # --- 복합 그래프 로직 (원본 기능 100% 복구) ---
    def get_period_summary(target_df, freq='M'):
        records = []
        group_key = ['년도', '월'] if freq == 'M' else ['년도', '주차']
        for key, group in target_df.groupby(group_key):
            label = f"{int(key[0])}-{key[1]}" if freq == 'M' else f"W{int(key[1])}"
            if freq == 'W' and int(key[1]) < 2: continue
            
            # 종합/공장별 수율 계산 (원본 multiplicative 로직)
            def _calc_y(sub):
                if sub.empty: return 1.0
                ps = sub.groupby('Process').agg({'양품수량':'sum','생산수량':'sum'})
                return np.prod([(ps.loc[p,'양품수량']/ps.loc[p,'생산수량']) if p in ps.index else 1.0 for p in PROCESS_ORDER])

            res = {'label': label, '양품량': group['양품수량'].sum(), '종합수율': _calc_y(group)}
            for p_nm in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
                res[p_nm.split('(')[0]] = _calc_y(group[group['공장'] == p_nm])
            records.append(res)
        return pd.DataFrame(records).set_index('label')

    st.subheader("생산 및 수율 트렌드 분석")
    c1, c2 = st.columns(2)
    with c1: _plot_combined_yield_chart(get_period_summary(df, 'M'), "월간 트렌드 (양품량 & 수율)")
    with c2: _plot_combined_yield_chart(get_period_summary(df, 'W'), "주간 트렌드 (양품량 & 수율)")

    # --- 상세 분석 탭 ---
    st.divider()
    t1, t2, t3 = st.tabs(["📅 일별 추이", "📦 공정별 비중", "📋 로우 데이터"])

    with t1:
        y_range = st.slider("수율 범위(%)", 0, 100, (80, 100))
        d_pivot = f_df.groupby(["생산일자", "Process"])["yield"].mean().unstack().fillna(1.0) * 100
        fig_d = px.line(d_pivot.reset_index(), x="생산일자", y=d_pivot.columns, markers=True)
        fig_d.update_yaxes(range=[y_range[0], y_range[1]])
        st.plotly_chart(fig_d, use_container_width=True)

    with t2:
        col_p1, col_p2 = st.columns(2)
        with col_p1: st.plotly_chart(px.pie(f_df, values="생산수량", names="Process", title="공정별 생산 비중", hole=0.4), use_container_width=True)
        with col_p2: st.plotly_chart(px.pie(f_df, values="생산수량", names="공장", title="공장별 생산 비중"), use_container_width=True)

    with t3:
        if 'show_data' not in st.session_state: st.session_state.show_data = False
        if st.button("전체 데이터 보기/숨기기"): st.session_state.show_data = not st.session_state.show_data
        
        if st.session_state.show_data:
            st.subheader(f"Filtered Dataset ({len(f_df)} rows)")
            st.dataframe(format_numbers(f_df.sort_values("생산일자", ascending=False)), use_container_width=True)

if __name__ == "__main__":
    main()
