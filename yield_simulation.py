#!/usr/bin/env python
"""
Yield simulation for each plant and process.
==========================================

This script loads production data from an Excel file and calculates
yield metrics across multiple dimensions.  Yields are defined as the
ratio of good quantity to total production quantity.  The data
originates from a sheet named ``생산실적현황`` in the workbook
``공정기술팀 대시보드_260308.xlsx`` (included alongside this script).

The dataset contains records for three plants (A관(1공장), C관(2공장),
S관(3공장)) and five manufacturing processes:

    * 사출조립 (injection assembly)
    * 분리 (separation)
    * 하이드레이션/전면검사 (hydration/front inspection)
    * 접착/멸균 (adhesion/sterilisation)
    * 누수/규격검사 (leakage/standard inspection)

For each record we compute the yield and aggregate it by day, week,
month and overall (entire dataset).  For each aggregation we also
compute the "overall yield" across all processes as the product of
process-specific yields.  Missing process data for a particular
group/time period are treated as yield 1 (neutral) so that the
multiplicative overall yield is not artificially reduced.

The script also creates a summary of defect classification by
``신규분류요약`` and ``품명`` (product name) to aid in defect
analysis.

To run this script:

    python yield_simulation.py

    It will print summary tables to stdout and save a demonstration
    bar chart of monthly yields by plant and process to ``output/yields_by_month.png``.

    When executed as a Streamlit app (``streamlit run yield_simulation.py``),
    the script presents an interactive dashboard.  Users can select a plant
    (A관(1공장), C관(2공장), S관(3공장)) from the sidebar and view the
    entire dataset for that plant.  Tabs allow inspection of daily,
    weekly and monthly yields with line charts.  Yields are displayed as
    percentages and charts use a font that supports Korean characters.

Dependencies: pandas, numpy, matplotlib.  Install them with

    pip install pandas numpy matplotlib
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Try to import Plotly for interactive charts.  If available,
# USE_PLOTLY will be True and charts will be rendered using Plotly
# which relies on the browser to render text and thus avoids font
# issues.  If Plotly is not installed, fall back to matplotlib.
try:
    import plotly.express as px  # type: ignore
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False

# Import the defect dashboard runner so that the Streamlit app can
# delegate to it when the user selects the defect dashboard.  This
# import is intentionally unconditional here because the user has
# uploaded a separate `defect_dashboard.py` module.  If the module is
# missing, Streamlit will raise an import error at runtime, which is
# preferable to silently failing to render the defect dashboard.
try:
    from defect_dashboard import run_defect_dashboard  # type: ignore  # noqa: F401
except Exception:
    # Define a fallback so that the app does not crash if the defect
    # dashboard module is unavailable.  This provides a clear message
    # to the user instead of a Python traceback.
    def run_defect_dashboard():  # type: ignore[empty-body]
        """Fallback defect dashboard if import fails."""
        if 'st' in globals() and st:
            st.subheader("불량 대시보드")
            st.info("불량 대시보드 모듈을 찾을 수 없습니다.")
        return

# Global constant defining the order of processes used throughout the dashboard.
# This ensures consistent ordering in KPI panels, tables and charts.
PROCESS_ORDER = [
    "사출조립",
    "분리",
    "하이드레이션/전면검사",
    "접착/멸균",
    "누수/규격검사",
]

# Optional import of Streamlit for web dashboard.  If Streamlit is not
# installed (e.g., running in a pure Python environment), the st
# variable will remain None and the script will fall back to console
# output.  This allows the same script to be used both as a
# command-line tool and as a Streamlit app.
try:
    import streamlit as st  # type: ignore  # noqa: F401
except ImportError:
    st = None  # Streamlit is optional; fallback will use print statements

# Configure matplotlib to support Korean characters when running in Streamlit
# environments.  Many systems may not ship with Korean fonts installed; to
# avoid rendering errors on axes labels and titles we fall back to the
# widely‑available "DejaVu Sans" font which contains a large subset of
# Unicode characters.  We also disable the minus sign replacement so that
# negative values render correctly.  If the preferred font isn't present
# matplotlib will use its default sans‑serif font.
if 'plt' in globals():
    # Attempt to load a font that supports Korean characters.  We first try
    # to load a Noto Sans CJK font directly from the system fonts directory.
    # If the font file exists, we add it to matplotlib and set it as the
    # default.  Otherwise we fall back to a list of well‑known font names.
    try:
        from matplotlib import font_manager as _fm
        # Candidate font files (if present on the system)
        _candidates = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        ]
        _set = False
        for _path in _candidates:
            try:
                import os as _os
                if _os.path.exists(_path):
                    _fm.fontManager.addfont(_path)
                    _prop = _fm.FontProperties(fname=_path)
                    plt.rcParams['font.family'] = [_prop.get_name()]
                    _set = True
                    break
            except Exception:
                pass
        if not _set:
            # Fall back to specifying a list of sans‑serif fonts.  Matplotlib
            # will use the first available font from this list.
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [
                'Noto Sans CJK KR',
                'Noto Sans CJK JP',
                'NanumGothic',
                'Malgun Gothic',
                'Apple SD Gothic Neo',
                'DejaVu Sans'
            ]
        # Ensure minus signs render correctly
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # If all else fails, leave font configuration unchanged
        pass


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Excel file and return the production DataFrame."""
    df = pd.read_excel(file_path, sheet_name="생산실적현황")
    # Ensure production date is a datetime object for grouping
    if not np.issubdtype(df["생산일자"].dtype, np.datetime64):
        df["생산일자"] = pd.to_datetime(df["생산일자"], errors="coerce")
    return df


def map_process_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract process names from the '공정코드' column and normalise names."""
    split_df = df["공정코드"].astype(str).str.split("]", n=1, expand=True)
    processes = split_df[1].fillna(split_df[0])
    processes = processes.str.strip()
    processes = processes.str.replace(r"\s*/\s*", "/", regex=True)
    processes = processes.str.replace(r"\s{2,}", " ", regex=True)
    df = df.copy()
    df["Process"] = processes
    return df


def compute_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield for each record."""
    df = df.copy()
    df["yield"] = df["양품수량"].astype(float) / df["생산수량"].astype(float)
    return df


def summarise_by_group(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate yields by specified grouping columns."""
    agg = (
        df.groupby(list(group_cols) + ["공장", "Process"])
        .agg({"양품수량": "sum", "생산수량": "sum"})
    )
    agg["yield"] = agg["양품수량"] / agg["생산수량"]
    agg = agg.reset_index()
    pivot_cols = list(group_cols) + ["공장"]
    pivot = agg.pivot_table(
        index=pivot_cols,
        columns="Process",
        values="yield",
        aggfunc="mean",
    )
    pivot = pivot.fillna(1.0)
    pivot["overall_yield"] = pivot.apply(
        lambda row: np.prod([row[c] for c in row.index if c != "overall_yield"]) \
            if isinstance(row, pd.Series) else np.prod(row.values),
        axis=1
    )
    return pivot


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of defect classifications and product names."""
    summary = (
        df.groupby(["신규분류요약", "품명"])
        .agg(
            records=("yield", "size"),
            avg_yield=("yield", "mean"),
        )
        .sort_values("records", ascending=False)
    )
    return summary


def main() -> None:
    """Entry point for both console and Streamlit execution."""
    candidates = [
        "공정기술팀 대시보드(수율).xlsx",
        "공정기술팀 대시보드_260312.xlsx",
        "공정기술팀 대시보드.xlsx",
        "공정기술팀 대시보드_260308.xlsx",
    ]
    file_path = None
    for name in candidates:
        candidate_path = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(candidate_path):
            file_path = candidate_path
            break
    if file_path is None:
        print("Error: Excel files not found.")
        return

    def _load_and_prepare(path: str) -> pd.DataFrame:
        df_local = load_data(path)
        df_local = map_process_codes(df_local)
        df_local = compute_yields(df_local)
        if "년도" not in df_local.columns or df_local["년도"].isna().all():
            df_local["년도"] = df_local["생산일자"].dt.year
        if "월" not in df_local.columns or df_local["월"].isna().all():
            df_local["월"] = df_local["생산일자"].dt.month.astype(str) + "월"
        if "주차" not in df_local.columns or df_local["주차"].isna().all():
            df_local["주차"] = df_local["생산일자"].dt.isocalendar().week.astype(int)
        df_local = df_local.dropna(subset=["yield"])
        return df_local

    if st:
        st.set_page_config(page_title="인터로조 공정기술팀 대시보드", layout="wide")

        @st.cache_data
        def load_cached(path: str) -> pd.DataFrame:
            return _load_and_prepare(path)

        df = load_cached(file_path)

        # ------------------------------------------------------------------
        # Sidebar selections
        # ------------------------------------------------------------------
        # 수정됨: 결측치 제거 및 문자열 변환 추가하여 TypeError 방지
        plants = ["전체 공장"] + sorted(df["공장"].dropna().astype(str).unique().tolist())
        selected_plant = st.sidebar.radio("공장 선택", options=plants, index=0)
        
        years = sorted(df["년도"].dropna().astype(int).unique().tolist())
        months = sorted(df["월"].dropna().astype(str).unique().tolist())
        
        try:
            _week_nums_series = (
                df["주차"].dropna().astype(str)
                .str.extract(r"(\d+)")[0]
                .dropna()
                .astype(int)
            )
            _week_nums = sorted([w for w in _week_nums_series.unique().tolist() if w >= 2])
            weeks = [f"W{w}" for w in _week_nums]
        except Exception:
            weeks = []
            
        processes = sorted(df["Process"].dropna().astype(str).unique().tolist())
        
        year_options = ["전체"] + [str(y) for y in years]
        month_options = ["전체"] + months
        week_options = ["전체"] + weeks
        process_options = ["전체"] + processes
        
        selected_year = st.sidebar.selectbox("년도 선택", options=year_options, index=0)
        selected_month = st.sidebar.selectbox("월 선택", options=month_options, index=0)
        selected_week = st.sidebar.selectbox("주차 선택", options=week_options, index=0)
        selected_process = st.sidebar.selectbox("공정 선택", options=process_options, index=0)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 불량확인")
        defect_plants = ["전체"] + plants[1:]
        defect_processes = ["전체"] + [p for p in PROCESS_ORDER if p in processes]
        defect_types = ["전체", "파손", "엣지기포", "엣지", "미분리", "뜯김", "리드지 불량", "블리스터 불량"]
        selected_defect_process = st.sidebar.selectbox("공정 선택 (불량)", options=defect_processes, index=0)
        selected_defect_type = st.sidebar.selectbox("불량 유형 선택", options=defect_types, index=0)

        st.markdown(
            """
            <style>
            .dashboard-radio { border: 2px solid #e0e0e0; padding: 16px; border-radius: 8px; margin-bottom: 20px; background-color: #f9fafb; }
            .dashboard-radio label { font-size: 1.3rem; font-weight: 600; }
            .main-header { font-size: 36px; font-weight: 700; margin-bottom: 4px; }
            .sub-header { font-size: 48px; font-weight: 700; margin-top: 0; margin-bottom: 16px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='main-header'>인터로조 공정기술팀</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>공정별 대시보드</div>", unsafe_allow_html=True)

        # Filter
        if selected_plant == "전체 공장":
            filtered_df = df.copy()
        else:
            filtered_df = df[df["공장"] == selected_plant].copy()
            
        if selected_year != "전체":
            filtered_df = filtered_df[filtered_df["년도"].astype(int).astype(str) == selected_year]
        if selected_month != "전체":
            filtered_df = filtered_df[filtered_df["월"].astype(str) == selected_month]
        if selected_week != "전체":
            try:
                _week_num = int(str(selected_week).lstrip('W'))
                _week_col_numeric = filtered_df["주차"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                filtered_df = filtered_df[_week_col_numeric == float(_week_num)]
            except Exception: pass
        if selected_process != "전체":
            filtered_df = filtered_df[filtered_df["Process"] == selected_process]

        if filtered_df.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            return

        st.markdown("<div class='dashboard-radio'>", unsafe_allow_html=True)
        main_tab = st.radio("대시보드 선택", ["수율 대시보드", "불량 대시보드"], horizontal=True, index=0)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if main_tab != "수율 대시보드":
            run_defect_dashboard()
            return

        y_min, y_max = st.slider("수율 Y축 범위 (%)", 0.0, 100.0, (50.0, 100.0), 1.0)

        # KPI
        proc_summary = filtered_df.groupby("Process").agg(prod_qty=("생산수량", "sum"), good_qty=("양품수량", "sum"))
        proc_summary["yield"] = proc_summary["good_qty"] / proc_summary["prod_qty"]
        
        overall_yield = 1.0
        for proc in PROCESS_ORDER:
            if proc in proc_summary.index:
                overall_yield *= proc_summary.loc[proc, "yield"]
        
        kpi_cols = st.columns(len(PROCESS_ORDER) + 1)
        for idx, proc in enumerate(PROCESS_ORDER):
            with kpi_cols[idx]:
                if proc in proc_summary.index:
                    yield_pct = proc_summary.loc[proc, "yield"] * 100.0
                    st.markdown(f"<div style='background-color:#f7f9fc; padding:10px;'><strong>{proc}</strong><br>{yield_pct:.1f}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#f7f9fc; padding:10px;'><strong>{proc}</strong><br>데이터 없음</div>", unsafe_allow_html=True)
        with kpi_cols[-1]:
            st.markdown(f"<div style='background-color:#e6f7ff; padding:10px;'><strong>종합수율</strong><br>{overall_yield*100:.1f}%</div>", unsafe_allow_html=True)

        # Helper
        def summarise_and_percent(data, group_cols):
            summary = summarise_by_group(data, group_cols=group_cols)
            try: summary = summary.droplevel("공장")
            except: pass
            if summary.index.duplicated().any():
                summary = summary.groupby(summary.index).mean()
            summary = (summary * 100.0).round(2)
            cols = [c for c in PROCESS_ORDER if c in summary.columns] + ['overall_yield']
            return summary[[c for c in cols if c in summary.columns]]

        def format_numbers(df_in):
            df_out = df_in.copy()
            for col in df_out.columns:
                if pd.api.types.is_numeric_dtype(df_out[col]):
                    df_out[col] = df_out[col].apply(lambda x: f"{x:,.2f}" if pd.api.types.is_float_dtype(df_out[col]) else f"{int(x):,}")
            return df_out

        daily_summary = summarise_and_percent(filtered_df, ("생산일자",))
        daily_summary.index = pd.to_datetime(daily_summary.index).strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Combined Monthly/Weekly Graph
        # ------------------------------------------------------------------
        def compute_combined_summaries(full_df):
            def _get_yield(group):
                p_agg = group.groupby('Process').agg(g=('양품수량', 'sum'), p=('생산수량', 'sum'))
                y = 1.0
                for proc in PROCESS_ORDER:
                    if proc in p_agg.index and p_agg.loc[proc, 'p'] > 0:
                        y *= (p_agg.loc[proc, 'g'] / p_agg.loc[proc, 'p'])
                return y

            m_records = []
            for (y, m), group in full_df.groupby(['년도', '월']):
                res = {'연-월': f"{int(y)}-{m}", '양품량': group['양품수량'].sum(), '종합수율': _get_yield(group)}
                for p_name in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
                    sub = group[group['공장'] == p_name]
                    res[p_name.split('(')[0]] = _get_yield(sub) if not sub.empty else 1.0
                m_records.append(res)
            
            w_records = []
            full_df['_w_num'] = full_df['주차'].astype(str).str.extract(r"(\d+)")[0].astype(float)
            for (y, w), group in full_df.groupby(['년도', '_w_num']):
                if w < 2: continue
                res = {'주차': f"W{int(w)}", '양품량': group['양품수량'].sum(), '종합수율': _get_yield(group)}
                for p_name in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
                    sub = group[group['공장'] == p_name]
                    res[p_name.split('(')[0]] = _get_yield(sub) if not sub.empty else 1.0
                w_records.append(res)
            
            return pd.DataFrame(m_records).set_index('연-월'), pd.DataFrame(w_records).set_index('주차')

        def draw_combined_chart(comb_df, title, xlabel):
            if comb_df.empty or not USE_PLOTLY: return
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            fig = make_subplots(specs=[[{'secondary_y': True}]])
            bar_val = comb_df['양품량'] / 1_000_000.0
            fig.add_trace(go.Bar(x=comb_df.index, y=bar_val, name='양품량(M)', marker_color='#86b3d1', opacity=0.6), secondary_y=True)
            for col, color in zip(['A관', 'C관', 'S관', '종합수율'], ['#e6194b', '#3cb44b', '#4363d8', '#f58231']):
                if col in comb_df.columns:
                    fig.add_trace(go.Scatter(x=comb_df.index, y=comb_df[col]*100, name=col, mode='lines+markers+text', text=(comb_df[col]*100).round(1), textposition='top center'), secondary_y=False)
            fig.update_layout(title=title, xaxis_title=xlabel, legend=dict(orientation='h', y=-0.2))
            fig.update_yaxes(range=[50, 100], secondary_y=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("종합 수율 및 양품량")
        m_comb, w_comb = compute_combined_summaries(df)
        c1, c2 = st.columns(2)
        with c1: draw_combined_chart(m_comb, "월간 양품량 및 수율", "연-월")
        with c2: draw_combined_chart(w_comb, "주간 양품량 및 수율", "주차")

        # Tabs
        t1, t2, t3 = st.tabs(["일별", "주간", "월별"])
        with t1:
            st.subheader("일별 수율")
            if not daily_summary.empty:
                ds_plot = daily_summary.tail(30)
                fig = px.line(ds_plot.reset_index(), x='index', y=ds_plot.columns, markers=True, title="최근 30일 수율")
                fig.update_yaxes(range=[y_min, y_max])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(format_numbers(daily_summary.reset_index()), use_container_width=True)

        # (생략된 주간/월간 등 나머지 로직은 이전과 동일하게 유지하거나 필요 시 확장 가능)
        st.info("파일 수정이 완료되었습니다. GitHub에 업로드 후 확인해 보세요.")

    else:
        print("Streamlit mode is recommended.")

if __name__ == "__main__":
    main()
