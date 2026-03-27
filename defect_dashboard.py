"""
Streamlit 대시보드: 불량율 분석
이 모듈은 불량 데이터 엑셀 파일을 불러와 공정별‧유형별 불량율을
일/주/월 단위로 계산하여 시각화하는 대시보드를 제공합니다.

주요 개선사항
- 엑셀 파일 수정 시 캐시가 자동 갱신되도록 파일 수정시간 기반 캐시 무효화
- 하드코딩된 공정/불량 유형 대신 실제 데이터에서 동적으로 목록 생성
- 데이터가 계속 누적되어도 최신 파일을 자동 탐색하고 전체 기간을 기준으로 집계
- 신규 공정/신규 불량 유형이 추가되어도 별도 코드 수정 없이 반영
"""

import glob
import os
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


FILE_PATTERNS = [
    "공정기술팀 대시보드(불량).xlsx",
    "공정기술팀 대시보드 (불량).xlsx",
    "공정기술팀 대시보드_불량.xlsx",
    "*불량*.xlsx",
]


def find_latest_data_file() -> Optional[str]:
    """후보 패턴 중 가장 최근 수정된 불량 엑셀 파일을 반환합니다."""
    search_roots = [os.getcwd(), os.path.join(os.getcwd(), "data"), os.path.dirname(__file__)]
    candidates: List[str] = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in FILE_PATTERNS:
            candidates.extend(glob.glob(os.path.join(root, pattern)))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    """엑셀 파일을 로드하고 분석에 필요한 전처리를 수행합니다.

    file_mtime 인자를 함께 받아 Streamlit 캐시가 파일 변경 시 자동으로
    무효화되도록 합니다.
    """
    _ = file_mtime  # cache key purpose
    if not os.path.exists(file_path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path, sheet_name=0)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

    required_cols = {"생산일자", "공정", "불량명", "양품수량", "공장"}
    missing = required_cols - set(df.columns)
    if missing:
        st.warning(f"데이터에 필요한 컬럼이 없습니다: {missing}")
        return pd.DataFrame()

    df = df.copy()
    df["생산일자"] = pd.to_datetime(df["생산일자"], errors="coerce")
    df = df.dropna(subset=["생산일자"])

    df["공정명"] = (
        df["공정"]
        .astype(str)
        .apply(lambda x: x.split("]")[-1].strip() if "]" in x else x.strip())
        .str.replace(" ", "", regex=False)
    )

    df["불량유형"] = (
        df["불량명"]
        .astype(str)
        .apply(lambda x: x.split(":")[-1].strip() if ":" in x else x.strip())
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip()
    )

    defect_qty_col = "불량수량.1" if "불량수량.1" in df.columns else "불량수량"
    if defect_qty_col not in df.columns:
        st.warning("데이터에 불량수량 컬럼이 없습니다.")
        return pd.DataFrame()

    for col in ["양품수량", defect_qty_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if defect_qty_col != "불량수량":
        df["불량수량"] = df[defect_qty_col]

    df["생산수량"] = df["양품수량"] + df["불량수량"]
    df["불량율"] = df.apply(
        lambda row: (row["불량수량"] / row["생산수량"]) * 100 if row["생산수량"] else 0,
        axis=1,
    )
    df["년도"] = df["생산일자"].dt.year
    df["월"] = df["생산일자"].dt.to_period("M").astype(str)
    df["주차"] = df["생산일자"].dt.isocalendar().week.astype(int)
    return df


def summarise_by_time(
    df: pd.DataFrame,
    time_unit: str,
    categories: List[str],
    group_filter: Optional[List[str]] = None,
    defect_filter: Optional[str] = None,
    process_filter: Optional[str] = None,
) -> pd.DataFrame:
    """일/주/월 단위로 불량율 피벗 테이블을 생성합니다."""
    data = df.copy()
    if defect_filter is not None:
        data = data[data["불량유형"] == defect_filter]
    if process_filter is not None:
        data = data[data["공정명"] == process_filter]
    if group_filter is not None:
        data = data[data["공정명"].isin(group_filter)]

    if data.empty:
        return pd.DataFrame(columns=categories)

    if time_unit == "D":
        data["기간"] = data["생산일자"].dt.strftime("%Y-%m-%d")
    elif time_unit == "W":
        iso = data["생산일자"].dt.isocalendar()
        data["기간"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str)
    elif time_unit == "M":
        data["기간"] = data["생산일자"].dt.to_period("M").astype(str)
    else:
        raise ValueError("지원하지 않는 time_unit")

    pivot_col = "공정명" if defect_filter else "불량유형"
    agg = (
        data.groupby(["기간", pivot_col], dropna=False)
        .agg({"양품수량": "sum", "불량수량": "sum"})
        .reset_index()
    )
    agg["불량율"] = agg.apply(
        lambda row: (row["불량수량"] / (row["양품수량"] + row["불량수량"])) * 100
        if (row["양품수량"] + row["불량수량"]) else 0,
        axis=1,
    )
    pivot = agg.pivot(index="기간", columns=pivot_col, values="불량율").fillna(0)

    for cat in categories:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[categories]

    if time_unit == "W":
        week_order = pivot.index.to_series().str.extract(r"(\d{4})-W(\d+)")
        week_order[0] = pd.to_numeric(week_order[0], errors="coerce").fillna(0)
        week_order[1] = pd.to_numeric(week_order[1], errors="coerce").fillna(0)
        pivot = pivot.assign(_year=week_order[0].values, _week=week_order[1].values)
        pivot = pivot.sort_values(["_year", "_week"]).drop(columns=["_year", "_week"])
    else:
        pivot = pivot.sort_index()

    return pivot


def plot_time_series(
    pivot: pd.DataFrame,
    title: str,
    series_names: List[str],
    y_range: Optional[List[float]] = None,
):
    """Plotly 꺾은선 그래프를 생성합니다."""
    fig = go.Figure()
    x_vals = pivot.index.tolist()
    colors = px.colors.qualitative.Set2 if px else None
    for i, series in enumerate(series_names):
        if series not in pivot.columns:
            continue
        y_vals = pivot[series].values
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers+text",
                name=series,
                text=[f"{v:.1f}" for v in y_vals],
                textposition="top center",
                marker=dict(color=colors[i % len(colors)]) if colors else None,
                line=dict(color=colors[i % len(colors)]) if colors else None,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="기간",
        yaxis_title="불량율 (%)",
        yaxis=dict(range=y_range),
        xaxis=dict(type="category", categoryorder="array", categoryarray=x_vals),
        legend_title_text="구분",
    )
    return fig


def run_defect_dashboard():
    """Streamlit 불량 대시보드 실행 함수."""
    st.title("공정별 불량율 대시보드")

    file_path = find_latest_data_file()
    if file_path is None:
        st.error("불량 데이터 파일을 찾을 수 없습니다. 불량 엑셀 파일을 프로젝트 폴더 또는 data 폴더에 업로드해 주세요.")
        return

    file_mtime = os.path.getmtime(file_path)
    df = load_data(file_path, file_mtime)
    if df.empty:
        st.warning("데이터를 불러올 수 없거나 필수 컬럼이 없습니다.")
        return

    st.caption(f"데이터 파일: {os.path.basename(file_path)} | 최종 수정시각: {pd.to_datetime(file_mtime, unit='s')}")

    process_names = sorted([p for p in df["공정명"].dropna().unique().tolist() if str(p).strip()])
    defect_by_process = {
        proc: sorted(
            df.loc[df["공정명"] == proc, "불량유형"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
        )
        for proc in process_names
    }
    all_defects = sorted(
        df["불량유형"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )

    st.markdown("## 공정별 불량율")
    st.info("엑셀 데이터가 추가되면 파일 수정시간을 기준으로 자동 새로고침되며, 공정/불량 유형도 실제 데이터 기준으로 자동 반영됩니다.")

    for proc_name in process_names:
        categories = defect_by_process.get(proc_name, [])
        if not categories:
            continue
        st.markdown(f"### {proc_name}")
        col1, col2, col3 = st.columns(3)
        with col1:
            pivot = summarise_by_time(df, "M", categories, process_filter=proc_name)
            fig = plot_time_series(pivot, f"{proc_name} 월별 불량율", categories, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pivot = summarise_by_time(df, "W", categories, process_filter=proc_name)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{proc_name} 주간 불량율", categories, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            pivot = summarise_by_time(df, "D", categories, process_filter=proc_name)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{proc_name} 일별 불량율", categories, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 유형별 불량율")
    st.info("불량 유형별로 공정 비교가 가능하며, 신규 불량 유형이 추가되어도 자동으로 표시됩니다.")

    for defect in all_defects:
        related_processes = sorted(
            df.loc[df["불량유형"] == defect, "공정명"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
        )
        if not related_processes:
            continue
        st.markdown(f"### {defect}")
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot = summarise_by_time(df, "M", related_processes, defect_filter=defect)
            fig = plot_time_series(pivot, f"{defect} 월별 불량율", related_processes, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pivot = summarise_by_time(df, "W", related_processes, defect_filter=defect)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{defect} 주간 불량율", related_processes, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            pivot = summarise_by_time(df, "D", related_processes, defect_filter=defect)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{defect} 일별 불량율", related_processes, y_range=[0, max(50, float(pivot.max().max()) + 5 if not pivot.empty else 50)])
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    run_defect_dashboard()
