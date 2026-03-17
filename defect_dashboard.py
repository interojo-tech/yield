"""
Streamlit 대시보드: 불량율 분석
이 모듈은 불량 데이터 엑셀 파일(`공정기술팀 대시보드(불량).xlsx`)을
불러와 공정별‧유형별 불량율을 일/주/월 단위로 계산하여 시각화하는 대시보드를 제공합니다.

사용법:
```
streamlit run defect_dashboard.py
```

불량 대시보드는 수율 대시보드와 병행해서 사용할 수 있도록 설계되었습니다.
본 파일은 기존 수율 코드에 영향을 주지 않습니다.
"""

import os
from typing import List, Optional

import pandas as pd
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    """불량 데이터 엑셀을 로드합니다.

    Args:
        file_path: 엑셀 파일 경로
    Returns:
        전처리된 데이터프레임 (생산일자, 공정명, 불량율 등 추가)
    """
    if not os.path.exists(file_path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=0)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()
    # 필수 컬럼 검사
    required = {"생산일자", "공정코드", "신규분류요약", "불량수량", "생산수량", "공장"}
    missing = required - set(df.columns)
    if missing:
        st.warning(f"필수 컬럼이 누락되었습니다: {missing}")
        return pd.DataFrame()
    # 날짜 변환
    if np.issubdtype(df["생산일자"].dtype, np.number):
        # 엑셀 숫자 날짜 -> datetime
        origin = pd.Timestamp('1899-12-30')
        df["생산일자"] = df["생산일자"].apply(lambda x: origin + pd.Timedelta(days=float(x)) if pd.notna(x) else pd.NaT)
    else:
        df["생산일자"] = pd.to_datetime(df["생산일자"], errors='coerce')
    df["년도"] = df["생산일자"].dt.year
    df["월"] = df["생산일자"].dt.to_period('M').astype(str)
    df["주차"] = df["생산일자"].dt.isocalendar().week
    # 공정명 매핑 (공정코드 -> 공정명)
    process_mapping = {
        '사출조립': '사출조립공정',
        '분리': '분리공정',
        '접착/멸균': '접착/멸균',
        '누수/규격검사': '누수/규격검사'
    }
    df["공정코드"] = df["공정코드"].astype(str).str.replace(" ", "").str.strip()
    df["공정명"] = df["공정코드"].map(process_mapping).fillna(df["공정코드"])
    # 불량율 계산
    df["불량율"] = df.apply(lambda row: (row["불량수량"] / row["생산수량"]) * 100 if row["생산수량"] else 0, axis=1)
    return df


def summarise_by_time(
    df: pd.DataFrame,
    time_unit: str,
    categories: List[str],
    group_filter: Optional[List[str]] = None,
    defect_filter: Optional[str] = None,
    process_filter: Optional[str] = None,
) -> pd.DataFrame:
    """일/주/월 단위로 불량율 피벗 테이블을 생성합니다.

    Args:
        df: 데이터프레임
        time_unit: 'D', 'W', 'M'
        categories: 피벗 열 순서 (공정명 또는 불량 유형)
        group_filter: 공정명 목록 (유형별 불량율에서 사용)
        defect_filter: 특정 불량 유형 필터
        process_filter: 특정 공정 필터
    Returns:
        피벗 테이블 (index: 기간, columns: categories)
    """
    data = df.copy()
    if defect_filter is not None:
        data = data[data["신규분류요약"] == defect_filter]
    if process_filter is not None:
        data = data[data["공정코드"] == process_filter]
    if group_filter is not None:
        data = data[data["공정명"].isin(group_filter)]
    # 기간 생성
    if time_unit == 'D':
        data['기간'] = data['생산일자'].dt.date
    elif time_unit == 'W':
        data['기간'] = data['생산일자'].dt.to_period('W').apply(lambda r: f"W{int(r.week)}")
    elif time_unit == 'M':
        data['기간'] = data['생산일자'].dt.to_period('M').astype(str)
    else:
        raise ValueError('지원하지 않는 time_unit')
    # 피벗
    pivot = pd.pivot_table(
        data,
        index='기간',
        columns='공정명' if defect_filter else '신규분류요약',
        values='불량율',
        aggfunc='mean',
        fill_value=0
    )
    # 누락된 컬럼 채움
    for cat in categories:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[categories]
    # 인덱스 정렬: 주간의 경우 숫자 기준으로
    if time_unit == 'W':
        pivot['week_num'] = pivot.index.map(lambda x: int(str(x).lstrip('W')) if str(x).lstrip('W').isdigit() else 0)
        pivot.sort_values('week_num', inplace=True)
        pivot.drop(columns='week_num', inplace=True)
    else:
        pivot.sort_index(inplace=True)
    return pivot


def plot_time_series(pivot: pd.DataFrame, title: str, series_names: List[str], y_range: Optional[List[float]] = None) -> go.Figure:
    """Plotly 꺾은선 그래프를 생성합니다.

    Args:
        pivot: 피벗 테이블 (index: 기간)
        title: 그래프 제목
        series_names: 그래프에 표시할 컬럼 순서
        y_range: y축 범위
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    x_vals = pivot.index.tolist()
    # 색상 팔레트 지정
    colors = px.colors.qualitative.Set2 if px else None
    for i, series in enumerate(series_names):
        y_vals = pivot[series].values
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers+text',
            name=series,
            text=[f"{v:.1f}" for v in y_vals],
            textposition='top center',
            marker=dict(color=colors[i % len(colors)]) if colors else None,
            line=dict(color=colors[i % len(colors)]) if colors else None
        ))
    fig.update_layout(
        title=title,
        xaxis_title='기간',
        yaxis_title='불량율 (%)',
        yaxis=dict(range=y_range)
    )
    return fig


def run_defect_dashboard():
    """Streamlit 불량 대시보드 실행 함수."""
    st.title("공정별 불량율 대시보드")
    # 데이터 경로 탐색
    candidates = [
        "공정기술팀 대시보드(불량).xlsx",
        "공정기술팀 대시보드 (불량).xlsx",
        "공정기술팀 대시보드_불량.xlsx"
    ]
    file_path = None
    for c in candidates:
        if os.path.exists(c):
            file_path = c
            break
        if os.path.exists(os.path.join('data', c)):
            file_path = os.path.join('data', c)
            break
    if file_path is None:
        st.error("불량 데이터 파일을 찾을 수 없습니다. '공정기술팀 대시보드(불량).xlsx' 파일을 프로젝트에 업로드해 주세요.")
        return
    df = load_data(file_path)
    if df.empty:
        st.warning("데이터를 불러올 수 없거나 필수 컬럼이 없습니다.")
        return
    # 공정 및 불량 유형 정의
    process_mapping = {
        '사출조립': '사출조립공정',
        '분리': '분리공정',
        '접착/멸균': '접착/멸균',
        '누수/규격검사': '누수/규격검사'
    }
    defect_types_general = ['파손', '엣지기포', '엣지', '미분리', '뜯김']
    defect_types_leak = ['리드지 불량', '블리스터 불량']
    # ────────────────────────────────────── 공정별 불량율 ─────────────────────────────────────
    st.markdown("## 공정별 불량율")
    st.info("각 공정에 대해 월별, 주간(최근 10주), 일별(최근 30일) 불량율을 확인합니다. 불량율 = 불량수량 / 생산수량 × 100")
    for proc_code, proc_label in process_mapping.items():
        st.markdown(f"### {proc_label}")
        if proc_code == '누수/규격검사':
            categories = defect_types_leak
        else:
            categories = defect_types_general
        col1, col2, col3 = st.columns(3)
        # 월별 그래프
        with col1:
            pivot = summarise_by_time(df, 'M', categories, process_filter=proc_code)
            fig = plot_time_series(pivot, f"{proc_label} 월별 불량율", categories, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        # 주간 그래프 (최근 10주)
        with col2:
            pivot = summarise_by_time(df, 'W', categories, process_filter=proc_code)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{proc_label} 주간 불량율", categories, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        # 일별 그래프 (최근 30일)
        with col3:
            pivot = summarise_by_time(df, 'D', categories, process_filter=proc_code)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{proc_label} 일별 불량율", categories, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    # ────────────────────────────────────── 유형별 불량율 ──────────────────────────────────────
    st.markdown("## 유형별 불량율")
    st.info("각 불량 유형에 대해 월별, 주간(최근 10주), 일별(최근 30일) 불량율을 확인합니다. 공정별 비교가 가능합니다.")
    # 일반 유형
    for defect in defect_types_general:
        st.markdown(f"### {defect}")
        # 비교할 공정들 (누수/규격검사 제외)
        compare_processes = [process_mapping[p] for p in process_mapping if p != '누수/규격검사']
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot = summarise_by_time(df, 'M', compare_processes, defect_filter=defect)
            fig = plot_time_series(pivot, f"{defect} 월별 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pivot = summarise_by_time(df, 'W', compare_processes, defect_filter=defect)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{defect} 주간 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            pivot = summarise_by_time(df, 'D', compare_processes, defect_filter=defect)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{defect} 일별 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    # 누수/규격검사 유형
    for defect in defect_types_leak:
        st.markdown(f"### {defect}")
        compare_processes = [process_mapping['누수/규격검사']]
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot = summarise_by_time(df, 'M', compare_processes, defect_filter=defect, process_filter='누수/규격검사')
            fig = plot_time_series(pivot, f"{defect} 월별 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pivot = summarise_by_time(df, 'W', compare_processes, defect_filter=defect, process_filter='누수/규격검사')
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{defect} 주간 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            pivot = summarise_by_time(df, 'D', compare_processes, defect_filter=defect, process_filter='누수/규격검사')
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{defect} 일별 불량율", compare_processes, y_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    run_defect_dashboard()