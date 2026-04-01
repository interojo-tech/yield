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
    """
    엑셀 파일을 로드하고 분석에 필요한 전처리를 수행합니다.

    현재 불량 대시보드의 데이터는 다음과 같은 구조를 가정합니다:
    - `생산일자`: 날짜 문자열 (예: '2026-01-05').
    - `공정`: '[10] 사출조립'과 같이 코드와 공정명이 함께 포함된 문자열.
    - `불량명`: 'H : 파손'과 같이 코드와 불량 유형이 ':'로 구분된 문자열.
    - `양품수량`, `불량수량`: 정수형 수량.

    이 함수에서는 위 컬럼을 기준으로:
    1. `생산일자`를 datetime 형식으로 변환합니다.
    2. `공정`에서 코드 부분을 제거하고 공정명만 추출해 `공정명` 컬럼을 생성합니다.
    3. `불량명`에서 불량 유형만 추출해 `불량유형` 컬럼을 생성합니다.
    4. `불량율`을 `불량수량 / (양품수량 + 불량수량) * 100`으로 계산합니다.
    5. 연도, 월, ISO 주차를 파생 컬럼으로 추가합니다.

    Args:
        file_path: Excel 파일 경로

    Returns:
        전처리된 pandas DataFrame
    """
    if not os.path.exists(file_path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=0)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()
    # 불량 데이터에서 필요한 컬럼이 존재하는지 확인합니다. 일부 엑셀 파일에는 `불량수량.1` 컬럼에
    # 실제 불량 수량이 저장되어 있으므로, 해당 컬럼을 우선적으로 사용합니다. 없으면 기존
    # `불량수량`을 사용합니다.
    required_cols = {"생산일자", "공정", "불량명", "양품수량", "공장"}
    missing = required_cols - set(df.columns)
    if missing:
        st.warning(f"데이터에 필요한 컬럼이 없습니다: {missing}")
        return pd.DataFrame()
    # 날짜 변환
    df["생산일자"] = pd.to_datetime(df["생산일자"], errors='coerce')
    # 공정명 추출: '[10] 사출조립' → '사출조립'
    # 일부 공정 이름에는 스페이스가 섞여 있으므로 모든 공백을 제거합니다.
    df["공정명"] = (
        df["공정"]
        .astype(str)
        .apply(lambda x: x.split("]")[-1].strip() if "]" in x else x.strip())
        .str.replace(" ", "")
    )
    # 불량 유형 추출: 'H : 파손' → '파손'
    df["불량유형"] = (
        df["불량명"]
        .astype(str)
        .apply(lambda x: x.split(":")[-1].strip() if ":" in x else x.strip())
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip()
    )
    # 사용할 불량수량 컬럼 선택: `불량수량.1`이 있으면 그것을 사용하고, 없으면 `불량수량`을 사용합니다.
    if "불량수량.1" in df.columns:
        df["불량수량"] = df["불량수량.1"]
    # 생산수량 계산: 양품 + 불량
    df["생산수량"] = df["양품수량"] + df["불량수량"]
    # 불량율 계산: 생산수량이 0이면 0. 행 단위로 불량율을 계산합니다.
    df["불량율"] = df.apply(
        lambda row: (row["불량수량"] / row["생산수량"]) * 100 if row["생산수량"] else 0, axis=1
    )
    # 연, 월, ISO 주차
    df["년도"] = df["생산일자"].dt.year
    df["월"] = df["생산일자"].dt.to_period('M').astype(str)
    df["주차"] = df["생산일자"].dt.isocalendar().week
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
    # defect_filter: 특정 불량 유형으로 필터링
    if defect_filter is not None:
        data = data[data["불량유형"] == defect_filter]
    # process_filter: 특정 공정으로 필터링
    if process_filter is not None:
        data = data[data["공정명"] == process_filter]
    # group_filter: 비교할 공정명 목록
    if group_filter is not None:
        data = data[data["공정명"].isin(group_filter)]
    # 기간 생성
    if time_unit == 'D':
        # 일별: 날짜를 그대로 사용
        data['기간'] = data['생산일자'].dt.date
    elif time_unit == 'W':
        # 주간: ISO 주차를 'W숫자' 형식으로 표시
        data['기간'] = data['생산일자'].dt.to_period('W').apply(lambda r: f"W{int(r.week)}")
    elif time_unit == 'M':
        # 월별: 월을 정수로 저장하여 나중에 월 이름으로 매핑
        data['기간'] = data['생산일자'].dt.month
    else:
        raise ValueError('지원하지 않는 time_unit')
    # 피벗을 생성하기 위해 공정/유형 별로 생산수량과 불량수량을 합산합니다.
    pivot_col = '공정명' if defect_filter else '불량유형'
    # 그룹화: 기간과 피벗컬럼으로 양품, 불량 수량 합계 구하기
    agg = (
        data
        .groupby(['기간', pivot_col])
        .agg({'양품수량': 'sum', '불량수량': 'sum'})
        .reset_index()
    )
    # 불량율 계산 (불량수량 합 / (양품수량 합 + 불량수량 합) * 100)
    agg['불량율'] = agg.apply(
        lambda row: (row['불량수량'] / (row['양품수량'] + row['불량수량'])) * 100 if (row['양품수량'] + row['불량수량']) else 0,
        axis=1
    )
    pivot = agg.pivot(index='기간', columns=pivot_col, values='불량율').fillna(0)
    # 누락된 컬럼 채움
    for cat in categories:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[categories]
    # 인덱스 정렬
    if time_unit == 'W':
        # 주간: W 접두사를 제거하여 숫자로 정렬
        pivot['week_num'] = pivot.index.map(lambda x: int(str(x).lstrip('W')) if str(x).lstrip('W').isdigit() else 0)
        pivot.sort_values('week_num', inplace=True)
        pivot.drop(columns='week_num', inplace=True)
    elif time_unit == 'M':
        # 월별: 정수 월(1~12) 기준으로 정렬하고 월 이름으로 인덱스 변경
        pivot.sort_index(inplace=True)
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        new_index = [month_map.get(int(idx), str(idx)) for idx in pivot.index]
        pivot.index = new_index
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
    # Configure axes: set x-axis to category mode to ensure all points are shown in order.
    fig.update_layout(
        title=title,
        xaxis_title='기간',
        yaxis_title='불량율 (%)',
        yaxis=dict(range=y_range),
        xaxis=dict(type='category', categoryorder='array', categoryarray=x_vals)
    )
    return fig




def get_process_y_range(proc_code: str) -> list[float]:
    """Return y-axis range for process defect charts."""
    if proc_code in ["분리", "접착/멸균"]:
        return [0, 25]
    if proc_code == "누수/규격검사":
        return [0, 5]
    return [0, 50]


def get_defect_y_range(defect: str) -> list[float]:
    """Return y-axis range for defect-type charts."""
    if defect in ["파손", "엣지기포", "미분리", "뜯김"]:
        return [0, 20]
    if defect in ["리드지 불량", "블리스터 불량"]:
        return [0, 5]
    return [0, 50]


def run_defect_dashboard():
    """Streamlit 불량 대시보드 실행 함수."""
    st.title("공정별 불량율 대시보드")
    # 데이터 경로 탐색
    #
    # 불량 데이터는 월별 파일로 제공되며 이름 형식은 '(26.xx)불량실적현황.xlsx'
    # 또는 '26.xx불량실적현황.xlsx'입니다. 여기서 xx는 01부터 12까지의 월을 의미합니다.
    # 프로젝트 루트와 'data' 폴더를 모두 검색하여 존재하는 모든 월 파일을 로드하고
    # 하나의 데이터프레임으로 결합합니다. 최소 하나의 파일이 있어야 대시보드를
    # 실행할 수 있습니다.
    monthly_paths: List[str] = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        # Two naming conventions: with parentheses and without
        names = [f"(26.{month_str})불량실적현황.xlsx", f"26.{month_str}불량실적현황.xlsx"]
        for name in names:
            # Check in current directory
            if os.path.exists(name):
                monthly_paths.append(name)
            # Check in 'data' subdirectory
            elif os.path.exists(os.path.join('data', name)):
                monthly_paths.append(os.path.join('data', name))
    if not monthly_paths:
        st.error("불량 데이터 파일을 찾을 수 없습니다. '(26.01)불량실적현황.xlsx' 형식의 파일을 프로젝트에 업로드해 주세요.")
        return
    # Load and concatenate all monthly data
    df_list: List[pd.DataFrame] = []
    for p in monthly_paths:
        df_month = load_data(p)
        if not df_month.empty:
            df_list.append(df_month)
    if not df_list:
        st.warning("데이터를 불러올 수 없거나 필수 컬럼이 없습니다.")
        return
    df = pd.concat(df_list, ignore_index=True)
    # 공정 및 불량 유형 정의
    # 데이터의 공정명은 '사출조립', '분리', '접착/멸균', '누수/규격검사' 등을 포함합니다.
    # 대시보드 표시용 라벨을 별도로 정의합니다.
    process_mapping = {
        '사출조립': '사출조립공정',
        '분리': '분리공정',
        '접착/멸균': '접착/멸균',
        '누수/규격검사': '누수/규격검사'
    }
    # 일반 공정에서 나타낼 불량 유형 목록
    defect_types_general = ['파손', '엣지기포', '엣지', '미분리', '뜯김']
    # 누수/규격검사 공정에서 나타낼 불량 유형 목록
    defect_types_leak = ['리드지 불량', '블리스터 불량']
    # ────────────────────────────────────── 공정별 불량율 ─────────────────────────────────────
    st.markdown("## 공정별 불량율")
    st.info("각 공정에 대해 월별, 주간(최근 10주), 일별(최근 30일) 불량율을 확인합니다. 불량율 = 불량수량 / (양품수량 + 불량수량) × 100")
    # 각 공정에 대해 그래프를 출력
    for proc_code, proc_label in process_mapping.items():
        st.markdown(f"### {proc_label}")
        # 공정별 사용할 불량 유형 목록 선정
        categories = defect_types_leak if proc_code == '누수/규격검사' else defect_types_general
        col1, col2, col3 = st.columns(3)
        # 월별 그래프
        with col1:
            pivot = summarise_by_time(df, 'M', categories, process_filter=proc_code)
            fig = plot_time_series(pivot, f"{proc_label} 월별 불량율", categories, y_range=get_process_y_range(proc_code))
            st.plotly_chart(fig, use_container_width=True)
        # 주간 그래프 (최근 10주)
        with col2:
            pivot = summarise_by_time(df, 'W', categories, process_filter=proc_code)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot, f"{proc_label} 주간 불량율", categories, y_range=get_process_y_range(proc_code))
            st.plotly_chart(fig, use_container_width=True)
        # 일별 그래프 (최근 30일)
        with col3:
            pivot = summarise_by_time(df, 'D', categories, process_filter=proc_code)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot, f"{proc_label} 일별 불량율", categories, y_range=get_process_y_range(proc_code))
            st.plotly_chart(fig, use_container_width=True)
    # ────────────────────────────────────── 유형별 불량율 ──────────────────────────────────────
    st.markdown("## 유형별 불량율")
    st.info("각 불량 유형에 대해 월별, 주간(최근 10주), 일별(최근 30일) 불량율을 확인합니다. 공정별 비교가 가능합니다.")
    # 공정별 비교에 사용할 공정명 리스트 (누수/규격검사 제외)
    compare_process_names = [p for p in process_mapping.keys() if p != '누수/규격검사']
    compare_process_labels = [process_mapping[p] for p in compare_process_names]
    # 일반 불량 유형에 대한 그래프
    for defect in defect_types_general:
        st.markdown(f"### {defect}")
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot = summarise_by_time(df, 'M', compare_process_names, defect_filter=defect)
            # pivot의 컬럼은 공정명 리스트(compare_process_names) 기준으로 생성되므로, 라벨과 매칭하여 그래프 생성
            fig = plot_time_series(pivot.rename(columns=process_mapping), f"{defect} 월별 불량율", compare_process_labels, y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pivot = summarise_by_time(df, 'W', compare_process_names, defect_filter=defect)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot.rename(columns=process_mapping), f"{defect} 주간 불량율", compare_process_labels, y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            pivot = summarise_by_time(df, 'D', compare_process_names, defect_filter=defect)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot.rename(columns=process_mapping), f"{defect} 일별 불량율", compare_process_labels, y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)
    # 누수/규격검사 불량 유형에 대한 그래프 (공정 하나)
    for defect in defect_types_leak:
        st.markdown(f"### {defect}")
        proc_name = '누수/규격검사'
        proc_label = process_mapping[proc_name]
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot = summarise_by_time(df, 'M', [proc_name], defect_filter=defect, process_filter=proc_name)
            fig = plot_time_series(pivot.rename(columns={proc_name: proc_label}), f"{defect} 월별 불량율", [proc_label], y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pivot = summarise_by_time(df, 'W', [proc_name], defect_filter=defect, process_filter=proc_name)
            if len(pivot) > 10:
                pivot = pivot.tail(10)
            fig = plot_time_series(pivot.rename(columns={proc_name: proc_label}), f"{defect} 주간 불량율", [proc_label], y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            pivot = summarise_by_time(df, 'D', [proc_name], defect_filter=defect, process_filter=proc_name)
            if len(pivot) > 30:
                pivot = pivot.tail(30)
            fig = plot_time_series(pivot.rename(columns={proc_name: proc_label}), f"{defect} 일별 불량율", [proc_label], y_range=get_defect_y_range(defect))
            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    run_defect_dashboard()