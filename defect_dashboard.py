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
    # 불량수량 컬럼 보존:
    # - 불량수량_유형: 불량유형별 수량(원본 `불량수량`)
    # - 불량수량_실적: 공정/집계용 실제 불량수량(`불량수량.1`이 있으면 우선 사용)
    df["불량수량_유형"] = pd.to_numeric(df.get("불량수량", 0), errors='coerce').fillna(0)
    if "불량수량.1" in df.columns:
        df["불량수량_실적"] = pd.to_numeric(df["불량수량.1"], errors='coerce').fillna(0)
    else:
        df["불량수량_실적"] = df["불량수량_유형"]
    df["양품수량"] = pd.to_numeric(df["양품수량"], errors='coerce').fillna(0)
    df["불량수량"] = df["불량수량_실적"]
    # 생산수량 계산: 양품 + 실제 불량
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



def resolve_machine_col(df: pd.DataFrame) -> Optional[str]:
    """호기/설비/기계 코드 컬럼명을 자동 탐색합니다."""
    candidates = [
        "공정기계코드",
        "기계코드",
        "설비코드",
        "호기",
        "호기번호",
        "설비번호",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def calc_defect_rate(good_qty: float, defect_qty: float) -> float:
    """불량율(%) 계산."""
    total = good_qty + defect_qty
    if total == 0:
        return 0.0
    return (defect_qty / total) * 100


def filter_root_cause_base(
    df: pd.DataFrame,
    plant: str,
    process: str,
    date_mode: str,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """이상 원인분석용 기본 필터 데이터셋을 반환합니다."""
    data = df.copy()

    if plant != "전체":
        data = data[data["공장"] == plant]

    if process != "전체":
        data = data[data["공정명"] == process]

    max_date = data["생산일자"].max()
    if pd.isna(max_date):
        return data.iloc[0:0]

    if date_mode == "이번주":
        start = max_date - pd.Timedelta(days=6)
        end = max_date
    elif date_mode == "최근 2주":
        start = max_date - pd.Timedelta(days=13)
        end = max_date
    elif date_mode == "최근 4주":
        start = max_date - pd.Timedelta(days=27)
        end = max_date
    else:
        start = pd.to_datetime(start_date) if start_date is not None else data["생산일자"].min()
        end = pd.to_datetime(end_date) if end_date is not None else data["생산일자"].max()

    data = data[
        (data["생산일자"] >= pd.to_datetime(start)) &
        (data["생산일자"] <= pd.to_datetime(end))
    ].copy()

    return data


def summarise_daily_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """일별 이상일 후보 요약."""
    if df.empty:
        return pd.DataFrame(columns=["날짜", "양품수량", "불량수량", "생산수량", "불량율", "평균대비"])

    grouped = (
        df.groupby(df["생산일자"].dt.date)
        .agg(
            양품수량=("양품수량", "sum"),
            불량수량=("불량수량", "sum"),
        )
        .reset_index()
        .rename(columns={"생산일자": "날짜"})
    )

    grouped["생산수량"] = grouped["양품수량"] + grouped["불량수량"]
    grouped["불량율"] = grouped.apply(
        lambda r: calc_defect_rate(r["양품수량"], r["불량수량"]),
        axis=1,
    )

    avg_rate = grouped["불량율"].mean() if not grouped.empty else 0.0
    grouped["평균대비"] = grouped["불량율"] - avg_rate
    grouped = grouped.sort_values("날짜")
    return grouped


def summarise_machine_day(df: pd.DataFrame, machine_col: str, selected_day) -> pd.DataFrame:
    """선택일 기준 호기별 불량율 요약. 높은 순으로 정렬합니다."""
    day_df = df[df["생산일자"].dt.date == pd.to_datetime(selected_day).date()].copy()

    if day_df.empty:
        return pd.DataFrame(columns=[machine_col, "양품수량", "불량수량", "생산수량", "불량율", "불량비중"])

    grouped = (
        day_df.groupby(machine_col)
        .agg(
            양품수량=("양품수량", "sum"),
            불량수량=("불량수량", "sum"),
        )
        .reset_index()
    )

    grouped["생산수량"] = grouped["양품수량"] + grouped["불량수량"]
    grouped["불량율"] = grouped.apply(
        lambda r: calc_defect_rate(r["양품수량"], r["불량수량"]),
        axis=1,
    )

    total_defect = grouped["불량수량"].sum()
    grouped["불량비중"] = np.where(
        total_defect > 0,
        grouped["불량수량"] / total_defect * 100,
        0,
    )

    grouped = grouped.sort_values(["불량율", "불량수량"], ascending=[False, False]).reset_index(drop=True)
    return grouped


def summarise_machine_defect_mix(
    df: pd.DataFrame,
    machine_col: str,
    selected_day,
    selected_machine,
) -> pd.DataFrame:
    """선택일 + 선택호기 기준 불량유형 분해.

    불량유형 구성에서는 유형별 불량수량만 분해해서 보여줍니다.
    일부 원본 파일은 `불량수량.1`에 공정/호기 총 불량이 반복 저장되고,
    원래 `불량수량` 컬럼에 불량유형별 수량이 들어있을 수 있으므로
    여기서는 `불량수량_유형`을 우선 사용합니다.
    """
    data = df[
        (df["생산일자"].dt.date == pd.to_datetime(selected_day).date()) &
        (df[machine_col].astype(str) == str(selected_machine))
    ].copy()

    if data.empty:
        return pd.DataFrame(columns=["불량유형", "불량수량", "불량비중"])

    defect_qty_col = "불량수량_유형" if "불량수량_유형" in data.columns else "불량수량"

    grouped = (
        data.groupby("불량유형")
        .agg(
            불량수량=(defect_qty_col, "sum"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["불량수량"] > 0].copy()
    total_defect = grouped["불량수량"].sum()
    grouped["불량비중"] = np.where(
        total_defect > 0,
        grouped["불량수량"] / total_defect * 100,
        0,
    )

    grouped = grouped.sort_values(["불량수량", "불량비중"], ascending=[False, False]).reset_index(drop=True)

    # 총합 행 추가
    if not grouped.empty:
        total_row = pd.DataFrame([{
            "불량유형": "총합",
            "불량수량": grouped["불량수량"].sum(),
            "불량비중": 100.0,
        }])
        grouped = pd.concat([grouped, total_row], ignore_index=True)

    return grouped


def plot_daily_anomaly_chart(daily_df: pd.DataFrame, selected_day=None) -> go.Figure:
    """선택 기간별 불량율 그래프."""
    fig = go.Figure()
    if daily_df.empty:
        fig.update_layout(title="선택 기간별 불량율")
        return fig

    avg_rate = daily_df["불량율"].mean()

    colors = []
    for _, row in daily_df.iterrows():
        if selected_day is not None and str(row["날짜"]) == str(selected_day):
            colors.append("#1f77b4")
        elif row["불량율"] > avg_rate:
            colors.append("#e15759")
        else:
            colors.append("#9ecae1")

    fig.add_trace(go.Bar(
        x=daily_df["날짜"].astype(str),
        y=daily_df["불량율"],
        marker_color=colors,
        text=[f"{v:.1f}" for v in daily_df["불량율"]],
        textposition="outside",
        name="불량율",
    ))

    fig.add_hline(
        y=avg_rate,
        line_dash="dash",
        line_color="black",
        annotation_text=f"평균 {avg_rate:.1f}%",
    )

    fig.update_layout(
        title="선택 기간별 불량율",
        xaxis_title="날짜",
        yaxis_title="불량율 (%)",
        xaxis=dict(type="category"),
        height=420,
    )
    return fig


def plot_machine_bar_chart(machine_df: pd.DataFrame, machine_col: str, selected_machine: str | None = None) -> go.Figure:
    """호기별 불량율 분석 그래프."""
    fig = go.Figure()
    if machine_df.empty:
        fig.update_layout(title="호기별 불량율 분석")
        return fig

    plot_df = machine_df.head(10).copy()
    avg_rate = plot_df["불량율"].mean()

    colors = [
        "#d62728" if selected_machine is not None and str(m) == str(selected_machine) else "#6baed6"
        for m in plot_df[machine_col]
    ]

    fig.add_trace(go.Bar(
        x=plot_df["불량율"],
        y=plot_df[machine_col].astype(str),
        orientation="h",
        marker_color=colors,
        text=[
            f"{r['불량율']:.1f}% / {int(r['불량수량'])}ea"
            for _, r in plot_df.iterrows()
        ],
        textposition="outside",
        name="호기별 불량율",
    ))

    fig.add_vline(
        x=avg_rate,
        line_dash="dash",
        line_color="black",
        annotation_text=f"평균 {avg_rate:.1f}%",
    )

    fig.update_layout(
        title="호기별 불량율 분석",
        xaxis_title="불량율 (%)",
        yaxis_title="공정기계코드",
        height=max(360, len(plot_df) * 45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_machine_defect_chart(defect_df: pd.DataFrame) -> go.Figure:
    """선택 호기 불량유형 구성 그래프."""
    if defect_df.empty:
        fig = go.Figure()
        fig.update_layout(title="선택 호기 불량유형 구성")
        return fig

    fig = px.bar(
        defect_df,
        x="불량유형",
        y="불량수량",
        text="불량수량",
        color="불량유형",
        title="선택 호기 불량유형 구성",
    )
    fig.update_layout(
        xaxis_title="불량유형",
        yaxis_title="불량수량",
        showlegend=False,
        height=380,
    )
    return fig


def render_root_cause_analysis(df: pd.DataFrame) -> None:
    """이상 원인분석 섹션 렌더링."""
    st.markdown("---")
    st.markdown("## 이상 원인분석")
    st.caption("선택 기간의 날짜를 고른 뒤 호기별 원인을 확인합니다.")

    machine_col = resolve_machine_col(df)
    if machine_col is None:
        st.warning("공정기계코드(호기) 컬럼이 없어 이상 원인분석은 표시하지 않습니다.")
        return

    with st.expander("분석 조건", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        plant_options = ["전체"] + sorted(df["공장"].dropna().astype(str).unique().tolist())
        process_options = ["전체"] + sorted(df["공정명"].dropna().astype(str).unique().tolist())

        with c1:
            selected_plant = st.selectbox("공장", plant_options, key="rc_plant")
        with c2:
            selected_process = st.selectbox("공정", process_options, key="rc_process")
        with c3:
            date_mode = st.selectbox("기간", ["이번주", "최근 2주", "최근 4주", "직접 선택"], key="rc_date_mode")
        with c4:
            ranking_mode = st.selectbox("정렬 기준", ["불량율 높은 순", "불량수량 높은 순", "평균 대비 큰 순"], key="rc_rank")

        start_date, end_date = None, None
        if date_mode == "직접 선택":
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("시작일", value=None, key="rc_start")
            with d2:
                end_date = st.date_input("종료일", value=None, key="rc_end")

    base_df = filter_root_cause_base(
        df=df,
        plant=selected_plant,
        process=selected_process,
        date_mode=date_mode,
        start_date=start_date,
        end_date=end_date,
    )

    if base_df.empty:
        st.info("선택 조건에 해당하는 데이터가 없습니다.")
        return

    daily_df = summarise_daily_anomaly(base_df)

    # 선택일 목록은 빠른 날짜부터 늦은 날짜 순으로 고정합니다.
    day_options_df = daily_df.copy()
    try:
        day_options_df["날짜"] = pd.to_datetime(day_options_df["날짜"])
        day_options_df = day_options_df.sort_values("날짜", ascending=True)
        day_options = day_options_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
    except Exception:
        day_options = sorted(daily_df["날짜"].astype(str).tolist())

    if not day_options:
        st.info("분석 가능한 날짜가 없습니다.")
        return

    # 그래프를 먼저 보여주고, 그 아래에서 선택일을 고르는 구조로 배치합니다.
    default_selected_day = day_options[0]
    selected_day = st.session_state.get("rc_selected_day", default_selected_day)
    if selected_day not in day_options:
        selected_day = default_selected_day

    st.plotly_chart(
        plot_daily_anomaly_chart(daily_df, selected_day=selected_day),
        use_container_width=True,
    )

    selected_day = st.selectbox(
        "선택일",
        options=day_options,
        index=day_options.index(selected_day),
        key="rc_selected_day",
    )

    selected_day_row = daily_df[daily_df["날짜"].astype(str) == str(selected_day)]
    if not selected_day_row.empty:
        row = selected_day_row.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("선택일", str(row["날짜"]))
        k2.metric("당일 불량율", f"{row['불량율']:.2f}%")
        k3.metric("불량수량", f"{int(row['불량수량']):,}")
        k4.metric("평균 대비", f"{row['평균대비']:+.2f}%p")

    machine_df = summarise_machine_day(base_df, machine_col, selected_day)

    if machine_df.empty:
        st.info("선택일 호기 데이터가 없습니다.")
        return

    machine_options = machine_df[machine_col].astype(str).tolist()
    selected_machine = st.selectbox(
        "상세 분석할 호기 선택 (기본값: 불량율 최고 호기)",
        machine_options,
        index=0,
        key="rc_machine",
    )

    left, right = st.columns([1.3, 1.0])

    with left:
        st.plotly_chart(
            plot_machine_bar_chart(machine_df, machine_col, selected_machine=selected_machine),
            use_container_width=True,
        )

    with right:
        machine_table = machine_df.copy()
        machine_table["불량율"] = machine_table["불량율"].map(lambda x: f"{x:.2f}")
        machine_table["불량비중"] = machine_table["불량비중"].map(lambda x: f"{x:.2f}")
        st.dataframe(machine_table, use_container_width=True, hide_index=True)

    defect_mix_df = summarise_machine_defect_mix(
        base_df, machine_col, selected_day, selected_machine
    )

    dcol1, dcol2 = st.columns([1.0, 1.2])

    with dcol1:
        st.plotly_chart(
            plot_machine_defect_chart(defect_mix_df),
            use_container_width=True,
        )

    with dcol2:
        if defect_mix_df.empty:
            st.info("선택 호기의 상세 불량유형 데이터가 없습니다.")
        else:
            machine_raw_df = base_df[
                (base_df["생산일자"].dt.date == pd.to_datetime(selected_day).date()) &
                (base_df[machine_col].astype(str) == str(selected_machine))
            ].copy()
            machine_good = pd.to_numeric(machine_raw_df["양품수량"], errors="coerce").fillna(0).sum()
            machine_defect_actual = pd.to_numeric(machine_raw_df["불량수량"], errors="coerce").fillna(0).sum()
            machine_rate = calc_defect_rate(machine_good, machine_defect_actual)

            st.markdown(
                f"**호기전체불량율:** {machine_rate:.2f}%"
            )

            view_df = defect_mix_df.copy()
            view_df["불량수량"] = view_df["불량수량"].map(lambda x: f"{int(x):,}")
            view_df["불량비중"] = view_df["불량비중"].map(lambda x: f"{x:.2f}%")
            st.dataframe(view_df, use_container_width=True, hide_index=True)

    with st.expander("원본 상세 데이터"):
        raw_df = base_df[
            (base_df["생산일자"].dt.date == pd.to_datetime(selected_day).date()) &
            (base_df[machine_col].astype(str) == str(selected_machine))
        ].copy()
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

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


    render_root_cause_analysis(df)


if __name__ == '__main__':
    run_defect_dashboard()