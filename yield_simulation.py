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
import glob
import re
from typing import Dict, List, Tuple

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

PROCESS_COLOR_MAP = {
    "사출조립": "#4E79A7",
    "분리": "#F28E2B",
    "하이드레이션/전면검사": "#E15759",
    "접착/멸균": "#59A14F",
    "누수/규격검사": "#76B7B2",
    "종합수율": "#111111",
    "overall_yield": "#111111",
}

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
    """Load the Excel file and return the production DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the sheet ``생산실적현황``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all columns from the Excel sheet.
    """
    df = pd.read_excel(file_path, sheet_name="생산실적현황")
    # Ensure production date is a datetime object for grouping
    if not np.issubdtype(df["생산일자"].dtype, np.datetime64):
        df["생산일자"] = pd.to_datetime(df["생산일자"], errors="coerce")
    return df


def map_process_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract process names from the '공정코드' column and normalise names.

    The column includes a numeric code in brackets followed by the
    process name.  This function strips the bracketed code and
    standardises spaces and slashes to match the expected categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a '공정코드' column with values like
        ``'[10] 사출조립'`` or ``'[55] 접착/ 멸균'``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional column ``Process``.
    """
    # Extract the substring after the closing bracket
    # Use expand=True to return a DataFrame of [before, after] when splitting.
    # The signature of str.split changed in recent pandas versions to limit
    # positional arguments; therefore we specify n and expand as keywords.
    split_df = df["공정코드"].astype(str).str.split("]", n=1, expand=True)
    # Take the second column (text after the bracket).  If no bracket is
    # present the second column will be NaN; replace NaN with the original
    # value to avoid missing process names.
    processes = split_df[1].fillna(split_df[0])
    processes = processes.str.strip()  # remove leading/trailing whitespace
    # Normalise multiple spaces and remove stray slashes/spaces
    # Normalize whitespace around slashes: remove any spaces before or after a slash
    # so that "접착/ 멸균" and "접착 /멸균" are unified as "접착/멸균".
    processes = processes.str.replace(r"\s*/\s*", "/", regex=True)
    # Collapse multiple spaces into a single space
    processes = processes.str.replace(r"\s{2,}", " ", regex=True)
    df = df.copy()
    df["Process"] = processes
    return df


def compute_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield for each record.

    Yield is defined as good quantity divided by production quantity.
    Rows with zero or missing production quantity will result in NaN
    yields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing '양품수량' and '생산수량' columns.

    Returns
    -------
    pd.DataFrame
        A copy of df with an additional column 'yield'.
    """
    df = df.copy()
    df["yield"] = df["양품수량"].astype(float) / df["생산수량"].astype(float)
    return df


def summarise_by_group(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate yields by specified grouping columns.

    For each group, compute the sum of good quantity and production
    quantity for each process, then compute the process yield
    (good / production).  The result is pivoted so that each process
    becomes a separate column.  Missing process categories are filled
    with 1.0 so they do not affect the overall yield when multiplied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns including group_cols, '공장', 'Process',
        '양품수량', and '생산수량'.
    group_cols : Tuple[str, ...]
        Columns to group by (e.g., ('공장', '생산일자')).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by group_cols and with columns for each
        process name plus an 'overall_yield' column.  Each cell
        represents the yield for that process or the overall yield
        (product of yields across processes) for that group.
    """
    # Group by the given columns, plus plant and process
    agg = (
        df.groupby(list(group_cols) + ["공장", "Process"])
        .agg({"양품수량": "sum", "생산수량": "sum"})
    )
    # Compute yield for each plant/process/time combination
    agg["yield"] = agg["양품수량"] / agg["생산수량"]
    # Reset index to pivot on process names
    agg = agg.reset_index()
    # Pivot: each process becomes a column
    pivot_cols = list(group_cols) + ["공장"]
    pivot = agg.pivot_table(
        index=pivot_cols,
        columns="Process",
        values="yield",
        aggfunc="mean",
    )
    # Fill missing process yields with 1 (neutral) so they do not affect the overall
    # yield calculation.  Missing processes imply a neutral (100%) yield.
    pivot = pivot.fillna(1.0)
    # ------------------------------------------------------------------
    # Compute the overall yield as the aggregated ratio of total good quantity to
    # total production quantity for each group.  This approach aligns the
    # combined summary charts and per‑period summaries by avoiding a
    # multiplicative combination of process yields.  Using a simple ratio
    # eliminates discrepancies that arise when processes have very low yields.
    # We reuse the original aggregated DataFrame ``agg`` to compute totals for
    # each group of ``pivot_cols`` (which correspond to the index of ``pivot``).
    try:
        # Sum good and production quantities across all processes within the same
        # group (group_cols + plant).  Use the same pivot index to align totals.
        totals = agg.groupby(pivot_cols)[["양품수량", "생산수량"]].sum()
        # Compute the aggregated ratio; handle zero production by treating the
        # overall yield as neutral (1.0).  Convert to float explicitly.
        overall_ratio = totals["양품수량"] / totals["생산수량"]
        # Replace NaN and infinite values resulting from division by zero with 1.0
        overall_ratio = overall_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # Assign the aggregated ratio to the pivot DataFrame.  Align on index.
        pivot["overall_yield"] = overall_ratio
    except Exception:
        # Fallback: if any error occurs, default all overall yields to 1.0 to
        # avoid raising unexpected exceptions during dashboard rendering.
        pivot["overall_yield"] = 1.0
    return pivot


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of defect classifications and product names.

    This function groups the data by '신규분류요약' and '품명' and counts
    the number of records and the mean yield for each combination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns '신규분류요약', '품명' and 'yield'.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by '신규분류요약' and '품명' with columns
        for record counts and average yield.
    """
    summary = (
        df.groupby(["신규분류요약", "품명"])
        .agg(
            records=("yield", "size"),
            avg_yield=("yield", "mean"),
        )
        .sort_values("records", ascending=False)
    )
    return summary


def create_example_chart(
    monthly_summary: pd.DataFrame, output_path: str
) -> None:
    """Create a bar chart of monthly yields by plant and process.

    The monthly_summary index should be a MultiIndex of the form
    ('년도', '월', '공장') and the columns should include the five
    processes and 'overall_yield'.  Only the five process columns
    are plotted.

    Parameters
    ----------
    monthly_summary : pd.DataFrame
        DataFrame with yields for each process and plant by month.
    output_path : str
        File path where the PNG image will be saved.
    """
    # Select process columns (exclude 'overall_yield')
    process_cols = [c for c in monthly_summary.columns if c != "overall_yield"]
    # Plot each plant in separate subplots
    plants = monthly_summary.index.get_level_values("공장").unique()
    n_plants = len(plants)
    fig, axes = plt.subplots(n_plants, 1, figsize=(10, 4 * n_plants), sharex=True)
    if n_plants == 1:
        axes = [axes]
    for ax, plant in zip(axes, plants):
        plant_data = monthly_summary.xs(plant, level="공장")
        # Ensure month order (1월, 2월, 3월) by converting to numeric
        month_order = plant_data.index.get_level_values("월")
        month_numbers = month_order.str.extract(r"(\d+)").astype(int).iloc[:, 0]
        plant_data = plant_data.copy()
        plant_data["month_number"] = month_numbers.values
        plant_data = plant_data.sort_values("month_number")
        plant_data[process_cols].plot(kind="bar", ax=ax)
        ax.set_title(f"월별 수율 by 공정 - {plant}")
        ax.set_ylabel("Yield (비율)")
        ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
        ax.set_xticklabels(plant_data.index.get_level_values("월"), rotation=0)
    fig.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _quarter_sort_key(path: str) -> tuple[int, int, str]:
    """Return a stable sort key for quarterly yield workbooks.

    Expected filenames look like:
    - 공정기술팀 대시보드(수율)_25.1q.xlsx
    - 공정기술팀 대시보드(수율)_26.4q.xlsx
    """
    name = os.path.basename(path)
    m = re.search(r"_(\d{2})\.(\d)q(?:\.xlsx)?$", name, flags=re.IGNORECASE)
    if not m:
        return (9999, 9999, name)
    return (int(m.group(1)), int(m.group(2)), name)


def discover_yield_files(base_dir: str) -> List[str]:
    """Discover quarterly yield workbooks first, then fall back to legacy names."""
    quarterly_patterns = [
        os.path.join(base_dir, "공정기술팀 대시보드(수율)_*.xlsx"),
        os.path.join(base_dir, "공정기술팀 대시보드(수율)_*.*q.xlsx"),
    ]
    quarterly_paths: List[str] = []
    for pattern in quarterly_patterns:
        quarterly_paths.extend(glob.glob(pattern))
    quarterly_paths = sorted(set(quarterly_paths), key=_quarter_sort_key)
    if quarterly_paths:
        return quarterly_paths

    # Legacy single-file fallback
    candidates = [
        "공정기술팀 대시보드(수율).xlsx",
        "공정기술팀 대시보드_260312.xlsx",
        "공정기술팀 대시보드.xlsx",
        "공정기술팀 대시보드_260308.xlsx",
    ]
    result: List[str] = []
    for name in candidates:
        candidate_path = os.path.join(base_dir, name)
        if os.path.exists(candidate_path):
            result.append(candidate_path)
    return result




def _safe_year_str(values) -> pd.Series:
    """Convert year-like values to string safely, preserving missing values as empty strings."""
    return (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
    )
def load_and_merge_yield_data(file_paths: List[str]) -> pd.DataFrame:
    """Load one or more yield workbooks and merge them into a single DataFrame."""
    if not file_paths:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for path in file_paths:
        df_part = load_data(path)
        df_part = df_part.copy()
        df_part["source_file"] = os.path.basename(path)
        frames.append(df_part)
    if len(frames) == 1:
        return frames[0]
    merged = pd.concat(frames, ignore_index=True)
    if "생산일자" in merged.columns and not np.issubdtype(merged["생산일자"].dtype, np.datetime64):
        merged["생산일자"] = pd.to_datetime(merged["생산일자"], errors="coerce")
    return merged


def main() -> None:
    """Entry point for both console and Streamlit execution.

    When run via ``streamlit run yield_simulation.py`` this function
    presents an interactive dashboard for exploring yield data.  When
    executed in a normal Python interpreter it prints summary tables
    similar to earlier versions of the script.
    """
    # Determine yield workbook paths.
    # Quarterly workbooks such as '공정기술팀 대시보드(수율)_25.1q.xlsx' are
    # discovered first and automatically merged.  If no quarterly files are
    # present, the dashboard falls back to the older single-file names.
    base_dir = os.path.dirname(__file__)
    file_paths = discover_yield_files(base_dir)
    if not file_paths:
        print(
            "Error: No yield workbook was found. Please place quarterly files such as "
            "'공정기술팀 대시보드(수율)_25.1q.xlsx' in the same directory as this script, "
            "or keep using one of the legacy workbook names."
        )
        return

    # Define a data loading function with optional caching when running in
    # Streamlit.  Caching avoids re-reading the Excel file on every
    # interaction.
    def _load_and_prepare(path: str | tuple[str, ...] | list[str]) -> pd.DataFrame:
        df_local = load_and_merge_yield_data([path] if isinstance(path, str) else list(path))
        df_local = map_process_codes(df_local)
        df_local = compute_yields(df_local)
        # Remove records corresponding to the '몰드인쇄' process (and variants such as '몰드 인쇄').
        # This ensures that the sidebar process selection and all yield/production charts exclude this process.
        if "Process" in df_local.columns:
            # Exclude all rows where the process name corresponds to the
            # '몰드인쇄' process.  Some datasets may use variants such as
            # '몰드 인쇄' (with a space) or other spacing differences.  We
            # normalise by removing spaces and then use a contains check to
            # filter out any process names containing the string '몰드인쇄'.
            # Using contains rather than equality ensures that unexpected
            # prefixes/suffixes (e.g. '[80] 몰드인쇄') are also removed.
            normalized = df_local["Process"].astype(str).str.replace(" ", "")
            df_local = df_local[~normalized.str.contains("몰드인쇄")].copy()
        # Ensure year/month/week columns exist; if not present or partially missing,
        # derive them from the production date column.  This allows the dashboard
        # to filter by year, month and week even if the source workbook lacks
        # these columns.
        # Add integer year from '생산일자'
        if "년도" not in df_local.columns or df_local["년도"].isna().all():
            df_local["년도"] = df_local["생산일자"].dt.year
        if "월" not in df_local.columns or df_local["월"].isna().all():
            # Use zero‑padded month strings to match typical naming (e.g. '1월', '2월')
            df_local["월"] = df_local["생산일자"].dt.month.astype(str) + "월"
        if "주차" not in df_local.columns or df_local["주차"].isna().all():
            # Compute ISO week number; convert to string without leading zeros
            df_local["주차"] = df_local["생산일자"].dt.isocalendar().week.astype(int)
        # Drop rows with NaN yields (due to zero production quantity)
        df_local = df_local.dropna(subset=["yield"])
        return df_local

    if st:
        # Configure the page
        st.set_page_config(page_title="인터로조 공정기술팀 대시보드", layout="wide")

        # Streamlit caching for data loading
        @st.cache_data  # type: ignore[misc]
        def load_cached(paths_key: tuple[str, ...]) -> pd.DataFrame:  # pragma: no cover
            return _load_and_prepare(paths_key)

        df = load_cached(tuple(file_paths))
        # Compute the last production date across all plants.  Convert to datetime
        # and then extract the most recent date to display as the data cutoff.
        try:
            _dt_series = pd.to_datetime(df["생산일자"], errors="coerce")
            last_prod_dt = _dt_series.max()
        except Exception:
            last_prod_dt = None
        if last_prod_dt is not None and pd.notnull(last_prod_dt):
            _yy = int(last_prod_dt.year) % 100
            _mm = int(last_prod_dt.month)
            _dd = int(last_prod_dt.day)
            last_date_label = f"{_yy:02d}년{_mm:02d}월{_dd:02d}일"
        else:
            last_date_label = ""

        # ------------------------------------------------------------------
        # Sidebar selections
        # ------------------------------------------------------------------
        # Include an "전체 공장" option so users can view all plants together.
        # Prepend the option to the sorted list of actual plant names.
        plants = ["전체 공장"] + sorted(df["공장"].dropna().astype(str).unique().tolist())
        selected_plant = st.sidebar.radio(
            "공장 선택",
            options=plants,
            index=0,
        )
        years = sorted(df["년도"].dropna().astype(int).unique().tolist())
        months = sorted(df["월"].dropna().astype(str).unique().tolist())
        # Determine available week options.  Convert to strings for display.
        # Build week labels by extracting numeric week numbers from the '주차' column.
        # The dataset may store weeks as strings like 'W3' or just numbers.  We
        # extract digits, convert to integers, filter out week 1, sort ascending,
        # and then prefix each with 'W' (e.g., 2 → 'W2').  This ensures W2 is
        # the first week displayed and that W10, W11, etc. appear after W9.
        try:
            _week_nums_series = (
                df["주차"].dropna().astype(str)
                .str.extract(r"(\d+)")[0]
                .dropna()
                .astype(int)
            )
            # Include week 1 and above when building week options.  Previously we
            # filtered out week 1 (w >= 2), but this caused the weekly
            # selection and charts to omit W1.  Keep all unique week numbers.
            _week_nums = sorted(_week_nums_series.unique().tolist())
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

        # 수율확인 영역의 기간 선택은 주차 선택 바로 아래에 배치합니다.
        # 여기서는 공장/년도/월/주차 기준으로만 임시 범위를 계산하여
        # 일별기간선택과 주간기간선택 위젯의 표시 범위를 정합니다.
        _date_base_df = df.copy()
        if selected_plant != "전체 공장":
            _date_base_df = _date_base_df[_date_base_df["공장"] == selected_plant]
        if selected_year != "전체":
            _date_base_df = _date_base_df[_safe_year_str(_date_base_df["년도"]) == selected_year]
        if selected_month != "전체":
            _date_base_df = _date_base_df[_date_base_df["월"].astype(str) == selected_month]
        if selected_week != "전체":
            try:
                _date_week_num = int(str(selected_week).lstrip('W'))
            except Exception:
                _date_week_num = None
            if _date_week_num is not None:
                _date_week_col = _date_base_df["주차"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                _date_base_df = _date_base_df[_date_week_col == float(_date_week_num)]
        try:
            _min_dt = _date_base_df['생산일자'].dropna().min().date()
            _max_dt = _date_base_df['생산일자'].dropna().max().date()
        except Exception:
            _min_dt = None
            _max_dt = None

        date_range_daily = st.sidebar.date_input(
            "일별 기간 선택",
            value=[],
            min_value=_min_dt if _min_dt else None,
            max_value=_max_dt if _max_dt else None,
            key='date_range_daily',
        )
        date_range_weekly = st.sidebar.date_input(
            "주간 기간 선택",
            value=[],
            min_value=_min_dt if _min_dt else None,
            max_value=_max_dt if _max_dt else None,
            key='date_range_weekly',
        )

        selected_process = st.sidebar.selectbox("공정 선택", options=process_options, index=0)

        # 추가: 불량확인 섹션.  추후 확장을 위해 공장, 공정, 불량 유형을 선택할 수 있는
        # 컨트롤을 배치합니다.  현재는 기본값만 제공하며 필터링에는 사용하지 않습니다.
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 불량확인")
        # 공장 옵션: 이후 다른 공장 데이터가 추가되면 여기에 반영합니다.
        defect_plants = ["전체"] + plants[1:]
        # Use the global process order for defect filtering to ensure consistent ordering
        defect_processes = ["전체"] + [p for p in PROCESS_ORDER if p in processes]
        # 간단한 불량 유형 목록 (placeholders)
        defect_types = ["전체", "파손", "엣지기포", "엣지", "미분리", "뜯김", "리드지 불량", "블리스터 불량"]
        _ = st.sidebar.selectbox("공장 선택 (불량)", options=defect_plants, index=0)
        selected_defect_process = st.sidebar.selectbox("공정 선택 (불량)", options=defect_processes, index=0)
        selected_defect_type = st.sidebar.selectbox("불량 유형 선택", options=defect_types, index=0)

        # Note: The filtered dataset for yield summaries (df_for_summaries) will be
        # created later after the main data filtering is applied.  This avoids
        # referencing filtered_df before it has been defined.

        # ------------------------------------------------------------------
        # Header and title
        # ------------------------------------------------------------------
        # Inject custom CSS to style the dashboard selector and enlarge fonts.  The
        # radio buttons are wrapped in a .dashboard-radio div so that we can
        # apply borders, padding and background colour.  We also enlarge the
        # header fonts to improve readability.
        st.markdown(
            """
            <style>
            /* Style a wrapper div for the dashboard radio with border and padding */
            .dashboard-radio {
                border: 2px solid #e0e0e0;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 20px;
                background-color: #f9fafb;
            }
            /* Enlarge radio labels within our custom wrapper */
            .dashboard-radio label {
                font-size: 1.3rem;
                font-weight: 600;
            }
            /* Enlarge header text */
            .main-header {
                font-size: 36px;
                font-weight: 700;
                margin-bottom: 4px;
                font-family: 'Noto Sans CJK KR', sans-serif;
            }
            .sub-header {
                font-size: 48px;
                font-weight: 700;
                margin-top: 0;
                margin-bottom: 16px;
                font-family: 'Noto Sans CJK KR', sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the organisation and dashboard title with larger font sizes.
        st.markdown(
            "<div class='main-header'>인터로조 공정기술팀</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sub-header'>공정별 대시보드</div>",
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # Data filtering based on sidebar selections
        # ------------------------------------------------------------------
        # Filter the dataset based on the selected plant.  When "전체 공장" is selected
        # we keep all plants; otherwise we filter to the chosen plant only.
        if selected_plant == "전체 공장":
            filtered_df = df.copy()
        else:
            filtered_df = df[df["공장"] == selected_plant].copy()
        if selected_year != "전체":
            filtered_df = filtered_df[_safe_year_str(filtered_df["년도"]) == selected_year]
        if selected_month != "전체":
            filtered_df = filtered_df[filtered_df["월"].astype(str) == selected_month]
        if selected_week != "전체":
            # Parse the selected week (e.g., 'W2') to its numeric part and
            # compare against the numeric component of the '주차' column.  This
            # allows filtering regardless of whether the underlying data stores
            # weeks as 'W3' or just '3'.
            try:
                _week_num = int(str(selected_week).lstrip('W'))
            except Exception:
                _week_num = None
            if _week_num is not None:
                _week_col_numeric = (
                    filtered_df["주차"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                )
                _mask = _week_col_numeric == float(_week_num)
                filtered_df = filtered_df[_mask]
        if selected_process != "전체":
            filtered_df = filtered_df[filtered_df["Process"] == selected_process]
        if filtered_df.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            return

        # Now that filtered_df is defined, build a separate DataFrame for yield
        # summaries.  This allows us to filter further by defect classification
        # if the user selected a specific defect type in the defect sidebar.
        df_for_summaries = filtered_df.copy()
        if 'selected_defect_type' in locals() and selected_defect_type != '전체':
            df_for_summaries = df_for_summaries[
                df_for_summaries['신규분류요약'] == selected_defect_type
            ]
        # If no data remains after filtering by defect type, inform the user.
        if df_for_summaries.empty:
            st.warning("선택한 조건과 불량 유형에 해당하는 데이터가 없습니다.")
            return

        # ------------------------------------------------------------------
        # Date range selectors for daily and weekly views
        #
        # 기간 선택 위젯은 이미 사이드바 상단(수율확인 영역)에 배치되어 있으며,
        # 여기서는 선택된 값을 해석해서 실제 조회 구간으로 변환합니다.

        # Normalise the date inputs into (start, end) tuples.  Streamlit may
        # return either a single date or a list/tuple of dates depending on
        # how the user interacts with the widget.  For single dates, we
        # interpret the start and end as that date; for empty or None, we
        # default to the full available range.  Use exceptions to handle
        # irregular values gracefully.
        def _parse_date_range(selected, default_start, default_end):
            if selected is None:
                return default_start, default_end
            try:
                if isinstance(selected, tuple) or isinstance(selected, list):
                    if len(selected) == 2:
                        start, end = selected[0], selected[1]
                    elif len(selected) == 1:
                        start = end = selected[0]
                    else:
                        start = default_start
                        end = default_end
                else:
                    # Single date selected
                    start = end = selected
            except Exception:
                start = default_start
                end = default_end
            # Fall back to defaults if any bound is missing
            start = start or default_start
            end = end or default_end
            return start, end

        start_d, end_d = _parse_date_range(date_range_daily, _min_dt, _max_dt)
        start_w, end_w = _parse_date_range(date_range_weekly, _min_dt, _max_dt)

        def _has_user_selected_range(selected) -> bool:
            # Return True when the user explicitly picked at least one date.
            if selected is None:
                return False
            try:
                if isinstance(selected, (tuple, list)):
                    return len(selected) > 0
            except Exception:
                return False
            return True

        daily_range_selected = _has_user_selected_range(date_range_daily)
        weekly_range_selected = _has_user_selected_range(date_range_weekly)

        # ------------------------------------------------------------------
        # Top-level page selection: choose between the yield dashboard and defect dashboard
        #
        # We present a radio button at the top of the main page so users can
        # switch between two high-level pages: "수율 대시보드" (the current
        # yield dashboard) and "불량 대시보드" (a placeholder for a future
        # defect analysis dashboard).  When the defect dashboard is selected,
        # we show a placeholder message and exit early so that the existing
        # yield dashboard content is not rendered.  When the yield dashboard
        # is selected, execution continues to render the existing content.
        # Render the dashboard selection inside a styled container.  Use an
        # empty label for the radio to avoid duplicating the title text.  The
        # surrounding <div> tag allows our custom CSS to style the radio.
        st.markdown("<div class='dashboard-radio'>", unsafe_allow_html=True)
        main_tab = st.radio(
            "대시보드 선택",
            ["수율 대시보드", "불량 대시보드"],
            horizontal=True,
            index=0,
            key="main_tab_selection",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if main_tab != "수율 대시보드":
            # Delegate to the defect dashboard when selected.  The
            # `run_defect_dashboard` function is imported from
            # `defect_dashboard.py` at the top of this file.  Executing
            # it here ensures that the yield dashboard content is not
            # rendered when the user wants to view defect analytics.
            run_defect_dashboard()
            return

        # Fix the yield y-axis range to 80–100% for all charts.  Removing the
        # slider prevents users from adjusting this range.  A consistent
        # high-range view focuses attention on yield variations within the
        # top quartile across all plants and time granularities.
        y_min, y_max = 80.0, 100.0

        # ------------------------------------------------------------------
        # Summary cards (KPI panels)
        # ------------------------------------------------------------------
        proc_summary = (
            filtered_df.groupby("Process")
            .agg(prod_qty=("생산수량", "sum"), good_qty=("양품수량", "sum"))
        )
        proc_summary["yield"] = proc_summary["good_qty"] / proc_summary["prod_qty"]
        # Compute overall yield as the product of process‑specific yields rather than
        # the ratio of total good quantity to total production.  Missing processes
        # are treated as yield 1.0 so they do not affect the product.
        # Use the global process order constant for consistency.  Compute the overall
        # yield as the product of process‑specific yields.  Missing processes
        # default to a yield of 1 (neutral) so they do not affect the product.
        process_order = PROCESS_ORDER.copy()
        overall_yield = 1.0
        for proc in process_order:
            if proc in proc_summary.index:
                overall_yield *= proc_summary.loc[proc, "yield"]
            else:
                overall_yield *= 1.0
        kpi_cols = st.columns(len(process_order) + 1)
        for idx, proc in enumerate(process_order):
            with kpi_cols[idx]:
                if proc in proc_summary.index:
                    prod_k = proc_summary.loc[proc, "prod_qty"] / 1000.0
                    good_k = proc_summary.loc[proc, "good_qty"] / 1000.0
                    yield_pct = proc_summary.loc[proc, "yield"] * 100.0
                    st.markdown(
                        f"<div style='background-color:#f7f9fc; border-radius:8px; padding:10px; font-family:\'Noto Sans CJK KR\', sans-serif;'>"
                        f"<strong>{proc} 생산현황</strong><br>"
                        f"생산량: {prod_k:,.0f}K<br>"
                        f"양품량: {good_k:,.0f}K<br>"
                        f"수율: {yield_pct:.1f}%"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:#f7f9fc; border-radius:8px; padding:10px; font-family:\'Noto Sans CJK KR\', sans-serif;'>"
                        f"<strong>{proc} 생산현황</strong><br>데이터 없음"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        with kpi_cols[-1]:
            overall_pct = overall_yield * 100.0
            st.markdown(
                f"<div style='background-color:#e6f7ff; border-radius:8px; padding:10px; font-family:\'Noto Sans CJK KR\', sans-serif;'>"
                f"<strong>종합수율</strong><br>{overall_pct:.1f}%"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # Helper to summarise and convert yields to percentage for a given grouping
        # ------------------------------------------------------------------
        def summarise_and_percent(
            data: pd.DataFrame, group_cols: Tuple[str, ...]
        ) -> pd.DataFrame:
            """Summarise yields by the specified grouping columns.

            This implementation aggregates good and production quantities across all
            plants for each time group and computes yields as percentages.  For
            each group (e.g. a specific day, week or month), the yield for
            each process is calculated as the ratio of total good quantity to
            total production quantity across all plants.  The overall yield
            is then computed as the product of these per‑process yields.  Results
            are expressed as percentages and rounded to two decimal places.

            Parameters
            ----------
            data : pd.DataFrame
                The filtered production data containing columns 'Process',
                '양품수량' and '생산수량'.  Additional columns used for
                grouping (specified in ``group_cols``) should also be present.
            group_cols : Tuple[str, ...]
                Columns to group by (e.g. ('생산일자',), ('year_week',),
                ('년도', '월')).  Each unique combination of these
                columns defines a separate time bucket.

            Returns
            -------
            pd.DataFrame
                A DataFrame indexed by the grouping columns.  Columns
                include one column per process (expressed as a yield
                percentage) and an 'overall_yield' column.  Missing
                process yields are treated as 100% (neutral).  Columns
                are ordered with standard processes first followed by any
                additional processes present in the data.
            """
            # Determine the full list of processes present in the data
            # so that we compute yields for all of them.  Use sorted
            # order for deterministic column ordering.
            processes = sorted(data['Process'].dropna().astype(str).unique().tolist())
            # Guarantee that standard processes (PROCESS_ORDER) are
            # considered first when ordering columns.  Some standard
            # processes may not appear in the data; include them
            # nonetheless so that missing processes yield a neutral yield.
            standard_procs = PROCESS_ORDER.copy()
            # Prepare a list of all process names (standard first, then
            # any additional processes not in the standard list) to avoid
            # duplicates.
            all_procs = [p for p in standard_procs] + [p for p in processes if p not in standard_procs]

            # Group the data by the specified columns.  If group_cols is empty,
            # treat the entire dataset as one group.
            result_dict: Dict[Tuple, Dict[str, float]] = {}
            if not group_cols:
                group_keys = [()]
                grouped = {(): data}
            else:
                grouped = dict(tuple(data.groupby(list(group_cols))))
                group_keys = list(grouped.keys())

            for key in group_keys:
                group_df = grouped[key]
                yields: Dict[str, float] = {}
                for proc in all_procs:
                    sub = group_df[group_df['Process'] == proc]
                    good_sum = sub['양품수량'].sum()
                    prod_sum = sub['생산수량'].sum()
                    # Handle missing or zero production: yield is neutral (100%)
                    if prod_sum == 0:
                        yield_val = 1.0
                    else:
                        yield_val = good_sum / prod_sum
                    # Convert to percentage
                    yields[proc] = yield_val * 100.0
                # Compute overall yield as the product of per-process yields.  This
                # reflects the combined yield across all processes (i.e., the
                # probability that an item passes through all processes without
                # defect).  Missing or zero production for a process is treated
                # as a neutral yield (1.0).
                overall_ratio = 1.0
                for proc in all_procs:
                    sub_proc = group_df[group_df['Process'] == proc]
                    g_sum = sub_proc['양품수량'].sum()
                    p_sum = sub_proc['생산수량'].sum()
                    if p_sum == 0:
                        proc_yield_ratio = 1.0
                    else:
                        proc_yield_ratio = g_sum / p_sum
                    overall_ratio *= proc_yield_ratio
                yields['overall_yield'] = overall_ratio * 100.0
                # Store results keyed by the group
                result_dict[key] = yields
            # Convert the result dictionary into a DataFrame.  The
            # resulting index will be the group keys; ensure that
            # ordering of columns follows all_procs + ['overall_yield'].
            summary = pd.DataFrame.from_dict(result_dict, orient='index')[all_procs + ['overall_yield']]
            # Round yields to two decimal places
            summary = summary.round(2)
            # ------------------------------------------------------------------
            # Flatten the index when grouping by a single column.  Pandas
            # groupby returns keys as tuples even when there is only one
            # grouping column.  This creates a tuple index that later
            # becomes a MultiIndex when converted to a DataFrame.  In
            # downstream charting code, such tuple indices cause
            # unintended additional columns (e.g. 'level_1') when
            # resetting the index.  To avoid this, collapse a single
            # element tuple into its single value and set the index name
            # explicitly.  When multiple grouping columns are present,
            # leave the index unchanged to preserve the MultiIndex.
            if group_cols and len(group_cols) == 1:
                new_index = []
                for idx in summary.index:
                    # If the index element is a tuple (e.g. ('2026-3',)),
                    # extract its first value.  Otherwise use it as is.
                    if isinstance(idx, tuple):
                        new_index.append(idx[0] if idx else idx)
                    else:
                        new_index.append(idx)
                summary.index = new_index
                summary.index.name = group_cols[0]
            return summary

        def format_numbers(df: pd.DataFrame) -> pd.DataFrame:
            """Format all numeric columns with thousand separators.

            Integers are formatted with commas (e.g., 1,000) and floats are
            formatted with comma separators and two decimal places. Non-numeric
            columns are left unchanged.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame whose numeric columns should be formatted.

            Returns
            -------
            pd.DataFrame
                A new DataFrame with numeric values formatted as strings.
            """
            formatted = df.copy()
            for col in formatted.columns:
                if pd.api.types.is_numeric_dtype(formatted[col]):
                    if pd.api.types.is_float_dtype(formatted[col]):
                        formatted[col] = formatted[col].apply(
                            lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
                        )
                    else:
                        formatted[col] = formatted[col].apply(
                            lambda x: f"{int(x):,}" if pd.notnull(x) else ""
                        )
            return formatted

        # Compute daily summary using the defect‑filtered dataset.  Filter
        # the production records to the user‑selected date range so that
        # only rows between start_d and end_d are included.  When the
        # filtered dataset is empty, the summary will also be empty.
        # Convert production date to a Python date object for comparison.
        df_daily_range = df_for_summaries[
            (df_for_summaries["생산일자"].dt.date >= start_d)
            & (df_for_summaries["생산일자"].dt.date <= end_d)
        ].copy()
        daily_summary = summarise_and_percent(
            df_daily_range, group_cols=("생산일자",)
        )
        # If a specific process is selected in the defect sidebar, reduce the
        # daily summary to that process only (plus overall yield) so that
        # subsequent charts display a single line.  Otherwise, keep all
        # process columns.
        if 'selected_defect_process' in locals() and selected_defect_process != '전체':
            if selected_defect_process in daily_summary.columns:
                cols_to_keep = [selected_defect_process]
                if 'overall_yield' in daily_summary.columns:
                    cols_to_keep.append('overall_yield')
                daily_summary = daily_summary[cols_to_keep]
        # Safely convert the index to a string date format.  Some index values may
        # not be datetime objects (e.g., strings or tuples).  Try to cast
        # to datetime and then format; if it fails, fall back to casting to
        # string.
        # Convert the index to a YYYY‑MM‑DD string.  Some index values may be tuples
        # (e.g., from grouping) or non‑datetime objects.  Attempt to cast
        # each element to datetime; fall back to string representation.
        try:
            _dt_index = pd.to_datetime(daily_summary.index, errors="coerce")
            if _dt_index.notna().all():
                daily_summary.index = _dt_index.strftime("%Y-%m-%d")
            else:
                raise ValueError
        except Exception:
            def _fmt_index(val):
                _v = val
                if isinstance(_v, tuple) and len(_v) > 0:
                    _v = _v[0]
                try:
                    _dt = pd.to_datetime(_v)
                    return _dt.strftime("%Y-%m-%d")
                except Exception:
                    return str(_v)
            daily_summary.index = [ _fmt_index(v) for v in daily_summary.index ]
        # Use the defect‑filtered dataset for weekly summaries.  Filter by the
        # user‑selected week date range (start_w to end_w) and compute
        # weekly yields based on the numeric week number.  This avoids
        # combining the year into the key and ensures labels such as
        # 'W1', 'W2', ... are generated in ascending order.  Drop rows
        # without a numeric week number.
        df_week = df_for_summaries[
            (df_for_summaries["생산일자"].dt.date >= start_w)
            & (df_for_summaries["생산일자"].dt.date <= end_w)
        ].copy()
        # Extract the numeric portion of the '주차' column.  Some entries may
        # include a 'W' prefix or other characters; extract digits only.  If
        # no digits are found, week_num will be NaN and the row will be
        # dropped before summarising.
        df_week["week_num"] = df_week["주차"].astype(str).str.extract(r"(\d+)")[0]
        df_week = df_week[df_week["week_num"].notna()].copy()
        df_week["week_num"] = df_week["week_num"].astype(int)
        # Compute weekly summary grouped by week number.  Use a single grouping
        # column tuple ("week_num",) so the summarise_and_percent helper
        # returns a DataFrame with index values equal to the week numbers.
        weekly_summary = summarise_and_percent(df_week, group_cols=("week_num",))
        # Filter weekly summary to selected defect process if specified
        if 'selected_defect_process' in locals() and selected_defect_process != '전체':
            if selected_defect_process in weekly_summary.columns:
                cols_to_keep = [selected_defect_process]
                if 'overall_yield' in weekly_summary.columns:
                    cols_to_keep.append('overall_yield')
                weekly_summary = weekly_summary[cols_to_keep]
        # Reorder weekly summary by the numeric week number and label rows
        # as 'W#'.  When the summary is empty, skip reindexing.  Sorting
        # ensures that W1, W2, ... appear in ascending order.
        if not weekly_summary.empty:
            try:
                # The index of weekly_summary is the numeric week_num; sort it
                sorted_indices = sorted(weekly_summary.index.to_list())
                weekly_summary = weekly_summary.loc[sorted_indices]
                # Rename index to W followed by the week number
                weekly_summary.index = [f"W{int(i)}" for i in sorted_indices]
                weekly_summary.index.name = '주차'
            except Exception:
                pass
        monthly_summary = summarise_and_percent(
            df_for_summaries, group_cols=("년도", "월")
        )
        # Filter monthly summary to selected defect process if specified
        if 'selected_defect_process' in locals() and selected_defect_process != '전체':
            if selected_defect_process in monthly_summary.columns:
                cols_to_keep = [selected_defect_process]
                if 'overall_yield' in monthly_summary.columns:
                    cols_to_keep.append('overall_yield')
                monthly_summary = monthly_summary[cols_to_keep]
        # Convert the (년도, 월) MultiIndex into a single string index "YYYY-월".
        # In some cases the index names may be lost (e.g., after grouping duplicates),
        # so fallback to using numeric positions or casting to string.
        try:
            if isinstance(monthly_summary.index, pd.MultiIndex):
                idx_frame = monthly_summary.index.to_frame(index=False)
                # Expect first column to be 년도 (year) and second to be 월 (month)
                monthly_summary.index = idx_frame.apply(
                    lambda row: f"{int(row.iloc[0])}-{row.iloc[1]}", axis=1
                )
            else:
                # Single-level index: just ensure it's string
                # When the index is a simple Index, convert elements to str.  Using
                # map(str) avoids issues when the index is a MultiIndex.
                monthly_summary.index = monthly_summary.index.map(str)
        except Exception:
            # Fallback: attempt to use level values by position
            try:
                monthly_summary.index = (
                    _safe_year_str(monthly_summary.index.get_level_values(0))
                    + "-" + monthly_summary.index.get_level_values(1).astype(str)
                )
            except Exception:
                # Final fallback: convert the index elements to string using map(str)
                monthly_summary.index = monthly_summary.index.map(str)

        # ------------------------------------------------------------------
        # Utility to draw a line chart for a summary DataFrame
        # ------------------------------------------------------------------
        def draw_line_chart(summary: pd.DataFrame, title: str, xlabel: str) -> None:
            """Render a line chart of yield summaries.

            If Plotly is available (USE_PLOTLY is True), use plotly.express to
            generate an interactive line chart.  Plotly charts rely on the web
            browser for font rendering, which avoids issues with missing
            system fonts and ensures Hangul characters display correctly.  If
            Plotly is not available, fall back to matplotlib.
            """
            if USE_PLOTLY:
                # Prepare data in long format. Rename overall_yield to 종합수율 for display.
                tmp = summary.copy()
                idx_name = tmp.index.name or "index"
                tmp = tmp.reset_index()
                # Ensure that the index column has the expected name for melting.  If
                # the automatically generated column name does not match idx_name,
                # rename the first column accordingly.  This prevents KeyError when
                # melting.
                if idx_name not in tmp.columns:
                    # The first column after reset_index holds the original index values
                    tmp = tmp.rename(columns={tmp.columns[0]: idx_name})
                if "overall_yield" in tmp.columns:
                    tmp = tmp.rename(columns={"overall_yield": "종합수율"})
                # Melt the DataFrame to long format for Plotly
                df_long = tmp.melt(id_vars=[idx_name], var_name="공정", value_name="수율")
                # Compute per-process min, max, and mean statistics for hover information
                stats_df = df_long.groupby("공정")["수율"].agg(["min", "max", "mean"]).reset_index()
                stats_df = stats_df.rename(columns={"min": "최소", "max": "최대", "mean": "평균"})
                # Merge statistics back into the long DataFrame
                df_long = df_long.merge(stats_df, on="공정", how="left")
                # Create interactive line chart with markers and custom hover data
                color_map = {
                    proc_name: PROCESS_COLOR_MAP.get(proc_name)
                    for proc_name in df_long["공정"].dropna().astype(str).unique().tolist()
                    if PROCESS_COLOR_MAP.get(proc_name)
                }
                fig = px.line(
                    df_long,
                    x=idx_name,
                    y="수율",
                    color="공정",
                    markers=True,
                    title=title,
                    labels={idx_name: xlabel, "수율": "수율 (%)", "공정": "공정"},
                    hover_data={
                        "최소": ':.2f',
                        "최대": ':.2f',
                        "평균": ':.2f',
                    },
                    color_discrete_map=color_map if color_map else None,
                    # Include the numeric value as text so that data labels can appear
                    text="수율",
                )
                # Apply the user-selected y-axis range.  Variables y_min and y_max
                # are defined in the outer scope (closure) via the slider.
                # Set y-axis range and colour to black and ensure tick labels are black
                fig.update_yaxes(range=[y_min, y_max], color='black', tickfont=dict(color='black'))
                # Position text labels above each point.  Labels will be shown
                # whenever a single trace is active (e.g. when isolating a line via
                # double‑clicking the legend).
                fig.update_traces(textposition="top center")
                # Apply black font for the overall chart and specify axis titles.  Title
                # fonts are defined via update_xaxes/update_yaxes to ensure the
                # title text itself is rendered in black.  The 'font' option
                # here sets the default colour for legend and other text.
                fig.update_layout(
                    legend_title_text="공정",
                    xaxis_title=xlabel,
                    yaxis_title="수율 (%)",
                    font=dict(color='black')
                )
                # Set axis tick and line colours to black and axis title fonts to black
                fig.update_xaxes(color='black', title_font=dict(color='black'), tickfont=dict(color='black'))
                fig.update_yaxes(color='black', title_font=dict(color='black'), tickfont=dict(color='black'))
                # Enhance line appearance: make the '종합수율' trace thicker and its
                # labels bold.  Other lines retain a moderate line width.  This
                # emphasises the overall yield relative to individual process
                # yields.  The modification applies only when the trace name
                # exactly matches '종합수율'.
                for i, trace in enumerate(fig.data):
                    try:
                        name = trace.name
                    except Exception:
                        name = ""
                    if isinstance(name, str) and name.strip() == "종합수율":
                        # Increase line width and marker size for the overall yield
                        fig.data[i].update(line=dict(width=5), marker=dict(size=9, color=PROCESS_COLOR_MAP.get('종합수율', '#111111')))
                        # Bold the text labels
                        # Convert each existing label to a bold HTML span
                        if hasattr(fig.data[i], 'text') and fig.data[i].text is not None:
                            fig.data[i].update(text=[f"<b>{t}</b>" for t in fig.data[i].text])
                        # Use black colour for the overall yield data labels
                        fig.data[i].update(textfont=dict(color='black'))
                    else:
                        # Set a moderate line width for individual processes
                        fig.data[i].update(line=dict(width=4), marker=dict(size=7, color=PROCESS_COLOR_MAP.get(name, '#4E79A7')))
                st.plotly_chart(fig, use_container_width=True)
            else:
                process_cols = [c for c in summary.columns if c != "overall_yield"]
                fig, ax = plt.subplots(figsize=(10, 4))
                for col in process_cols:
                    ax.plot(
                        summary.index,
                        summary[col],
                        marker="o",
                        markersize=5,
                        linewidth=3.0,
                        label=col,
                        color=PROCESS_COLOR_MAP.get(col, '#4E79A7'),
                    )
                if "overall_yield" in summary.columns:
                    ax.plot(
                        summary.index,
                        summary["overall_yield"],
                        marker="o",
                        markersize=6,
                        linewidth=4,
                        linestyle="--",
                        label="overall_yield",
                        color=PROCESS_COLOR_MAP.get('overall_yield', 'black'),
                    )
                ax.set_xlabel(xlabel, color='black')
                ax.set_ylabel("수율 (%)", color='black')
                ax.set_title(title, color='black')
                try:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                except Exception:
                    pass
                # Set y-axis range based on the selected slider values
                ax.set_ylim(y_min, y_max)
                ax.tick_params(axis="x", rotation=45, colors='black')
                ax.tick_params(axis="y", colors='black')
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)


        def draw_product_chart(pivot_df: pd.DataFrame, title: str, xlabel: str) -> None:
            """Render a line chart for product yield summaries.

            This helper takes a pivot table where each column represents a
            product group (from the '신규분류요약' column) and each row index
            represents a time period (daily, weekly or monthly).  It
            generates an interactive line chart showing the yield (%) over
            time for each product group.  Hovering over a line will show
            the minimum, maximum and average yield for that product group.

            Parameters
            ----------
            pivot_df : pd.DataFrame
                Pivoted DataFrame with index as time labels and columns
                representing product groups.  Values should be yield (%)
                already multiplied by 100.
            title : str
                Title of the chart.
            xlabel : str
                Label for the x-axis (e.g., '날짜', '주차', '연-월').
            """
            if pivot_df.empty:
                st.write("데이터가 없습니다.")
                return
            # Use Plotly if available for better font rendering
            if USE_PLOTLY:
                tmp = pivot_df.copy()
                # Reset index so the time period becomes a column
                tmp = tmp.reset_index()
                idx_name = tmp.columns[0]
                # Melt to long format
                df_long = tmp.melt(id_vars=[idx_name], var_name="제품군", value_name="수율")
                # Compute stats (min, max, mean) per product group
                stats_df = df_long.groupby("제품군")["수율"].agg(["min", "max", "mean"]).reset_index()
                stats_df = stats_df.rename(columns={"min": "최소", "max": "최대", "mean": "평균"})
                df_long = df_long.merge(stats_df, on="제품군", how="left")
                # Create interactive line chart
                fig = px.line(
                    df_long,
                    x=idx_name,
                    y="수율",
                    color="제품군",
                    markers=True,
                    title=title,
                    labels={idx_name: xlabel, "수율": "수율 (%)", "제품군": "제품군"},
                    hover_data={
                        "최소": ':.2f',
                        "최대": ':.2f',
                        "평균": ':.2f',
                    },
                )
                # Set y-axis range to 50–100 and colour to black.  Also ensure the
                # axis title uses a black font and tick labels are black.
                fig.update_yaxes(
                    range=[50, 100],
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black'),
                )
                # Update layout with black font and maintain legend title
                fig.update_layout(
                    legend_title_text="제품군",
                    font=dict(color='black')
                )
                # Show data labels above points with black text
                fig.update_traces(text=df_long["수율"], textposition="top center", textfont=dict(color='black'))
                # Set x-axis label and ticks to black and specify title and tick fonts
                fig.update_xaxes(
                    title=xlabel,
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig, ax = plt.subplots(figsize=(10, 4))
                for col in pivot_df.columns:
                    ax.plot(
                        pivot_df.index,
                        pivot_df[col],
                        marker="o",
                        markersize=3,
                        label=col,
                    )
                ax.set_xlabel(xlabel, color='black')
                ax.set_ylabel("수율 (%)", color='black')
                ax.set_title(title, color='black')
                try:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                except Exception:
                    pass
                # Align y-axis range with other charts (50–100)
                ax.set_ylim(50, 100)
                ax.tick_params(axis="x", rotation=45, colors='black')
                ax.tick_params(axis="y", colors='black')
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

        # ------------------------------------------------------------------
        # Compute product summary using the '신규분류요약' column.  We place this
        # before the tab definitions so that product_summary_df is defined
        # when it is referenced inside each tab.  The summary groups records
        # by defect classification and summarises production quantity, good
        # quantity and yield percentage.
        prod_group = "신규분류요약"
        product_summary_df = (
            filtered_df.groupby(prod_group)
            .agg(생산량=("생산수량", "sum"), 양품량=("양품수량", "sum"))
        )
        product_summary_df["수율(%)"] = (
            product_summary_df["양품량"] / product_summary_df["생산량"] * 100
        ).round(2)
        product_summary_df = (
            product_summary_df.reset_index()
            .rename(columns={prod_group: "신규분류요약"})
            .sort_values("신규분류요약")
        )

        # ------------------------------------------------------------------
        # Compute pivot tables for product yields by time unit (daily, weekly, monthly)
        # ------------------------------------------------------------------
        # Prepare a copy of the filtered data with additional time keys
        _prod_df = filtered_df.copy()
        # Daily key: convert production date to string (YYYY-MM-DD)
        _prod_df["_day_key"] = _prod_df["생산일자"].dt.strftime("%Y-%m-%d")
        # Weekly key: year-week format.  Use existing '주차' and '년도'
        _prod_df["_week_key"] = (
            _safe_year_str(_prod_df["년도"])
            + "-W"
            + _prod_df["주차"].astype(str)
        )
        # Monthly key: year-month format
        _prod_df["_month_key"] = (
            _safe_year_str(_prod_df["년도"])
            + "-"
            + _prod_df["월"].astype(str)
        )
        # Function to create a pivot table: index is the time key, columns are product groups
        def _pivot_product(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
            tmp = (
                df.groupby([time_col, "신규분류요약"])
                .agg(생산량=("생산수량", "sum"), 양품량=("양품수량", "sum"))
                .reset_index()
            )
            tmp["수율(%)"] = (tmp["양품량"] / tmp["생산량"] * 100).round(2)
            pivot = tmp.pivot(index=time_col, columns="신규분류요약", values="수율(%)")
            # Fill missing values with zero to avoid NaNs
            pivot = pivot.fillna(0).round(2)
            # Sort index for clarity
            pivot = pivot.sort_index()
            return pivot
        product_daily_pivot = _pivot_product(_prod_df, "_day_key")
        product_weekly_pivot = _pivot_product(_prod_df, "_week_key")
        product_monthly_pivot = _pivot_product(_prod_df, "_month_key")

        # Reorder weekly pivot to ensure weeks start from W2 and follow ascending order
        if not product_weekly_pivot.empty:
            # Extract week numbers and sort
            def _extract_week(x: str) -> int:
                # Expect format 'YYYY-W##' or similar; extract numeric part after 'W'
                try:
                    return int(x.split('W')[-1])
                except Exception:
                    return 0
            sorted_week_index = sorted(product_weekly_pivot.index, key=_extract_week)
            product_weekly_pivot = product_weekly_pivot.loc[sorted_week_index]

        # ------------------------------------------------------------------
        # Combined monthly/weekly good quantity & yield summary
        #
        # Insert a composite chart showing five metrics: total good quantity
        # (양품량) and the overall yields for A관, C관, S관 and all plants
        # combined.  We compute these summaries using the full dataset
        # ``df`` so that all three plants are represented regardless of
        # the current plant selection.  The resulting charts allow users
        # to compare monthly and weekly performance across plants and the
        # overall process.  A bar is used for 양품량 (quantity) on the
        # secondary y-axis and lines are used for yields on the primary
        # y-axis.

        def compute_combined_summaries(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Compute monthly and weekly combined summaries for all plants.

            Parameters
            ----------
            full_df : pd.DataFrame
                The complete production dataset across all plants and processes.

            Returns
            -------
            Tuple[pd.DataFrame, pd.DataFrame]
                Two DataFrames for monthly and weekly summaries.  Each DataFrame
                has the time period as the index and columns:
                - 양품량: total good quantity across all plants and processes
                - A관: overall yield for plant A across all processes (product of process yields)
                - C관: overall yield for plant C
                - S관: overall yield for plant S
                - 종합수율: overall yield across all plants and processes (product of process yields)
            """
            # Helper to compute plant and overall yields for a given group.
            # The overall yield is defined as the product of the yields of each
            # process within the group.  For each process, compute the ratio
            # of total good quantity to total production quantity and multiply
            # these ratios.  If a process has zero production, treat its yield
            # as 1.0 (neutral) so it does not affect the product.
            def _compute_yields_product(group: pd.DataFrame) -> float:
                # Determine the list of processes present in this group
                procs_in_group = group['Process'].dropna().astype(str).unique().tolist()
                # Use the global PROCESS_ORDER first, then any additional
                # processes appearing in the data to ensure all relevant
                # processes are considered
                all_procs = [p for p in PROCESS_ORDER] + [p for p in procs_in_group if p not in PROCESS_ORDER]
                overall_ratio = 1.0
                for proc in all_procs:
                    sub = group[group['Process'] == proc]
                    good_sum = sub['양품수량'].sum()
                    prod_sum = sub['생산수량'].sum()
                    if prod_sum == 0:
                        proc_ratio = 1.0
                    else:
                        proc_ratio = float(good_sum / prod_sum)
                    overall_ratio *= proc_ratio
                return overall_ratio

            # Monthly summary
            # Build a pivot with index (year, month) and compute metrics
            monthly_records = []
            # Group by year and month to compute good quantity and yields
            for (year, month), group in full_df.groupby(['년도', '월']):
                # Total good quantity across all plants for the leak/standard inspection process only
                total_good = group[group['Process'] == '누수/규격검사']['양품수량'].sum()
                # Compute yields for each plant using product of process yields
                plant_yields: Dict[str, float] = {}
                for plant in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
                    sub = group[group['공장'] == plant]
                    if not sub.empty:
                        plant_yields[plant] = _compute_yields_product(sub)
                    else:
                        plant_yields[plant] = 1.0
                # Overall yield across all plants: product of process yields across the entire group
                overall = _compute_yields_product(group)
                monthly_records.append({
                    '년도': year,
                    '월': month,
                    '양품량': total_good,
                    'A관': plant_yields['A관(1공장)'],
                    'C관': plant_yields['C관(2공장)'],
                    'S관': plant_yields['S관(3공장)'],
                    '종합수율': overall,
                })
            monthly_combined = pd.DataFrame(monthly_records)
            # Create a string index "YYYY-월" for display
            monthly_combined['연-월'] = monthly_combined.apply(
                lambda row: f"{int(row['년도'])}-{row['월']}", axis=1
            )
            monthly_combined = monthly_combined.set_index('연-월')[
                ['양품량', 'A관', 'C관', 'S관', '종합수율']
            ]

            # Weekly summary
            weekly_records = []
            # Extract numeric week number from '주차' (which may include 'W' prefix)
            full_df['_week_num'] = full_df['주차'].astype(str).str.extract(r"(\d+)")[0].astype(float)
            for (year, week_num), group in full_df.groupby(['년도', '_week_num']):
                # Include all week numbers (including week 1) and use leak/standard inspection good quantity only
                total_good = group[group['Process'] == '누수/규격검사']['양품수량'].sum()
                plant_yields = {}
                for plant in ['A관(1공장)', 'C관(2공장)', 'S관(3공장)']:
                    sub = group[group['공장'] == plant]
                    plant_yields[plant] = _compute_yields_product(sub) if not sub.empty else 1.0
                overall = _compute_yields_product(group)
                weekly_records.append({
                    '년도': year,
                    '주차번호': int(week_num),
                    '양품량': total_good,
                    'A관': plant_yields['A관(1공장)'],
                    'C관': plant_yields['C관(2공장)'],
                    'S관': plant_yields['S관(3공장)'],
                    '종합수율': overall,
                })
            weekly_combined = pd.DataFrame(weekly_records)
            if not weekly_combined.empty:
                # Sort by numeric week number
                weekly_combined = weekly_combined.sort_values(['년도', '주차번호'])
                # Create string index like 'W2', 'W3', etc.
                weekly_combined['주차'] = weekly_combined['주차번호'].apply(lambda x: f"W{int(x)}")
                weekly_combined = weekly_combined.set_index('주차')[
                    ['양품량', 'A관', 'C관', 'S관', '종합수율']
                ]
            else:
                # No weekly data
                weekly_combined = pd.DataFrame(columns=['양품량', 'A관', 'C관', 'S관', '종합수율'])
            # Clean up the temporary week number column to avoid side effects on the input DataFrame
            if '_week_num' in full_df.columns:
                full_df.drop(columns=['_week_num'], inplace=True)
            return monthly_combined, weekly_combined


        def compute_process_period_summary(
            full_df: pd.DataFrame, period: str
        ) -> Dict[str, pd.DataFrame]:
            """Compute per-process yield summaries for monthly or weekly periods.

            Parameters
            ----------
            full_df : pd.DataFrame
                Full production dataset.
            period : str
                Either 'monthly' or 'weekly'.

            Returns
            -------
            Dict[str, pd.DataFrame]
                Mapping from process name to a summary DataFrame indexed by
                period label with one process-named yield column in percent.
            """
            result: Dict[str, pd.DataFrame] = {}
            for proc in PROCESS_ORDER:
                proc_df = full_df[full_df["Process"] == proc].copy()
                if proc_df.empty:
                    empty_df = pd.DataFrame(columns=[proc])
                    empty_df.index.name = "label"
                    result[proc] = empty_df
                    continue
                if period == "monthly":
                    grouped = (
                        proc_df.groupby(["년도", "월"])
                        .agg(양품수량=("양품수량", "sum"), 생산수량=("생산수량", "sum"))
                        .reset_index()
                    )
                    grouped["label"] = grouped.apply(
                        lambda row: f"{int(row['년도'])}-{row['월']}", axis=1
                    )
                else:
                    proc_df["_week_num"] = (
                        proc_df["주차"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                    )
                    grouped = (
                        proc_df.groupby(["년도", "_week_num"])
                        .agg(양품수량=("양품수량", "sum"), 생산수량=("생산수량", "sum"))
                        .reset_index()
                        .dropna(subset=["_week_num"])
                    )
                    grouped = grouped.sort_values(["년도", "_week_num"])
                    grouped["label"] = grouped["_week_num"].apply(
                        lambda x: f"W{int(x)}"
                    )
                grouped[proc] = np.where(
                    grouped["생산수량"] > 0,
                    grouped["양품수량"] / grouped["생산수량"] * 100.0,
                    100.0,
                )
                summary = grouped.set_index("label")[[proc]].round(2)
                summary.index.name = "label"
                result[proc] = summary
            return result

        def draw_combined_chart(combined_df: pd.DataFrame, title: str, xlabel: str, y_range: Tuple[float, float] = (80.0, 100.0)) -> None:
            """Draw a combined bar/line chart for good quantity and plant yields.

            The DataFrame must have columns: '양품량', 'A관', 'C관', 'S관', '종합수율'.  The
            index will be used as the x-axis labels.  A bar is drawn for
            '양품량' on the secondary y-axis (right), and lines are drawn for
            the yields on the primary y-axis (left).  Data labels are shown
            for both bars and lines.  The yield axis is constrained to the
            range 50–100 to align with other yield charts.
            """
            if combined_df.empty:
                st.write("데이터가 없습니다.")
                return
            if USE_PLOTLY:
                from plotly.subplots import make_subplots  # type: ignore
                import plotly.graph_objects as go  # type: ignore
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                # Bars for good quantity (scaled to millions).  Bars use the secondary y-axis (right)
                bar_values = combined_df['양품량'] / 1_000_000.0
                fig.add_trace(
                    go.Bar(
                        x=combined_df.index,
                        y=bar_values,
                        name='양품량',
                        marker_color='#5F90B5',
                        # Format bar labels without unit suffix (one decimal)
                        text=[f"{v:.1f}" for v in bar_values],
                        textposition='outside',
                        textfont=dict(color='#3F6F95'),
                        # Use a lower opacity to ensure the line remains fully visible across
                        # the entire bar area
                        opacity=0.32,
                    ),
                    secondary_y=True,
                )
                # Define distinct colours for the yield lines.  Use the original
                # colour palette without darkening so that the lines remain
                # vibrant.  Colours correspond to A관, C관, S관 and 종합수율.
                base_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
                line_colors: list[str] = list(base_colors)
                # Lines for yields (use primary y-axis on the left)
                for col, color in zip(['A관', 'C관', 'S관', '종합수율'], line_colors):
                    # Coerce values to numeric to avoid plotting non‑numeric types.  Any
                    # non‑numeric values are converted to NaN.  Multiply by 100 to
                    # convert ratios to percentages.
                    try:
                        numeric_vals = pd.to_numeric(combined_df[col], errors='coerce').astype(float)
                    except Exception:
                        numeric_vals = combined_df[col].astype(float)
                    line_values = (numeric_vals * 100.0).round(2)
                    # Bold the text for the combined overall yield line to make it stand out.
                    text_labels = [
                        f"<b>{val:.1f}</b>" if col == '종합수율' else f"{val:.1f}"
                        for val in line_values
                    ]
                    # For the combined yield line, show data labels in black; otherwise use
                    # the same colour as the line.  Maintain larger font size for
                    # the combined yield.  This emphasises the overall yield value.
                    if col == '종합수율':
                        label_color = 'black'
                        label_size = 14
                    else:
                        label_color = color
                        label_size = 12
                    fig.add_trace(
                        go.Scatter(
                            x=combined_df.index,
                            y=line_values,
                            name=col,
                            mode='lines+markers+text',
                            marker=dict(size=6, color=color),
                            # Make lines thicker to differentiate them from bars
                            line=dict(color=color, width=4),
                            # Display the yield as a label above each point (one decimal, no %)
                            text=text_labels,
                            textposition='top center',
                            # Set text font colour and size (black for 종합수율)
                            textfont=dict(color=label_color, size=label_size),
                            hovertemplate='%{y:.2f}%'
                        ),
                        secondary_y=False,
                    )
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    bargap=0.2,
                    bargroupgap=0.1,
                    legend=dict(orientation='h', y=-0.25),
                    font=dict(color='black'),
                )
                # Left y-axis for yield (%) with configurable range and black labels/title and tick labels
                fig.update_yaxes(
                    title_text='수율 (%)',
                    title_font=dict(color='black'),
                    secondary_y=False,
                    range=list(y_range),
                    color='black',
                    tickfont=dict(color='black')
                )
                # Right y-axis for quantity (in millions) with black labels/title and tick labels
                fig.update_yaxes(
                    title_text='수량 (M)',
                    title_font=dict(color='black'),
                    secondary_y=True,
                    color='black',
                    tickfont=dict(color='black')
                )
                # Set x-axis tick and title colours to black and tick labels to black
                fig.update_xaxes(color='black', title_font=dict(color='black'), tickfont=dict(color='black'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to matplotlib: draw bars on secondary axis (right) and lines on primary axis (left)
                fig, ax_yield = plt.subplots(figsize=(8, 4))
                x = np.arange(len(combined_df))
                # Secondary axis for good quantity in millions (bars drawn first so lines appear on top)
                ax_qty = ax_yield.twinx()
                bar_vals = (combined_df['양품량'] / 1_000_000.0).values
                # Draw bars first with lower zorder so lines appear on top
                # Draw bars with low opacity so that the overlaid lines remain clearly
                # visible across the entire bar area
                bars = ax_qty.bar(x, bar_vals, color='#5F90B5', label='양품량', alpha=0.35, zorder=1)
                ax_qty.set_ylabel('수량 (M)')
                # Annotate bars with values (one decimal, no unit)
                for i, b in enumerate(bars):
                    height = b.get_height()
                    ax_qty.annotate(
                        f"{height:.1f}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color='#3F6F95',
                        zorder=2,
                    )
                # Bring the yield axis in front of the quantity axis so the line color
                # stays visually consistent from start to end, including where it crosses bars.
                ax_yield.set_zorder(ax_qty.get_zorder() + 1)
                ax_yield.patch.set_visible(False)
                # Plot yield lines on the primary axis (left) after bars to ensure lines are on top.
                # Use the original colour palette for the lines without darkening.
                base_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
                darkened = list(base_colors)
                for col, color in zip(['A관', 'C관', 'S관', '종합수율'], darkened):
                    y_vals = (combined_df[col] * 100.0).values
                    ax_yield.plot(x, y_vals, marker='o', markersize=6, linewidth=4.0, label=col, color=color, zorder=4)
                    # Annotate each point with its value (one decimal, no %); bold for 종합수율
                    for j, v in enumerate(y_vals):
                        # Use black text for the combined yield data labels; otherwise use the line colour
                        label_color = 'black' if col == '종합수율' else color
                        ax_yield.annotate(
                            f"{v:.1f}",
                            xy=(j, v),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=8 if col != '종합수율' else 10,
                            fontweight='bold' if col == '종합수율' else 'normal',
                            color=label_color,
                            zorder=4,
                        )
                # Set axis labels, ranges, and colours
                ax_yield.set_ylabel('수율 (%)', color='black')
                # Constrain yield axis to the provided range to align with other yield charts
                ax_yield.set_ylim(y_range)
                ax_yield.set_xticks(x)
                ax_yield.set_xticklabels(combined_df.index, rotation=45, color='black')
                ax_qty.set_ylabel('수량 (M)', color='black')
                # Colour ticks for both axes
                ax_yield.tick_params(axis='y', colors='black')
                ax_qty.tick_params(axis='y', colors='black')
                # Combine legends from both axes
                lines1, labels1 = ax_yield.get_legend_handles_labels()
                lines2, labels2 = ax_qty.get_legend_handles_labels()
                ax_yield.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax_yield.set_xlabel(xlabel, color='black')
                ax_yield.set_title(title, color='black')
                fig.tight_layout()
                st.pyplot(fig)


        def _limit_recent_months_combined(combined_df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
            """Sort a combined monthly summary by year/month and keep only recent rows."""
            if combined_df.empty:
                return combined_df
            tmp = combined_df.reset_index().copy()
            idx_col = tmp.columns[0]
            idx_series = tmp[idx_col].astype(str)
            tmp["_year_sort"] = pd.to_numeric(idx_series.str.extract(r"^(\d{4})")[0], errors="coerce")
            tmp["_month_sort"] = pd.to_numeric(idx_series.str.extract(r"-(\d{1,2})")[0], errors="coerce")
            tmp = tmp.dropna(subset=["_year_sort", "_month_sort"])
            if tmp.empty:
                return combined_df.tail(min(months, len(combined_df)))
            tmp = tmp.sort_values(["_year_sort", "_month_sort"])
            tmp = tmp.tail(min(months, len(tmp)))
            tmp = tmp.drop(columns=["_year_sort", "_month_sort"]).set_index(idx_col)
            return tmp

        # ------------------------------------------------------------------
        # Insert combined yield & quantity graphs here (before the time tabs)
        # ------------------------------------------------------------------
        # Display the combined header with the data cutoff date inline.
        if last_date_label:
            # Display the main header and append the data cutoff date on the same
            # line.  The date text is rendered slightly smaller than the main
            # header and follows a hyphen.  Use a reduced font size to
            # distinguish it from the title while keeping the alignment neat.
            st.markdown(
                f"<h3 style='margin-bottom:0;'>종합 수율 및 양품량 - <span style='font-size:0.75em; font-weight:500;'>Data 기준일: {last_date_label}</span></h3>",
                unsafe_allow_html=True,
            )
        else:
            st.subheader("종합 수율 및 양품량")

        # Compute combined summaries using the full dataset rather than the
        # filtered view.  This ensures that A관, C관, S관 and the overall
        # yield are represented even when a specific plant is selected.
        monthly_combined, weekly_combined = compute_combined_summaries(df)
        monthly_combined_plot = _limit_recent_months_combined(monthly_combined, 12)
        # Display monthly and weekly combined charts side by side using columns
        comb_cols = st.columns(2)
        with comb_cols[0]:
            st.markdown("**월간**")
            # For monthly combined chart, use a yield range of 65–80 to reflect lower overall yields
            draw_combined_chart(monthly_combined_plot, "월간 양품량 및 수율", "연-월", (65.0, 80.0))
        with comb_cols[1]:
            st.markdown("**주간**")
            # For weekly combined chart, use a yield range of 65–85 to capture more variation
            draw_combined_chart(weekly_combined, "주간 양품량 및 수율", "주차", (65.0, 85.0))


        # ------------------------------------------------------------------
        # Tabs for daily, weekly, monthly yield views and production check
        # ------------------------------------------------------------------
        tab_daily, tab_weekly, tab_monthly, tab_prod = st.tabs([
            "일별 수율", "주간 수율", "월별 수율", "생산량 확인"
        ])
        with tab_daily:
            st.subheader(f"{selected_plant} - 일별 수율")
            # Draw the daily summary chart using only the most recent 30 days.  Convert
            # index to datetime for sorting, then take the last 30 records.  If
            # conversion fails (e.g., non-standard format), fall back to tail.
            if not daily_summary.empty:
                try:
                    ds = daily_summary.copy()
                    ds.index = pd.to_datetime(ds.index)
                    ds = ds.sort_index()
                    if daily_range_selected:
                        daily_summary_plot = ds
                    else:
                        daily_summary_plot = ds.tail(min(30, len(ds)))
                    daily_summary_plot.index = daily_summary_plot.index.strftime("%Y-%m-%d")
                except Exception:
                    if daily_range_selected:
                        daily_summary_plot = daily_summary
                    else:
                        daily_summary_plot = daily_summary.tail(min(30, len(daily_summary)))
            else:
                daily_summary_plot = daily_summary
            # 일별 수율은 전체 공장뿐 아니라 A관/C관/S관 개별 선택 시에도 동일하게 70~100 범위를 사용합니다.
            _orig_ymin, _orig_ymax = y_min, y_max
            y_min = 70.0
            y_max = 100.0
            draw_line_chart(daily_summary_plot, f"{selected_plant} 일별 수율 (공정별)", "날짜")
            # Restore the original y-axis range for subsequent charts
            y_min, y_max = _orig_ymin, _orig_ymax

            # After the chart, provide a production date search.  Selecting a specific
            # date filters the yield summary table to that date; selecting '전체'
            # shows all dates.  The search and table are positioned below the chart.
            unique_dates = sorted(
                filtered_df['생산일자'].dt.strftime('%Y-%m-%d').unique()
            )
            selected_date_daily = st.selectbox(
                '생산일자 검색',
                options=(['검색안함'] + unique_dates),
                index=0,
                key='date_search_daily'
            )
            if selected_date_daily == '검색안함':
                summary_daily_filtered = None
            else:
                # Filter the daily summary to the selected date if present
                if selected_date_daily in daily_summary.index:
                    summary_daily_filtered = daily_summary.loc[[selected_date_daily]]
                else:
                    summary_daily_filtered = daily_summary.iloc[0:0]
            # Reorder columns only when a table should actually be shown.
            # '검색안함'을 선택한 경우 summary_daily_filtered는 None이며
            # 이때는 표를 숨긴다.
            if summary_daily_filtered is not None:
                col_order = [
                    '사출조립', '분리', '하이드레이션/전면검사', '접착/멸균', '누수/규격검사', 'overall_yield'
                ]
                summary_daily_filtered = summary_daily_filtered.reindex(columns=col_order)
                summary_daily_filtered = summary_daily_filtered.rename(columns={
                    'overall_yield': 'Overall_Yield'
                })
                # Convert index to a column named '생산일자'
                daily_df = summary_daily_filtered.reset_index()
                # Rename the first column (previous index) to '생산일자'
                if daily_df.columns[0] != '생산일자':
                    daily_df = daily_df.rename(columns={daily_df.columns[0]: '생산일자'})
                # Format numbers with thousand separators and two decimal places
                daily_df_fmt = format_numbers(daily_df)
                # Display the filtered summary table below the search box
                st.dataframe(daily_df_fmt, use_container_width=True)
        with tab_weekly:
            st.subheader(f"{selected_plant} - 주간 수율")
            # Draw the weekly summary chart using only the most recent 10 weeks.  If
            # fewer than 10 weeks are available, display all.  This keeps the
            # chart focused on recent performance.
            if not weekly_summary.empty:
                if weekly_range_selected:
                    weekly_summary_plot = weekly_summary
                else:
                    weekly_summary_plot = weekly_summary.tail(min(10, len(weekly_summary)))
            else:
                weekly_summary_plot = weekly_summary
            _orig_ymin, _orig_ymax = y_min, y_max
            y_min, y_max = 70.0, 100.0
            draw_line_chart(weekly_summary_plot, f"{selected_plant} 주간 수율 (공정별)", "주차")
            y_min, y_max = _orig_ymin, _orig_ymax

            # Provide a production date search to filter the weekly summary table.
            unique_dates = sorted(
                filtered_df['생산일자'].dt.strftime('%Y-%m-%d').unique()
            )
            selected_date_weekly = st.selectbox(
                '생산일자 검색',
                options=(['검색안함'] + unique_dates),
                index=0,
                key='date_search_weekly'
            )
            if selected_date_weekly == '검색안함':
                summary_weekly_filtered = None
            else:
                # Determine week key from the selected date.  Weekly summary index
                # uses 'W#' format (e.g., 'W2', 'W3'), so extract the ISO week number
                # and build the key accordingly.
                dt_sel = pd.to_datetime(selected_date_weekly)
                week_no = int(dt_sel.isocalendar().week)
                week_key = f"W{week_no}"
                if week_key in weekly_summary.index:
                    summary_weekly_filtered = weekly_summary.loc[[week_key]]
                else:
                    summary_weekly_filtered = weekly_summary.iloc[0:0]
            # Reorder columns and rename overall_yield to Overall_Yield only when
            # a table should actually be shown. '검색안함'을 선택한
            # 경우 summary_weekly_filtered는 None이며 이때는 표를 숨긴다.
            if summary_weekly_filtered is not None:
                col_order = [
                    '사출조립', '분리', '하이드레이션/전면검사', '접착/멸균', '누수/규격검사', 'overall_yield'
                ]
                summary_weekly_filtered = summary_weekly_filtered.reindex(columns=col_order)
                summary_weekly_filtered = summary_weekly_filtered.rename(columns={'overall_yield': 'Overall_Yield'})
                # Reset index to a column named '주차'
                weekly_df = summary_weekly_filtered.reset_index()
                if weekly_df.columns[0] != '주차':
                    weekly_df = weekly_df.rename(columns={weekly_df.columns[0]: '주차'})
                weekly_df_fmt = format_numbers(weekly_df)
                st.dataframe(weekly_df_fmt, use_container_width=True)
        with tab_monthly:
            st.subheader(f"{selected_plant} - 월별 수율")
            # Draw the monthly summary chart before the search and table.  This
            # displays all months regardless of the selected date.
            _orig_ymin, _orig_ymax = y_min, y_max
            y_min, y_max = 70.0, 100.0
            draw_line_chart(monthly_summary, f"{selected_plant} 월별 수율 (공정별)", "연-월")
            y_min, y_max = _orig_ymin, _orig_ymax

            # Allow users to filter the monthly summary by selecting a specific
            # production date.  Selecting '전체' shows the full summary.
            unique_dates = sorted(
                filtered_df['생산일자'].dt.strftime('%Y-%m-%d').unique()
            )
            selected_date_monthly = st.selectbox(
                '생산일자 검색',
                options=(['검색안함'] + unique_dates),
                index=0,
                key='date_search_monthly'
            )
            if selected_date_monthly == '검색안함':
                summary_monthly_filtered = None
            else:
                dt_sel_m = pd.to_datetime(selected_date_monthly)
                month_key = f"{dt_sel_m.year}-{dt_sel_m.month}"
                if month_key in monthly_summary.index:
                    summary_monthly_filtered = monthly_summary.loc[[month_key]]
                else:
                    summary_monthly_filtered = monthly_summary.iloc[0:0]
            # Reorder columns and rename overall_yield to Overall_Yield only when
            # a table should actually be shown. '검색안함'을 선택한
            # 경우 summary_monthly_filtered는 None이며 이때는 표를 숨긴다.
            if summary_monthly_filtered is not None:
                col_order = [
                    '사출조립', '분리', '하이드레이션/전면검사', '접착/멸균', '누수/규격검사', 'overall_yield'
                ]
                summary_monthly_filtered = summary_monthly_filtered.reindex(columns=col_order)
                summary_monthly_filtered = summary_monthly_filtered.rename(columns={'overall_yield': 'Overall_Yield'})
                # Convert index to a column named '연-월'
                monthly_df = summary_monthly_filtered.reset_index()
                if monthly_df.columns[0] != '연-월':
                    monthly_df = monthly_df.rename(columns={monthly_df.columns[0]: '연-월'})
                # Format and show the table below the search box
                monthly_df_fmt = format_numbers(monthly_df)
                st.dataframe(
                    monthly_df_fmt,
                    use_container_width=True,
                )
        with tab_prod:
            # Production quantity check: show table and bar chart of production quantities by process
            st.subheader(f"{selected_plant} - 생산량 확인")
            # Aggregate production and good quantity by process and compute yield
            prod_df = (
                filtered_df.groupby('Process').agg(
                    생산량=('생산수량', 'sum'),
                    양품량=('양품수량', 'sum'),
                )
                .reset_index()
                .sort_values('생산량', ascending=False)
            )
            prod_df['수율(%)'] = (
                prod_df['양품량'] / prod_df['생산량'] * 100
            ).round(2)
            # Format numbers with thousand separators
            prod_df_fmt = format_numbers(prod_df)
            # Show the table with yield column
            st.dataframe(prod_df_fmt, use_container_width=True)
            # Bar/line combo chart: production and good quantities as bars; yield as line with secondary y-axis
            if USE_PLOTLY:
                from plotly.subplots import make_subplots  # type: ignore
                import plotly.graph_objects as go  # type: ignore
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                # Bar for 생산량
                fig.add_trace(
                    go.Bar(
                        x=prod_df['Process'],
                        y=prod_df['생산량'],
                        name='생산량',
                        marker_color='#5470c6',
                        width=0.4,
                    ),
                    secondary_y=False,
                )
                # Bar for 양품량
                fig.add_trace(
                    go.Bar(
                        x=prod_df['Process'],
                        y=prod_df['양품량'],
                        name='양품량',
                        marker_color='#91cc75',
                        width=0.4,
                    ),
                    secondary_y=False,
                )
                # Line for 수율(%)
                fig.add_trace(
                    go.Scatter(
                        x=prod_df['Process'],
                        y=prod_df['수율(%)'],
                        name='수율(%)',
                        mode='lines+markers',
                        line=dict(color='#ee6666'),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    title=f"{selected_plant} 공정별 생산량, 양품량 및 수율",
                    xaxis_title='공정',
                    yaxis_title='수량',
                    legend=dict(orientation='h', y=-0.25),
                    bargap=0.2,
                    bargroupgap=0.1,
                    font=dict(color='black')
                )
                # Set axis labels and colours; specify title font colours
                fig.update_yaxes(title_text='수량', title_font=dict(color='black'), secondary_y=False, color='black')
                # Set the yield axis to range between 50 and 100 and colour to black
                fig.update_yaxes(title_text='수율 (%)', title_font=dict(color='black'), secondary_y=True, range=[50, 100], color='black')
                fig.update_xaxes(color='black', title_font=dict(color='black'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to matplotlib: bar chart with two series and line on secondary axis
                fig, ax1 = plt.subplots(figsize=(8, 4))
                processes = prod_df['Process']
                x = np.arange(len(processes))
                width = 0.35
                ax1.bar(x - width/2, prod_df['생산량'], width=width, label='생산량', color='#5470c6')
                ax1.bar(x + width/2, prod_df['양품량'], width=width, label='양품량', color='#91cc75')
                ax1.set_xlabel('공정', color='black')
                ax1.set_ylabel('수량', color='black')
                ax1.set_xticks(x)
                ax1.set_xticklabels(processes, rotation=45, color='black')
                # Secondary axis for yield
                ax2 = ax1.twinx()
                ax2.plot(x, prod_df['수율(%)'], color='#ee6666', marker='o', label='수율(%)')
                ax2.set_ylabel('수율 (%)', color='black')
                # Set consistent y-axis range for yield
                ax2.set_ylim(50, 100)
                # Colour tick labels
                ax1.tick_params(axis='x', colors='black')
                ax1.tick_params(axis='y', colors='black')
                ax2.tick_params(axis='y', colors='black')
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1.set_title(f"{selected_plant} 공정별 생산량, 양품량 및 수율", color='black')
                fig.tight_layout()
                st.pyplot(fig)


        # (Dataset toggle moved to the bottom of the page.  See later code.)

        st.markdown("---")
        st.subheader("공정별 수율")
        # 전체 공장뿐 아니라 A관/C관/S관 개별 선택 시에도 현재 선택된 공장 데이터 기준으로
        # 동일한 위치와 구성으로 공정별 월간/주간 수율을 표시합니다.
        monthly_process_summaries = compute_process_period_summary(filtered_df, "monthly")
        weekly_process_summaries = compute_process_period_summary(filtered_df, "weekly")
        process_y_ranges = {
            "사출조립": (85.0, 100.0),
            "분리": (80.0, 95.0),
            "하이드레이션/전면검사": (95.0, 100.0),
            "접착/멸균": (90.0, 100.0),
            "누수/규격검사": (95.0, 100.0),
        }
        for proc in PROCESS_ORDER:
            st.markdown(f"**{proc}**")
            proc_cols = st.columns(2)
            proc_monthly_summary = monthly_process_summaries.get(proc, pd.DataFrame(columns=[proc]))
            proc_weekly_summary = weekly_process_summaries.get(proc, pd.DataFrame(columns=[proc]))
            _orig_ymin, _orig_ymax = y_min, y_max
            y_min, y_max = process_y_ranges.get(proc, (70.0, 100.0))
            with proc_cols[0]:
                draw_line_chart(proc_monthly_summary, f"{proc} 월간 수율", "연-월")
            with proc_cols[1]:
                draw_line_chart(proc_weekly_summary, f"{proc} 주간 수율", "주차")
            y_min, y_max = _orig_ymin, _orig_ymax

        # ------------------------------------------------------------------
        # Product-level yield views with category and time selection
        # ------------------------------------------------------------------
        st.subheader("제품군별 수율 현황")
        # Define the major categories and their matching prefixes.  This will
        # allow grouping of finer product categories (e.g., '1-Day_Sph',
        # 'FRP_Color_Sph') into broader categories (1-DAY, FRP, Si-1-DAY, Si-FRP).
        category_prefix_map = {
            "1-DAY": ["1-Day", "1-DAY"],
            "FRP": ["FRP"],
            "Si-1-DAY": ["Si_1-Day", "Si-1-Day", "Si1-Day"],
            "Si-FRP": ["Si_FRP", "Si-FRP"],
        }
        # Provide buttons (radio) for selecting the major product category
        selected_category = st.radio(
            "대분류 선택",
            list(category_prefix_map.keys()),
            horizontal=True,
            index=0,
        )
        # Provide buttons (radio) for selecting the time granularity
        selected_period = st.radio(
            "기간 선택",
            ["일별", "주간", "월별"],
            horizontal=True,
            index=0,
        )
        # Choose the appropriate pivot table and axis labels
        if selected_period == "일별":
            pivot_source = product_daily_pivot
            index_label = "날짜"
            title_suffix = "일별"
        elif selected_period == "주간":
            pivot_source = product_weekly_pivot
            index_label = "연-주"
            title_suffix = "주간"
        else:
            pivot_source = product_monthly_pivot
            index_label = "연-월"
            title_suffix = "월별"
        # Determine which columns correspond to the selected major category
        prefixes = category_prefix_map[selected_category]
        category_columns = [c for c in pivot_source.columns if any(c.startswith(p) for p in prefixes)]
        if not category_columns:
            st.info("선택한 분류에 해당하는 데이터가 없습니다.")
        else:
            pivot_filtered = pivot_source[category_columns]
            # Display the filtered product yield table
            idx_name = pivot_filtered.index.name or pivot_source.index.name or "index"
            df_view = pivot_filtered.reset_index().rename(columns={idx_name: index_label})
            df_view_fmt = format_numbers(df_view)
            st.dataframe(df_view_fmt, use_container_width=True)
            # Draw the line chart for the selected category and period
            draw_product_chart(
                pivot_filtered,
                f"{selected_plant} - {selected_category} {title_suffix} 수율",
                index_label,
            )

        # ------------------------------------------------------------------
        # Toggle to show or hide the full filtered dataset at the bottom of the page
        # ------------------------------------------------------------------
        # Use two separate buttons depending on whether the dataset is currently
        # visible.  Each button has its own key to avoid interference.  When
        # clicked, the state is updated and the UI is rerendered with the
        # appropriate button label.
        if 'show_data' not in st.session_state:
            st.session_state.show_data = False
        if not st.session_state.show_data:
            # Show button to reveal the dataset
            if st.button('전체 데이터 보기', key='show_data_btn'):
                st.session_state.show_data = True
        else:
            # Button to hide the dataset
            if st.button('전체 데이터 숨기기', key='hide_data_btn'):
                st.session_state.show_data = False
            # Display the dataset while visible
            data_to_show = filtered_df.copy()
            data_to_show["수율(%)"] = (data_to_show["yield"] * 100).round(2)
            # Format numeric columns with thousand separators
            data_to_show_fmt = format_numbers(data_to_show)
            st.subheader(f"{selected_plant} 전체 데이터")
            st.dataframe(data_to_show_fmt, use_container_width=True)
    else:
        # Non-Streamlit fallback: compute simple summaries and print to stdout
        df = _load_and_prepare(tuple(file_paths))
        # Summarise yields overall (no time grouping)
        total_summary = summarise_by_group(df, group_cols=())
        print("\n=== 전체 수율 요약 (공장별 & 공정별) ===")
        print(total_summary.reset_index())
        print("\nRun this script with 'streamlit run yield_simulation.py' for an interactive dashboard.")


if __name__ == "__main__":
    main()
