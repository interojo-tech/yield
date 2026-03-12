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
# environments.  We choose the Noto Sans CJK KR font (bundled with many
# Linux distributions) which includes Hangul glyphs.  If this font is not
# available the default font will be used and some labels may not render
# correctly.  The unicode_minus setting ensures minus signs appear properly.
if 'plt' in globals():
    # Only apply font settings when matplotlib has been imported
    try:
        plt.rcParams['font.family'] = 'Noto Sans CJK KR'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # Silently ignore if the font is unavailable
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
    processes = processes.str.replace(r"\s+/\s+", "/", regex=True)
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
    # Fill missing process yields with 1 (neutral) for multiplication
    pivot = pivot.fillna(1.0)
    # Compute overall yield as the product of yields across all processes
    pivot["overall_yield"] = pivot.apply(
        lambda row: np.prod([row[c] for c in row.index if c != "overall_yield"]) \
            if isinstance(row, pd.Series) else np.prod(row.values),
        axis=1
    )
    # The above lambda includes 'overall_yield' when computing for each row; so exclude it.
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


def main() -> None:
    """Entry point for both console and Streamlit execution.

    When run via ``streamlit run yield_simulation.py`` this function
    presents an interactive dashboard for exploring yield data.  When
    executed in a normal Python interpreter it prints summary tables
    similar to earlier versions of the script.
    """
    # Determine the Excel file path.  Prefer the more recent workbook
    # if it exists, but fall back to the older filename for backward
    # compatibility.
    candidates = [
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
        print(
            "Error: None of the expected Excel files were found. Please "
            "place '공정기술팀 대시보드.xlsx' or '공정기술팀 대시보드_260308.xlsx' "
            "in the same directory as this script."
        )
        return

    # Define a data loading function with optional caching when running in
    # Streamlit.  Caching avoids re-reading the Excel file on every
    # interaction.
    def _load_and_prepare(path: str) -> pd.DataFrame:
        df_local = load_data(path)
        df_local = map_process_codes(df_local)
        df_local = compute_yields(df_local)
        # Drop rows with NaN yields (due to zero production quantity)
        df_local = df_local.dropna(subset=["yield"])
        return df_local

    if st:
        # Configure the page as soon as possible.  This must be the first
        # Streamlit command executed in the app.
        st.set_page_config(page_title="인터로조 공정기술팀 대시보드", layout="wide")

        # Use Streamlit caching for efficient data loading.  The
        # @st.cache_data decorator ensures that the data is loaded only
        # once per session.
        @st.cache_data  # type: ignore[misc]
        def load_cached(path: str) -> pd.DataFrame:  # pragma: no cover
            return _load_and_prepare(path)

        df = load_cached(file_path)
        # Header
        st.markdown(
            "<h1 style='text-align: left;'>인터로조 공정기술팀</h1>",
            unsafe_allow_html=True,
        )
        # Sidebar plant selection
        plants = sorted(df["공장"].unique().tolist())
        selected_plant = st.sidebar.radio(
            "공장을 선택하세요",
            options=plants,
            index=0,
        )
        # Filter data for the selected plant
        plant_df = df[df["공장"] == selected_plant].copy()
        # Append a percentage column for the raw data
        plant_df["수율(%)"] = (plant_df["yield"] * 100).round(2)
        # Display the full raw dataset for the selected plant
        st.subheader(f"{selected_plant} 전체 데이터")
        st.dataframe(plant_df, use_container_width=True)

        # Helper to summarise and convert yields to percentage for a given grouping
        def summarise_and_percent(
            data: pd.DataFrame, group_cols: Tuple[str, ...]
        ) -> pd.DataFrame:
            summary = summarise_by_group(data, group_cols=group_cols)
            # Drop the plant index level (constant for this subset)
            try:
                summary = summary.droplevel("공장")
            except Exception:
                pass
            # Convert yields to percentage
            summary = summary * 100.0
            # Round to two decimal places
            summary = summary.round(2)
            return summary

        # Prepare summarised data for each time granularity
        # Daily
        daily_summary = summarise_and_percent(
            plant_df, group_cols=("생산일자",)
        )
        daily_summary.index = daily_summary.index.strftime("%Y-%m-%d")
        # Weekly: create a year-week column on a copy to avoid modifying plant_df
        plant_week = plant_df.copy()
        plant_week["year_week"] = (
            plant_week["년도"].astype(int).astype(str) + "-" + plant_week["주차"].astype(str)
        )
        weekly_summary = summarise_and_percent(plant_week, group_cols=("year_week",))
        # Monthly
        monthly_summary = summarise_and_percent(
            plant_df, group_cols=("년도", "월")
        )
        # Combine year and month for index display
        monthly_summary.index = (
            monthly_summary.index.get_level_values("년도").astype(int).astype(str)
            + "-"
            + monthly_summary.index.get_level_values("월").astype(str)
        )

        # Utility to draw a line chart for a summary DataFrame
        def draw_line_chart(summary: pd.DataFrame, title: str, xlabel: str) -> None:
            process_cols = [c for c in summary.columns if c != "overall_yield"]
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot each process as a line
            for col in process_cols:
                ax.plot(
                    summary.index,
                    summary[col],
                    marker="o",
                    label=col,
                )
            # Also plot the overall yield
            if "overall_yield" in summary.columns:
                ax.plot(
                    summary.index,
                    summary["overall_yield"],
                    marker="o",
                    linewidth=3,
                    linestyle="--",
                    label="overall_yield",
                    color="black",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("수율 (%)")
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)

        # Tabs for daily, weekly, and monthly views
        tab_daily, tab_weekly, tab_monthly = st.tabs([
            "일별 수율", "주간 수율", "월별 수율"
        ])
        with tab_daily:
            st.subheader(f"{selected_plant} - 일별 수율")
            st.dataframe(daily_summary.reset_index().rename(columns={"index": "날짜"}), use_container_width=True)
            draw_line_chart(daily_summary, f"{selected_plant} 일별 수율 (공정별)", "날짜")
        with tab_weekly:
            st.subheader(f"{selected_plant} - 주간 수율")
            st.dataframe(weekly_summary.reset_index().rename(columns={"index": "주차"}), use_container_width=True)
            draw_line_chart(weekly_summary, f"{selected_plant} 주간 수율 (공정별)", "주차")
        with tab_monthly:
            st.subheader(f"{selected_plant} - 월별 수율")
            st.dataframe(monthly_summary.reset_index().rename(columns={"index": "연-월"}), use_container_width=True)
            draw_line_chart(monthly_summary, f"{selected_plant} 월별 수율 (공정별)", "연-월")
    else:
        # Non-Streamlit fallback: compute simple summaries and print to stdout
        df = _load_and_prepare(file_path)
        # Summarise yields overall (no time grouping)
        total_summary = summarise_by_group(df, group_cols=())
        print("\n=== 전체 수율 요약 (공장별 & 공정별) ===")
        print(total_summary.reset_index())
        print("\nRun this script with 'streamlit run yield_simulation.py' for an interactive dashboard.")


if __name__ == "__main__":
    main()