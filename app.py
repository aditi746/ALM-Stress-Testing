import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Interactive Data Report", layout="wide")
st.title("📊 Interactive Data Report")
st.caption(
    "Upload a CSV and explore dynamic, filter-aware visualizations. "
    "This report includes a time-series line chart, a category bar chart, "
    "and a proportional pie chart, each with automatic insights."
)

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    """Read CSV and attempt to parse dates loosely."""
    df = pd.read_csv(file)
    # Try to detect/parse a Date column (case-insensitive match)
    date_cols = [c for c in df.columns if c.strip().lower() in {"date", "day", "dt", "timestamp"}]
    if date_cols:
        for dc in date_cols:
            with pd.option_context("mode.chained_assignment", None):
                df[dc] = pd.to_datetime(df[dc], errors="coerce", infer_datetime_format=True)
        # pick the first recognized date column as the canonical Date
        df.rename(columns={date_cols[0]: "Date"}, inplace=True)
    # Normalize a likely Category column if present
    cat_cols_guess = [c for c in df.columns if c.strip().lower() in {"category", "segment", "class", "group", "product"}]
    if cat_cols_guess:
        df.rename(columns={cat_cols_guess[0]: "Category"}, inplace=True)
    return df


def _safe_num_cols(df: pd.DataFrame):
    """Return numeric columns that are meaningful metrics (exclude all-null or constant)."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # filter out columns that are all NaN or constant
    filtered = []
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        if s.nunique() <= 1:
            continue
        filtered.append(c)
    return filtered


def _trend_text(series: pd.Series, label: str) -> str:
    """Return a short interpretation of a trend."""
    s = series.dropna()
    if len(s) < 2:
        return f"Not enough data to evaluate the {label} trend."
    first, last = s.iloc[0], s.iloc[-1]
    change = last - first
    pct = (change / first * 100) if first != 0 else np.nan
    direction = "upward 📈" if change > 0 else ("downward 📉" if change < 0 else "flat ➖")
    msg = f"{label} shows an **{direction}** movement from {first:,.2f} to {last:,.2f}"
    if np.isfinite(pct):
        msg += f" (**{pct:+.1f}%**)."
    else:
        msg += "."
    return msg


def _share_table(df: pd.DataFrame, by: str, metric: str) -> pd.DataFrame:
    """Return a table with totals and shares by 'by'."""
    grp = df.groupby(by, dropna=False)[metric].sum().reset_index()
    grp.rename(columns={metric: "Value"}, inplace=True)
    total = grp["Value"].sum()
    grp["Share_%"] = np.where(total != 0, grp["Value"] / total * 100, 0.0)
    grp = grp.sort_values("Value", ascending=False)
    return grp


def _ensure_columns(df: pd.DataFrame):
    """Check that we have at least one numeric metric, and preferably Date & Category."""
    errs = []
    num_cols = _safe_num_cols(df)
    if not num_cols:
        errs.append("No usable numeric columns found.")
    has_date = "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"])
    has_cat = "Category" in df.columns
    return errs, num_cols, has_date, has_cat


# ---------------------------
# Sidebar: Upload & Filters
# ---------------------------
st.sidebar.header("📥 Upload & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if not uploaded:
    st.info("👈 Please upload a CSV to begin. Expected columns (flexible names): a **Date** field, a **Category** field, and at least one **numeric metric**.")
    st.stop()

df_raw = _read_csv(uploaded)
errors, numeric_cols, has_date, has_cat = _ensure_columns(df_raw)

if errors:
    st.error(" • ".join(errors))
    st.write("Columns detected:", list(df_raw.columns))
    st.stop()

df = df_raw.copy()

# Date filter
if has_date:
    min_d, max_d = df["Date"].min(), df["Date"].max()
    # guard for invalid datetimes
    if pd.isna(min_d) or pd.isna(max_d):
        has_date = False
    else:
        start_d, end_d = st.sidebar.date_input(
            "Date range", [min_d.date(), max_d.date()],
            min_value=min_d.date(), max_value=max_d.date()
        )
        # Ensure tuple unpack is robust
        if isinstance(start_d, (list, tuple)):
            start, end = start_d
        else:
            start, end = start_d, end_d
        # Filter by date
        mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
        df = df.loc[mask].copy()

# Category filter
if has_cat:
    all_cats = df["Category"].dropna().astype(str).unique().tolist()
    selected_cats = st.sidebar.multiselect(
        "Categories", options=sorted(all_cats), default=sorted(all_cats)
    )
    if selected_cats:
        df = df[df["Category"].astype(str).isin(selected_cats)]

# Metric selection
metric = st.sidebar.selectbox("Metric", numeric_cols)

# Optional time granularity
if has_date:
    freq = st.sidebar.selectbox("Time aggregation", ["Auto", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)
else:
    freq = "None"

# ---------------------------
# Section 1: Line graph (Time Trend)
# ---------------------------
st.markdown("## 1) Trend Over Time")
if has_date:
    # Aggregate by frequency
    df_ts = df.copy()
    df_ts = df_ts[["Date", metric] + (["Category"] if has_cat else [])].dropna(subset=["Date"])
    df_ts = df_ts.sort_values("Date")

    def resample_frame(frame: pd.DataFrame, rule: str):
        if rule == "Auto":
            # heuristic: if period longer than ~200 days, use M; else W
            span = (frame["Date"].max() - frame["Date"].min()).days
            rule = "M" if span > 200 else "W"
        rule_map = {
            "Daily": "D", "Weekly": "W", "Monthly": "M",
            "Quarterly": "Q", "Yearly": "Y", "Auto": rule
        }
        code = rule_map.get(rule, "W")
        if has_cat:
            out = (frame
                   .set_index("Date")
                   .groupby("Category")[metric]
                   .resample(code).sum()
                   .reset_index())
        else:
            out = (frame
                   .set_index("Date")[metric]
                   .resample(code).sum()
                   .reset_index())
        return out

    df_res = resample_frame(df_ts, freq)

    if has_cat:
        line_fig = px.line(
            df_res, x="Date", y=metric, color="Category",
            markers=True, title="Time Series by Category"
        )
    else:
        line_fig = px.line(
            df_res, x="Date", y=metric,
            markers=True, title="Time Series"
        )

    st.plotly_chart(line_fig, use_container_width=True)

    # Interpretation
    try:
        # For insight, aggregate across categories for total trend
        df_total = df_res if not has_cat else df_res.groupby("Date")[metric].sum().reset_index()
        st.caption("**Insight:** " + _trend_text(df_total[metric], metric))
    except Exception:
        st.caption("**Insight:** Trend interpretation unavailable for this selection.")
else:
    st.info("No valid Date column detected. Skipping the time trend chart.")

st.divider()

# ---------------------------
# Section 2: Bar chart (Category Comparison)
# ---------------------------
st.markdown("## 2) Category Comparison")
if has_cat:
    comp = _share_table(df, by="Category", metric=metric)

    # Top N toggle
    colL, colR = st.columns([0.6, 0.4])
    with colR:
        top_n = st.slider("Top N categories", 3, min(10, len(comp)), min(5, len(comp)))
    comp_top = comp.head(top_n)

    bar_fig = px.bar(
        comp_top, x="Category", y="Value",
        color="Category", text_auto=True,
        title=f"{metric} by Category (Top {top_n})"
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Interpretation
    try:
        leader_row = comp.iloc[0]
        leader = str(leader_row["Category"])
        share = leader_row["Share_%"]
        st.caption(f"**Insight:** '{leader}' contributes the most, accounting for **{share:.1f}%** of total {metric} in the selected filters.")
    except Exception:
        st.caption("**Insight:** Category ranking not available.")
else:
    st.info("No Category column detected. Skipping category comparison.")

st.divider()

# ---------------------------
# Section 3: Pie chart (Proportional Breakdown)
# ---------------------------
st.markdown("## 3) Proportional Breakdown")
if has_cat:
    comp_full = _share_table(df, by="Category", metric=metric)

    # Small share aggregation option to reduce clutter
    with st.expander("Pie options"):
        min_share = st.slider("Group small slices below (%)", 0.0, 10.0, 0.0, 0.5)
    if min_share > 0:
        major = comp_full[comp_full["Share_%"] >= min_share].copy()
        minor = comp_full[comp_full["Share_%"] < min_share].copy()
        if not minor.empty:
            other = pd.DataFrame({"Category": ["Other (<" + f"{min_share:.1f}%" + ")"],
                                  "Value": [minor["Value"].sum()],
                                  "Share_%": [minor["Share_%"].sum()]})
            comp_plot = pd.concat([major, other], ignore_index=True)
        else:
            comp_plot = major
    else:
        comp_plot = comp_full

    pie_fig = px.pie(
        comp_plot, names="Category", values="Value", hole=0.35,
        title=f"{metric} Proportional Breakdown"
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # Interpretation
    try:
        lead = comp_full.iloc[0]["Category"]
        lead_pct = comp_full.iloc[0]["Share_%"]
        tail_pct = comp_full.iloc[1:]["Share_%"].sum()
        st.caption(
            f"**Insight:** '{lead}' holds the largest share at **{lead_pct:.1f}%**. "
            f"The remaining categories together account for **{tail_pct:.1f}%** of total {metric}."
        )
    except Exception:
        st.caption("**Insight:** Proportion interpretation not available.")
else:
    st.info("No Category column detected. Skipping proportional pie chart.")

st.divider()

# ---------------------------
# Download & Data peek
# ---------------------------
st.markdown("## 🔎 Data Summary & Download")
cols = st.columns(3)
with cols[0]:
    st.metric("Rows (filtered)", value=len(df))
with cols[1]:
    st.metric("Categories (filtered)" if has_cat else "Numeric metrics", value=len(df["Category"].unique()) if has_cat else len(numeric_cols))
with cols[2]:
    st.metric("Date coverage" if has_date else "Time dimension", value=f"{df['Date'].min().date()} → {df['Date'].max().date()}" if has_date and not df.empty else "N/A")

with st.expander("Preview filtered dataset"):
    st.dataframe(df.head(50), use_container_width=True)

st.download_button(
    "⬇️ Download filtered data as CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.info(
    "Tips: Use the **Date range** and **Category** filters on the left to refine all visuals. "
    "Pick the **Metric** most relevant to your question (e.g., Sales, Cost, Units). "
    "Switch **Time aggregation** to Monthly/Quarterly/Yearly for smoother trends."
)
