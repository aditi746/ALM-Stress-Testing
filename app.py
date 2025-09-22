import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Interactive Data Report (Auto-Generated Data)", layout="wide")
st.title("📊 Interactive Data Report")
st.caption(
    "This dashboard auto-generates realistic sample data and provides filter-aware visualizations. "
    "You’ll see a time-series line chart, a category bar chart, and a proportional pie chart — each with clear, auto-generated insights."
)

# =========================================================
# Sidebar — Data Generator & Global Filters
# =========================================================
st.sidebar.header("🧪 Data Generator")

seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999_999, value=1234, step=1)
np.random.seed(seed)

freq = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=2)
periods = st.sidebar.slider("Number of Periods", min_value=12, max_value=730,
                            value=36 if freq == "Monthly" else (104 if freq == "Weekly" else 365), step=1)

start_dt = st.sidebar.date_input("Start Date", date(2023, 1, 1))
n_cats = st.sidebar.slider("Number of Categories", 3, 10, 5, 1)

# Generate simple category names
cat_names = [f"Category {chr(ord('A')+i)}" for i in range(n_cats)]

# =========================================================
# Synthetic data generation
# =========================================================
def make_date_index(start: date, periods: int, freq: str) -> pd.DatetimeIndex:
    if freq == "Daily":
        rule = "D"
    elif freq == "Weekly":
        rule = "W"
    else:
        rule = "MS"  # month start
    return pd.date_range(pd.to_datetime(start), periods=periods, freq=rule)

def generate_synthetic_dataframe(start: date, periods: int, freq: str, categories: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = make_date_index(start, periods, freq)

    # Base global trend & seasonality for "Sales"
    t = np.arange(len(idx))
    trend = 1 + 0.002 * t  # gentle upward trend
    if freq == "Monthly":
        # seasonality by months
        season = 1 + 0.15*np.sin(2*np.pi*(t/12.0)) + 0.05*np.cos(2*np.pi*(t/6.0))
    elif freq == "Weekly":
        season = 1 + 0.10*np.sin(2*np.pi*(t/52.0))
    else:
        season = 1 + 0.05*np.sin(2*np.pi*(t/365.0))

    total_baseline = 1000.0  # baseline level for Sales

    # Category shares via Dirichlet — vary over time slightly
    # Start with static base shares…
    base_alpha = np.ones(len(categories))
    base_shares = rng.dirichlet(base_alpha)
    base_shares = base_shares / base_shares.sum()

    rows = []
    for i, ts in enumerate(idx):
        # small jitter to shares over time
        jitter = rng.normal(0, 0.02, size=len(categories))
        shares = np.clip(base_shares + jitter, 0.01, None)
        shares = shares / shares.sum()

        # global shocks (random events)
        shock = rng.normal(loc=1.0, scale=0.06)  # ~6% std
        total_sales = total_baseline * trend[i] * season[i] * shock

        for c_idx, cat in enumerate(categories):
            # per-category multiplier and noise
            cat_mult = 0.8 + 0.4*rng.random()  # 0.8–1.2
            sales = total_sales * shares[c_idx] * cat_mult
            # Costs ~ 60–80% of sales depending on category efficiency
            gross_margin_pct = 0.2 + 0.25*rng.random()  # 20–45%
            cost = sales * (1 - gross_margin_pct)
            # Units scale with sales but with randomness
            price_per_unit = 10 + 20*rng.random()
            units = max(sales / price_per_unit + rng.normal(0, 5), 0)

            rows.append({
                "Date": ts,
                "Category": cat,
                "Sales": sales,
                "Cost": cost,
                "Units": units
            })

    df = pd.DataFrame(rows)
    df["Profit"] = df["Sales"] - df["Cost"]
    # Guard against divide-by-zero; Margin% useful for extra insights
    df["Margin_%"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, 0.0)
    return df

df_raw = generate_synthetic_dataframe(start_dt, periods, freq, cat_names, seed=seed)

# =========================================================
# Global Filters (apply to generated data)
# =========================================================
st.sidebar.header("🔎 Filters")

# Date range filter
min_d, max_d = df_raw["Date"].min(), df_raw["Date"].max()
date_range = st.sidebar.date_input(
    "Date Range",
    [min_d.date(), max_d.date()],
    min_value=min_d.date(), max_value=max_d.date()
)
start_sel, end_sel = date_range if isinstance(date_range, (list, tuple)) else (date_range, date_range)

df = df_raw[(df_raw["Date"].dt.date >= start_sel) & (df_raw["Date"].dt.date <= end_sel)].copy()

# Category filter
all_categories = df["Category"].unique().tolist()
sel_categories = st.sidebar.multiselect("Categories", options=all_categories, default=all_categories)
if sel_categories:
    df = df[df["Category"].isin(sel_categories)]

# Metric selection
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# Prioritize business metrics order
metric_choices = [m for m in ["Sales", "Profit", "Cost", "Units", "Margin_%"] if m in numeric_cols] or numeric_cols
metric = st.sidebar.selectbox("Metric", metric_choices)

# Time aggregation for the line chart
time_agg = st.sidebar.selectbox("Time Aggregation", ["Auto", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)

# =========================================================
# Helper functions for insights
# =========================================================
def trend_text(series: pd.Series, label: str) -> str:
    s = series.dropna()
    if len(s) < 2:
        return f"Not enough data to evaluate the {label} trend."
    first, last = s.iloc[0], s.iloc[-1]
    change = last - first
    pct = (change / first * 100) if first != 0 else np.nan
    direction = "upward 📈" if change > 0 else ("downward 📉" if change < 0 else "flat ➖")
    base = f"{label} shows an **{direction}** movement from {first:,.2f} to {last:,.2f}"
    return base + (f" (**{pct:+.1f}%**)." if np.isfinite(pct) else ".")

def share_table(df: pd.DataFrame, by: str, val: str) -> pd.DataFrame:
    grp = df.groupby(by, dropna=False)[val].sum().reset_index().rename(columns={val: "Value"})
    total = grp["Value"].sum()
    grp["Share_%"] = np.where(total != 0, grp["Value"] / total * 100, 0.0)
    grp = grp.sort_values("Value", ascending=False)
    return grp

def resample_frame(frame: pd.DataFrame, rule: str, metric: str, has_cat=True) -> pd.DataFrame:
    """Aggregate time series by chosen frequency."""
    if rule == "Auto":
        span = (frame["Date"].max() - frame["Date"].min()).days
        rule = "M" if span > 200 else "W"
    map_rule = {"Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q","Yearly":"Y"}
    code = map_rule.get(rule, "W")
    if has_cat:
        out = (frame.set_index("Date")
                     .groupby("Category")[metric]
                     .resample(code).sum()
                     .reset_index())
    else:
        out = (frame.set_index("Date")[metric]
                     .resample(code).sum()
                     .reset_index())
    return out

# =========================================================
# SECTION 1 — Time Trend (Line)
# =========================================================
st.markdown("## 1) Trend Over Time")

if df.empty:
    st.warning("No data after filters. Adjust your date range or categories.")
else:
    has_cat = "Category" in df.columns
    df_ts = df[["Date", metric] + (["Category"] if has_cat else [])].dropna(subset=["Date"])
    df_ts = df_ts.sort_values("Date")

    # Resample
    df_res = resample_frame(df_ts, time_agg, metric, has_cat=has_cat)

    # Draw chart
    if has_cat:
        line_fig = px.line(df_res, x="Date", y=metric, color="Category", markers=True,
                           title=f"{metric} over Time ({time_agg})")
    else:
        line_fig = px.line(df_res, x="Date", y=metric, markers=True,
                           title=f"{metric} over Time ({time_agg})")

    st.plotly_chart(line_fig, use_container_width=True)

    # Interpretation (total trend across categories)
    try:
        total_trend = df_res if not has_cat else df_res.groupby("Date")[metric].sum().reset_index()
        st.caption("**Insight:** " + trend_text(total_trend[metric], metric))
    except Exception:
        st.caption("**Insight:** Trend interpretation unavailable for this selection.")

st.divider()

# =========================================================
# SECTION 2 — Category Comparison (Bar)
# =========================================================
st.markdown("## 2) Category Comparison")

if df.empty:
    st.info("No data to display. Adjust filters.")
else:
    comp = share_table(df, by="Category", val=metric)

    left, right = st.columns([0.6, 0.4])
    with right:
        top_n = st.slider("Top-N categories", 3, min(10, len(comp)), min(5, len(comp)))
    comp_top = comp.head(top_n)

    bar_fig = px.bar(comp_top, x="Category", y="Value", color="Category",
                     text_auto=True, title=f"{metric} by Category (Top {top_n})")
    st.plotly_chart(bar_fig, use_container_width=True)

    # Interpretation: leader + relative gap
    try:
        leader = comp.iloc[0]
        leader_cat = str(leader["Category"])
        leader_share = leader["Share_%"]
        runner_up_share = comp.iloc[1]["Share_%"] if len(comp) > 1 else 0.0
        diff = leader_share - runner_up_share
        st.caption(
            f"**Insight:** '{leader_cat}' leads with **{leader_share:.1f}%** share of {metric}. "
            f"Lead over next category: **{diff:.1f} p.p.**"
        )
    except Exception:
        st.caption("**Insight:** Could not determine a clear category leader.")

st.divider()

# =========================================================
# SECTION 3 — Proportional Breakdown (Pie)
# =========================================================
st.markdown("## 3) Proportional Breakdown")

if df.empty:
    st.info("No data to display. Adjust filters.")
else:
    comp_full = share_table(df, by="Category", val=metric)

    with st.expander("Pie options"):
        min_share = st.slider("Group slices below (%) into 'Other'", 0.0, 10.0, 0.0, 0.5)

    if min_share > 0:
        major = comp_full[comp_full["Share_%"] >= min_share].copy()
        minor = comp_full[comp_full["Share_%"] < min_share].copy()
        if not minor.empty:
            other = pd.DataFrame({"Category": [f"Other (<{min_share:.1f}%)"],
                                  "Value": [minor["Value"].sum()],
                                  "Share_%": [minor["Share_%"].sum()]})
            comp_plot = pd.concat([major, other], ignore_index=True)
        else:
            comp_plot = major
    else:
        comp_plot = comp_full

    pie_fig = px.pie(comp_plot, names="Category", values="Value", hole=0.35,
                     title=f"{metric} — Proportional Breakdown")
    st.plotly_chart(pie_fig, use_container_width=True)

    # Interpretation
    try:
        lead = comp_full.iloc[0]["Category"]
        lead_pct = comp_full.iloc[0]["Share_%"]
        tail_pct = comp_full.iloc[1:]["Share_%"].sum()
        st.caption(
            f"**Insight:** '{lead}' holds the largest share at **{lead_pct:.1f}%** of total {metric}. "
            f"All other categories combined account for **{tail_pct:.1f}%**."
        )
    except Exception:
        st.caption("**Insight:** Proportion interpretation unavailable.")

st.divider()

# =========================================================
# Data Summary & Download
# =========================================================
st.markdown("## 🔎 Data Summary & Download")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Rows (filtered)", value=len(df))
with colB:
    st.metric("Categories (filtered)", value=df["Category"].nunique())
with colC:
    st.metric("Start → End",
              value=f"{df['Date'].min().date() if not df.empty else '-'} → {df['Date'].max().date() if not df.empty else '-'}")
with colD:
    st.metric("Metric", value=metric)

with st.expander("Preview generated & filtered dataset"):
    st.dataframe(df.head(50), use_container_width=True)

st.download_button(
    "⬇️ Download current filtered data as CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"generated_filtered_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.info(
    "Tips: Use **Date Range**, **Categories**, and **Metric** to control all visuals. "
    "Adjust **Frequency / Periods / Seed** in the sidebar to regenerate a fresh synthetic dataset."
)
