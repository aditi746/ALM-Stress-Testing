import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# Page
# -----------------------------------------------------------
st.set_page_config(page_title="ALM Stress Testing Dashboard (IBR)", layout="wide")
st.title("💊 ALM Stress Testing Dashboard")
st.caption("R&D Intensity • Regulatory Shocks • IP Expiry → Financial Risk in Pharma")

# -----------------------------------------------------------
# Synthetic Data (same schema as your version)
# -----------------------------------------------------------
default_df = pd.DataFrame({
    "Year": list(range(2015, 2025)),
    "Revenue": [10.5,12.1,15.1,18.2,18.7,20.9,24.9,26.3,27.3,30.5],
    "COGS":    [4.2,5.2,5.6,7.3,7.6,7.4,10.2,9.7,9.7,13.6],
    "Opex_exR&D": [2.6,2.9,3.2,3.7,4.4,4.6,5.1,5.9,5.5,7.5],
    "R&D":     [2.0,2.3,3.0,3.9,3.9,4.2,5.6,5.0,5.5,6.3],
    "Capex":   [0.5,0.7,0.7,1.0,1.1,0.9,1.7,1.7,1.9,2.0],
    "Cash_Begin": [3.4,4.7,1.4,1.8,1.2,2.3,2.6,2.1,4.3,2.4],
    "Debt_Begin": [10.0,11.8,9.0,13.6,8.5,14.9,13.4,9.4,8.0,13.7],
    "InterestRate": [0.044,0.045,0.045,0.032,0.037,0.032,0.047,0.043,0.037,0.031],
    "PrincipalDue_12m": [0.8,1.0,1.1,1.6,1.2,1.5,0.8,1.1,1.0,1.5],
    "DSO": [73,66,68,74,74,69,72,78,66,76],
    "DPO": [69,53,54,56,62,64,60,53,62,56],
    "DIO": [98,81,89,92,85,91,91,99,90,86],
    "Undrawn_Revolver": [2.8,1.5,2.3,1.0,1.7,1.6,1.3,2.1,2.0,2.4],
    "Depreciation": [0.4,0.5,0.6,0.7,0.8,0.8,1.0,1.0,1.1,1.2]
})

st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional, same columns)", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else default_df.copy()
df = df.sort_values("Year").reset_index(drop=True)

# -----------------------------------------------------------
# Controls
# -----------------------------------------------------------
st.sidebar.header("🎛️ Core Controls")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
rd_cash_now_pct = st.sidebar.slider("R&D Cash Now (%)", 0, 100, 80, 5)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)
comp_cost_pct = st.sidebar.slider("Compliance Cost (% of Revenue)", 0, 10, 0, 1)
preset = st.sidebar.selectbox("Scenario Preset", ["Base", "R&D Surge", "Reg Shock", "Patent Cliff", "Black Swan"])
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)

with st.sidebar.expander("Advanced Working Capital"):
    dso_delta = st.slider("Δ DSO (days)", -15, 30, 0, 1)
    dpo_delta = st.slider("Δ DPO (days)", -15, 30, 0, 1)
    dio_delta = st.slider("Δ DIO (days)", -30, 30, 0, 1)

# -----------------------------------------------------------
# Scenario Engine (latest-year snapshot)
# -----------------------------------------------------------
def run_scenario(df,
                 rd_bps=0, rd_cash_now_pct=80,
                 price_cap=0, comp_cost_pct=0,
                 preset="Base",
                 new_debt=0, new_equity=0,
                 tax_rate=18,
                 dso_delta=0, dpo_delta=0, dio_delta=0):
    scn = df.copy()

    # Revenue impact from regulation
    scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)

    # Presets
    if preset == "Patent Cliff":
        scn["Revenue_scn"] *= 0.8
        scn["COGS"] *= 1.10
        scn["R&D"] *= 1.15
    elif preset == "R&D Surge":
        rd_bps = max(rd_bps, 800)  # ensure surge
    elif preset == "Reg Shock":
        price_cap = max(price_cap, 15)
        comp_cost_pct = max(comp_cost_pct, 3)
    elif preset == "Black Swan":
        rd_bps = max(rd_bps, 600)
        rd_cash_now_pct = max(rd_cash_now_pct, 95)
        price_cap = max(price_cap, 10)
        comp_cost_pct = max(comp_cost_pct, 4)
        scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)

    # Expense effects
    scn["R&D_scn"] = scn["R&D"] + scn["Revenue"] * (rd_bps/10000.0)
    scn["Opex_exR&D"] = scn["Opex_exR&D"] + scn["Revenue"] * (comp_cost_pct/100.0)

    # Working capital deltas (timing)
    dso = scn["DSO"] + dso_delta
    dpo = scn["DPO"] + dpo_delta
    dio = scn["DIO"] + dio_delta
    ar = scn["Revenue_scn"] * (dso/365.0)
    inv = scn["COGS"] * (dio/365.0)
    ap = scn["COGS"] * (dpo/365.0)
    wc = ar + inv - ap
    scn["WC_Delta"] = wc - wc.shift(1).fillna(wc.iloc[0])

    # Profit & cash flow
    scn["EBIT"] = scn["Revenue_scn"] - scn["COGS"] - scn["Opex_exR&D"] - scn["R&D_scn"]
    scn["OCF"] = scn["EBIT"] * (1 - tax_rate/100.0) + scn["Depreciation"] - scn["WC_Delta"]

    # Funding & gap
    scn["RD_CashNow"] = scn["R&D_scn"] * (rd_cash_now_pct/100.0)
    scn["NewFunding"] = new_debt + new_equity
    scn["Interest"] = scn["Debt_Begin"] * scn["InterestRate"]
    scn["DebtService"] = scn["PrincipalDue_12m"] + scn["Interest"]

    # Gap = Uses – Sources
    uses = scn["RD_CashNow"] + scn["WC_Delta"] + scn["Capex"] + scn["DebtService"]
    sources = scn["OCF"] + scn["NewFunding"]
    scn["FundingGap"] = uses - sources

    scn["Cash_End"] = scn["Cash_Begin"] - scn["FundingGap"]

    # Ratios (guard divide-by-zero)
    scn["RD_Intensity"] = (scn["R&D_scn"] / scn["Revenue_scn"]).replace([np.inf, -np.inf], 0).fillna(0)
    scn["ALM_Stress"] = scn["FundingGap"].abs() * (1 + scn["RD_Intensity"])
    scn["ICR"] = (scn["EBIT"].clip(lower=1e-6) / scn["Interest"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scn["DSCR"] = (scn["OCF"] / scn["DebtService"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scn["Liquidity12m"] = (scn["Cash_Begin"] + scn["Undrawn_Revolver"]) / (scn["Capex"] + scn["RD_CashNow"] + scn["DebtService"]).replace(0, np.nan)
    scn["Runway_months"] = np.where(scn["FundingGap"] > 0, scn["Cash_Begin"] / (scn["FundingGap"]/12.0), np.inf)

    # Return latest year snapshot only
    return scn.iloc[-1]

# Utility to format B with 2 decimals
fmtB = lambda x: f"{x:.2f}"

# -----------------------------------------------------------
# KPIs (selected scenario)
# -----------------------------------------------------------
sel = run_scenario(
    df, rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
    price_cap=price_cap, comp_cost_pct=comp_cost_pct,
    preset=preset, new_debt=new_debt, new_equity=new_equity,
    tax_rate=tax_rate, dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta
)

st.subheader(f"📊 Key Metrics — {preset} Scenario (latest year)")
k = st.columns(6)
k[0].metric("Funding Gap ($B)", fmtB(sel["FundingGap"]))
k[1].metric("EBIT ($B)", fmtB(sel["EBIT"]))
k[2].metric("OCF ($B)", fmtB(sel["OCF"]))
k[3].metric("DSCR (×)", fmtB(sel["DSCR"]))
k[4].metric("ICR (×)", fmtB(sel["ICR"]))
k[5].metric("Liquidity 12m (×)", fmtB(sel["Liquidity12m"] if np.isfinite(sel["Liquidity12m"]) else 0))

st.markdown("---")

# ----------------------------
# Sources vs Uses by Scenario (composition view) — FIXED
# ----------------------------
def scenario_breakdown_with_current_sliders(preset_name: str):
    """Run scenario using current sliders so comparison stays consistent with KPI cards."""
    rr = run_scenario(
        df,
        rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
        price_cap=price_cap, comp_cost_pct=comp_cost_pct,
        preset=preset_name, new_debt=new_debt, new_equity=new_equity,
        tax_rate=tax_rate, dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta
    )
    # Build uses/sources consistent with the engine
    uses = {
        "R&D cash now": rr["R&D_scn"] * (rd_cash_now_pct/100.0),
        "Δ Working capital": rr["WC_Delta"],
        "Capex": rr["Capex"],
        "Debt service": rr["DebtService"],
    }
    sources = {
        "Operating CF": rr["OCF"],
        "New funding": rr["NewFunding"],  # <-- standardized name with space
    }
    return rr, uses, sources

rows = []
for name in ["Base", "R&D Surge", "Reg Shock", "Patent Cliff"]:
    rr, uses, sources = scenario_breakdown_with_current_sliders(name)
    row = {
        "Scenario": name,
        "FundingGap": float(rr["FundingGap"]),
        # Sources (keep human-friendly names)
        "Operating CF": float(sources["Operating CF"]),
        "New funding": float(sources["New funding"]),
        # Uses
        "R&D cash now": float(uses["R&D cash now"]),
        "Δ Working capital": float(uses["Δ Working capital"]),
        "Capex": float(uses["Capex"]),
        "Debt service": float(uses["Debt service"]),
    }
    rows.append(row)

ss = pd.DataFrame(rows)

# Melt to long form using the exact column names we just created
value_cols = ["Operating CF", "New funding", "R&D cash now", "Δ Working capital", "Capex", "Debt service"]
missing = [c for c in value_cols if c not in ss.columns]
if missing:
    st.error(f"Internal error: missing columns for melt: {missing}")
else:
    ss_long = ss.melt(
        id_vars=["Scenario", "FundingGap"],
        value_vars=value_cols,
        var_name="Component", value_name="Value"
    )
    # Sources are negative for relative stacking; Uses are positive
    ss_long["Sign"] = np.where(ss_long["Component"].isin(["Operating CF", "New funding"]), "Source (−)", "Use (+)")
    ss_long["PlotValue"] = np.where(ss_long["Sign"] == "Source (−)", -ss_long["Value"], ss_long["Value"])

    stack_fig = px.bar(
        ss_long, x="Scenario", y="PlotValue", color="Component",
        barmode="relative", title="Uses (+) vs Sources (−) by Scenario"
    )
    stack_fig.update_layout(yaxis_title="$B (relative stacking)", xaxis_title="")
    st.plotly_chart(stack_fig, use_container_width=True)
    st.caption(
        "Positive bars are uses (R&D, working capital, capex, debt service); "
        "negative bars are sources (OCF, new funding). The net height equals the Funding Gap."
    )

# -----------------------------------------------------------
# SENSITIVITY — Funding Gap + EBIT + OCF vs R&D Intensity
# -----------------------------------------------------------
grid = list(range(0, 1201, 150))
sens_rows = []
for x in grid:
    r = run_scenario(df, rd_bps=x, preset="Base", rd_cash_now_pct=rd_cash_now_pct,
                     price_cap=price_cap, comp_cost_pct=comp_cost_pct,
                     new_debt=new_debt, new_equity=new_equity,
                     tax_rate=tax_rate, dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta)
    sens_rows.append({"R&D_bps": x, "FundingGap": r["FundingGap"], "EBIT": r["EBIT"], "OCF": r["OCF"]})
sens = pd.DataFrame(sens_rows)

st.subheader("📈 Objective 1: Sensitivity to R&D Intensity")
sens_fig = go.Figure()
sens_fig.add_trace(go.Scatter(x=sens["R&D_bps"], y=sens["FundingGap"], mode="lines+markers", name="Funding Gap ($B)"))
sens_fig.add_trace(go.Scatter(x=sens["R&D_bps"], y=sens["EBIT"], mode="lines+markers", name="EBIT ($B)"))
sens_fig.add_trace(go.Scatter(x=sens["R&D_bps"], y=sens["OCF"], mode="lines+markers", name="OCF ($B)"))
sens_fig.update_layout(xaxis_title="R&D change (bps of Revenue)", yaxis_title="$B", title="Funding Gap, EBIT & OCF vs R&D Intensity")
st.plotly_chart(sens_fig, use_container_width=True)
st.caption("Mechanism: as R&D intensity rises, EBIT and OCF fall (short-term), which widens the Funding Gap — unless balanced by stronger sources (revenue, efficiency) or new funding.")

st.markdown("---")

# ----------------------------
# Custom Waterfall (replicating WHO style with per-bar colors)
# ----------------------------
st.subheader("💧 Funding Gap Waterfall — Need vs Secured vs Expected")

wf_scn = st.selectbox(
    "Select scenario for waterfall",
    ["Base", "R&D Surge", "Reg Shock", "Patent Cliff"],
    index=3
)

wf_row = run_scenario(
    df,
    rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
    price_cap=price_cap, comp_cost_pct=comp_cost_pct,
    preset=wf_scn, new_debt=new_debt, new_equity=new_equity,
    tax_rate=tax_rate, dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta
)

need = float(
    wf_row["R&D_scn"] * (rd_cash_now_pct/100.0)
  + wf_row["WC_Delta"]
  + wf_row["Capex"]
  + wf_row["DebtService"]
)

secured_oper_cf   = max(float(wf_row["OCF"]), 0.0)
secured_committed = max(float(wf_row["NewFunding"]), 0.0)
expected_pipeline = max(float(wf_row["Undrawn_Revolver"]), 0.0)

gap_val = max(need - (secured_oper_cf + secured_committed + expected_pipeline), 0.0)
gap_pct = (gap_val/need*100.0) if need > 0 else 0.0

# Bar definitions
steps = [
    {"label": "Total cash need", "value": need, "color": "#0b2b53"},
    {"label": "Operating CF (secured)", "value": -secured_oper_cf, "color": "#1aa1ff"},
    {"label": "Committed funding (secured)", "value": -secured_committed, "color": "#1aa1ff"},
    {"label": "Pipeline / Undrawn (expected)", "value": -expected_pipeline, "color": "#9aa0a6"},
    {"label": f"Funding gap ({gap_pct:.0f}%)", "value": gap_val, "color": "#d61f45"}
]

# Compute cumulative positions
x_labels, bar_values, bar_base, bar_colors = [], [], [], []
running = 0
for step in steps:
    x_labels.append(step["label"])
    bar_values.append(step["value"])
    bar_colors.append(step["color"])
    if step["label"].startswith("Total"):
        bar_base.append(0)
        running = step["value"]
    elif step["label"].startswith("Funding"):
        bar_base.append(0)  # final total shown as standalone
    else:
        bar_base.append(running)
        running += step["value"]

# Plot with go.Bar (mimic waterfall)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=x_labels,
    y=bar_values,
    base=bar_base,
    marker_color=bar_colors,
    text=[f"{abs(v):,.0f}" for v in bar_values],
    textposition="outside"
))

fig.update_layout(
    title=f"{wf_scn}: Need vs Secured vs Expected → Funding Gap",
    yaxis_title="$B",
    showlegend=False,
    height=420,
    margin=dict(l=20,r=20,t=50,b=20)
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Total cash need = R&D cash now + ΔWC + Capex + Debt service. "
    f"Secured (Operating CF, committed funding) and expected (pipeline) reduce the requirement. "
    f"Remaining shortfall is the **Funding Gap = {gap_val:.2f}B ({gap_pct:.0f}% of need)**."
)
