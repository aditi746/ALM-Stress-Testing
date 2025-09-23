import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="ALM Stress Testing Dashboard (IBR)", layout="wide")
st.title("💊 ALM Stress Testing Dashboard")
st.caption("R&D Intensity • Regulatory Shocks • Patent Cliff → Financial Risk in Pharma")

# ----------------------------
# Synthetic dataset (2015–2024)
# ----------------------------
df0 = pd.DataFrame({
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

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🎛️ Core Controls")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
rd_cash_now_pct = st.sidebar.slider("R&D Cash Now (%)", 0, 100, 80, 5)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)
comp_cost_pct = st.sidebar.slider("Compliance Cost (% of Revenue)", 0, 10, 0, 1)
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)

with st.sidebar.expander("Advanced (optional)"):
    st.markdown("These are optional shocks for sensitivity only.")
    dso_delta = st.slider("Δ DSO (days)", -15, 30, 0, 1)
    dpo_delta = st.slider("Δ DPO (days)", -15, 30, 0, 1)
    dio_delta = st.slider("Δ DIO (days)", -30, 30, 0, 1)
    ocf_adj = st.slider("OCF Shock (%)", -50, 50, 0, 5)  # optional; leave 0 to ignore

# ----------------------------
# Scenario Engine
# ----------------------------
def run_scenario(df,
                 rd_bps=0, rd_cash_now_pct=80,
                 price_cap=0, comp_cost_pct=0,
                 new_debt=0, new_equity=0, tax_rate=18,
                 dso_delta=0, dpo_delta=0, dio_delta=0,
                 ocf_adj=0, preset="Base"):
    scn = df.copy()
    scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)

    # Presets modify drivers on top of sliders
    if preset == "R&D Surge":
        rd_bps = max(rd_bps, 800)
    elif preset == "Reg Shock":
        price_cap = max(price_cap, 15); comp_cost_pct = max(comp_cost_pct, 3)
        scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)
    elif preset == "Patent Cliff":
        scn["Revenue_scn"] *= 0.80
        scn["COGS"] *= 1.10
        scn["R&D"] *= 1.15

    # R&D & Opex
    scn["R&D_scn"] = scn["R&D"] + scn["Revenue"] * (rd_bps/10000)
    scn["Opex_exR&D"] += scn["Revenue"] * (comp_cost_pct/100)

    # Working capital (cash view)
    dso = scn["DSO"] + dso_delta
    dpo = scn["DPO"] + dpo_delta
    dio = scn["DIO"] + dio_delta
    ar = scn["Revenue_scn"] * (dso/365)
    inv = scn["COGS"] * (dio/365)
    ap = scn["COGS"] * (dpo/365)
    wc = ar + inv - ap
    scn["WC_Delta"] = wc - wc.shift(1).fillna(wc.iloc[0])

    # EBIT & OCF (accrual to cash)
    scn["EBIT"] = scn["Revenue_scn"] - scn["COGS"] - scn["Opex_exR&D"] - scn["R&D_scn"]
    scn["Interest"] = scn["Debt_Begin"] * scn["InterestRate"]
    scn["OCF"] = scn["EBIT"] * (1 - tax_rate/100) + scn["Depreciation"] - scn["WC_Delta"]
    scn["OCF"] *= (1 + ocf_adj/100)  # optional sensitivity

    # Cash uses/sources
    scn["RD_CashNow"] = scn["R&D_scn"] * (rd_cash_now_pct/100.0)
    scn["NewFunding"] = new_debt + new_equity
    scn["FundingGap"] = scn["RD_CashNow"] + scn["WC_Delta"] + scn["Capex"] + scn["PrincipalDue_12m"] - (scn["OCF"] + scn["NewFunding"])
    scn["Cash_End"] = scn["Cash_Begin"] - scn["FundingGap"]

    # Risk metrics
    scn["RD_Intensity"] = scn["R&D_scn"] / scn["Revenue_scn"]     # fraction
    scn["DSCR"] = scn["OCF"] / (scn["PrincipalDue_12m"] + scn["Interest"] + 1e-6)
    scn["ICR"]  = scn["EBIT"].clip(lower=1e-6) / (scn["Interest"] + 1e-6)
    scn["Runway"] = np.where(scn["FundingGap"] > 0, scn["Cash_Begin"] / (scn["FundingGap"]/12), np.inf)

    return scn.iloc[-1]

# ----------------------------
# KPI Cards (based on CURRENT sliders)
# ----------------------------
curr = run_scenario(df0,
                    rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
                    price_cap=price_cap, comp_cost_pct=comp_cost_pct,
                    new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
                    dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
                    ocf_adj=ocf_adj)

st.subheader("📊 Key Metrics — Current Settings")
k = st.columns(6)
k[0].metric("Funding Gap ($B)", f"{curr['FundingGap']:.2f}")
k[1].metric("OCF ($B)", f"{curr['OCF']:.2f}")
k[2].metric("EBIT ($B)", f"{curr['EBIT']:.2f}")
k[3].metric("DSCR (x)", f"{curr['DSCR']:.2f}")
k[4].metric("R&D Intensity (%)", f"{curr['RD_Intensity']*100:.1f}")
runway_disp = 60 if np.isinf(curr["Runway"]) else curr["Runway"]
k[5].metric("Liquidity Runway (mo)", f"{runway_disp:.1f}")

st.markdown("---")

# ============================
# Objective 1 — Dynamic Bubble: R&D → Risk (current-year logic)
# ============================
grid = list(range(0, 1201, 200))  # R&D stress points
grid_runs = [
    run_scenario(
        df0,
        rd_bps=x, rd_cash_now_pct=rd_cash_now_pct,
        price_cap=price_cap, comp_cost_pct=comp_cost_pct,
        new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
        dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
        ocf_adj=ocf_adj
    ) for x in grid
]
df_bub = pd.DataFrame(grid_runs)
df_bub["RD_Intensity_pct"] = df_bub["RD_Intensity"] * 100
df_bub["Runway"] = df_bub["Runway"].replace([np.inf, -np.inf], 60).fillna(60)
df_bub["DSCR"] = df_bub["DSCR"].replace([np.inf, -np.inf], 10).fillna(10)

fig1 = px.scatter(
    df_bub, x="RD_Intensity_pct", y="FundingGap",
    size="Runway", color="DSCR", size_max=60,
    color_continuous_scale="RdYlGn", range_color=[0,3],
    labels={"RD_Intensity_pct":"R&D Intensity (%)", "FundingGap":"Funding Gap ($B)", "DSCR":"DSCR (x)", "Runway":"Runway (mo)"},
    title="Objective 1: R&D Intensity vs Funding Gap (bubble size = Runway, color = DSCR)"
)
# highlight the CURRENT slider point
fig1.add_trace(go.Scatter(
    x=[curr["RD_Intensity"]*100], y=[curr["FundingGap"]],
    mode="markers+text",
    marker=dict(symbol="star", size=18, color="white", line=dict(color="black", width=2)),
    text=["Current"], textposition="top center", showlegend=False
))
st.plotly_chart(fig1, use_container_width=True)
st.caption("As R&D intensity rises, funding gaps grow, DSCR weakens (red), and runway shrinks (smaller bubbles). The star marks your current settings.")

st.markdown("---")

# ============================
# Objective 2 — Regulatory Impact on Profitability
# ============================

st.markdown("---")
st.subheader("💰 Objective 2: Financial Impact of Regulatory Changes on Profitability")

# Run base and regulatory shock scenarios
base_case = run_scenario(df0,
                         rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
                         new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
                         dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
                         ocf_adj=ocf_adj, preset="Base")

reg_case = run_scenario(df0,
                        rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
                        new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
                        dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
                        ocf_adj=ocf_adj, preset="Reg Shock")

# ----------------------------
# Waterfall Chart
# ----------------------------
revenue_base = base_case["Revenue_scn"]
price_cap_effect = reg_case["Revenue_scn"] - base_case["Revenue_scn"]  # negative
compliance_effect = -(reg_case["Opex_exR&D"] - base_case["Opex_exR&D"])
ebit_reg = reg_case["EBIT"]

wf_labels = ["Revenue (Base)", "Price Cap Impact", "Compliance Costs", "EBIT (Reg Shock)"]
wf_values = [revenue_base, price_cap_effect, compliance_effect, ebit_reg]
wf_measures = ["absolute", "relative", "relative", "total"]

wf_fig2 = go.Figure(go.Waterfall(
    x=wf_labels, 
    y=wf_values, 
    measure=wf_measures,
    connector={"line": {"color": "gray"}},
    text=[f"{v:.2f}" for v in wf_values],
    textposition="outside"
))
wf_fig2.update_layout(
    title="Regulatory Shock: Step-by-Step Profitability Erosion",
    yaxis_title="$B",
    height=420
)

st.plotly_chart(wf_fig2, use_container_width=True)

st.caption("The waterfall shows how regulation reduces profitability: Revenue falls with price caps, compliance costs increase Opex, and final EBIT is lower.")

# ============================
# Objective 3 — IP Expiry & Competition Impact (Waterfall)
# ============================

st.markdown("---")
st.subheader("⚖️ Objective 3: Impact of IP Expiry & Competition on Financial Performance")

# Run base and patent cliff scenario
base_case = run_scenario(df0,
                         rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
                         new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
                         dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
                         ocf_adj=ocf_adj, preset="Base")

cliff_case = run_scenario(df0,
                          rd_bps=rd_bps, rd_cash_now_pct=rd_cash_now_pct,
                          new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate,
                          dso_delta=dso_delta, dpo_delta=dpo_delta, dio_delta=dio_delta,
                          ocf_adj=ocf_adj, preset="Patent Cliff")

# Calculate impacts
revenue_base = base_case["Revenue_scn"]
revenue_loss = cliff_case["Revenue_scn"] - base_case["Revenue_scn"]   # negative
margin_erosion = -( (cliff_case["COGS"] + cliff_case["Opex_exR&D"]) -
                    (base_case["COGS"] + base_case["Opex_exR&D"]) )   # extra costs/erosion
ebit_cliff = cliff_case["EBIT"]

# Build waterfall
wf_labels = ["Revenue (Base)", "Patent Expiry Impact", "Competition / Margin Erosion", "EBIT (Patent Cliff)"]
wf_values = [revenue_base, revenue_loss, margin_erosion, ebit_cliff]
wf_measures = ["absolute", "relative", "relative", "total"]

wf_fig3 = go.Figure(go.Waterfall(
    x=wf_labels,
    y=wf_values,
    measure=wf_measures,
    connector={"line": {"color": "gray"}},
    text=[f"{v:.2f}" for v in wf_values],
    textposition="outside"
))
wf_fig3.update_layout(
    title="Patent Cliff: Step-by-Step Erosion of Profitability",
    yaxis_title="$B",
    height=420
)

st.plotly_chart(wf_fig3, use_container_width=True)

st.caption(
    "This waterfall shows how IP expiry reduces revenues, adds competition-driven margin erosion, "
    "and lowers EBIT in the Patent Cliff scenario compared to the Base Case."
)
