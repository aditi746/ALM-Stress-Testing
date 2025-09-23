import streamlit as st
import pandas as pd
import numpy as np
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
df = pd.DataFrame({
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
    "Undrawn_Revolver": [2.8,1.5,2.3,1.0,1.7,1.6,1.3,2.1,2.0,2.4],
    "Depreciation": [0.4,0.5,0.6,0.7,0.8,0.8,1.0,1.0,1.1,1.2]
})

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🎛️ Controls")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)

# ----------------------------
# Scenario Engine
# ----------------------------
def run_scenario(df, rd_bps=0, price_cap=0, new_debt=0, new_equity=0, tax_rate=18, preset="Base"):
    scn = df.copy()
    scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)

    # Scenario presets
    if preset == "R&D Surge":
        rd_bps = 800
    elif preset == "Reg Shock":
        price_cap = 15
    elif preset == "Patent Cliff":
        scn["Revenue_scn"] *= 0.8
        scn["R&D"] *= 1.15

    scn["R&D_scn"] = scn["R&D"] + scn["Revenue"] * (rd_bps/10000)
    scn["EBIT"] = scn["Revenue_scn"] - scn["COGS"] - scn["Opex_exR&D"] - scn["R&D_scn"]
    scn["OCF"] = scn["EBIT"] * (1 - tax_rate/100) + scn["Depreciation"]
    scn["NewFunding"] = new_debt + new_equity
    scn["FundingGap"] = scn["R&D_scn"] + scn["Capex"] + scn["PrincipalDue_12m"] - (scn["OCF"] + scn["NewFunding"])

    return scn.iloc[-1]  # latest year snapshot

# ----------------------------
# KPI Cards
# ----------------------------
sel = run_scenario(df, rd_bps=rd_bps, price_cap=price_cap, new_debt=new_debt, new_equity=new_equity, tax_rate=tax_rate)
st.subheader("📊 Key Metrics (Latest Year)")
k = st.columns(4)
k[0].metric("Funding Gap ($B)", f"{sel['FundingGap']:.2f}")
k[1].metric("EBIT ($B)", f"{sel['EBIT']:.2f}")
k[2].metric("OCF ($B)", f"{sel['OCF']:.2f}")
k[3].metric("R&D Intensity (%)", f"{(sel['R&D_scn']/sel['Revenue_scn']*100):.1f}")

st.markdown("---")

# ----------------------------
# Objective 1: R&D Intensity → Risk
# ----------------------------
grid = list(range(0, 1201, 200))
sens = pd.DataFrame([run_scenario(df, rd_bps=x).to_dict() for x in grid])
sens["R&D_bps"] = grid

st.subheader("📈 Objective 1: R&D Intensity vs Funding Gap")
st.line_chart(sens.set_index("R&D_bps")[["FundingGap"]])
st.caption("Higher R&D intensity widens funding gaps by reducing EBIT and OCF, stressing liquidity.")

st.markdown("---")

# ----------------------------
# Objective 2: Regulatory Shocks
# ----------------------------
scenarios = {
    "Base": run_scenario(df, preset="Base"),
    "Reg Shock": run_scenario(df, preset="Reg Shock"),
    "R&D Surge": run_scenario(df, preset="R&D Surge"),
    "Patent Cliff": run_scenario(df, preset="Patent Cliff")
}
scn_df = pd.DataFrame(scenarios).T

st.subheader("🏛 Objective 2: Regulatory Shocks and Scenario Comparison")
st.bar_chart(scn_df[["FundingGap"]])
st.caption("Regulatory caps compress revenues, worsening funding gaps compared to the base case.")

st.markdown("---")

# ----------------------------
# Objective 3: Patent Cliff
# ----------------------------
pc = scenarios["Patent Cliff"]
labels = ["Operating CF", "New funding", "R&D cash now", "Capex", "Debt service", "Gap"]
values = [
    pc["OCF"], pc["NewFunding"], -pc["R&D_scn"], -pc["Capex"], -pc["PrincipalDue_12m"],
    pc["FundingGap"]
]

wf = go.Figure(go.Waterfall(
    x=labels,
    y=values,
    measure=["relative", "relative", "relative", "relative", "relative", "total"],
    connector={"line": {"color": "gray"}},
))
wf.update_layout(title="Patent Cliff Cash Flow Breakdown", height=400)
st.subheader("💧 Objective 3: Patent Cliff — Sources vs Uses of Cash")
st.plotly_chart(wf, use_container_width=True)
st.caption("Patent expiry reduces revenues but obligations remain — creating the largest funding gap.")

