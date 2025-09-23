import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ALM Stress Testing Dashboard (IBR)", layout="wide")
st.title("💊 ALM Stress Testing Dashboard")
st.caption("R&D Intensity • Regulatory Shocks • IP Expiry → Financial Risk in Pharma")

# ----------------------------
# Default synthetic dataset
# ----------------------------
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

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🎛️ Core Controls")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
rd_cash_now_pct = st.sidebar.slider("R&D Cash Now (%)", 0, 100, 80, 5)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)
comp_cost_pct = st.sidebar.slider("Compliance Cost (% of Revenue)", 0, 10, 0, 1)
preset = st.sidebar.selectbox("Scenario Preset", ["Base", "R&D Surge", "Reg Shock", "Patent Cliff", "Black Swan"])
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)
ocf_adj = st.sidebar.slider("Δ Operating Cash Flow (%)", -50, 50, 0, 5)

with st.sidebar.expander("Advanced Working Capital"):
    dso_delta = st.slider("Δ DSO (days)", -15, 30, 0, 1)
    dpo_delta = st.slider("Δ DPO (days)", -15, 30, 0, 1)
    dio_delta = st.slider("Δ DIO (days)", -30, 30, 0, 1)

# ----------------------------
# Scenario Engine
# ----------------------------
def run_scenario(df, rd_bps=0, rd_cash_now_pct=80, price_cap=0, comp_cost_pct=0,
                 preset="Base", new_debt=0, new_equity=0, tax_rate=18,
                 dso_delta=0, dpo_delta=0, dio_delta=0, ocf_adj=0):
    scn = df.copy()
    scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100)

    if preset == "Patent Cliff":
        scn["Revenue_scn"] *= 0.8
        scn["COGS"] *= 1.10
        scn["R&D"] *= 1.15
    elif preset == "R&D Surge":
        rd_bps = 800
    elif preset == "Reg Shock":
        price_cap = 15; comp_cost_pct = 3
    elif preset == "Black Swan":
        rd_bps = 600; rd_cash_now_pct = 95
        price_cap = 10; comp_cost_pct = 4

    scn["R&D_scn"] = scn["R&D"] + scn["Revenue"] * (rd_bps/10000)
    scn["Opex_exR&D"] += scn["Revenue"] * (comp_cost_pct/100)

    # Working capital
    dso = scn["DSO"] + dso_delta
    dpo = scn["DPO"] + dpo_delta
    dio = scn["DIO"] + dio_delta
    ar = scn["Revenue_scn"] * (dso/365)
    inv = scn["COGS"] * (dio/365)
    ap = scn["COGS"] * (dpo/365)
    wc = ar + inv - ap
    scn["WC_Delta"] = wc - wc.shift(1).fillna(wc.iloc[0])

    # EBIT & OCF
    scn["EBIT"] = scn["Revenue_scn"] - scn["COGS"] - scn["Opex_exR&D"] - scn["R&D_scn"]
    scn["OCF"] = scn["EBIT"] * (1 - tax_rate/100) + scn["Depreciation"] - scn["WC_Delta"]
    scn["OCF"] *= (1 + ocf_adj/100)

    # Funding
    scn["RD_CashNow"] = scn["R&D_scn"] * (rd_cash_now_pct/100.0)
    scn["NewFunding"] = new_debt + new_equity
    scn["FundingGap"] = scn["RD_CashNow"] + scn["WC_Delta"] + scn["Capex"] + scn["PrincipalDue_12m"] - (scn["OCF"] + scn["NewFunding"])
    scn["Cash_End"] = scn["Cash_Begin"] - scn["FundingGap"]

    # Ratios
    scn["RD_Intensity"] = scn["R&D_scn"]/scn["Revenue_scn"]
    scn["DSCR"] = (scn["OCF"] / (scn["PrincipalDue_12m"] + scn["Debt_Begin"]*scn["InterestRate"]+1e-6))
    scn["Runway"] = np.where(scn["FundingGap"]>0, scn["Cash_Begin"]/(scn["FundingGap"]/12), np.inf)

    return scn.iloc[-1]  # latest year snapshot

# ----------------------------
# Chart 1: Bubble Chart (Objective 1)
# ----------------------------
grid = list(range(0, 1201, 200))  # stress points
results = [run_scenario(default_df, rd_bps=x, tax_rate=tax_rate, new_debt=new_debt, new_equity=new_equity) for x in grid]
df_bubble = pd.DataFrame(results)
df_bubble["Runway"] = df_bubble["Runway"].replace([np.inf, -np.inf], 60).fillna(60)
df_bubble["DSCR"] = df_bubble["DSCR"].replace([np.inf, -np.inf], 10).fillna(10)

fig1 = px.scatter(
    df_bubble, x="RD_Intensity", y="FundingGap",
    size="Runway", color="DSCR",
    color_continuous_scale=["red", "orange", "green"],
    size_max=60,
    title="Objective 1: Impact of R&D Intensity on Financial Risk (Current Year)"
)

# ----------------------------
# Chart 2: Bar Chart (Objective 2)
# ----------------------------
scenarios = {
    "Base": run_scenario(default_df, preset="Base"),
    "R&D Surge": run_scenario(default_df, preset="R&D Surge"),
    "Reg Shock": run_scenario(default_df, preset="Reg Shock"),
    "Patent Cliff": run_scenario(default_df, preset="Patent Cliff"),
}
df_bar = pd.DataFrame(scenarios).T
fig2 = px.bar(df_bar, x=df_bar.index, y="FundingGap", title="Objective 2: Funding Gap Across Scenarios")

# ----------------------------
# Chart 3: Waterfall Chart (Objective 3)
# ----------------------------
pc = scenarios["Patent Cliff"]
wf_labels = ["Operating CF", "New funding", "R&D cash now", "Δ Working capital", "Capex", "Debt service", "Funding Gap"]
wf_values = [
    pc["OCF"], pc["NewFunding"], -pc["RD_CashNow"], -pc["WC_Delta"], -pc["Capex"], -pc["PrincipalDue_12m"], pc["FundingGap"]
]
wf_measures = ["relative"] * (len(wf_labels)-1) + ["total"]

fig3 = go.Figure(go.Waterfall(
    x=wf_labels, y=wf_values, measure=wf_measures,
    text=[f"{v:.2f}" for v in wf_values],
    textposition="outside"
))
fig3.update_layout(title="Objective 3: Patent Cliff — Sources vs Uses of Cash")

# ----------------------------
# Show charts
# ----------------------------
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)
