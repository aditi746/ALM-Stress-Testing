import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ALM Stress Testing Dashboard (IBR)", layout="wide")
st.title("💊 ALM Stress Testing Dashboard")
st.caption("R&D Intensity • Regulatory Shocks • IP Expiry → Financial Risk in Pharma")

# ----------------------------
# Data (synthetic defaults, can replace with CSV)
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

st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional, same columns)", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else default_df.copy()
df = df.sort_values("Year").reset_index(drop=True)

# ----------------------------
# Core scenario controls
# ----------------------------
st.sidebar.header("🎛️ Core Controls")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
rd_cash_now_pct = st.sidebar.slider("R&D Cash Now (%)", 0, 100, 80, 5)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)
compliance_cost_pct = st.sidebar.slider("Compliance Cost (% of Revenue)", 0, 10, 0, 1)
preset = st.sidebar.selectbox("Scenario Preset", ["Base", "R&D Surge", "Reg Shock", "Patent Cliff", "Black Swan"])
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)

# Advanced (WC changes)
with st.sidebar.expander("Advanced Working Capital"):
    dso_delta = st.slider("Δ DSO (days)", -15, 30, 0, 1)
    dpo_delta = st.slider("Δ DPO (days)", -15, 30, 0, 1)
    dio_delta = st.slider("Δ DIO (days)", -30, 30, 0, 1)

# ----------------------------
# Scenario engine (returns latest year snapshot)
# ----------------------------
def run_scenario(df, rd_bps=0, rd_cash_now_pct=80, price_cap=0, comp_cost_pct=0,
                 preset="Base", new_debt=0, new_equity=0, tax_rate=18,
                 dso_delta=0, dpo_delta=0, dio_delta=0):
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

    # Funding
    scn["RD_CashNow"] = scn["R&D_scn"] * (rd_cash_now_pct/100.0)
    scn["NewFunding"] = new_debt + new_equity
    scn["FundingGap"] = scn["RD_CashNow"] + scn["WC_Delta"] + scn["Capex"] + scn["PrincipalDue_12m"] - (scn["OCF"] + scn["NewFunding"])
    scn["Cash_End"] = scn["Cash_Begin"] - scn["FundingGap"]

    # Ratios
    scn["RD_Intensity"] = scn["R&D_scn"]/scn["Revenue_scn"]
    scn["ALM_Stress"] = scn["FundingGap"].abs() * (1 + scn["RD_Intensity"])
    scn["ICR"] = (scn["EBIT"].clip(lower=1e-6) / (scn["Debt_Begin"]*scn["InterestRate"]+1e-6))
    scn["DSCR"] = (scn["OCF"] / (scn["PrincipalDue_12m"] + scn["Debt_Begin"]*scn["InterestRate"]+1e-6))
    scn["Liquidity12m"] = (scn["Cash_Begin"] + scn["Undrawn_Revolver"]) / (scn["Capex"] + scn["RD_CashNow"] + scn["PrincipalDue_12m"])
    scn["Runway_months"] = np.where(scn["FundingGap"]>0, scn["Cash_Begin"]/(scn["FundingGap"]/12), np.inf)

    return scn.iloc[-1]  # return only latest year snapshot

# ----------------------------
# KPI cards for selected scenario
# -------------------------
