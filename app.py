import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ALM What-If Dashboard (IBR)", layout="wide")
st.title("💊 ALM What-If Dashboard — R&D, Regulatory, IP Expiry")

st.caption(
    "Objective 1: R&D-driven financial risk • "
    "Objective 2: Regulatory impact on profitability • "
    "Objective 3: IP expiry & competition effects"
)

# ----------------------------
# Data: default synthetic if no upload (2015–2024)
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
uploaded = st.sidebar.file_uploader("Upload CSV (same columns as default)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = default_df.copy()

df = df.sort_values("Year").reset_index(drop=True)

# ----------------------------
# Scenario controls
# ----------------------------
st.sidebar.header("🎛️ Scenario Controls")

# Presets
preset = st.sidebar.selectbox(
    "Preset",
    [
        "Base",
        "R&D Surge",
        "Regulatory Shock",
        "Patent Cliff (IP Expiry)",
        "Black Swan"
    ],
)

# Core levers
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
rd_cash_now_pct = st.sidebar.slider("R&D Cash Now (%)", 0, 100, 80, 5)
price_cap = st.sidebar.slider("Regulatory Price Cap (%)", 0, 50, 0, 5)          # cuts revenue
compliance_cost_pct = st.sidebar.slider("Compliance Cost (% of Revenue)", 0, 10, 0, 1)
revenue_shock_pct = st.sidebar.slider("Revenue Shock (%)", -50, 20, 0, 1)
debt_spread_bps = st.sidebar.slider("Debt Spread (bps)", -100, 500, 200, 25)
dso_delta = st.sidebar.slider("Δ DSO (days)", -15, 30, 0, 1)
dpo_delta = st.sidebar.slider("Δ DPO (days)", -15, 30, 0, 1)
dio_delta = st.sidebar.slider("Δ DIO (days)", -30, 30, 0, 1)
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)

# Apply preset overrides (makes demo fast)
if preset == "R&D Surge":
    rd_bps = 800; rd_cash_now_pct = 90
elif preset == "Regulatory Shock":
    price_cap = 15; compliance_cost_pct = 3
elif preset == "Patent Cliff (IP Expiry)":
    revenue_shock_pct = -10; # additional reductions apply below
elif preset == "Black Swan":
    rd_bps = 600; rd_cash_now_pct = 95
    price_cap = 10; compliance_cost_pct = 4
    revenue_shock_pct = -15
    debt_spread_bps = 350
    dso_delta = 10; dpo_delta = -5; dio_delta = 10

# ----------------------------
# Scenario engine
# ----------------------------
scn = df.copy()

# Regulatory cap + general shock on revenue
scn["Revenue_scn"] = scn["Revenue"] * (1 - price_cap/100.0) * (1 + revenue_shock_pct/100.0)

# IP expiry preset transforms
if preset == "Patent Cliff (IP Expiry)":
    scn["Revenue_scn"] *= 0.8      # -20% over horizon
    scn["COGS"] *= 1.10            # margin compression
    scn["R&D"] *= 1.15             # pipeline rebuild pressure

# Compliance cost to opex
scn["Opex_exR&D"] = scn["Opex_exR&D"] + scn["Revenue"] * (compliance_cost_pct/100.0)

# R&D stress
scn["R&D_scn"] = scn["R&D"] + scn["Revenue"] * (rd_bps/10000.0)

# Working capital with deltas
dso = scn["DSO"] + dso_delta
dpo = scn["DPO"] + dpo_delta
dio = scn["DIO"] + dio_delta
cogs = scn["COGS"]

ar = scn["Revenue_scn"] * (dso/365.0)
inv = cogs * (dio/365.0)
ap = cogs * (dpo/365.0)
wc = ar + inv - ap
wc_shift = wc.shift(1).fillna(wc.iloc[0])
scn["WC_Delta"] = wc - wc_shift

# EBIT & interest
scn["EBIT"] = scn["Revenue_scn"] - scn["COGS"] - scn["Opex_exR&D"] - scn["R&D_scn"]
rate = scn["InterestRate"] + (debt_spread_bps/10000.0)
scn["Interest"] = scn["Debt_Begin"] * rate
scn["DebtService"] = scn["Interest"] + scn["PrincipalDue_12m"]

# OCF (simplified but defensible)
t = tax_rate/100.0
scn["OCF"] = scn["EBIT"] * (1 - t) + scn["Depreciation"] - scn["WC_Delta"]

# Funding structure
scn["RD_CashNow"] = scn["R&D_scn"] * (rd_cash_now_pct/100.0)
scn["NewFunding"] = new_debt + new_equity

# Funding gap (positive => cash need)
scn["FundingGap"] = scn["RD_CashNow"] + scn["WC_Delta"] + scn["Capex"] + scn["DebtService"] - (scn["OCF"] + scn["NewFunding"])
scn["Cash_End"] = scn["Cash_Begin"] - scn["FundingGap"]

# Risk metrics
safe_interest = scn["Interest"].replace(0, np.nan)
scn["ICR"] = (scn["EBIT"].clip(lower=1e-6) / safe_interest).replace([np.inf,-np.inf], np.nan)

den_dscr = scn["DebtService"].replace(0, np.nan)
scn["DSCR"] = (scn["OCF"] / den_dscr).replace([np.inf,-np.inf], np.nan)

den_liq = (scn["DebtService"] + scn["Capex"] + scn["RD_CashNow"]).replace(0, np.nan)
scn["Liquidity12m"] = (scn["Cash_Begin"] + scn["Undrawn_Revolver"]) / den_liq

scn["RD_Intensity"] = (scn["R&D_scn"] / scn["Revenue_scn"]).replace([np.inf,-np.inf], 0).fillna(0.0)
scn["ALM_Stress"] = scn["FundingGap"].abs() * (1 + scn["RD_Intensity"])

burn = scn["FundingGap"].clip(lower=0) / 12.0
scn["Runway_months"] = np.where(burn>0, scn["Cash_Begin"]/burn, np.inf)

# ----------------------------
# KPI header (latest year)
# ----------------------------
latest = scn.iloc[-1]
k = st.columns(6)
k[0].metric("Funding Gap (latest, $B)", f"{latest['FundingGap']:.2f}")
k[1].metric("ALM Stress (latest)", f"{latest['ALM_Stress']:.2f}")
k[2].metric("Cash Runway", "∞" if np.isinf(latest["Runway_months"]) else f"{latest['Runway_months']:.1f} mo")
k[3].metric("DSCR", f"{latest['DSCR']:.2f}")
k[4].metric("ICR", f"{latest['ICR']:.2f}")
k[5].metric("Liquidity 12m", f"{latest['Liquidity12m']:.2f}×")

st.markdown("---")

# ----------------------------
# Charts
# ----------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Funding Gap by Year ($B)")
    st.bar_chart(scn.set_index("Year")["FundingGap"])
with c2:
    st.subheader("ALM Stress Ratio by Year")
    st.line_chart(scn.set_index("Year")["ALM_Stress"])

# Waterfall (latest year) of cash uses/sources
st.subheader("Funding Gap Waterfall — Latest Year")
y = latest["Year"]
uses = {
    "R&D cash now": latest["RD_CashNow"],
    "Δ Working capital": latest["WC_Delta"],
    "Capex": latest["Capex"],
    "Debt service": latest["DebtService"],
}
sources = {
    "Operating CF": -latest["OCF"],   # negative to show as source
    "New funding": -(latest["NewFunding"]),
}
water_items = list(uses.items()) + list(sources.items())
labels = [k for k,_ in water_items] + ["Gap"]
vals = [v for _,v in water_items]
gap = sum(vals)
vals.append(gap)

wf = go.Figure(go.Waterfall(
    x=labels,
    measure=["relative"]*len(water_items) + ["total"],
    y=vals
))
wf.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(wf, use_container_width=True)

# ----------------------------
# Table & download
# ----------------------------
st.subheader("Scenario Table")
st.dataframe(scn.round(3), use_container_width=True)

st.download_button(
    "⬇️ Download Scenario CSV",
    scn.to_csv(index=False).encode("utf-8"),
    file_name="alm_scenario_output.csv",
    mime="text/csv"
)

st.info(
    "Methodology: We simulate R&D timing/intensity, regulatory price caps & compliance costs, "
    "IP expiry (revenue drop, margin compression, R&D uplift), and funding mix. "
    "Outputs show Funding Gap, ALM Stress, DSCR, ICR, Liquidity coverage, and Cash runway."
)
