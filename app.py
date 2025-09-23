import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Sample Latest-Year Snapshot
# ----------------------------
base_data = {
    "Revenue": 30.5,
    "COGS": 13.6,
    "Opex_exR&D": 7.5,
    "R&D": 6.3,
    "Capex": 2.0,
    "Depreciation": 1.2,
    "Debt_Begin": 13.7,
    "InterestRate": 0.031,
    "PrincipalDue_12m": 1.5,
    "Cash_Begin": 2.4
}

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🎛️ R&D Stress Testing")
rd_bps = st.sidebar.slider("Δ R&D (bps of Revenue)", -500, 1500, 300, 25)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 40, 18, 1)
new_debt = st.sidebar.number_input("New Funding — Debt ($B)", 0.0, 100.0, 0.0, 0.1)
new_equity = st.sidebar.number_input("New Funding — Equity ($B)", 0.0, 100.0, 0.0, 0.1)

# ----------------------------
# Scenario Engine (Current Year Only)
# ----------------------------
def run_current(rd_bps=0, tax_rate=18, new_debt=0, new_equity=0):
    revenue = base_data["Revenue"]
    cogs = base_data["COGS"]
    opex = base_data["Opex_exR&D"]
    capex = base_data["Capex"]
    rd = base_data["R&D"] + revenue * (rd_bps / 10000)  # apply stress
    dep = base_data["Depreciation"]
    debt = base_data["Debt_Begin"]
    ir = base_data["InterestRate"]
    principal = base_data["PrincipalDue_12m"]
    cash = base_data["Cash_Begin"]

    ebit = revenue - cogs - opex - rd
    ocf = ebit * (1 - tax_rate/100) + dep
    funding_gap = rd + capex + principal - (ocf + new_debt + new_equity)

    # Risk metrics
    dscr = ocf / (principal + debt*ir + 1e-6)
    runway = np.where(funding_gap>0, cash / (funding_gap/12), np.inf)

    return {
        "R&D_intensity": rd / revenue * 100,
        "FundingGap": funding_gap,
        "DSCR": dscr,
        "Runway": runway
    }

# ----------------------------
# Generate Bubble Data
# ----------------------------
grid = list(range(0, 1201, 200))  # stress points
results = [run_current(rd_bps=x, tax_rate=tax_rate, new_debt=new_debt, new_equity=new_equity) for x in grid]
df = pd.DataFrame(results)

# ----------------------------
# Bubble Chart
# ----------------------------
fig = px.scatter(
    df, x="R&D_intensity", y="FundingGap",
    size="Runway", color="DSCR",
    color_continuous_scale=["red","orange","green"],
    size_max=60,
    labels={"R&D_intensity":"R&D Intensity (%)","FundingGap":"Funding Gap ($B)","DSCR":"Debt Service Coverage"},
    title="Impact of R&D Intensity on Financial Risk (Current Year)"
)

st.plotly_chart(fig, use_container_width=True)
