import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import newton

# ==========================================
# 🧠 CORE QUANT ENGINE (UNCHANGED)
# ==========================================

class IndianBondQuant:
    """
    Institutional Grade Engine for Indian Fixed Income.
    Conventions: 30/360 Day Count, T+1 Settlement.
    """
    def __init__(self, face_value=100, freq=2):
        self.fv = face_value
        self.freq = freq

    @staticmethod
    def day_count_30_360(start_date, end_date):
        d1 = min(30, start_date.day)
        d2 = min(30, end_date.day) if d1 == 30 else end_date.day
        return 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + (d2 - d1)

    def get_cash_flows(self, maturity_date, settle_date, coupon_rate):
        dates = []
        curr = maturity_date
        while curr > settle_date:
            dates.append(curr)
            curr = curr - datetime.timedelta(days=365//self.freq)
        dates.sort()
        
        cf_amt = (self.fv * coupon_rate) / self.freq
        flows = []
        for i, d in enumerate(dates):
            amt = cf_amt
            if i == len(dates) - 1: amt += self.fv
            days = self.day_count_30_360(settle_date, d)
            t = days / 360.0
            flows.append({'date': d, 't': t, 'cf': amt})
        return pd.DataFrame(flows)

    def price_bond(self, flows, ytm):
        flows['df'] = 1 / ((1 + ytm/self.freq) ** (flows['t'] * self.freq))
        flows['pv'] = flows['cf'] * flows['df']
        return flows['pv'].sum()

    def calculate_risk_metrics(self, flows, ytm, price):
        mac_num = (flows['t'] * flows['pv']).sum()
        mac_dur = mac_num / price
        mod_dur = mac_dur / (1 + ytm/self.freq)
        
        dy = 0.0001
        p_up = self.price_bond(flows, ytm + dy)
        p_down = self.price_bond(flows, ytm - dy)
        convexity = (p_down + p_up - 2 * price) / (price * dy**2)
        pv01 = (p_down - p_up) / 2
        return mod_dur, convexity, pv01

    def z_spread_solver(self, flows, market_price, risk_free_curve_func):
        def objective(z):
            pv_sum = 0
            for _, row in flows.iterrows():
                r_t = risk_free_curve_func(row['t'])
                discount_rate = r_t + z
                pv_sum += row['cf'] / ((1 + discount_rate/self.freq) ** (row['t'] * self.freq))
            return pv_sum - market_price
        try:
            return newton(objective, 0.01)
        except:
            return 0.0

# Simulation Engine
def vasicek_simulation(r0, kappa, theta, sigma, T=1, dt=1/252, paths=100):
    N = int(T / dt)
    rates = np.zeros((paths, N))
    rates[:, 0] = r0
    for t in range(1, N):
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, paths)
        rates[:, t] = rates[:, t-1] + dr
    return rates

# ==========================================
# 🖥️ FRIDAY TERMINAL UI (Tabbed Architecture)
# ==========================================
st.set_page_config(page_title="Friday FIT", layout="wide", page_icon="📈")

# Dark Mode Terminal CSS
st.markdown("""
<style>
    /* Main Background */
    .stApp {background-color: #0e1117;}
    
    /* Metrics */
    div[data-testid="stMetricValue"] {font-size: 24px; color: #00d2d3;}
    div[data-testid="stMetricLabel"] {font-size: 14px; color: #a4b0be;}
    
    /* Tabs */
    button[data-baseweb="tab"] {font-size: 18px; font-weight: 600; color: #dfe6e9;}
    button[data-baseweb="tab"][aria-selected="true"] {color: #00d2d3; border-bottom-color: #00d2d3;}
    
    /* Headers */
    h1 {color: #ffffff; font-family: 'Roboto', sans-serif;}
    h2, h3 {color: #f1f2f6;}
    
    /* Cards */
    .card {padding: 15px; border-radius: 10px; background-color: #1e272e; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# Header
c_logo, c_title = st.columns([1, 10])
with c_logo:
    st.markdown("## 🇮🇳")
with c_title:
    st.title("Friday Fixed Income Terminal (FIT)")
    st.caption("Institutional Analytics | T+1 Settlement | Z-Spread | Monte Carlo")

st.markdown("---")

# MAIN TABS
tab_pricing, tab_credit, tab_sim = st.tabs(["📊 Pricing & Risk Desk", "📉 Curve & Credit Analysis", "🎲 Monte Carlo Simulator"])

quant = IndianBondQuant(face_value=100, freq=2)

# ==========================================
# TAB 1: PRICING & RISK
# ==========================================
with tab_pricing:
    col_input, col_metrics = st.columns([1, 3])
    
    with col_input:
        st.markdown("### 📝 Security Definition")
        with st.container():
            cr = st.number_input("Coupon Rate (%)", 6.0, 12.0, 7.50, step=0.1) / 100
            ytm = st.number_input("Market Yield (%)", 5.0, 15.0, 7.20, step=0.05) / 100
            mat_date = st.date_input("Maturity Date", datetime.date.today() + datetime.timedelta(days=365*5))
            settle_date = st.date_input("Settlement Date", datetime.date.today() + datetime.timedelta(days=1))
            st.caption(f"Days to Maturity: {(mat_date - settle_date).days}")

    with col_metrics:
        # Calculate
        flows = quant.get_cash_flows(mat_date, settle_date, cr)
        price = quant.price_bond(flows, ytm)
        mod_dur, conv, pv01 = quant.calculate_risk_metrics(flows, ytm, price)
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean Price", f"₹ {price:,.2f}", delta=f"{price-100:.2f}")
        m2.metric("Mod Duration", f"{mod_dur:.2f}", help="Sensitivity to 1% rate change")
        m3.metric("Convexity", f"{conv:.2f}", help="Curvature risk")
        m4.metric("PV01", f"₹ {pv01:.3f}", help="Value of 1 basis point change")
        
        st.markdown("### 🚨 Sensitivity Matrix (Stress Test)")
        
        # Stress Test Table
        shocks = [-50, -25, -10, 0, 10, 25, 50]
        stress_data = []
        for s in shocks:
            sy = ytm + (s/10000)
            sp = quant.price_bond(flows, sy)
            pnl = sp - price
            stress_data.append({
                "Shift (bps)": s,
                "Yield (%)": f"{sy*100:.2f}%",
                "Price (₹)": f"{sp:.2f}",
                "P&L (₹)": pnl
            })
        
        st.dataframe(pd.DataFrame(stress_data).style.format({"P&L (₹)": "{:,.2f}"}).background_gradient(subset=["P&L (₹)"], cmap="RdYlGn"), use_container_width=True)

# ==========================================
# TAB 2: CREDIT & CURVE
# ==========================================
with tab_credit:
    c_curve, c_spread = st.columns([2, 1])
    
    with c_curve:
        st.markdown("### 📈 Sovereign Yield Curve Construction")
        c1, c2, c3, c4 = st.columns(4)
        r1 = c1.number_input("1Y T-Bill", value=6.50)/100
        r5 = c2.number_input("5Y G-Sec", value=7.10)/100
        r10 = c3.number_input("10Y G-Sec", value=7.40)/100
        r30 = c4.number_input("30Y G-Sec", value=7.80)/100
        
        # Curve Viz
        x_curve = np.linspace(1, 30, 50)
        y_curve = np.interp(x_curve, [1, 5, 10, 30], [r1, r5, r10, r30]) * 100
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(x_curve, y_curve, color='#00d2d3', lw=3)
        ax.fill_between(x_curve, y_curve, alpha=0.1, color='#00d2d3')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.grid(color='#444', linestyle=':')
        ax.set_title("Bootstrapped Zero-Coupon Curve", color='white')
        st.pyplot(fig)
        
        def rf_curve_func(t):
            return np.interp(t, [1, 5, 10, 30], [r1, r5, r10, r30])

    with c_spread:
        st.markdown("### 🔎 Z-Spread Solver")
        st.info("Calculates the Credit Premium over the Sovereign Curve.")
        
        corp_price = st.number_input("Corp Bond Price (₹)", value=98.00)
        corp_cpn = st.number_input("Corp Coupon (%)", value=8.50) / 100
        
        if st.button("Solve Z-Spread"):
            corp_flows = quant.get_cash_flows(mat_date, settle_date, corp_cpn)
            z_spread = quant.z_spread_solver(corp_flows, corp_price, rf_curve_func)
            st.metric("Z-Spread", f"{z_spread*10000:.0f} bps", delta="Risk Premium")

# ==========================================
# TAB 3: SIMULATION
# ==========================================
with tab_sim:
    st.markdown("### 🎲 Stochastic Interest Rate Modeling (Vasicek)")
    
    col_params, col_plot = st.columns([1, 3])
    
    with col_params:
        st.markdown("**Model Config**")
        r0_sim = st.slider("Initial Rate", 0.05, 0.10, 0.07)
        theta_sim = st.slider("Long Term Mean", 0.05, 0.10, 0.075)
        sigma_sim = st.slider("Volatility", 0.01, 0.05, 0.02)
        kappa_sim = 0.5
    
    with col_plot:
        paths = vasicek_simulation(r0_sim, kappa_sim, theta_sim, sigma_sim, T=5, paths=200)
        
        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
        ax_sim.plot(paths.T, color='#6c5ce7', alpha=0.05)
        ax_sim.plot(paths.mean(axis=0), color='#ffffff', lw=2, label='Mean Path')
        ax_sim.set_facecolor('#0e1117')
        fig_sim.patch.set_facecolor('#0e1117')
        ax_sim.tick_params(colors='white')
        ax_sim.set_title("5-Year Rate Projection (200 Paths)", color='white')
        st.pyplot(fig_sim)
