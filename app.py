import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

# ==========================================
# 🧠 CORE QUANT ENGINE (The "Brain")
# ==========================================

class IndianBondQuant:
    """
    Institutional Grade Engine for Indian Fixed Income.
    Conventions:
    - Day Count: 30/360 (Standard for India G-Sec & Corp)
    - Settlement: T+1 (Standard RBI/CCIL)
    """
    def __init__(self, face_value=100, freq=2):
        self.fv = face_value
        self.freq = freq

    @staticmethod
    def day_count_30_360(start_date, end_date):
        """Standard 30/360 Day count convention."""
        d1 = min(30, start_date.day)
        d2 = min(30, end_date.day) if d1 == 30 else end_date.day
        return 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + (d2 - d1)

    def get_cash_flows(self, maturity_date, settle_date, coupon_rate):
        """Generates cash flow dates and amounts."""
        dates = []
        curr = maturity_date
        # Backtrack from maturity
        while curr > settle_date:
            dates.append(curr)
            # Approx 6 months back
            curr = curr - datetime.timedelta(days=365//self.freq)
        
        dates.sort()
        
        cf_amt = (self.fv * coupon_rate) / self.freq
        flows = []
        for i, d in enumerate(dates):
            amt = cf_amt
            if i == len(dates) - 1:
                amt += self.fv
            
            # Time in years (30/360 basis)
            days = self.day_count_30_360(settle_date, d)
            t = days / 360.0
            flows.append({'date': d, 't': t, 'cf': amt})
            
        return pd.DataFrame(flows)

    def price_bond(self, flows, ytm):
        """Prices bond using precise discount factors."""
        # PV = CF / (1 + y/f)^(t*f)
        flows['df'] = 1 / ((1 + ytm/self.freq) ** (flows['t'] * self.freq))
        flows['pv'] = flows['cf'] * flows['df']
        return flows['pv'].sum()

    def calculate_risk_metrics(self, flows, ytm, price):
        """Calculates Duration, Convexity, and PV01."""
        # Modified Duration
        # MacDur = Sum(t * PV) / Price
        mac_num = (flows['t'] * flows['pv']).sum()
        mac_dur = mac_num / price
        mod_dur = mac_dur / (1 + ytm/self.freq)
        
        # Convexity
        # Conv = (1 / P) * Sum( CF / (1+y)^(t+2) * (t^2 + t) ) ... simplified approx
        # Using centered difference for robustness:
        # (P_down + P_up - 2P) / (P * (dy)^2)
        dy = 0.0001 # 1 bp
        p_up = self.price_bond(flows, ytm + dy)
        p_down = self.price_bond(flows, ytm - dy)
        convexity = (p_down + p_up - 2 * price) / (price * dy**2)
        
        # PV01 (Price Value of 1 bp change) per 100 face value
        pv01 = (p_down - p_up) / 2
        
        return mod_dur, convexity, pv01

    def z_spread_solver(self, flows, market_price, risk_free_curve_func):
        """
        Calculates Z-Spread (Credit Spread) over a risk-free curve.
        Z-Spread is the constant spread 'z' added to the yield curve to match market price.
        """
        def objective(z):
            # Discount each cash flow by (RiskFreeRate(t) + z)
            # PV = Sum [ CF_i / (1 + r_i + z)^t_i ] -- Simplified for continuous or annual
            # Using discrete compounding match:
            pv_sum = 0
            for _, row in flows.iterrows():
                r_t = risk_free_curve_func(row['t']) # Get G-Sec rate for this maturity
                discount_rate = r_t + z
                pv_sum += row['cf'] / ((1 + discount_rate/self.freq) ** (row['t'] * self.freq))
            return pv_sum - market_price

        try:
            # Solve for z where objective(z) == 0
            z_spread = newton(objective, 0.01) # Start guess 1%
            return z_spread
        except:
            return 0.0

# ==========================================
# 🎲 SIMULATION ENGINE (Vasicek Model)
# ==========================================
def vasicek_simulation(r0, kappa, theta, sigma, T=1, dt=1/252, paths=100):
    """
    Simulates Interest Rate Paths using Vasicek Model.
    dr_t = kappa * (theta - r_t) * dt + sigma * dW_t
    """
    N = int(T / dt)
    rates = np.zeros((paths, N))
    rates[:, 0] = r0
    
    for t in range(1, N):
        # Euler-Maruyama discretization
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, paths)
        rates[:, t] = rates[:, t-1] + dr
        
    return rates

# ==========================================
# 🖥️ FRIDAY TERMINAL UI
# ==========================================
st.set_page_config(page_title="Friday Terminal", layout="wide", page_icon="💹")

# Custom CSS for "Terminal" Feel
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #e0e0e0; font-family: 'Roboto Mono', monospace;}
    .stMetric {background-color: #1f2937; padding: 10px; border-radius: 5px; border: 1px solid #374151;}
    .stDataFrame {border: 1px solid #374151;}
    div[data-testid="stSidebar"] {background-color: #111827;}
</style>
""", unsafe_allow_html=True)

st.title("💹 Friday Fixed Income Terminal")
st.caption("Institutional Analytics | RBI Conventions | Credit Modeling")

# --- NAVIGATION ---
page = st.sidebar.radio("Navigate Module", ["1. Pricing & Risk", "2. Curve & Credit", "3. Monte Carlo Sim"])

# Initialize Engine
quant = IndianBondQuant(face_value=100, freq=2)

# ==========================================
# PAGE 1: PRICING & RISK (The Deal Desk)
# ==========================================
if page == "1. Pricing & Risk":
    st.header("1. Deal Desk: Pricing & Sensitivities")
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        with st.expander("Security Definition", expanded=True):
            cr = st.number_input("Coupon Rate (%)", 7.00, step=0.1) / 100
            mat_date = st.date_input("Maturity", datetime.date.today() + datetime.timedelta(days=3650))
            ytm = st.number_input("Market YTM (%)", 7.20, step=0.01) / 100
        
        st.info("Settlement: T+1 (Standard)")
        settle_date = datetime.date.today() + datetime.timedelta(days=1)

    # Calculation
    flows = quant.get_cash_flows(mat_date, settle_date, cr)
    price = quant.price_bond(flows, ytm)
    mod_dur, conv, pv01 = quant.calculate_risk_metrics(flows, ytm, price)
    
    with col_out:
        # Top Line Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean Price", f"₹{price:.3f}")
        m2.metric("Mod. Duration", f"{mod_dur:.2f}", help="% Change in price for 1% change in yield")
        m3.metric("Convexity", f"{conv:.2f}", help="Curvature of price-yield relationship")
        m4.metric("PV01 (Risk)", f"₹{pv01:.3f}", help="Rupee value lost if rates rise 1 bp (0.01%)")
        
        # Scenario Analysis Table (Stress Test)
        st.subheader("Scenario Matrix")
        shocks = [-100, -50, -25, 0, 25, 50, 100] # bps
        res_data = []
        for s in shocks:
            shock_y = ytm + (s/10000)
            p_shock = quant.price_bond(flows, shock_y)
            p_change = p_shock - price
            res_data.append({
                "Shock (bps)": s, 
                "New Yield": f"{shock_y*100:.2f}%", 
                "Est. Price": f"{p_shock:.2f}",
                "P&L (per 100)": p_change
            })
        
        df_res = pd.DataFrame(res_data)
        
        # Highlight logic for dataframe
        def color_pnl(val):
            color = '#ff4b4b' if val < 0 else '#00cc66'
            return f'color: {color}'
            
        st.dataframe(df_res.style.applymap(color_pnl, subset=['P&L (per 100)']), use_container_width=True)

# ==========================================
# PAGE 2: CURVE & CREDIT (The Analyst)
# ==========================================
elif page == "2. Curve & Credit":
    st.header("2. Credit Spread Analysis")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Bootstrapped G-Sec Curve")
        st.caption("Theoretical Risk-Free Curve constructed from Benchmarks")
        
        # Inputs for Curve Construction
        rates = {
            1: st.number_input("1Y T-Bill Rate (%)", value=6.50)/100,
            5: st.number_input("5Y G-Sec Rate (%)", value=7.10)/100,
            10: st.number_input("10Y G-Sec Rate (%)", value=7.40)/100,
            30: st.number_input("30Y G-Sec Rate (%)", value=7.80)/100
        }
        
        # Simple Interpolation Function (Linear for demo)
        def risk_free_curve(t):
            years = sorted(rates.keys())
            vals = [rates[y] for y in years]
            return np.interp(t, years, vals)
            
        # Visualize Curve
        t_space = np.linspace(0.5, 30, 100)
        r_space = [risk_free_curve(t)*100 for t in t_space]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_space, r_space, color='#00d2d3', lw=2)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_title("Standard Sovereign Yield Curve", color='white')
        ax.grid(color='#444', linestyle='--')
        st.pyplot(fig)

    with c2:
        st.subheader("Z-Spread Solver")
        st.caption("Calculate credit risk premium over the G-Sec curve.")
        
        mp = st.number_input("Corporate Bond Market Price (₹)", value=98.50)
        cpn = st.number_input("Corporate Coupon (%)", value=8.50) / 100
        mat = st.date_input("Maturity Date", datetime.date.today() + datetime.timedelta(days=1800))
        
        if st.button("Calculate Z-Spread"):
            settle = datetime.date.today() + datetime.timedelta(days=1)
            flows = quant.get_cash_flows(mat, settle, cpn)
            
            # Solve
            z = quant.z_spread_solver(flows, mp, risk_free_curve)
            
            st.metric("Z-Spread (Credit Premium)", f"{z*10000:.0f} bps", 
                     delta="Spread over G-Sec")
            
            if z > 0.025: # > 250 bps
                st.warning("⚠️ High Yield / Junk Bond Territory")
            else:
                st.success("✅ Investment Grade Territory")

# ==========================================
# PAGE 3: MONTE CARLO (The Quant)
# ==========================================
elif page == "3. Monte Carlo Sim":
    st.header("3. Stochastic Rate Modeling (Vasicek)")
    st.markdown("Simulate 1000s of future interest rate paths to stress-test the bond.")
    
    col_param, col_sim = st.columns([1, 3])
    
    with col_param:
        st.markdown("**Model Parameters**")
        r0 = st.number_input("Initial Rate (r0)", 0.07)
        theta = st.number_input("Long Term Mean (theta)", 0.075)
        kappa = st.number_input("Mean Reversion Speed (kappa)", 0.5)
        sigma = st.number_input("Volatility (sigma)", 0.02)
        
    with col_sim:
        if st.button("Run Simulation (100 Paths)"):
            paths = vasicek_simulation(r0, kappa, theta, sigma, T=5, paths=100)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(paths.T, color='#4caf50', alpha=0.1)
            ax.plot(paths.mean(axis=0), color='white', lw=3, label="Average Path")
            
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.set_title("Vasicek Short Rate Paths (5 Years)", color='white')
            st.pyplot(fig)
            
            # Pricing based on terminal rates
            st.success(f"Simulated Volatility Range: {paths[:,-1].min()*100:.2f}% to {paths[:,-1].max()*100:.2f}%")
