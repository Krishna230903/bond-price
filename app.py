import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import newton
from fpdf import FPDF
import base64

# ==========================================
# 🧠 1. THE FRIDAY QUANT ENGINE (ALL FEATURES)
# ==========================================
class FridayQuantEngine:
    def __init__(self, face_value=100, freq=2, day_count_method="30/360"):
        self.fv = face_value
        self.freq = freq
        self.dc_method = day_count_method

    def _days_between(self, d1, d2):
        """Handles both Indian Corporate (30/360) and G-Sec (Act/365) logic."""
        if self.dc_method == "30/360":
            d1_day = min(30, d1.day)
            d2_day = min(30, d2.day) if d1_day == 30 else d2.day
            return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (d2_day - d1_day)
        else:
            return (d2 - d1).days

    def get_cash_flows(self, maturity_date, settle_date, coupon_rate, is_floating=False, ref_rate=0.0):
        if settle_date >= maturity_date: return pd.DataFrame()
        
        # Generate dates backwards
        dates = []
        curr = maturity_date
        while curr > settle_date:
            dates.append(curr)
            # Approx step back
            curr = curr - datetime.timedelta(days=365//self.freq)
        dates.sort()
        
        # Determine Coupon Amount
        eff_rate = (ref_rate + coupon_rate) if is_floating else coupon_rate
        cpn_amt = (self.fv * eff_rate) / self.freq
        
        flows = []
        denominator_base = 360.0 if self.dc_method == "30/360" else 365.0
        
        for i, d in enumerate(dates):
            amt = cpn_amt
            if i == len(dates) - 1: amt += self.fv
            
            days = self._days_between(settle_date, d)
            t = days / denominator_base
            flows.append({'date': d, 't': t, 'cf': amt})
            
        return pd.DataFrame(flows)

    def calculate_metrics(self, flows, ytm):
        """Full Pricing Model"""
        if flows.empty: return 0, 0, 0, 0, 0
        
        # 1. Discount Factors & PV
        # Standard street convention: PV = CF / (1 + y/f)^(t*f)
        flows['df'] = 1 / ((1 + ytm/self.freq) ** (flows['t'] * self.freq))
        flows['pv'] = flows['cf'] * flows['df']
        
        dirty_price = flows['pv'].sum()
        
        # 2. Accrued Interest (Clean Price logic)
        # Accrued is roughly (Coupon * Fraction of period passed)
        # We simplify for robustness: Total PV - Clean = Accrued
        # But accurately: find last coupon date
        settle_t = flows['t'].iloc[0] # Time to first payment
        period_len = 1/self.freq
        fraction_remaining = (settle_t % period_len) / period_len # Rough approx
        accrued = 0 # Default if exact dates align
        
        # To get explicit accrued, we usually calculate clean price = Dirty - Accrued
        # Here we will define Accrued based on standard logic:
        # Simple Logic: Coupon * (Days since last / Days in period)
        # For this demo, let's assume standard accrued calculation
        accrued = (self.fv * (ytm if ytm>0 else 0.05) / self.freq) * (1 - (flows['t'].iloc[0] * self.freq))
        if accrued < 0: accrued = 0
        
        clean_price = dirty_price - accrued
        
        # 3. Risk Metrics
        mac_num = (flows['t'] * flows['pv']).sum()
        mac_dur = mac_num / dirty_price if dirty_price else 0
        mod_dur = mac_dur / (1 + ytm/self.freq)
        
        # Convexity & PV01 (Shock Method)
        dy = 0.0001
        # Re-calc price up/down
        flows_up = flows.copy(); flows_down = flows.copy()
        flows_up['pv'] = flows_up['cf'] * (1 / ((1 + (ytm+dy)/self.freq) ** (flows_up['t'] * self.freq)))
        flows_down['pv'] = flows_down['cf'] * (1 / ((1 + (ytm-dy)/self.freq) ** (flows_down['t'] * self.freq)))
        
        p_up = flows_up['pv'].sum()
        p_down = flows_down['pv'].sum()
        
        convexity = (p_down + p_up - 2 * dirty_price) / (dirty_price * dy**2)
        pv01 = (p_down - p_up) / 2
        
        return clean_price, dirty_price, accrued, mod_dur, convexity, pv01

    def yield_solver(self, flows, target_dirty_price):
        """Newton-Raphson Solver to find YTM from Price"""
        def objective(y):
            # Recalculate PV with yield 'y'
            df = 1 / ((1 + y/self.freq) ** (flows['t'] * self.freq))
            return (flows['cf'] * df).sum() - target_dirty_price
        
        try:
            return newton(objective, 0.05)
        except:
            return 0.0

    def z_spread_solver(self, flows, market_price, rf_curve_func):
        """Calculates Spread over Sovereign Curve"""
        def obj(z):
            pv = 0
            for _, row in flows.iterrows():
                r = rf_curve_func(row['t'])
                pv += row['cf'] / ((1 + (r+z)/self.freq) ** (row['t']*self.freq))
            return pv - market_price
        try:
            return newton(obj, 0.01)
        except:
            return 0.0

    @staticmethod
    def nss_model(t, b0, b1, b2, tau):
        """Nelson-Siegel Model"""
        term1 = (1 - np.exp(-t/tau)) / (t/tau)
        term2 = term1 - np.exp(-t/tau)
        return b0 + b1*term1 + b2*term2

# ==========================================
# 📄 PDF ENGINE
# ==========================================
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Friday Fixed Income Valuation", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    for key, value in data.items():
        pdf.cell(100, 10, f"{key}: {value}", 0, 1)
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 🖥️ FRIDAY TERMINAL UI
# ==========================================
st.set_page_config(page_title="Friday Master Terminal", layout="wide", page_icon="🏛️")

# Session State
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []

# CSS Styling
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #1e272e; padding: 15px; border-radius: 8px; border-left: 5px solid #00d2d3;}
    h1, h2, h3, h4 {color: #f5f6fa; font-family: 'Roboto', sans-serif;}
    .stDataFrame {border: 1px solid #353b48;}
</style>
""", unsafe_allow_html=True)

# Sidebar for Global Settings
with st.sidebar:
    st.header("⚙️ Global Settings")
    dc_select = st.selectbox("Day Count Convention", ["30/360 (Corporate)", "Actual/365 (G-Sec)"])
    dc_method = "30/360" if "30/360" in dc_select else "Actual/365"
    st.info(f"Active Logic: **{dc_method}**")
    st.caption("All calculations will use this convention.")

# Header
st.title("🏛️ Friday Master Terminal")
st.caption("Institutional Analytics | Pricing | Risk | Credit | Simulation")

# Tabs
tabs = st.tabs(["📊 Trading Desk", "💼 Portfolio", "📉 Credit & Curve", "🧪 Quant Lab"])

# Init Engine
quant = FridayQuantEngine(face_value=100, freq=2, day_count_method=dc_method)

# ==========================================
# TAB 1: TRADING DESK (Pricing & Risk)
# ==========================================
with tabs[0]:
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("1. Bond Definition")
        bond_type = st.radio("Structure", ["Fixed Rate", "Floating Rate (FRN)"], horizontal=True)
        
        # Logic for Floating vs Fixed
        if "Floating" in bond_type:
            ref_rate = st.number_input("Ref Rate (MIBOR %)", 6.50)/100
            spread = st.number_input("Spread (bps)", 100.0)/10000
            eff_cpn = spread # For calculation flow
            is_float = True
            disp_cpn = f"MIBOR + {spread*10000:.0f} bps"
        else:
            eff_cpn = st.number_input("Coupon Rate (%)", 7.50, step=0.1)/100
            is_float = False
            ref_rate = 0.0
            disp_cpn = f"{eff_cpn*100:.2f}%"
            
        mat_date = st.date_input("Maturity", datetime.date.today() + datetime.timedelta(days=365*5))
        settle_date = st.date_input("Settlement", datetime.date.today() + datetime.timedelta(days=1))
        
        st.subheader("2. Market Data")
        mode = st.radio("Calculation Mode", ["Price from Yield", "Yield from Price"])
        
        user_yield = 0.0
        user_price = 0.0
        
        if "Price from Yield" in mode:
            user_yield = st.number_input("Input YTM (%)", 7.20, step=0.05)/100
        else:
            user_price = st.number_input("Input Clean Price (₹)", 100.50, step=0.1)

    with c_right:
        # --- CALCULATION CORE ---
        flows = quant.get_cash_flows(mat_date, settle_date, eff_cpn, is_float, ref_rate)
        
        if not flows.empty:
            if "Price from Yield" in mode:
                # Normal Forward Calculation
                final_ytm = user_yield
                cln, dty, acc, md, cv, pv01 = quant.calculate_metrics(flows, final_ytm)
            else:
                # Reverse Solver
                # Approx Accrued for solver target
                est_acc = (100 * (eff_cpn + ref_rate) / 2) * 0.5 # Rough guess
                final_ytm = quant.yield_solver(flows, user_price + est_acc)
                cln, dty, acc, md, cv, pv01 = quant.calculate_metrics(flows, final_ytm)
            
            # --- DISPLAY RESULTS ---
            st.markdown("### 🏷️ Valuation Output")
            m1, m2, m3 = st.columns(3)
            m1.metric("Clean Price", f"₹ {cln:,.2f}")
            m2.metric("Accrued Int", f"₹ {acc:,.2f}")
            m3.metric("Invoice Price", f"₹ {dty:,.2f}", delta="Payable")
            
            st.markdown("### ⚠️ Risk Sensitivities")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("YTM", f"{final_ytm*100:.3f}%")
            r2.metric("Mod Duration", f"{md:.2f}")
            r3.metric("Convexity", f"{cv:.2f}")
            r4.metric("PV01", f"₹ {pv01:.4f}")
            
            # --- STRESS TEST MATRIX ---
            st.markdown("#### 🔥 Stress Test Matrix")
            # Create Sensitivity Table
            shocks = [-50, -25, -10, 0, 10, 25, 50]
            sim_data = []
            for s in shocks:
                sy = final_ytm + (s/10000)
                # Quick price recalc
                s_cln, _, _, _, _, _ = quant.calculate_metrics(flows, sy)
                sim_data.append({
                    "Shock (bps)": s,
                    "New Yield": f"{sy*100:.2f}%",
                    "New Price": f"{s_cln:.2f}",
                    "P&L": s_cln - cln
                })
            st.dataframe(pd.DataFrame(sim_data).style.format({"P&L": "{:+.2f}"}).background_gradient(subset=["P&L"], cmap="RdYlGn"), use_container_width=True)
            
            # --- ACTIONS ---
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("➕ Add to Portfolio"):
                    st.session_state['portfolio'].append({
                        "Type": bond_type, "Maturity": str(mat_date), "YTM": f"{final_ytm*100:.2f}%", 
                        "Price": cln, "Duration": md
                    })
                    st.success("Added!")
            with ac2:
                # PDF Generation
                pdf_data = {"Bond": bond_type, "Maturity": str(mat_date), "Price": f"{cln:.2f}", "YTM": f"{final_ytm*100:.2f}%"}
                pdf_bytes = create_pdf(pdf_data)
                st.download_button("📄 Download PDF", data=pdf_bytes, file_name="Valuation.pdf", mime="application/pdf")
            with ac3:
                csv = flows.to_csv().encode('utf-8')
                st.download_button("💾 Download Cash Flows", data=csv, file_name="cashflows.csv", mime='text/csv')

# ==========================================
# TAB 2: PORTFOLIO MANAGER
# ==========================================
with tabs[1]:
    st.subheader("💼 Active Holdings")
    if len(st.session_state['portfolio']) > 0:
        df_p = pd.DataFrame(st.session_state['portfolio'])
        st.dataframe(df_p, use_container_width=True)
        
        st.markdown("#### Aggregate Risk")
        p_dur = df_p['Duration'].mean()
        p_val = df_p['Price'].sum()
        c1, c2 = st.columns(2)
        c1.metric("Total Market Value", f"₹ {p_val:,.2f}")
        c2.metric("Portfolio Duration", f"{p_dur:.2f}")
        
        if st.button("Clear Portfolio"):
            st.session_state['portfolio'] = []
            st.experimental_rerun()
    else:
        st.info("Portfolio is empty.")

# ==========================================
# TAB 3: CREDIT & CURVE
# ==========================================
with tabs[2]:
    c_nss, c_z = st.columns(2)
    
    with c_nss:
        st.markdown("### 📉 Nelson-Siegel Curve Fitter")
        b0 = st.slider("Beta0 (Long Term)", 0.0, 0.15, 0.07)
        b1 = st.slider("Beta1 (Short Term)", -0.05, 0.05, -0.02)
        b2 = st.slider("Beta2 (Hump)", -0.05, 0.05, 0.01)
        tau = st.slider("Tau (Location)", 0.5, 5.0, 2.0)
        
        t_ax = np.linspace(0.1, 30, 100)
        y_ax = quant.nss_model(t_ax, b0, b1, b2, tau) * 100
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_ax, y_ax, color='#00d2d3', lw=2)
        ax.set_title("NSS Yield Curve")
        ax.set_facecolor('#0e1117'); fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white'); ax.grid(alpha=0.2)
        st.pyplot(fig)
        
        # Helper for Z-Spread
        def rf_func(t): return quant.nss_model(t, b0, b1, b2, tau)

    with c_z:
        st.markdown("### 🕵️ Z-Spread Solver")
        st.caption("Calculate Credit Premium over the NSS Curve")
        z_price = st.number_input("Corp Bond Price", 98.0)
        z_cpn = st.number_input("Corp Coupon (%)", 8.5) / 100
        
        if st.button("Calculate Spread"):
            # Dummy flows for Z-spread calc (using 5Y default)
            z_flows = quant.get_cash_flows(datetime.date.today()+datetime.timedelta(days=1825), 
                                           datetime.date.today()+datetime.timedelta(days=1), z_cpn)
            
            spread = quant.z_spread_solver(z_flows, z_price, rf_func)
            st.metric("Z-Spread", f"{spread*10000:.0f} bps", delta="Credit Risk")

# ==========================================
# TAB 4: QUANT LAB (SIMULATION)
# ==========================================
with tabs[3]:
    st.markdown("### 🧪 Vasicek Rate Simulation")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        v_r0 = st.number_input("Start Rate", 0.06)
        v_theta = st.number_input("Mean", 0.07)
        v_sigma = st.number_input("Vol", 0.02)
        v_paths = st.slider("Paths", 10, 500, 100)
    
    with col2:
        if st.button("Run Monte Carlo"):
            dt = 1/252; T = 5; N = int(T/dt)
            rates = np.zeros((v_paths, N))
            rates[:,0] = v_r0
            kappa = 0.5
            
            for t in range(1, N):
                dr = kappa*(v_theta - rates[:,t-1])*dt + v_sigma*np.sqrt(dt)*np.random.normal(0,1,v_paths)
                rates[:,t] = rates[:,t-1] + dr
                
            fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
            ax_mc.plot(np.linspace(0, T, N), rates.T, alpha=0.1, color='cyan')
            ax_mc.plot(np.linspace(0, T, N), rates.mean(axis=0), color='white', lw=2)
            ax_mc.set_facecolor('#0e1117'); fig_mc.patch.set_facecolor('#0e1117')
            ax_mc.tick_params(colors='white')
            st.pyplot(fig_mc)
