import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import newton
from fpdf import FPDF
import base64

# ==========================================
# 🧠 FRIDAY QUANT ENGINE (FIXED LOGIC)
# ==========================================
class FridayQuantEngine:
    def __init__(self, face_value=100, freq=2, day_count_method="30/360"):
        self.fv = face_value
        self.freq = freq
        self.dc_method = day_count_method

    def _days_between(self, d1, d2):
        """Day count logic: 30/360 vs Actual/365"""
        if self.dc_method == "30/360":
            d1_day = min(30, d1.day)
            d2_day = min(30, d2.day) if d1_day == 30 else d2.day
            return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (d2_day - d1_day)
        else:
            return (d2 - d1).days

    def get_cash_flows(self, maturity_date, settle_date, coupon_rate, is_floating=False, ref_rate=0.0):
        if settle_date >= maturity_date: return pd.DataFrame()
        
        # 1. Generate Dates (Backwards from Maturity)
        dates = []
        curr = maturity_date
        while curr > settle_date:
            dates.append(curr)
            curr = curr - datetime.timedelta(days=365//self.freq)
        dates.sort()
        
        # 2. Determine Coupon Amount
        eff_rate = (ref_rate + coupon_rate) if is_floating else coupon_rate
        cpn_amt = (self.fv * eff_rate) / self.freq
        
        flows = []
        denominator = 360.0 if self.dc_method == "30/360" else 365.0
        
        for i, d in enumerate(dates):
            amt = cpn_amt
            if i == len(dates) - 1: amt += self.fv
            
            # Time (t) in years from settlement
            days = self._days_between(settle_date, d)
            t = days / denominator
            flows.append({'date': d, 't': t, 'cf': amt})
            
        return pd.DataFrame(flows)

    def calculate_accrued_interest(self, settle_date, flows, coupon_rate):
        """Calculates Exact Accrued Interest independent of Yield."""
        if flows.empty: return 0.0
        
        # Find the "Previous Coupon Date"
        # Logic: It is exactly 1 period before the Next Coupon Date (flows[0])
        next_date = flows['date'].iloc[0]
        days_in_period = 360 // self.freq if self.dc_method == "30/360" else 365 // self.freq
        
        # Days accrued = Days in Period - Days remaining to next payment
        # (This is a robust approximation for generic tools)
        days_remaining = self._days_between(settle_date, next_date)
        days_accrued = max(0, days_in_period - days_remaining)
        
        # Fraction
        fraction = days_accrued / days_in_period
        
        # Accrued Amount
        cpn_amt = (self.fv * coupon_rate) / self.freq
        return cpn_amt * fraction

    def calculate_metrics_from_yield(self, flows, ytm, accrued_val):
        """Calculates Price given Yield"""
        if flows.empty: return 0, 0, 0, 0, 0
        
        # Discount Factors
        # PV = CF / (1 + y/f)^(t*f)
        flows['df'] = 1 / ((1 + ytm/self.freq) ** (flows['t'] * self.freq))
        flows['pv'] = flows['cf'] * flows['df']
        
        dirty_price = flows['pv'].sum()
        clean_price = dirty_price - accrued_val
        
        # Risk Metrics
        mac_num = (flows['t'] * flows['pv']).sum()
        mac_dur = mac_num / dirty_price if dirty_price else 0
        mod_dur = mac_dur / (1 + ytm/self.freq)
        
        # Convexity & PV01 (Perturbation Method)
        dy = 0.0001
        p_up = (flows['cf'] / ((1 + (ytm+dy)/self.freq) ** (flows['t'] * self.freq))).sum()
        p_down = (flows['cf'] / ((1 + (ytm-dy)/self.freq) ** (flows['t'] * self.freq))).sum()
        
        convexity = (p_down + p_up - 2 * dirty_price) / (dirty_price * dy**2)
        pv01 = (p_down - p_up) / 2
        
        return clean_price, dirty_price, mod_dur, convexity, pv01

    def solve_yield_from_price(self, flows, target_clean_price, accrued_val):
        """Solves for YTM given a Clean Price"""
        target_dirty = target_clean_price + accrued_val
        
        def objective(y):
            # Calculate PV at yield 'y'
            if y <= -1.0: return 99999 # Prevent div by zero
            pv = (flows['cf'] / ((1 + y/self.freq) ** (flows['t'] * self.freq))).sum()
            return pv - target_dirty
        
        try:
            # Newton solver starting at 5%
            return newton(objective, 0.05, tol=1e-5, maxiter=50)
        except:
            return 0.0

    @staticmethod
    def nss_model(t, b0, b1, b2, tau):
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

# --- 1. SESSION STATE FIX (Prevents KeyError) ---
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

# Validate Portfolio Structure (Auto-fix for old versions)
if len(st.session_state['portfolio']) > 0:
    first_item = st.session_state['portfolio'][0]
    if 'Duration' not in first_item: 
        st.session_state['portfolio'] = [] # Clear corrupted data
        st.toast("⚠️ Portfolio cleared to update to new format.", icon="🔄")

# CSS Styling
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #1e272e; padding: 15px; border-radius: 8px; border-left: 5px solid #00d2d3;}
    h1, h2, h3, h4 {color: #f5f6fa; font-family: 'Roboto', sans-serif;}
    .stDataFrame {border: 1px solid #353b48;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Global Settings")
    dc_select = st.selectbox("Day Count Convention", ["30/360 (Corporate)", "Actual/365 (G-Sec)"])
    dc_method = "30/360" if "30/360" in dc_select else "Actual/365"
    st.info(f"Active Logic: **{dc_method}**")

# Header
st.title("🏛️ Friday Master Terminal")
st.caption("Institutional Analytics | Pricing | Risk | Credit | Simulation")

# Tabs
tabs = st.tabs(["📊 Trading Desk", "💼 Portfolio", "📉 Credit & Curve", "🧪 Quant Lab"])

# Init Engine
quant = FridayQuantEngine(face_value=100, freq=2, day_count_method=dc_method)

# ==========================================
# TAB 1: TRADING DESK
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
            eff_cpn = spread # For flow generation, pass spread
            eff_rate_calc = ref_rate + spread # For accrued calc
            is_float = True
            disp_cpn = f"MIBOR + {spread*10000:.0f} bps"
        else:
            eff_cpn = st.number_input("Coupon Rate (%)", 7.50, step=0.1)/100
            eff_rate_calc = eff_cpn
            is_float = False
            ref_rate = 0.0
            disp_cpn = f"{eff_cpn*100:.2f}%"
            
        mat_date = st.date_input("Maturity", datetime.date.today() + datetime.timedelta(days=365*5))
        settle_date = st.date_input("Settlement", datetime.date.today() + datetime.timedelta(days=1))
        
        st.subheader("2. Market Data")
        mode = st.radio("Calculation Mode", ["Price from Yield", "Yield from Price"])
        
        if "Price from Yield" in mode:
            user_input = st.number_input("Input YTM (%)", 7.20, step=0.05)/100
        else:
            user_input = st.number_input("Input Clean Price (₹)", 100.00, step=0.1)

    with c_right:
        # --- CALCULATION CORE ---
        flows = quant.get_cash_flows(mat_date, settle_date, eff_cpn, is_float, ref_rate)
        
        # Init variables to safe defaults
        final_ytm = 0.0; cln = 0.0; dty = 0.0; acc = 0.0; md = 0.0; cv = 0.0; pv01 = 0.0
        
        if not flows.empty:
            # 1. Calculate Accrued Interest First (Independent of Yield)
            acc = quant.calculate_accrued_interest(settle_date, flows, eff_rate_calc)
            
            # 2. Branch Logic
            if "Price from Yield" in mode:
                final_ytm = user_input
                cln, dty, md, cv, pv01 = quant.calculate_metrics_from_yield(flows, final_ytm, acc)
            else:
                # Yield from Price Solver
                target_price = user_input
                final_ytm = quant.solve_yield_from_price(flows, target_price, acc)
                # Recalculate metrics based on solved yield to get Duration/Risk
                cln, dty, md, cv, pv01 = quant.calculate_metrics_from_yield(flows, final_ytm, acc)

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
            shocks = [-50, -25, -10, 0, 10, 25, 50]
            sim_data = []
            for s in shocks:
                sy = final_ytm + (s/10000)
                s_cln, _, _, _, _ = quant.calculate_metrics_from_yield(flows, sy, acc)
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
                        "Type": bond_type, 
                        "Maturity": str(mat_date), 
                        "YTM": f"{final_ytm*100:.2f}%", 
                        "Price": float(cln), 
                        "Duration": float(md)
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
        
        # Check if Dataframe has correct columns (Extra safety)
        if 'Price' in df_p.columns and 'Duration' in df_p.columns:
            st.dataframe(df_p, use_container_width=True)
            
            st.markdown("#### Aggregate Risk")
            # Safe calculation
            p_dur = df_p['Duration'].astype(float).mean()
            p_val = df_p['Price'].astype(float).sum()
            
            c1, c2 = st.columns(2)
            c1.metric("Total Market Value", f"₹ {p_val:,.2f}")
            c2.metric("Avg Portfolio Duration", f"{p_dur:.2f}")
            
            if st.button("Clear Portfolio"):
                st.session_state['portfolio'] = []
                st.experimental_rerun()
        else:
            st.error("Data structure error. Clearing portfolio...")
            st.session_state['portfolio'] = []
            st.experimental_rerun()
    else:
        st.info("Portfolio is empty. Go to 'Trading Desk' to add bonds.")

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
        
        def rf_func(t): return quant.nss_model(t, b0, b1, b2, tau)

    with c_z:
        st.markdown("### 🕵️ Z-Spread Solver")
        st.caption("Calculate Credit Premium over the NSS Curve")
        z_price = st.number_input("Corp Bond Price", 98.0)
        z_cpn = st.number_input("Corp Coupon (%)", 8.5) / 100
        
        if st.button("Calculate Spread"):
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
