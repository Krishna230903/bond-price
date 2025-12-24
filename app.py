import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import newton
from fpdf import FPDF
import base64

# ==========================================
# CORE ANALYTICS ENGINE
# ==========================================
class FixedIncomeEngine:
    def __init__(self, face_value=100, freq=2, day_count_method="30/360"):
        self.fv = face_value
        self.freq = freq
        self.dc_method = day_count_method

    def _days_between(self, d1, d2):
        if self.dc_method == "30/360":
            d1_day = min(30, d1.day)
            d2_day = min(30, d2.day) if d1_day == 30 else d2.day
            return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (d2_day - d1_day)
        else:
            return (d2 - d1).days

    def get_cash_flows(self, maturity_date, settle_date, coupon_rate, is_floating=False, ref_rate=0.0):
        if settle_date >= maturity_date: return pd.DataFrame()
        
        dates = []
        curr = maturity_date
        while curr > settle_date:
            dates.append(curr)
            curr = curr - datetime.timedelta(days=365//self.freq)
        dates.sort()
        
        eff_rate = (ref_rate + coupon_rate) if is_floating else coupon_rate
        cpn_amt = (self.fv * eff_rate) / self.freq
        
        flows = []
        denominator = 360.0 if self.dc_method == "30/360" else 365.0
        
        for i, d in enumerate(dates):
            amt = cpn_amt
            if i == len(dates) - 1: amt += self.fv
            days = self._days_between(settle_date, d)
            t = days / denominator
            flows.append({'date': d, 't': t, 'cf': amt})
            
        return pd.DataFrame(flows)

    def calculate_accrued_interest(self, settle_date, flows, coupon_rate):
        if flows.empty: return 0.0
        next_date = flows['date'].iloc[0]
        days_in_period = 360 // self.freq if self.dc_method == "30/360" else 365 // self.freq
        days_remaining = self._days_between(settle_date, next_date)
        days_accrued = max(0, days_in_period - days_remaining)
        fraction = days_accrued / days_in_period
        cpn_amt = (self.fv * coupon_rate) / self.freq
        return cpn_amt * fraction

    def calculate_metrics_from_yield(self, flows, ytm, accrued_val):
        if flows.empty: return 0, 0, 0, 0, 0
        
        # Discount Factors
        flows['df'] = 1 / ((1 + ytm/self.freq) ** (flows['t'] * self.freq))
        flows['pv'] = flows['cf'] * flows['df']
        
        dirty_price = flows['pv'].sum()
        clean_price = dirty_price - accrued_val
        
        # Risk Metrics
        mac_num = (flows['t'] * flows['pv']).sum()
        mac_dur = mac_num / dirty_price if dirty_price else 0
        mod_dur = mac_dur / (1 + ytm/self.freq)
        
        # Convexity & PV01
        dy = 0.0001
        p_up = (flows['cf'] / ((1 + (ytm+dy)/self.freq) ** (flows['t'] * self.freq))).sum()
        p_down = (flows['cf'] / ((1 + (ytm-dy)/self.freq) ** (flows['t'] * self.freq))).sum()
        
        convexity = (p_down + p_up - 2 * dirty_price) / (dirty_price * dy**2)
        pv01 = (p_down - p_up) / 2
        
        return clean_price, dirty_price, mod_dur, convexity, pv01

    def solve_yield_from_price(self, flows, target_clean_price, accrued_val):
        target_dirty = target_clean_price + accrued_val
        def objective(y):
            if y <= -1.0: return 99999 
            pv = (flows['cf'] / ((1 + y/self.freq) ** (flows['t'] * self.freq))).sum()
            return pv - target_dirty
        try:
            return newton(objective, 0.05, tol=1e-5, maxiter=50)
        except:
            return 0.0

    def z_spread_solver(self, flows, market_price, rf_curve_func):
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
        term1 = (1 - np.exp(-t/tau)) / (t/tau)
        term2 = term1 - np.exp(-t/tau)
        return b0 + b1*term1 + b2*term2

# ==========================================
# REPORTING ENGINE
# ==========================================
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Fixed Income Valuation Report", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for key, value in data.items():
        pdf.cell(100, 10, f"{key}: {value}", 0, 1)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# USER INTERFACE
# ==========================================
st.set_page_config(page_title="Fixed Income Analytics", layout="wide")

# BLUE THEME CSS
st.markdown("""
<style>
    /* Main Background - Deep Navy */
    .stApp {background-color: #0c1829; color: #e6f1ff;}
    
    /* Headers - Light Blue */
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; color: #64b5f6 !important;}
    
    /* Metrics - Darker Blue Cards */
    .stMetric {background-color: #16263b !important; border: 1px solid #2c4a70 !important; border-radius: 4px;}
    div[data-testid="stMetricValue"] {font-size: 24px; color: #64b5f6 !important;}
    div[data-testid="stMetricLabel"] {font-size: 14px; color: #b0bec5 !important;}
    
    /* Tabs */
    button[data-baseweb="tab"] {color: #b0bec5; font-weight: bold;}
    button[data-baseweb="tab"][aria-selected="true"] {color: #64b5f6; border-bottom-color: #64b5f6;}
    
    /* Inputs */
    .stTextInput>div>div>input {color: white; background-color: #16263b;}
    .stNumberInput>div>div>input {color: white; background-color: #16263b;}
    
    /* Dataframes */
    .stDataFrame {border: 1px solid #2c4a70;}
    
    /* Expander */
    .streamlit-expanderHeader {color: #e6f1ff !important; background-color: #16263b !important;}
</style>
""", unsafe_allow_html=True)

# Session State
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

# Top Bar
with st.expander("Terminal Settings", expanded=False):
    dc_select = st.selectbox("Day Count Convention", ["30/360 (Corporate)", "Actual/365 (Government)"])
    dc_method = "30/360" if "30/360" in dc_select else "Actual/365"

st.title("Fixed Income Analytics Terminal")
st.markdown("Institutional Valuation | Risk Analysis | Credit Modeling")

# Initialize Engine
quant = FixedIncomeEngine(face_value=100, freq=2, day_count_method=dc_method)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Pricing & Risk", "Portfolio Manager", "Credit & Curve", "Simulation"])

# --- TAB 1: PRICING ---
with tab1:
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.subheader("Security Definition")
        bond_type = st.selectbox("Structure", ["Fixed Rate Bond", "Floating Rate Note"])
        
        if bond_type == "Floating Rate Note":
            ref_rate = st.number_input("Reference Rate (%)", 6.50)/100
            spread = st.number_input("Spread (bps)", 100.0)/10000
            eff_cpn = spread
            eff_rate_calc = ref_rate + spread
            is_float = True
        else:
            eff_cpn = st.number_input("Coupon Rate (%)", 7.50, step=0.1)/100
            eff_rate_calc = eff_cpn
            is_float = False
            ref_rate = 0.0
            
        mat_date = st.date_input("Maturity Date", datetime.date.today() + datetime.timedelta(days=365*5))
        settle_date = st.date_input("Settlement Date", datetime.date.today() + datetime.timedelta(days=1))
        
        st.subheader("Market Inputs")
        calc_mode = st.radio("Calculation Target", ["Price (from Yield)", "Yield (from Price)"])
        
        if "Price" in calc_mode:
            user_val = st.number_input("Yield to Maturity (%)", 7.20, step=0.05)/100
        else:
            user_val = st.number_input("Clean Price", 100.00, step=0.1)

    with col_output:
        st.subheader("Valuation Output")
        flows = quant.get_cash_flows(mat_date, settle_date, eff_cpn, is_float, ref_rate)
        
        if not flows.empty:
            acc = quant.calculate_accrued_interest(settle_date, flows, eff_rate_calc)
            
            if "Price" in calc_mode:
                final_ytm = user_val
                cln, dty, md, cv, pv01 = quant.calculate_metrics_from_yield(flows, final_ytm, acc)
            else:
                final_ytm = quant.solve_yield_from_price(flows, user_val, acc)
                cln, dty, md, cv, pv01 = quant.calculate_metrics_from_yield(flows, final_ytm, acc)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Clean Price", f"{cln:,.2f}")
            m2.metric("Accrued Interest", f"{acc:,.2f}")
            m3.metric("Invoice Price", f"{dty:,.2f}")
            
            st.markdown("---")
            st.subheader("Risk Metrics")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Yield", f"{final_ytm*100:.3f}%")
            r2.metric("Mod Duration", f"{md:.2f}")
            r3.metric("Convexity", f"{cv:.2f}")
            r4.metric("PV01", f"{pv01:.4f}")
            
            st.subheader("Sensitivity Matrix")
            shocks = [-50, -25, -10, 0, 10, 25, 50]
            sim_data = []
            for s in shocks:
                sy = final_ytm + (s/10000)
                s_cln, _, _, _, _ = quant.calculate_metrics_from_yield(flows, sy, acc)
                sim_data.append({
                    "Shift (bps)": s,
                    "Yield (%)": f"{sy*100:.2f}",
                    "Price": f"{s_cln:.2f}",
                    "P&L": s_cln - cln
                })
            
            df_sim = pd.DataFrame(sim_data)
            st.dataframe(df_sim.style.background_gradient(subset=['P&L'], cmap='RdYlGn'), use_container_width=True)

            # Actions
            c_btn1, c_btn2, c_btn3 = st.columns(3)
            with c_btn1:
                if st.button("Add to Portfolio"):
                    st.session_state['portfolio'].append({
                        "Maturity": str(mat_date),
                        "Coupon": f"{eff_cpn*100:.2f}%",
                        "Yield": f"{final_ytm*100:.2f}%",
                        "Price": cln,
                        "Duration": md
                    })
                    st.success("Position Added")
            with c_btn2:
                pdf_bytes = create_pdf({"Maturity": str(mat_date), "Price": f"{cln:.2f}", "Yield": f"{final_ytm*100:.2f}%"})
                st.download_button("Download Report", data=pdf_bytes, file_name="valuation_report.pdf")
            with c_btn3:
                csv = flows.to_csv().encode('utf-8')
                st.download_button("Download Cash Flows", data=csv, file_name="cash_flows.csv")

# --- TAB 2: PORTFOLIO ---
with tab2:
    st.subheader("Current Holdings")
    if st.session_state['portfolio']:
        df_port = pd.DataFrame(st.session_state['portfolio'])
        st.dataframe(df_port, use_container_width=True)
        
        if 'Price' in df_port.columns and 'Duration' in df_port.columns:
            total_mv = df_port['Price'].sum()
            avg_dur = df_port['Duration'].mean()
            
            k1, k2 = st.columns(2)
            k1.metric("Total Market Value", f"{total_mv:,.2f}")
            k2.metric("Portfolio Duration", f"{avg_dur:.2f}")
            
        if st.button("Clear Portfolio"):
            st.session_state['portfolio'] = []
            st.experimental_rerun()
    else:
        st.info("No active positions.")

# --- TAB 3: CREDIT & CURVE ---
with tab3:
    c_curve, c_spread = st.columns(2)
    
    with c_curve:
        st.subheader("NSS Curve Construction")
        b0 = st.slider("Beta 0", 0.0, 0.15, 0.07)
        b1 = st.slider("Beta 1", -0.05, 0.05, -0.02)
        b2 = st.slider("Beta 2", -0.05, 0.05, 0.01)
        tau = st.slider("Tau", 0.5, 5.0, 2.0)
        
        t_seq = np.linspace(0.1, 30, 100)
        y_seq = quant.nss_model(t_seq, b0, b1, b2, tau) * 100
        
        fig, ax = plt.subplots(figsize=(10, 4))
        # Blue Theme Chart Colors
        fig.patch.set_facecolor('#0c1829')
        ax.set_facecolor('#0c1829')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        ax.plot(t_seq, y_seq, color='#00e5ff', linewidth=2)
        ax.set_title("Sovereign Yield Curve")
        ax.grid(True, alpha=0.2, color='white')
        st.pyplot(fig)
        
        def rf_func(t): return quant.nss_model(t, b0, b1, b2, tau)

    with c_spread:
        st.subheader("Z-Spread Solver")
        z_price = st.number_input("Market Price", 98.0)
        z_cpn = st.number_input("Coupon (%)", 8.5)/100
        
        if st.button("Calculate Z-Spread"):
            z_flows = quant.get_cash_flows(datetime.date.today()+datetime.timedelta(days=1825), 
                                           datetime.date.today()+datetime.timedelta(days=1), z_cpn)
            z_spread = quant.z_spread_solver(z_flows, z_price, rf_func)
            st.metric("Z-Spread", f"{z_spread*10000:.0f} bps")

# --- TAB 4: SIMULATION ---
with tab4:
    st.subheader("Vasicek Rate Simulation")
    c_sim1, c_sim2 = st.columns([1, 3])
    
    with c_sim1:
        v_r0 = st.number_input("Initial Rate", 0.06)
        v_theta = st.number_input("Long Term Mean", 0.07)
        v_sigma = st.number_input("Volatility", 0.02)
        v_paths = st.slider("Paths", 10, 500, 100)
        
    with c_sim2:
        if st.button("Run Simulation"):
            dt = 1/252; T = 5; N = int(T/dt)
            rates = np.zeros((v_paths, N))
            rates[:,0] = v_r0
            kappa = 0.5
            for t in range(1, N):
                dr = kappa*(v_theta - rates[:,t-1])*dt + v_sigma*np.sqrt(dt)*np.random.normal(0,1,v_paths)
                rates[:,t] = rates[:,t-1] + dr
                
            fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
            
            # Blue Theme Chart Colors
            fig_mc.patch.set_facecolor('#0c1829')
            ax_mc.set_facecolor('#0c1829')
            ax_mc.tick_params(colors='white')
            ax_mc.xaxis.label.set_color('white')
            ax_mc.yaxis.label.set_color('white')
            ax_mc.title.set_color('white')
            ax_mc.spines['bottom'].set_color('white')
            ax_mc.spines['top'].set_color('white')
            ax_mc.spines['left'].set_color('white')
            ax_mc.spines['right'].set_color('white')

            ax_mc.plot(np.linspace(0, T, N), rates.T, alpha=0.1, color='#00e5ff')
            ax_mc.plot(np.linspace(0, T, N), rates.mean(axis=0), color='white', linewidth=2)
            ax_mc.set_title("Interest Rate Paths")
            st.pyplot(fig_mc)
