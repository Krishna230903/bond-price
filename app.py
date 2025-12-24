import streamlit as st
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# ==========================================
# 🧠 FRIDAY'S "DATE-AWARE" ENGINE (Fixed Logic)
# ==========================================
class IndianBondEngine:
    def __init__(self, face_value, coupon_rate, maturity_date, settlement_date, frequency, day_count="30/360"):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_date = maturity_date
        self.settlement_date = settlement_date
        self.frequency = frequency
        self.day_count_method = day_count
        
        # Derived basics
        self.coupon_amt_per_period = (self.face_value * self.coupon_rate) / self.frequency

    def _days_between(self, d1, d2):
        """Calculates days based on convention."""
        if self.day_count_method == "30/360":
            # Indian Corporate / Standard 30/360 NASD logic
            d1_day = min(30, d1.day)
            d2_day = min(30, d2.day) if d1_day == 30 else d2.day
            return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2_day - d1_day)
        else:
            # Actual/365 or Actual/Actual (G-Sec simplified)
            return (d2 - d1).days

    def get_coupon_schedule(self):
        """Generates all FUTURE coupon dates from Settlement to Maturity."""
        if self.settlement_date >= self.maturity_date:
            return []

        # 1. Backtrack from Maturity Date to find all coupon dates
        # We go backwards to ensure the Maturity Date is a coupon date
        dates = []
        current_date = self.maturity_date
        
        # Step back in months (12/freq) until we pass the settlement date
        months_step = int(12 / self.frequency)
        
        while current_date > self.settlement_date:
            dates.append(current_date)
            current_date = current_date - relativedelta(months=months_step)
        
        # The list is [Maturity, Mat-6m, Mat-12m...]. Reverse it to be chronological.
        dates.reverse()
        return dates

    def calculate_metrics(self, ytm_percent):
        """Calculates Dirty Price, Clean Price, and Accrued Interest using Date Logic."""
        future_dates = self.get_coupon_schedule()
        
        if not future_dates:
            return 0.0, 0.0, 0.0, pd.DataFrame()

        # 1. Identify Coupon Boundaries
        next_coupon_date = future_dates[0]
        months_step = int(12 / self.frequency)
        prev_coupon_date = next_coupon_date - relativedelta(months=months_step)
        
        # 2. Calculate Accrued Interest Fractions
        # Fraction = (Settlement - Prev) / (Next - Prev)
        days_accrued = self._days_between(prev_coupon_date, self.settlement_date)
        days_in_period = self._days_between(prev_coupon_date, next_coupon_date)
        
        # Safety for division by zero or negative dates
        if days_in_period <= 0: days_in_period = 180 # Fallback
        
        fraction_accumulated = days_accrued / days_in_period
        accrued_interest = fraction_accumulated * self.coupon_amt_per_period

        # 3. Discounting Cash Flows
        # We discount to the SETTLEMENT date.
        # Time (t) for specific cash flow = (Date - Settlement) / 365 (or 360)
        
        df_data = []
        dirty_price = 0.0
        
        # Determine year denominator for discounting (usually market uses Act/365 for Yield)
        year_base = 360.0 if self.day_count_method == "30/360" else 365.0
        
        for i, date in enumerate(future_dates):
            # Cash Flow is Coupon, plus Face Value if it's the last date
            cf = self.coupon_amt_per_period
            if i == len(future_dates) - 1:
                cf += self.face_value
            
            # Time in years from settlement
            days_from_settle = self._days_between(self.settlement_date, date)
            t_years = days_from_settle / year_base
            
            # PV Calculation
            # Standard Street Convention: PV = CF / (1 + YTM/freq)^(n - fraction + i)
            # But let's use continuous compounding or standard annual periods for clarity in generic tool
            # Using: PV = CF / (1 + YTM)^t
            
            pv = cf / ((1 + ytm_percent) ** t_years)
            dirty_price += pv
            
            df_data.append({
                "Date": date,
                "Days Away": days_from_settle,
                "Cash Flow": cf,
                "PV Factor": 1 / ((1 + ytm_percent) ** t_years),
                "PV": pv
            })
            
        clean_price = dirty_price - accrued_interest
        
        return clean_price, dirty_price, accrued_interest, pd.DataFrame(df_data)

    def yield_solver(self, target_clean_price):
        """Solves for YTM given a Clean Price."""
        # Simple bisection search (more robust than Newton for generic dates)
        low = 0.0001
        high = 1.00 # 100%
        for _ in range(50):
            mid = (low + high) / 2
            p, _, _, _ = self.calculate_metrics(mid)
            if abs(p - target_clean_price) < 0.01:
                return mid
            if p > target_clean_price:
                low = mid # Need higher yield to lower price
            else:
                high = mid
        return (low + high) / 2

# ==========================================
# 🇮🇳 UI CONFIGURATION (Beautified)
# ==========================================
st.set_page_config(page_title="Friday's Pro Terminal", layout="wide", page_icon="📈")

# CSS for a High-End Fintech Look
st.markdown("""
<style>
    /* Global Background */
    .stApp {background-color: #f8f9fa;}
    
    /* Header Styling */
    h1 {color: #0f2c4a; font-family: 'Helvetica Neue', sans-serif;}
    .stMarkdown h3 {color: #444;}
    
    /* Input Cards */
    .input-card {
        background-color: white; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #ff9f43;
    }
    
    /* Result Cards */
    .result-metric {
        background: linear-gradient(135deg, #ffffff 0%, #f1f2f6 100%);
        border: 1px solid #dcdde1;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    .metric-value {font-size: 28px; font-weight: 700; color: #2f3542;}
    .metric-label {font-size: 14px; color: #747d8c; text-transform: uppercase; letter-spacing: 1px;}
    
    /* Table Styling */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("🇮🇳 India Fixed Income Analytics")
    st.caption("Professional Date-Aware Valuation (RBI / SEBI Standards)")
with col_h2:
    st.markdown("## 🗓️ " + datetime.date.today().strftime("%d %b %Y"))

st.markdown("---")

# --- 1. CONFIGURATION PANEL (Top Ribbon) ---
with st.container():
    # We use a container with a white background effect via CSS potential or just Streamlit columns
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.subheader("1. Bond Specs")
        fv = st.number_input("Face Value (₹)", 1000, step=100)
        cr = st.number_input("Coupon Rate (%)", value=7.50, step=0.1) / 100
        freq = st.selectbox("Frequency", [1, 2], format_func=lambda x: "Annual" if x==1 else "Semi-Annual")

    with c2:
        st.subheader("2. Dates")
        # Logic: Bond usually matures 5 years from now
        default_mat = datetime.date.today() + relativedelta(years=5)
        mat_date = st.date_input("Maturity Date", value=default_mat)
        
        settle_date = st.date_input("Settlement Date", value=datetime.date.today())

    with c3:
        st.subheader("3. Convention")
        bond_type = st.radio("Market Standard", ["G-Sec (Act/365)", "Corporate (30/360)"])
        dc_method = "Actual/365" if "G-Sec" in bond_type else "30/360"
        
    with c4:
        st.subheader("4. Mode")
        calc_mode = st.radio("Solver Mode", ["Price from Yield", "Yield from Price"])

# Initialize Engine
engine = IndianBondEngine(fv, cr, mat_date, settle_date, freq, dc_method)

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. MAIN CALCULATION BLOCK ---

# Variables to hold results
final_clean = 0.0
final_dirty = 0.0
final_accrued = 0.0
final_ytm = 0.0
schedule_df = pd.DataFrame()

# Perform Calculations based on Mode
if "Price from Yield" in calc_mode:
    # Input YTM -> Get Price
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        user_ytm = st.number_input("📉 Input Market Yield (YTM %)", value=7.25, step=0.05) / 100
        st.markdown('</div>', unsafe_allow_html=True)
        final_ytm = user_ytm
        final_clean, final_dirty, final_accrued, schedule_df = engine.calculate_metrics(final_ytm)
        
else:
    # Input Price -> Get Yield
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        user_price = st.number_input("💰 Input Clean Price (₹)", value=1005.00, step=0.50)
        st.markdown('</div>', unsafe_allow_html=True)
        final_clean = user_price
        final_ytm = engine.yield_solver(final_clean)
        final_clean, final_dirty, final_accrued, schedule_df = engine.calculate_metrics(final_ytm)

# --- 3. DISPLAY RESULTS (Full Width) ---

if not schedule_df.empty:
    with col_out:
        # Using custom HTML for cards to ensure they look "Beautified"
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            st.markdown(f"""
            <div class="result-metric">
                <div class="metric-label">Clean Price (Quote)</div>
                <div class="metric-value" style="color:#2980b9">₹{final_clean:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc2:
            st.markdown(f"""
            <div class="result-metric">
                <div class="metric-label">Accrued Interest</div>
                <div class="metric-value" style="color:#27ae60">+ ₹{final_accrued:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc3:
            st.markdown(f"""
            <div class="result-metric">
                <div class="metric-label">Invoice Price (Pay)</div>
                <div class="metric-value" style="color:#c0392b">₹{final_dirty:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"<div style='text-align:center; margin-top:10px; font-weight:bold; color:#555'>Yield to Maturity: {final_ytm*100:.3f}%</div>", unsafe_allow_html=True)

    # --- 4. DETAILS & CASH FLOWS ---
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Cash Flow Schedule", "📊 Valuation Analysis"])
    
    with tab1:
        st.write("This table proves the valuation by discounting every specific future date.")
        
        # Formatting for display
        display_df = schedule_df.copy()
        display_df['Date'] = display_df['Date'].apply(lambda x: x.strftime('%d-%b-%Y'))
        display_df['Cash Flow'] = display_df['Cash Flow'].apply(lambda x: f"₹{x:,.2f}")
        display_df['PV'] = display_df['PV'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Days Away'] = display_df['Days Away'].astype(int)
        display_df['PV Factor'] = display_df['PV Factor'].apply(lambda x: f"{x:.4f}")
        
        st.table(display_df)
        
    with tab2:
        c_chart, c_text = st.columns([2, 1])
        with c_chart:
            # Simple PV Bar Chart
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(schedule_df['Date'], schedule_df['PV'], color='#3498db', width=20)
            ax.set_title("Present Value of Each Future Payment")
            ax.set_ylabel("PV (₹)")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            # Format x-axis dates
            fig.autofmt_xdate()
            st.pyplot(fig)
            
        with c_text:
            st.info(f"""
            **Logic Check:**
            
            1. **Day Count:** {dc_method}
            2. **Next Coupon:** {schedule_df['Date'].iloc[0].strftime('%d-%b-%Y')}
            3. **Days Accrued:** {(schedule_df['Date'].iloc[0] - settle_date).days} days remaining until next coupon.
            """)

else:
    st.error("⚠️ Error: Settlement Date cannot be after Maturity Date.")
