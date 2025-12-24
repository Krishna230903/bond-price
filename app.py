import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# ==========================================
# 🧠 FRIDAY'S INDIA FINANCIAL ENGINE
# ==========================================
class IndianBondAnalytics:
    def __init__(self, face_value, coupon_rate, years, frequency, day_count_method="30/360"):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years = years
        self.frequency = frequency
        self.day_count_method = day_count_method
        
        # Standardize periods
        self.total_periods = int(self.years * self.frequency)
        self.coupon_amount = (self.face_value * self.coupon_rate) / self.frequency

    def calculate_accrued_interest(self, days_held):
        """
        Calculates Accrued Interest based on Indian Day Count Conventions.
        - 30/360: Standard for Corporate Bonds.
        - Actual/365: Standard for G-Secs.
        """
        if self.day_count_method == "30/360":
            days_in_year = 360
        else: # Actual/365 (G-Sec)
            days_in_year = 365
            
        # Accrued = (Days / Days in Year) * (Coupon * Frequency) ? 
        # Actually simpler: Accrued = (Days Held / Days in Period) * Coupon Payment
        
        days_in_period = days_in_year / self.frequency
        fraction = days_held / days_in_period
        return fraction * self.coupon_amount, fraction

    def price_from_yield(self, ytm_percent, days_held=0):
        """Calculates CLEAN Price given a YTM."""
        r = ytm_percent / self.frequency
        accrued_val, fraction_period = self.calculate_accrued_interest(days_held)
        
        # Time remaining for each cash flow
        # If we held for 30 days, next coupon is closer.
        time_to_flows = np.arange(1, self.total_periods + 1) - fraction_period
        
        # Cash Flows
        cash_flows = np.full(self.total_periods, self.coupon_amount)
        cash_flows[-1] += self.face_value
        
        # Dirty Price (PV of future cash flows)
        dirty_price = np.sum(cash_flows * ((1 + r) ** -time_to_flows))
        
        # Clean Price = Dirty - Accrued
        clean_price = dirty_price - accrued_val
        return clean_price, dirty_price, accrued_val

    def yield_from_price(self, market_clean_price, days_held=0):
        """Solves for YTM (XIRR equivalent) given a Market Price."""
        guess = self.coupon_rate
        for _ in range(50): # Newton-Raphson limit
            price_est, _, _ = self.price_from_yield(guess, days_held)
            diff = price_est - market_clean_price
            if abs(diff) < 1e-6: return guess
            
            # Derivative
            shock_price, _, _ = self.price_from_yield(guess + 0.0001, days_held)
            derivative = (shock_price - price_est) / 0.0001
            guess = guess - (diff / derivative)
        return guess

# ==========================================
# 🇮🇳 UI CONFIGURATION (No Sidebar)
# ==========================================
st.set_page_config(page_title="Friday's India Bond Terminal", layout="wide", page_icon="🇮🇳")

# CSS for "Clean Professional" Look
st.markdown("""
<style>
    /* Remove Sidebar spacing */
    .css-1d391kg {padding-top: 0rem;}
    
    /* Metrics Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .big-font {font-size: 20px; font-weight: 600; color: #1a73e8;}
    
    /* Section Headers */
    .section-head {
        font-size: 18px; 
        font-weight: bold; 
        color: #333; 
        border-bottom: 2px solid #FF9933; /* Saffron color */
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("🇮🇳 India Fixed Income Terminal")
    st.caption("Advanced Analytics for G-Secs, SDLs, and Corporate Bonds.")
with c_head2:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=60)

# --- 1. TOP RIBBON: BOND INPUTS ---
st.markdown('<div class="section-head">1. Security Definition (Face Value & Coupon)</div>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        bond_type = st.selectbox("Bond Type", ["Corporate Bond", "Govt Security (G-Sec)"])
        # Auto-select day count based on type
        dc_method = "Actual/365" if "Govt" in bond_type else "30/360"
        
    with col2:
        fv = st.number_input("Face Value (₹)", value=1000, step=100, help="Usually ₹1,000 for Retail, ₹10 Lakhs for Private.")
    
    with col3:
        cr = st.number_input("Coupon Rate (%)", value=7.50, step=0.10) / 100
        
    with col4:
        mat = st.number_input("Years to Maturity", value=5.0, step=0.5)
        
    with col5:
        freq = st.selectbox("Frequency", [1, 2], format_func=lambda x: "Annual" if x==1 else "Semi-Annual")

# Initialize Engine
bond = IndianBondAnalytics(fv, cr, mat, freq, dc_method)

# --- 2. MARKET DATA & CALCULATIONS ---
st.markdown(f'<div class="section-head">2. Market Valuation (Day Count: {dc_method})</div>', unsafe_allow_html=True)

# Layout: Left side (Controls), Right side (Results)
c_left, c_right = st.columns([1, 2])

with c_left:
    st.info("👇 **Enter Market Data Here**")
    
    # Toggle Calculation Mode
    calc_mode = st.radio("I want to calculate:", ["Fair Price (from Yield)", "Implied Yield (from Price)"], horizontal=True)
    
    st.markdown("---")
    
    days_held = st.slider("Days Since Last Coupon", 0, 180, 0, help="Used for Accrued Interest calculation.")
    
    final_ytm = 0.0
    final_clean = 0.0
    
    if "Price" in calc_mode:
        user_ytm = st.number_input("Market Yield (YTM %)", value=7.20, step=0.05) / 100
        final_ytm = user_ytm
        final_clean, dirty, accrued = bond.price_from_yield(final_ytm, days_held)
    else:
        user_price = st.number_input("Clean Market Price (₹)", value=1005.00, step=1.0)
        final_clean = user_price
        final_ytm = bond.yield_from_price(final_clean, days_held)
        _, dirty, accrued = bond.price_from_yield(final_ytm, days_held)

with c_right:
    # RESULT CARDS
    rc1, rc2, rc3 = st.columns(3)
    
    with rc1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #666; font-size: 14px;">Clean Price (Quote)</div>
            <div style="font-size: 26px; font-weight: bold; color: #2c3e50;">₹{final_clean:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with rc2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #666; font-size: 14px;">Accrued Interest</div>
            <div style="font-size: 26px; font-weight: bold; color: #27ae60;">+ ₹{accrued:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with rc3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #666; font-size: 14px;">Total Invoice Price</div>
            <div style="font-size: 26px; font-weight: bold; color: #c0392b;">₹{dirty:,.2f}</div>
            <div style="font-size: 12px;">(What you pay)</div>
        </div>
        """, unsafe_allow_html=True)

    # Secondary Metrics Line
    st.write("") # Spacer
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Yield to Maturity (YTM)", f"{final_ytm*100:.3f}%")
    
    # Quick Check
    if final_clean < fv:
        sm2.warning(f"Discount Bond (Cheap)")
    elif final_clean > fv:
        sm2.success(f"Premium Bond (Expensive)")
    else:
        sm2.info("Par Bond")

# --- 3. CHARTS & SENSITIVITY ---
st.markdown('<div class="section-head">3. Analysis & Scenarios</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Sensitivity Graph", "🚨 Stress Test Matrix"])

with tab1:
    # Plotting Price vs Yield
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Generate range around current YTM
    y_range = np.linspace(max(0.01, final_ytm - 0.02), final_ytm + 0.02, 50)
    prices = [bond.price_from_yield(y, days_held)[0] for y in y_range]
    
    ax.plot(y_range*100, prices, color="#1a73e8", linewidth=2.5)
    ax.scatter([final_ytm*100], [final_clean], color="#e74c3c", s=100, zorder=5, label="Current Position")
    
    # Styling
    ax.set_title("Price Sensitivity to Interest Rates", fontsize=10)
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Price (₹)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    
    # Remove top and right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

with tab2:
    st.write("How does the price change if RBI changes rates?")
    
    # Horizontal Scenario Table
    scenarios = [-0.50, -0.25, 0.0, +0.25, +0.50]
    cols = st.columns(len(scenarios))
    
    for i, shock in enumerate(scenarios):
        new_y = final_ytm + (shock / 100)
        new_p, _, _ = bond.price_from_yield(new_y, days_held)
        diff = new_p - final_clean
        color = "green" if diff > 0 else "red"
        
        with cols[i]:
            st.metric(f"Rate {shock:+.2f}%", f"₹{new_p:,.0f}", f"{diff:+.0f}", delta_color="normal")
