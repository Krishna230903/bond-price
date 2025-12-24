import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==========================================
# 🧠 FRIDAY'S FINANCIAL ENGINE (BACKEND)
# ==========================================
class BondAnalytics:
    def __init__(self, face_value, coupon_rate, years, frequency=1, days_since_last_coupon=0):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years = years
        self.frequency = frequency
        self.days_since = days_since_last_coupon
        
        # Standardize time periods
        self.total_periods = int(self.years * self.frequency)
        self.coupon_amount = (self.face_value * self.coupon_rate) / self.frequency
        
        # Calculate Accrued Interest
        # (Days Since / Days in Period) * Coupon Amount
        days_in_period = 360 / self.frequency # Standard US Day Count Convention (30/360 approx)
        self.fraction_period = self.days_since / days_in_period
        self.accrued_interest = self.fraction_period * self.coupon_amount

    def price_from_yield(self, ytm_percent):
        """Calculates CLEAN Price given a YTM."""
        r = ytm_percent / self.frequency
        
        # Time remaining for each cash flow
        # If we are 30 days into a period, the next coupon comes sooner (1 - fraction)
        time_to_flows = np.arange(1, self.total_periods + 1) - self.fraction_period
        
        # Discount Factors
        discount_factors = (1 + r) ** -time_to_flows
        
        # Cash Flows
        cash_flows = np.full(self.total_periods, self.coupon_amount)
        cash_flows[-1] += self.face_value
        
        # Dirty Price (Present Value of all future cash flows)
        dirty_price = np.sum(cash_flows * discount_factors)
        
        # Clean Price = Dirty Price - Accrued Interest
        clean_price = dirty_price - self.accrued_interest
        return clean_price, dirty_price

    def yield_from_price(self, market_clean_price):
        """
        Solves for YTM given a Market Price using Newton-Raphson method.
        This effectively 'reverse engineers' the bond.
        """
        # Initial guess (Coupon Rate is usually a good starting point)
        guess = self.coupon_rate
        tolerance = 1e-6
        max_iter = 100
        
        for _ in range(max_iter):
            price_est, _ = self.price_from_yield(guess)
            diff = price_est - market_clean_price
            
            if abs(diff) < tolerance:
                return guess
            
            # Derivative approx (small shock) to find slope
            shock_price, _ = self.price_from_yield(guess + 0.0001)
            derivative = (shock_price - price_est) / 0.0001
            
            # Newton Step
            guess = guess - (diff / derivative)
            
        return guess # Return best guess if not converged

    def get_risk_metrics(self, ytm):
        """Calculates Duration and Convexity based on the YTM."""
        clean, dirty = self.price_from_yield(ytm)
        r = ytm / self.frequency
        
        # Cash Flow Arrays
        time_to_flows = np.arange(1, self.total_periods + 1) - self.fraction_period
        cash_flows = np.full(self.total_periods, self.coupon_amount)
        cash_flows[-1] += self.face_value
        
        pv_flows = cash_flows * ((1 + r) ** -time_to_flows)
        
        # Macaulay Duration (Weighted Average Time)
        # Sum( Time * PV ) / Total Price
        mac_duration = np.sum(time_to_flows * pv_flows) / dirty
        mac_duration_years = mac_duration / self.frequency
        
        # Modified Duration
        mod_duration = mac_duration_years / (1 + r)
        
        # Convexity
        # Sum ( CF / (1+y)^t * (t^2+t) )
        # Simplified approximate calculation for code robustness
        convexity_term = np.sum(pv_flows * (time_to_flows ** 2 + time_to_flows))
        convexity = convexity_term / (dirty * (1 + r)**2 * self.frequency**2)
        
        return mod_duration, convexity, mac_duration_years

# ==========================================
# 🖥️ STREAMLIT UI (FRONTEND)
# ==========================================
st.set_page_config(page_title="Friday's Master Terminal", layout="wide", page_icon="🏦")

# Custom Styling
st.markdown("""
<style>
    .metric-container {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #2e86de; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .highlight {color: #2e86de; font-weight: bold;}
    .big-num {font-size: 28px; font-weight: bold; color: #333;}
    .sub-text {font-size: 14px; color: #666;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Institutional Bond Pricing Terminal")
st.markdown("Advanced analytics including **Accrued Interest**, **Implied Yield Solving**, and **Risk Matrices**.")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("1. Bond Contract Specs")
    fv = st.number_input("Face Value ($)", value=1000, step=100)
    cr = st.number_input("Annual Coupon Rate (%)", value=5.0, step=0.1) / 100
    mat = st.number_input("Years to Maturity", value=10.0, step=0.5)
    freq = st.selectbox("Payment Frequency", [1, 2], format_func=lambda x: "Annual" if x==1 else "Semi-Annual")
    
    st.markdown("---")
    st.header("2. Transaction Date")
    days_since = st.number_input("Days Since Last Coupon", min_value=0, max_value=int(360/freq)-1, value=0,
                                help="Used to calculate Accrued Interest (Dirty Price).")
    
    st.markdown("---")
    st.info("💡 **Friday's Tip:** 'Clean Price' is what you see quoted. 'Dirty Price' is what you actually pay.")

# Initialize Engine
bond = BondAnalytics(fv, cr, mat, freq, days_since)

# --- MODE SELECTION ---
mode = st.radio("Select Analysis Mode:", ["Calculate Price (from Yield)", "Calculate Yield (from Price)"], horizontal=True)

st.markdown("---")

# ==========================================
# 📊 MAIN LOGIC BRANCHING
# ==========================================

final_ytm = 0.0
final_clean_price = 0.0

if mode == "Calculate Price (from Yield)":
    # INPUT: Yield -> OUTPUT: Price
    col_in, col_res = st.columns([1, 2])
    with col_in:
        user_ytm = st.number_input("Enter Market Yield (YTM %)", value=4.0, step=0.01) / 100
        final_ytm = user_ytm
        final_clean_price, dirty_price = bond.price_from_yield(final_ytm)
        
    with col_res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean Price (Quote)", f"${final_clean_price:,.2f}")
        c2.metric("Accrued Interest", f"+ ${bond.accrued_interest:,.2f}")
        c3.metric("Invoice Price (Dirty)", f"${dirty_price:,.2f}", delta="Amount to Pay")

else:
    # INPUT: Price -> OUTPUT: Yield
    col_in, col_res = st.columns([1, 2])
    with col_in:
        user_price = st.number_input("Enter Market Price ($)", value=1050.0, step=10.0)
        final_clean_price = user_price
        
        # Run Solver
        final_ytm = bond.yield_from_price(final_clean_price)
        _, dirty_price = bond.price_from_yield(final_ytm)
        
    with col_res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Implied YTM (Yield)", f"{final_ytm*100:.3f}%", delta="Annual Return")
        c2.metric("Accrued Interest", f"${bond.accrued_interest:,.2f}")
        c3.metric("Invoice Price (Dirty)", f"${dirty_price:,.2f}")

# ==========================================
# 📉 RISK & SENSITIVITY ANALYSIS
# ==========================================
st.subheader("Risk Analytics")

mod_d, conv, mac_d = bond.get_risk_metrics(final_ytm)

# Risk Cards
r1, r2, r3, r4 = st.columns(4)
r1.info(f"**Modified Duration:** {mod_d:.2f}\n\n(1% Rate Hike = {mod_d:.2f}% Price Drop)")
r2.info(f"**Convexity:** {conv:.2f}\n\n(Curvature Adjustment)")
r3.info(f"**Macaulay Duration:** {mac_d:.2f} Yrs\n\n(Time to recover capital)")
r4.info(f"**DV01:** ${mod_d * final_clean_price * 0.0001:,.3f}\n\n(Value of 1 basis point)")

# --- SENSITIVITY MATRIX ---
st.markdown("### Sensitivity Matrix (Price Scenarios)")
with st.expander("Open Price Scenario Table", expanded=True):
    # Create a range of Yields (+/- 1%)
    base_bps = final_ytm * 10000
    scenarios = [-100, -50, -25, 0, 25, 50, 100]
    
    matrix_data = []
    for s in scenarios:
        test_y = (base_bps + s) / 10000
        p, _ = bond.price_from_yield(test_y)
        change_pct = (p - final_clean_price) / final_clean_price * 100
        matrix_data.append({
            "Shift (bps)": f"{s:+}",
            "New Yield": f"{test_y*100:.3f}%",
            "Est. Price": f"${p:,.2f}",
            "% Change": f"{change_pct:+.2f}%",
            "P/L ($)": f"${p - final_clean_price:,.2f}"
        })
    
    st.table(pd.DataFrame(matrix_data).set_index("Shift (bps)"))

# --- VISUALIZATION ---
st.markdown("### Price-Yield Curve Analysis")
fig, ax = plt.subplots(figsize=(10, 4))
y_range = np.linspace(max(0.001, final_ytm-0.03), final_ytm+0.03, 100)
p_curve = [bond.price_from_yield(y)[0] for y in y_range]

ax.plot(y_range*100, p_curve, color="#2e86de", linewidth=2, label="Price Curve")
ax.scatter([final_ytm*100], [final_clean_price], color="red", s=100, zorder=5, label="Current Bond")

# Annotations
ax.set_title("Bond Price vs Market Yield", fontsize=12)
ax.set_xlabel("Yield (%)")
ax.set_ylabel("Clean Price ($)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

st.pyplot(fig)
