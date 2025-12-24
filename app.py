import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE MATH ENGINE (Backend Logic) ---
class BondAnalytics:
    def __init__(self, face_value, coupon_rate, ytm, years, frequency=1):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.ytm = ytm
        self.years = years
        self.frequency = frequency
        
        self.periods = self.years * self.frequency
        self.coupon_payment = (self.face_value * self.coupon_rate) / self.frequency
        self.rate_per_period = self.ytm / self.frequency

    def generate_cash_flows(self):
        periods_arr = np.arange(1, self.periods + 1)
        cash_flows = np.full(self.periods, self.coupon_payment)
        cash_flows[-1] += self.face_value
        
        df = pd.DataFrame({
            'Period': periods_arr,
            'Time (Years)': periods_arr / self.frequency,
            'Cash Flow': cash_flows
        })
        return df

    def calculate_metrics(self):
        df = self.generate_cash_flows()
        df['PV Factor'] = (1 + self.rate_per_period) ** -df['Period']
        df['PV of Cash Flow'] = df['Cash Flow'] * df['PV Factor']
        
        price = df['PV of Cash Flow'].sum()
        
        # Macaulay Duration
        numerator_mac = (df['Period'] * df['PV of Cash Flow']).sum()
        mac_duration_periods = numerator_mac / price if price != 0 else 0
        mac_duration_years = mac_duration_periods / self.frequency
        
        # Modified Duration
        mod_duration = mac_duration_years / (1 + self.rate_per_period)
        
        # Convexity
        term = df['PV of Cash Flow'] * (df['Period'] ** 2 + df['Period'])
        convexity_adj = term.sum()
        convexity = convexity_adj / (price * (1 + self.rate_per_period)**2 * (self.frequency**2))

        return price, mac_duration_years, mod_duration, convexity, df

    def get_curve_data(self):
        ytm_range = np.linspace(0.001, 0.20, 100) # 0.1% to 20%
        prices = []
        original_ytm = self.ytm
        
        for y in ytm_range:
            self.ytm = y
            self.rate_per_period = y / self.frequency
            # Quick calc for price only
            df = self.generate_cash_flows()
            df['PV Factor'] = (1 + self.rate_per_period) ** -df['Period']
            prices.append((df['Cash Flow'] * df['PV Factor']).sum())
            
        # Reset
        self.ytm = original_ytm
        self.rate_per_period = original_ytm / self.frequency
        
        return ytm_range * 100, prices

# --- 2. THE APP INTERFACE (Streamlit) ---
st.set_page_config(page_title="Friday's Bond Analyzer", layout="wide")

st.title("📊 Advanced Bond Pricing & Analytics")
st.markdown("Adjust the inputs in the sidebar to value the bond and analyze its risk.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Bond Characteristics")

face_value = st.sidebar.number_input("Face Value ($)", value=1000, step=100)
coupon_rate = st.sidebar.slider("Annual Coupon Rate (%)", 0.0, 15.0, 5.0, 0.1) / 100
years = st.sidebar.slider("Years to Maturity", 1, 30, 10, 1)
frequency = st.sidebar.selectbox("Payment Frequency", options=[1, 2], format_func=lambda x: "Annual" if x==1 else "Semi-Annual")

st.sidebar.header("Market Conditions")
ytm = st.sidebar.slider("Yield to Maturity (YTM %)", 0.0, 20.0, 4.0, 0.1) / 100

# --- CALCULATIONS ---
bond = BondAnalytics(face_value, coupon_rate, ytm, years, frequency)
price, mac_d, mod_d, conv, df_cashflows = bond.calculate_metrics()

# --- MAIN DASHBOARD ---

# Row 1: Key Metrics using st.metric for nice visual cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Bond Price", value=f"${price:,.2f}", 
              delta=f"{price - face_value:,.2f} vs Par")
with col2:
    st.metric(label="Macaulay Duration", value=f"{mac_d:.2f} Yrs")
with col3:
    st.metric(label="Modified Duration", value=f"{mod_d:.2f}")
with col4:
    st.metric(label="Convexity", value=f"{conv:.2f}")

st.markdown("---")

# Row 2: Charts and Data
col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.subheader("Price-Yield Sensitivity Curve")
    
    # Generate Plot
    yields, prices = bond.get_curve_data()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yields, prices, label='Price Curve', color='#1f77b4', linewidth=2)
    
    # Plot current point
    ax.scatter([ytm * 100], [price], color='red', s=100, zorder=5, label='Current Position')
    
    # Styling
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Price ($)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    
    st.info(f"💡 **Insight:** Since Modified Duration is **{mod_d:.2f}**, if market rates go UP by 1%, this bond's price will drop by approximately **{mod_d:.2f}%**.")

with col_data:
    st.subheader("Cash Flow Schedule")
    st.dataframe(df_cashflows[['Period', 'Time (Years)', 'Cash Flow', 'PV of Cash Flow']].style.format("{:.2f}"), height=400)
