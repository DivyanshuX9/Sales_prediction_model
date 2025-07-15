import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import plotly.express as px
import time

# Load pre-trained model and scaler
model = joblib.load('hyper_tuned_sales_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load Walmart dataset for display
walmart_df = pd.read_csv('walmart.csv')

# Feature engineering for graph
walmart_df['Date'] = pd.to_datetime(walmart_df['Date'], format='%d-%m-%Y')
walmart_df['Month'] = walmart_df['Date'].dt.month
monthly_sales = walmart_df.groupby('Month')['Weekly_Sales'].sum().reset_index()

# Streamlit App
st.set_page_config(page_title="Walmart Sales Predictor", page_icon="🛒", layout="wide")

# Subtle animation (confetti on load)
st.balloons()

# Custom CSS for dark theme and white text
st.markdown("""
    <style>
    body, .stApp {
        background: #181818 !important;
        color: #fff !important;
    }
    .big-font { font-size:28px !important; font-weight: 700; color: #fff !important; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; padding: 0.5rem 2rem; color: #fff !important; }
    .stButton>button { background-color: #ff6600; color: white; font-size: 18px; border-radius: 8px; }
    .stButton>button:hover { background-color: #ff944d; }
    .stDataFrame { border-radius: 10px; color: #fff !important; }
    .stMarkdown, .stHeader, .stText, .stDataFrame, .stTable, .stCode, .stInfo, .stSuccess, .stError, .stWarning {
        color: #fff !important;
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div>div {
        background: #222 !important;
        color: #fff !important;
    }
    .stDataFrame, .stTable { background: #222 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">🛒 Walmart Weekly Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:18px; color:#fff;">Predict weekly sales based on store inputs or your own dataset. Visualize sales trends and more!</div>', unsafe_allow_html=True)

# Tabs for navigation
main_tab, data_tab, upload_tab, graph_tab = st.tabs([
    "Single Prediction", "View Walmart Dataset", "Upload Your Dataset", "Sales Trend Graph"
])

with main_tab:
    st.header("🔢 Single Prediction")
    # Take date input and extract features
    date_input = st.date_input("Select a Date")
    year = date_input.year
    month = date_input.month
    week = date_input.isocalendar()[1]

    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.number_input("Store ID", min_value=1, step=1)
        holiday_flag = st.selectbox("Is it a Holiday Week?", options=[0, 1])
    with col2:
        temperature = st.number_input("Average Temperature (°F)")
        fuel_price = st.number_input("Fuel Price (USD)")
    with col3:
        cpi = st.number_input("Consumer Price Index")
        unemployment = st.number_input("Unemployment Rate (%)")

    predict_btn = st.button("Predict Sales", use_container_width=True)
    if predict_btn:
        with st.spinner('Fetching prediction...'):
            time.sleep(1.5)  # Simulate loading
            input_data = pd.DataFrame([[store, holiday_flag, temperature, fuel_price, cpi, unemployment, year, month, week]],
                                       columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week'])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            st.success(f"🧾 Predicted Weekly Sales: ${prediction:,.2f}")
            st.snow()

with data_tab:
    st.header("📊 Walmart Dataset Preview")
    st.dataframe(walmart_df, use_container_width=True)

with upload_tab:
    st.header("📁 Upload Your Own Dataset for Batch Prediction")
    st.write("Upload a CSV file with the following columns:")
    st.code("Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Week")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        required_cols = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week']
        if all(col in user_df.columns for col in required_cols):
            with st.spinner('Generating predictions for your dataset...'):
                time.sleep(1.5)
                user_scaled = scaler.transform(user_df[required_cols])
                user_df['Predicted_Weekly_Sales'] = model.predict(user_scaled)
                st.success("Predictions generated! See below:")
                st.dataframe(user_df, use_container_width=True)
                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", data=csv, file_name="predicted_sales.csv", mime="text/csv")
        else:
            st.error(f"Uploaded file must contain columns: {', '.join(required_cols)}")

with graph_tab:
    st.header("📈 Monthly Sales Trend (All Stores)")
    # Add a reset zoom button
    reset = st.button("🔄 Reset Zoom", key="reset_zoom")
    fig = px.line(monthly_sales, x='Month', y='Weekly_Sales', markers=True,
                  title='Total Walmart Weekly Sales by Month',
                  labels={'Month': 'Month', 'Weekly_Sales': 'Total Weekly Sales'},
                  template='plotly_dark')
    # Highlight dips
    min_idx = monthly_sales['Weekly_Sales'].idxmin()
    min_month = monthly_sales.loc[min_idx, 'Month']
    min_sales = monthly_sales.loc[min_idx, 'Weekly_Sales']
    fig.add_scatter(x=[min_month], y=[min_sales], mode='markers+text',
                    marker=dict(size=16, color='red'),
                    text=[f"Dip: {min_sales:,.0f}"], textposition="top center",
                    name='Sales Dip')
    # If reset button is pressed, re-create the figure (Streamlit reruns the script)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"The lowest sales occur in month {min_month}.")



