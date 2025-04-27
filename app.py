import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import joblib
import datetime

# Load and clean data
df = pd.read_csv("walmart.csv")
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df.drop(columns=['Date'], inplace=True)

# Splitting dataset
X = df.drop(columns=['Weekly_Sales'])
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
]

# Define meta model
meta_model = LinearRegression()

# Create stacking regressor
hybrid_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
hybrid_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(hybrid_model, 'sales_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit App
st.title("🛒 Walmart Weekly Sales Predictor")
st.write("Predict weekly sales based on store inputs")

# Take date input and extract features
date_input = st.date_input("Select a Date")
year = date_input.year
month = date_input.month
week = date_input.isocalendar()[1]

store = st.number_input("Store ID", min_value=1, step=1)
holiday_flag = st.selectbox("Is it a Holiday Week?", options=[0, 1])
temperature = st.number_input("Average Temperature (°F)")
fuel_price = st.number_input("Fuel Price (USD)")
cpi = st.number_input("Consumer Price Index")
unemployment = st.number_input("Unemployment Rate (%)")

if st.button("Predict Sales"):
    # Load model and scaler
    model = joblib.load('sales_model.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data = pd.DataFrame([[store, holiday_flag, temperature, fuel_price, cpi, unemployment, year, month, week]],
                               columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"🧾 Predicted Weekly Sales: ${prediction:,.2f}")



