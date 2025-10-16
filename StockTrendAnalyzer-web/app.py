import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------
# Technical Indicator Functions
# ----------------------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def compute_bollinger_bands(data, window=20):
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    data['BB_Middle'] = middle_band
    data['BB_Upper'] = middle_band + 2 * std_dev
    data['BB_Lower'] = middle_band - 2 * std_dev
    return data

# ----------------------
# Streamlit Interface
# ----------------------
st.title("📈 Stock Trend Analyzer Web App")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button("Analyze"):
    with st.spinner("Fetching and analyzing data..."):
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            st.error("No data found. Try another stock symbol or date range.")
        else:
            # Compute Features
            data['MA10'] = data['Close'].rolling(10).mean()
            data['MA20'] = data['Close'].rolling(20).mean()
            data['RSI'] = compute_rsi(data)
            data = compute_macd(data)
            data = compute_bollinger_bands(data)
            data['Price_Change'] = data['Close'].pct_change()
            data.dropna(inplace=True)

            if len(data) < 30:
                st.error("Not enough data to analyze. Use a longer date range.")
            else:
                features = ['MA10','MA20','RSI','MACD','Signal_Line',
                            'BB_Middle','BB_Upper','BB_Lower','Price_Change','Volume']
                X = data[features]
                y = data['Close']

                # Train RandomForest model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X_train, y_train)

                # Predict next day
                last_row = data[features].iloc[[-1]].values
                next_day_price = float(model.predict(last_row))
                last_close = float(data['Close'].iloc[-1])

                trend = "Uptrend 📈" if next_day_price > last_close else "Downtrend 📉" if next_day_price < last_close else "Stable ➡️"
                rsi_value = float(data['RSI'].iloc[-1])
                rsi_status = "Overbought ⚠️" if rsi_value > 70 else "Oversold 💰" if rsi_value < 30 else "Neutral ✅"

                # Display Results
                st.subheader("Prediction")
                st.write(f"Last Closing Price: ${last_close:.2f}")
                st.write(f"Predicted Next-Day Price: ${next_day_price:.2f}")
                st.write(f"Trend: {trend}")
                st.write(f"RSI Status: {rsi_status}")

                # Plot
                st.subheader("Actual vs Predicted Prices")
                plt.figure(figsize=(10,5))
                plt.plot(data.index, data['Close'], label='Actual Close', color='skyblue')
                plt.plot(data.index, model.predict(X), label='Predicted Close', color='orange')
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.legend()
                st.pyplot(plt)
