import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Predictor Pro")

st.title("📈 Stock Predictor Pro (AI Based App)")
st.write("Enter any company stock to predict next price")

stock = st.text_input("Enter Stock Symbol (Example: TCS.NS, RELIANCE.NS, AAPL)")

if st.button("Predict"):

    if stock == "":
        st.warning("Please enter stock symbol")
    else:
        with st.spinner("Downloading data..."):
            data = yf.download(stock, period="2y")  # Increased to 2 years for MA200

        if data.empty:
            st.error("Invalid stock symbol or no data available")
        else:
            # Fix MultiIndex columns issue from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            st.subheader("📊 Latest Data")
            st.write(data.tail())

            # Chart
            st.subheader("📈 Price Chart")
            fig = plt.figure(figsize=(10, 4))
            plt.plot(data["Close"], label="Closing Price")
            plt.title(stock + " Closing Price")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

            # ML prediction
            data["Prediction"] = data["Close"].shift(-1)
            data_clean = data.dropna()

            if len(data_clean) < 2:
                st.error("Not enough data for prediction")
            else:
                # Use .values to ensure proper numpy array conversion
                X = data_clean[["Close"]].values
                y = data_clean["Prediction"].values

                model = LinearRegression()
                model.fit(X, y)

                # Get last price and ensure it's 2D array
                last_close = float(data["Close"].iloc[-1])
                last_price = np.array([[last_close]])
                
                prediction = model.predict(last_price)

                # Detect currency
                currency = "₹" if stock.endswith((".NS", ".BO")) else "$"
                
                st.success(f"📌 Next Day Predicted Price: {currency}{prediction[0]:.2f}")
                
                # Show confidence metric
                current_price = data["Close"].iloc[-1]
                change_pct = ((prediction[0] - current_price) / current_price) * 100
                st.info(f"Expected change: {change_pct:+.2f}%")

            # Buy/Sell signal
            if len(data) >= 200:
                data["MA50"] = data["Close"].rolling(50).mean()
                data["MA200"] = data["Close"].rolling(200).mean()

                if pd.notna(data["MA50"].iloc[-1]) and pd.notna(data["MA200"].iloc[-1]):
                    if data["MA50"].iloc[-1] > data["MA200"].iloc[-1]:
                        st.success("📢 BUY Signal (Golden Cross)")
                    else:
                        st.error("📢 SELL Signal (Death Cross)")
                else:
                    st.warning("⚠️ Not enough data for moving average signals")
            else:
                st.warning(f"⚠️ Need 200+ days of data for signals (have {len(data)} days)")