import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up the app title and input for stock ID
st.title("Stock Market Forecasting App")
st.title("Prediction")

# Input field for the user to enter a stock symbol (default is GOOG")
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set up the start and end dates for fetching historical data (last 20 years)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Function to fetch stock data with error handling and caching
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(stock, start, end):
    try:
        data = yf.download(stock, start, end, auto_adjust=False)  # Ensure 'Close' column is available
        if data.empty:
            st.error(f"No data found for {stock}. Check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Failed to download data for {stock}: {e}")
        return None

# Fetch historical stock data using yfinance
google_data = fetch_stock_data(stock, start, end)

# Ensure 'Close' column exists
if google_data is not None and not google_data.empty:
    if 'Close' not in google_data.columns:
        st.error("'Close' column not found in the stock data. Please check the stock symbol or try again later.")
    else:
        # Load the pre-trained LSTM model for prediction
        model = load_model("Latest_stock_price_model.keras")

        # Display the fetched stock data
        st.subheader("Stock Data")
        st.write(google_data)

        # Split the data for training and testing (70% training, 30% testing)
        splitting_len = int(len(google_data) * 0.7)
        x_test = google_data[['Close']][splitting_len:]

        # Function to plot graphs
        def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
            fig = plt.figure(figsize=figsize)
            plt.plot(values, 'Orange')  # Plot rolling average or any other values
            plt.plot(full_data['Close'], 'b')  # Plot original closing price
            if extra_data:
                plt.plot(extra_dataset)
            return fig

        # Plot Moving Averages
        for days in [250, 200, 100]:
            st.subheader(f'Original Close Price and MA for {days} days')
            ma_col = f'MA_for_{days}_days'
            google_data[ma_col] = google_data['Close'].rolling(days).mean()
            st.pyplot(plot_graph((15, 6), google_data[ma_col], google_data, 0))

        # Scale the test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))

        # Prepare input data
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])
        x_data, y_data = np.array(x_data), np.array(y_data)

        # Model predictions
        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Plot original vs predicted values
        st.subheader("Original values vs Predicted values")
        ploting_data = pd.DataFrame(
            {'original_test_data': inv_y_test.reshape(-1), 'predictions': inv_pre.reshape(-1)},
            index=google_data.index[splitting_len + 100:]
        )
        st.write(ploting_data)

        # Forecasting
        st.title("Forecast")
        days = st.text_input("Enter number of days to forecast within a Year", 30)
        if days.isdigit():
            days = int(days)
        else:
            st.error("Please enter a valid number.")

        # Prepare data for forecasting
        last_730_days = scaled_data[-730:].reshape(1, -1, 1)

        # Predict future days
        def predict_days(model, last_730_days, days=days):
            pred_list = []
            input_sequence = last_730_days
            for _ in range(days):
                next_pred = model.predict(input_sequence, verbose=0)
                pred_list.append(next_pred[0][0])
                input_sequence = np.append(input_sequence[:, 1:, :], [[next_pred[0]]], axis=1)
            return pred_list

        # Make predictions for the next N days
        next_days_scaled = predict_days(model, last_730_days)
        next_days = scaler.inverse_transform(np.array(next_days_scaled).reshape(-1, 1))

        # Plot forecast
        start_date = datetime.now()
        date_range = pd.date_range(start=start_date, periods=days, freq='B').date
        df = pd.DataFrame(data=next_days, index=date_range, columns=['forecast_price'])
        df_filtered = google_data.loc[google_data.index >= '2023-07-01', ['Close']]
        st.subheader(f'Forecasting for {days} days')
        fig = plt.figure(figsize=(15, 6))
        plt.plot(df_filtered.index, df_filtered['Close'], label='Close Price')
        plt.plot(df.index, df['forecast_price'], label='Forecasted Price')
        plt.legend()
        st.pyplot(fig)

        # "Buy Now" Button
        buy_url = f"https://www.indmoney.com/us-stocks/{stock}/apple-inc-share-price"
        if st.button("Buy Now", key="buy_now"):
            st.markdown(f"[Click here to Buy {stock}]( {buy_url} )", unsafe_allow_html=True)

        # "Stay Updated" Button
        stay_updated_url = "https://dalalstreetindia.com/"
        if st.button("Stay Updated", key="stay_updated"):
            st.markdown(f"[Stay Updated with Market Trends]( {stay_updated_url} )", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        ## Disclaimer
        The information provided by this app is for educational purposes only.
        Please do your own research before making any investment decisions.
        """)
else:
    st.error("No data available for the given stock symbol.")
