import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import io
import base64
import logging
import time
import threading

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a lock for thread safety
plot_lock = threading.Lock()

def create_plot():
    plt.figure(figsize=(10, 6))
    return plt

def close_plot(plt):
    plt.close()

def fetch_stock_data(stock, start, end):
    try:
        # Create a Ticker object
        ticker = yf.Ticker(stock)
        
        # Get historical data with more detailed error handling
        try:
            data = ticker.history(start=start, end=end, auto_adjust=True)
        except Exception as e:
            logging.error(f"Failed to fetch history for {stock}: {str(e)}")
            return None
        
        if data is None:
            logging.error(f"Received None data for {stock}")
            return None
            
        if data.empty:
            logging.error(f"No data found for {stock}. Please verify the stock symbol.")
            return None
            
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns for {stock}: {', '.join(missing_columns)}")
            return None
            
        # Fill any missing values
        data = data.fillna(method='ffill')
        
        # Additional validation
        if len(data) < 2:
            logging.error(f"Insufficient data points for {stock}. Need at least 2 data points.")
            return None
            
        # Log success
        logging.info(f"Successfully fetched data for {stock} from {data.index[0]} to {data.index[-1]}")
        
        return data
    except Exception as e:
        logging.error(f"Unexpected error while fetching data for {stock}: {str(e)}")
        return None

def plot_graph(values, full_data, extra_data=0, extra_dataset=None):
    plt.figure(figsize=(15, 6))
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def predict_days(model, last_730_days, days):
    try:
        pred_list = []
        input_sequence = last_730_days.copy()  # Make a copy to avoid modifying original
        
        for _ in range(days):
            # Ensure input sequence has correct shape
            if input_sequence.shape[1] < 100:
                logging.error("Input sequence too short for prediction")
                return None
                
            # Get prediction
            next_pred = model.predict(input_sequence, verbose=0)
            pred_list.append(next_pred[0][0])
            
            # Update input sequence for next prediction
            input_sequence = np.append(input_sequence[:, 1:, :], 
                                     [[next_pred[0]]], axis=1)
        
        return pred_list
    except Exception as e:
        logging.error(f"Error in predict_days: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        logging.info("Starting stock analysis...")
        stock = request.form.get('stock', 'GOOG').upper().strip()
        days = int(request.form.get('days', 30))
        
        logging.info(f"Processing request for stock: {stock}, days: {days}")
        
        # Validate stock symbol - allow up to 6 characters for some international stocks
        if not stock or len(stock) < 1 or len(stock) > 6:
            logging.error(f"Invalid stock symbol: {stock}")
            return jsonify({
                'error': 'Invalid stock symbol. Please enter a valid stock symbol (1-6 characters).'
            })
        
        # Validate days
        if days < 1 or days > 365:
            logging.error(f"Invalid number of days: {days}")
            return jsonify({
                'error': 'Invalid number of days. Please enter a number between 1 and 365.'
            })
        
        # Set up dates - fetch 3 years of data
        end = datetime.now()
        start = datetime(end.year - 3, end.month, end.day)
        logging.info(f"Fetching data from {start} to {end}")
        
        # Fetch data with retry
        max_retries = 3
        retry_count = 0
        stock_data = None
        
        while retry_count < max_retries and stock_data is None:
            try:
                stock_data = fetch_stock_data(stock, start, end)
                if stock_data is None:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.info(f"Retry {retry_count} for {stock}")
                        time.sleep(1)  # Wait 1 second before retry
            except Exception as e:
                logging.error(f"Error during data fetch attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
        
        if stock_data is None or stock_data.empty:
            logging.error(f"No data available for {stock} after {max_retries} attempts")
            return jsonify({
                'error': f'Unable to fetch data for {stock}. Please check if the stock symbol is correct and try again.'
            })
        
        logging.info(f"Successfully fetched {len(stock_data)} data points for {stock}")
        
        # Load model
        try:
            logging.info("Loading prediction model...")
            model = load_model("Latest_stock_price_model.keras")
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return jsonify({
                'error': 'Error loading the prediction model. Please try again later.'
            })
        
        # Calculate moving averages
        logging.info("Calculating moving averages...")
        ma_plots = {}
        with plot_lock:
            for window in [20, 50, 200]:
                if len(stock_data) >= window:
                    ma = stock_data['Close'].rolling(window=window).mean()
                    plt = create_plot()
                    plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
                    plt.plot(stock_data.index, ma, label=f'{window}-day MA')
                    plt.title(f'{stock} {window}-day Moving Average')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot to base64 string
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode()
                    ma_plots[f'ma_{window}'] = plot_data
                    close_plot(plt)
        
        # Prepare data for prediction
        logging.info("Preparing data for prediction...")
        try:
            splitting_len = int(len(stock_data) * 0.7)
            x_test = stock_data[['Close']][splitting_len:]
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))
            
            # Prepare input data
            x_data, y_data = [], []
            for i in range(100, len(scaled_data)):
                x_data.append(scaled_data[i - 100:i])
                y_data.append(scaled_data[i])
            x_data, y_data = np.array(x_data), np.array(y_data)
            
            logging.info(f"Prepared {len(x_data)} sequences for prediction")
        except Exception as e:
            logging.error(f"Error preparing prediction data: {str(e)}")
            return jsonify({
                'error': 'Error preparing data for prediction. Please try again.'
            })
        
        # Make predictions
        logging.info("Making predictions...")
        try:
            predictions = model.predict(x_data)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return jsonify({
                'error': 'Error making predictions. Please try again.'
            })
        
        # Prepare forecast
        logging.info("Preparing forecast...")
        if len(scaled_data) < 100:
            logging.error(f"Insufficient data points: {len(scaled_data)} < 100")
            return jsonify({
                'error': 'Not enough historical data for forecasting. Please try a different stock.'
            })
        
        try:
            last_100_days = scaled_data[-100:].reshape(1, -1, 1)
            next_days_scaled = predict_days(model, last_100_days, days)
            
            if next_days_scaled is None:
                logging.error("Failed to generate forecast")
                return jsonify({
                    'error': 'Error generating forecast. Please try again.'
                })
            
            next_days = scaler.inverse_transform(np.array(next_days_scaled).reshape(-1, 1))
        except Exception as e:
            logging.error(f"Error preparing forecast: {str(e)}")
            return jsonify({
                'error': 'Error preparing forecast. Please try again.'
            })
        
        # Create forecast plot
        logging.info("Creating forecast plot...")
        try:
            # Get the last trading day from the stock data
            last_trading_day = stock_data.index[-1]
            
            # Generate business days for the forecast period
            date_range = pd.bdate_range(start=last_trading_day, periods=days+1)[1:]
            
            # Create DataFrame with forecast data
            df = pd.DataFrame(data=next_days, index=date_range, columns=['forecast_price'])
            
            # Get recent data for comparison
            recent_data = stock_data.tail(100)  # Last 100 days
            
            plt = create_plot()
            plt.plot(recent_data.index, recent_data['Close'], label='Historical Close Price')
            plt.plot(df.index, df['forecast_price'], label='Forecasted Price', linestyle='--')
            plt.title(f'{stock} Stock Price Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Save forecast plot
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            forecast_plot = base64.b64encode(buffer.getvalue()).decode()
            close_plot(plt)
            
            # Format dates for JSON response
            forecast_dates = [d.strftime('%Y-%m-%d') for d in date_range]
            
            # Format stock data dates
            stock_data_dates = [d.strftime('%Y-%m-%d') for d in stock_data.tail(10).index]
            stock_data_values = stock_data.tail(10).to_dict('records')
            formatted_stock_data = []
            for date, data in zip(stock_data_dates, stock_data_values):
                data['Date'] = date
                formatted_stock_data.append(data)
            
        except Exception as e:
            logging.error(f"Error creating forecast plot: {str(e)}")
            return jsonify({
                'error': 'Error creating forecast visualization. Please try again.'
            })
        
        logging.info("Analysis completed successfully")
        return jsonify({
            'stock_data': formatted_stock_data,
            'ma_plots': ma_plots,
            'forecast_plot': forecast_plot,
            'predictions': inv_pre.reshape(-1).tolist(),
            'actual_values': inv_y_test.reshape(-1).tolist(),
            'forecast_dates': forecast_dates,
            'forecast_prices': next_days.reshape(-1).tolist()
        })
    except Exception as e:
        logging.error(f"Unexpected error during analysis: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred during analysis: {str(e)}'
        })

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        date = data.get('date')
        
        # Convert date string to datetime
        search_date = datetime.strptime(date, '%d-%m-%y')
        
        # Fetch stock data using yfinance
        stock = yf.Ticker(symbol)
        hist = stock.history(start=search_date, end=search_date + timedelta(days=1))
        
        if hist.empty:
            return jsonify({'error': 'No data available for this date'})
            
        # Get the first (and only) row of data
        stock_data = {
            'Date': hist.index[0].strftime('%Y-%m-%d'),
            'Open': float(hist['Open'][0]),
            'High': float(hist['High'][0]),
            'Low': float(hist['Low'][0]),
            'Close': float(hist['Close'][0]),
            'Volume': int(hist['Volume'][0])
        }
        
        return jsonify({'stock_data': stock_data})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
