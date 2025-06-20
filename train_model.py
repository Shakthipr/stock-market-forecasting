import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def fetch_training_data(symbol='AAPL', years=5):
    """Fetch historical data for training"""
    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)
    
    data = yf.download(symbol, start=start, end=end, progress=False)
    return data

def prepare_data(data, sequence_length=100):
    """Prepare data for LSTM model"""
    # Use only the 'Close' prices
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i])
        y.append(scaled_prices[i])
    
    return np.array(X), np.array(y), scaler

def create_model(sequence_length):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model():
    """Main function to train and save the model"""
    try:
        # Fetch data
        logging.info("Fetching training data...")
        data = fetch_training_data()
        
        if data.empty:
            raise Exception("No data fetched")
        
        # Prepare data
        logging.info("Preparing data...")
        X, y, scaler = prepare_data(data)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train model
        logging.info("Creating and training model...")
        model = create_model(sequence_length=100)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save the model
        logging.info("Saving model...")
        model.save("Latest_stock_price_model.keras")
        logging.info("Model saved successfully!")
        
        # Save the scaler for later use
        np.save('scaler.npy', scaler)
        logging.info("Scaler saved successfully!")
        
        return True
        
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 