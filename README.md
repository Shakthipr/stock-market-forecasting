 # Stock Market Forecasting App

A web application for forecasting stock prices using deep learning (LSTM) and interactive visualizations. The app allows users to analyze historical stock data, view moving averages, compare actual vs. predicted prices, and forecast future prices for any stock symbol.

## Features
- Fetches historical stock data using Yahoo Finance (yfinance)
- Visualizes stock price, moving averages (20, 50, 200 days)
- Predicts and forecasts future stock prices using a trained LSTM model
- Interactive web interface (Flask + Bootstrap)
- Date-based search for specific stock data
- Comparison table for actual vs. predicted prices
- Buy/Update stock quick links
- Educational disclaimer

## Demo
![App Screenshot](static/app_screenshot.png) <!-- Add a screenshot if available -->

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shakthipr/stock-market-forecasting.git
   cd stock-market-forecasting
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model file `Latest_stock_price_model.keras` and `scaler.npy` are present in the project root.

### Running the App
```bash
python app.py
```
The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Usage
- Enter a stock symbol (e.g., GOOG, AAPL) and number of days to forecast.
- Click **Analyze** to view stock data, moving averages, predictions, and forecasts.
- Use the **Buy Stock** and **Stock Update** buttons for quick access to market resources.
- Search for specific dates using the date search bar.

## File Structure
- `app.py` - Main Flask backend
- `templates/index.html` - Main web UI
- `static/` - Static assets (CSS, images)
- `Latest_stock_price_model.keras` - Trained LSTM model
- `scaler.npy` - Scaler for data normalization
- `requirements.txt` - Python dependencies

## Model Training
To retrain the model, use `train_model.py` (ensure you have sufficient historical data and adjust parameters as needed).

## Disclaimer
The information provided by this app is for educational purposes only. Please do your own research before making any investment decisions.

## License
This project is owned and licensed by shakthipr this project can be used for fair use only.

## Author
- [shakthipr](https://github.com/Shakthipr)


