import yfinance as yf
import pandas as pd
from pathlib import Path

def download_tickers(stocks, start_date):
    data = yf.download(stocks,
                   start = start_date,
                   auto_adjust = True
                   )["Close"]
    
    # Path("data").mkdir(exist_ok=True)
    data.to_csv("data/data.csv")
    
def load_data(path):
    return pd.read_csv(path,
            index_col=0,
            parse_dates=True
            )
    
def validate_data(data):
    """Basic validation of downloaded data."""
    if data.empty:
        raise ValueError("No data was downloaded.")

    if data.isnull().all().any():
        raise ValueError("One or more tickers contain only missing values.")

    return True

def save_data(data, filename):
    data.to_csv(filename)