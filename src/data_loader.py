import yfinance as yf
import pandas as pd

def download_tickers(stocks, start_date):
    data = yf.download(stocks,
                   start = start_date,
                   auto_adjust = True
                   )["Close"]
    
    data.to_csv("/data")
    
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