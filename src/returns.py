import numpy as np
import pandas as pd

def calculate_log_returns(ticker_data):
    return np.log(ticker_data/ticker_data.shift(1)).dropna()

def calculate_simple_returns(ticker_data):
    return ticker_data.pct_change().dropna()

def calculate_mean_returns(ticker_data):
    return ticker_data.pct_change().dropna().mean()

