from src.data_loader import download_tickers, load_data
from src.monte_carlo import simulate_single_path
import numpy as np

def increase_volatility(sigma, factor=2):
    '''
    Increase the volatility of the simulator
    factor: how many times the volatility should change 
    '''

    return sigma*factor

def change_expected_return(mu, factor=3):
    '''
    Simulate a bearish or bullish market
    factor: factors by which the returns should change for loong term
            -ve: simulates a bearish market
            +ve: simulates a bullish market
    '''
    
    return mu*factor

def simulate_market_crash(stock, start_date, 
                          mu, sigma,
                          mu_factor, sigma_factor,c=0.6):
    
    download_tickers(stocks=stock, 
                     start_date=start_date)
    
    data = load_data(path="/data")
    data.loc[data.index[-1], stock] = data.loc[data.index[-1], stock] * (1 - c)
    log_returns = calculate_log_returns(data)
    S0 = data[stock].iloc[-1]
    
    mu = change_expected_return(mu, mu_factor)
    sigma = increase_volatility(sigma, sigma_factor)
    
    path = simulate_single_path(S0, mu, sigma)
    return path