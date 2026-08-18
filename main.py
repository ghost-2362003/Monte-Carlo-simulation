import numpy as np

from src.data_loader import (
    download_tickers,
    load_data
)
from src.returns import calculate_log_returns
from src.monte_carlo import simulate_correlated_paths
from src.portfolio import (
    calculate_portfolio_value_path,
    calculate_terminal_portfolio_values
)
from src.risk_metrics import (
    calculate_var,
    calculate_cvar,
    probability_of_loss
)

def main():
    # -------------------------
    # 1. Portfolio configuration
    # -------------------------

    stocks = ["AAPL", "MSFT", "NVDA"]
    weights = np.array([0.4, 0.3, 0.3])

    initial_capital = 1000000000

    days = 252
    simulations = 10000
    
    # -------------------------
    # 2. Load historical data
    # -------------------------

    download_tickers(stocks=stocks, start_date="2026-07-01")
    data = load_data(path="/data")
    
    # -------------------------
    # 3. Calculate returns
    # -------------------------

    log_returns = calculate_log_returns(data)
 
    mu = log_returns.mean().values * days

    cov_matrix = log_returns.cov().values * days
    
    # -------------------------
    # 4. Initial stock prices
    # -------------------------

    S0 = data.iloc[-1]
    
if __name__ == "__main__":
    main()