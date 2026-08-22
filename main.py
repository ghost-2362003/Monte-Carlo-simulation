import numpy as np
import pandas as pd
from src.data_loader import (
    download_tickers,
    load_data
)
from src.returns import (
    calculate_log_returns,
    calculate_simple_returns
)
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

    download_tickers(stocks=stocks, start_date="2016-07-01")
    data = load_data(path="data/data.csv")
    
    # -------------------------
    # 3. Calculate returns
    # -------------------------

    log_returns = calculate_log_returns(data)
 
    mu = log_returns.mean().values * days

    cov_matrix = log_returns.cov().values * days
    
    print("mu:", mu)
    print("annual volatility:", np.sqrt(np.diag(cov_matrix)))
    print("covariance matrix:")
    print(cov_matrix)
    
    # -------------------------
    # 4. Initial stock prices
    # -------------------------

    S0 = data.iloc[-1].to_numpy()
    
    # -------------------------
    # 5. Monte Carlo simulation
    # -------------------------
    
    # np.random.seed(42)
    paths = simulate_correlated_paths(
        S0,
        mu,
        cov_matrix,
        days,
        simulations
    )
    print("paths shape:", paths.shape)
    print("paths min:", np.min(paths))
    print("paths max:", np.max(paths))
    print("paths first:", paths[0].shape)
    print("paths last:", paths[-1].shape)
    
    # -------------------------
    # 6. Portfolio simulation
    # -------------------------

    portfolio_paths = calculate_portfolio_value_path(
        paths,
        weights,
        initial_capital
    )
    
    # -------------------------
    # 7. Terminal values
    # -------------------------

    terminal_values = calculate_terminal_portfolio_values(
        portfolio_paths
    )
    
    print("Portfolio paths shape:", portfolio_paths.shape)
    print("Initial portfolio values:", portfolio_paths[0, :10])
    print("Terminal values:", terminal_values[:10])
    # -------------------------
    # 8. Portfolio returns
    # -------------------------
    
    portfolio_returns = (terminal_values - initial_capital) / initial_capital
    
    print("portfolio returns:", portfolio_returns)
    print("Mean:", np.mean(portfolio_returns))
    print("Std:", np.std(portfolio_returns))
    print("Min:", np.min(portfolio_returns))
    print("Max:", np.max(portfolio_returns))
     
    # -------------------------
    # 9. Risk metrics
    # -------------------------

    var = calculate_var(portfolio_returns)

    cvar = calculate_cvar(portfolio_returns)

    probability_loss = probability_of_loss(
        portfolio_returns
    )


    # -------------------------
    # 10. Display results
    # -------------------------

    print("\n===== Portfolio Risk Analysis =====")

    print(f"VaR (95%): {var:.2%}")

    print(f"CVaR (95%): {cvar:.2%}")

    print(
        f"Probability of Loss: "
        f"{probability_loss:.2%}"
    )
    
if __name__ == "__main__":
    main()