import numpy as np


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("Weights cannot sum to zero.")
    return weights / total


def calculate_portfolio_return(weights, mean_returns):
    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)
    return np.dot(weights, mean_returns)


def calculate_portfolio_volatility(weights, cov_matrix):
    weights = np.asarray(weights, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    return np.sqrt(weights.T @ cov_matrix @ weights)


def calculate_portfolio_value_path(price_paths, weights, initial_value=100000):
    """
    price_paths shape: (days + 1, simulations, n_assets)
    weights shape: (n_assets,)
    Returns: portfolio value paths of shape (days + 1, simulations)
    """
    weights = np.asarray(weights, dtype=float)

    if price_paths.shape[2] != len(weights):
        raise ValueError("Number of weights must match number of assets.")

    intial_prices = price_paths[0]
    relative_prices = price_paths / intial_prices
    weighted_prices = relative_prices @ weights
    portfolio_values = initial_value * weighted_prices
    
    return portfolio_values


def calculate_terminal_portfolio_values(portfolio_paths):
    return portfolio_paths[-1]