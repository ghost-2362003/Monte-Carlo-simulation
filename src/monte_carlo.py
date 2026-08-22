import numpy as np

def simulate_single_path(intial_price,
                         mu, sigma, 
                         days=252):
    
    t = 1/days
    path = [intial_price]
    
    for day in range(days):
        z = np.random.normal()
        
        next_price = path[-1]*np.exp(       ## Geometric Brownian Motion (GBM) in action
        (mu - sigma**2/2)*t + 
        sigma*np.sqrt(t)*z)
        
        path.append(next_price)
        
    return path
        
def simulate_price_paths(intial_price,
                         mu, sigma, 
                         simulations, days=252):
    
    t = 1/days
    paths = np.zeros((days, simulations))
    paths[0] = intial_price
    
    for sim in simulations:
        for day in range(1, days+1):
            
            z = np.random.normal()
            
            paths[day, sim] = paths[day-1, sim]*np.exp(
                (mu - sigma**2/2)*t + 
                sigma*np.sqrt(t)*z
            )
            
    return paths
            
def simulate_multiple_stocks(intial_price,
                             expected_returns,
                             volatilies,
                             days, simulations):
    
    paths = np.zeros((len(expected_returns), days, simulations))
    paths[0] = intial_price
    
    for stock in range(len(expected_returns)):
        paths[stock] = simulate_price_paths(intial_price=intial_price[stock],
                                            mu=expected_returns[stock],
                                            sigma=volatilies[stock],
                                            simulations=simulations)
        
    return paths

def simulate_correlated_paths(S0, mu, cov_matrix, days, simulations, dt=1/252, seed=None):
    """
    Simulate correlated stock price paths using Geometric Brownian Motion.

    Parameters
    ----------
    S0 : array-like
        Initial prices of the stocks, shape (n_assets,)
    mu : array-like
        Expected annual returns for each stock, shape (n_assets,)
    cov_matrix : array-like
        Annual covariance matrix of returns, shape (n_assets, n_assets)
    days : int
        Number of trading days to simulate
    simulations : int
        Number of Monte Carlo simulations
    dt : float
        Time step in years. Default = 1/252
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Simulated paths with shape (days + 1, simulations, n_assets)
    """
    if seed is not None:
        np.random.seed(seed)

    S0 = np.asarray(S0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    n_assets = len(S0)

    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError("cov_matrix must have shape (n_assets, n_assets)")

    # Cholesky factor of covariance matrix
    L = np.linalg.cholesky(cov_matrix)

    # Store paths: time x simulations x assets
    paths = np.zeros((days, simulations, n_assets))
    paths[0] = S0

    # Drift term for each asset
    drift = (mu - 0.5 * np.diag(cov_matrix)) * dt

    # Simulate day by day
    for t in range(1, days):
        # Independent standard normals: shape (simulations, n_assets)
        Z = np.random.normal(size=(simulations, n_assets))

        # Correlated shocks
        correlated_shocks = Z @ L.T * np.sqrt(dt)

        # GBM update
        paths[t] = paths[t - 1] * np.exp(drift + correlated_shocks)

    return paths