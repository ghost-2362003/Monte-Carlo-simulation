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

