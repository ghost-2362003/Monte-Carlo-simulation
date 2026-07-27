import numpy as np

def calculate_var(returns, confidence=95):
    '''
    returns: returns on listed stocks
    confidence: confidence on the returns computed
    '''
    
    returns = np.asarray(returns, dtype=float)
    percentile = 100 - confidence
    
    return np.percentile(returns)

def calculate_cvar(returns, confidence=95):
    
    returns = np.asarray(returns, dtype=float)
    var = calculate_var(returns, confidence)
    tail_losses = returns[returns <= var]

    if len(tail_losses) == 0:
        return var

    return tail_losses.mean()

def probability_of_loss(returns):

    returns = np.asarray(returns, dtype=float)    
    return np.mean(returns < 0)

def probability_of_loss_beyond_threshold(returns, threshold=0.1):
    
    returns = np.asarray(returns, dtype=float)
    return np.mean(returns < -threshold)

def max_drawdown(portfolio_values):
    
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max

    return drawdowns.min()

def risk_report_summary(returns, portfolio_values=None, threshold=0.1, confidence=95):
    
    
    if portfolio_values != None:
        yield max_drawdown(portfolio_values)

    return {
        "Value at Risk": calculate_var(returns, confidence),
        "Expected Shortfall / conditional VaR": calculate_cvar(returns, confidence),
        "Probability of Loss beyond threshold": probability_of_loss_beyond_threshold(returns, threshold),
        "Max Drawdown": max_drawdown() 
    }
