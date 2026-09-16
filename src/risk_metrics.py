import numpy as np

def calculate_var(returns, confidence=99):
    '''
    returns: returns on listed stocks
    confidence: confidence on the returns computed
    '''
    
    returns = np.asarray(returns, dtype=float)
    percentile = 100 - confidence
    
    return np.percentile(returns, percentile)

def calculate_cvar(returns, confidence=99):
    
    returns = np.asarray(returns, dtype=float)
    var = calculate_var(returns, confidence)
    tail_losses = returns[returns <= var]

    if len(tail_losses) == 0:
        return var

    return tail_losses.mean()

def probability_of_loss(returns):

    returns = np.asarray(returns, dtype=float)    
    return np.mean(returns < 0)

def probability_of_loss_beyond_threshold(returns, threshold=0.01):
    
    returns = np.asarray(returns, dtype=float)
    return np.mean(returns < -threshold)

def max_drawdown(portfolio_values=None):
    
    if portfolio_values == None:
        return None
    
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max

    return drawdowns.min()

def risk_report_summary(returns, portfolio_values=None, threshold=0.0000001, confidence=95):
    
    
    ''' if portfolio_values != None:
        yield max_drawdown(portfolio_values)
    '''
    
    return {
        "Value at Risk": round(float(calculate_var(returns, confidence)), 4),
        "Expected Shortfall / conditional VaR": round(float(calculate_cvar(returns, confidence)), 4),
        "Probability of Loss beyond threshold": round(float(probability_of_loss_beyond_threshold(returns, threshold)), 4),
        "Max Drawdown": max_drawdown() 
    }
