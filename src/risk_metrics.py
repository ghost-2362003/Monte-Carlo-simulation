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

