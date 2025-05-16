import pandas as pd
import numpy as np
import joblib

def optimise_portfolio(predicted_returns):
    predicted_returns = np.maximum(predicted_returns, 0)
    weights = predicted_returns / np.sum(predicted_returns)
    return weights
