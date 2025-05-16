from src.portfolio_optimiser import optimise_portfolio
import numpy as np

def test_weights_sum_to_one():
    predicted_returns = np.array([0.2, 0.3, 0.5])
    weights = optimise_portfolio(predicted_returns)
    assert abs(weights.sum()) - 1.0 < 1e-6, "Weights do not sum to 1"


    