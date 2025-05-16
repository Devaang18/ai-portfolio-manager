import pandas as pd
import numpy as np
import joblib

def optimise_portfolio(predicted_returns):
    predicted_returns = np.maximum(predicted_returns, 0)
    weights = predicted_returns / np.sum(predicted_returns)
    return weights

model = joblib.load("models/model.joblib")
df = pd.read_csv("data/stock_returns.csv", index_col=0)
latest = df.iloc[-1].values.reshape(1, -1)
predicted = model.predict(latest)[0]


weights = optimise_portfolio(predicted)
print("Optimised Portfolio Weights:", weights)