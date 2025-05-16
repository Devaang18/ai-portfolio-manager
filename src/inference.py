import pandas as pd
import joblib
from portfolio_optimiser import optimise_portfolio

def predict_next_returns():
    model = joblib.load("models/model.joblib")
    df = pd.read_csv("data/stock_returns.csv", index_col=0)
    latest = df.iloc[-1].values.reshape(1, -1)
    predicted = model.predict(latest)[0]
    return predicted

if __name__ == "__main__":
    predicted = predict_next_returns()
    weights = optimise_portfolio(predicted)
    print("Optimized Portfolio Weights:")
    print(weights)
