import pandas as pd

def calculate_returns(filepath):
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    returns.to_csv("data/stock_returns.csv")
    print("Returns calculated and saved")
    return returns

if __name__ == "__main__":
    calculate_returns("data/stock_data.csv")
