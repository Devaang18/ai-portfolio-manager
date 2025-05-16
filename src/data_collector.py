import yfinance as yf
import pandas as pd
from datetime import datetime

def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close']


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    df = get_data(tickers, '2020-01-01', datetime.today('%Y-%m-%d'))
    df.to_csv("data/stock_data.csv")
    print("Data downloaded and saved")

