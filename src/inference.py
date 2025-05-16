import joblib
import pandas as pd

def predict_next_returns():
    model = joblib.load("models/model.joblib")
    df = pd.read_csv("data/stock_returns.csv", index_col=0)
    latest = df.iloc[-1].values.reshape(1, -1)
    return model.predict(latest)[0]

if __name__ == "__main__":
    print(predict_next_returns())

