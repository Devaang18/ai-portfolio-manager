from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

def train_model():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
    x = df.shift(1).dropna()
    y = df.loc[x.index]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    joblib.dump(model, "models/model.joblib")


if __name__ == "__main__":
    train_model()